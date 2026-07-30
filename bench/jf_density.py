"""S1 Stage-0 probe: forced-run (jump-forward) density measurement.

Answers, BEFORE any serving delivery is built, whether singleton-mask chains
are frequent enough to be worth injecting as draft tokens (the S1 decision
gate: proceed if forced steps >= ~5% of decode steps on at least one target
workload; publish the histogram either way).

Method: replay JSONSchemaBench-style test instances through guide states —
the maskbench replay shape (bench/maskbench_grid.py), not model generation.
Along a VALID instance the replay token at a forced position is by definition
the forced token (it is in the mask and the mask is a singleton), so forced
density and run lengths measured along the replay equal what any generation
producing that output would see. Per step:

- forced step  := mask (including eos when the state can terminate) has
  cardinality 1 and its element is not eos — exactly the condition under
  which GridGuide.get_next_instruction emits a Write span (guide.py:166);
- runs         := maximal sequences of consecutive forced steps; the length
  histogram is reported against j_max caps {4, 8, 16, 32} as the saved-step
  fraction: a run of length L verified as drafts of <= j tokens per step
  costs ceil(L/j) steps instead of L (each verification rides one step);
- byte-run leg := the coverage GAP between token-level chains and the
  deferred byte-level jump-forward (DESIGN.md §12: v2, re-tokenization):
  at each byte of the instance, the byte is FORCED iff it is the unique
  viable next byte and the state cannot terminate. Viability mirrors
  guide._advance's byte machinery (lexer.advance + event shifts + the
  partial-lexeme tail rule) — measurement-grade copy, not a product path.
  Reported: bytes inside forced-byte runs vs bytes covered by forced TOKEN
  runs; their ratio bounds what the token-level lever captures of the
  byte-level ceiling.

Workloads:
- JSONSchemaBench splits (local: tmp/jsb-src/data synced in this tree);
- Spider SQL replay needs the Spider dataset + generations — NOT in this
  repo; run on the GPU box alongside the serving bake (--help stays honest).

Run (local, gpt2 is the cached tokenizer; the box run should pass the
serving tokenizer, e.g. Qwen/Qwen2.5-0.5B-Instruct):

  .venv-bench/bin/python bench/jf_density.py --data tmp/jsb-src/data \
      --sample 6 --tokenizer gpt2 --out tmp/jf-density.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))

J_CAPS = (4, 8, 16, 32)
RUN_BUCKETS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32)  # ">=last" tail bucket


# ------------------------------------------------------------- byte viability


def _viable_byte(guide, stack, lexer, data: bytes):
    """(stack, lexer) after one byte, or None — a measurement-grade copy of
    guide._advance's byte machinery (events + shifts + partial-lexeme tail
    rule), needed because _advance takes token ids, not bytes."""
    from grid.lexer.dfa import DEAD
    from grid.lexer.run import ScanReject
    from grid.trie.walk import pick_viable

    try:
        lexer2, events = lexer.advance(guide.dfa, data)
    except ScanReject:
        return None
    buf = lexer.remainder + data
    node = stack
    offset = 0
    for ev in events:
        seg = buf[offset:offset + ev.length]
        offset += ev.length
        viable = ev.candidates & guide.producer.allowed(node)
        pick = pick_viable(ev, seg, viable, guide.tables.ignored_terminal_ids,
                           guide._priority, guide.lexicons)
        if pick is None:
            return None
        if pick in guide.tables.ignored_terminal_ids:
            continue
        node = guide.producer.shift(node, pick)
        if node is None:
            return None
    if lexer2.remainder:
        st = guide.dfa.scan_state(lexer2.remainder)
        if st == DEAD:
            return None
        a_now = guide.producer.allowed(node)
        ok = False
        for t in guide.dfa.live[st]:
            if t in guide.tables.ignored_terminal_ids:
                ok = True
                break
            if t in a_now and (guide.lexicons is None
                               or guide.lexicons.prefix_ok(t, lexer2.remainder)):
                ok = True
                break
        if not ok:
            return None
    return node, lexer2


def byte_run_stats(guide, payload: bytes, cap: int) -> dict:
    """Walk `payload` byte-by-byte; per position, count viable next bytes
    (early-exit at 2) and mark the position forced iff exactly one is viable
    and the state cannot terminate. Returns bytes probed / forced."""
    stack, lexer = guide.initial_state.stack, guide.initial_state.lexer
    probed = forced = 0
    for i, byte in enumerate(payload):
        if i >= cap:
            break
        n_viable = 0
        for b in range(256):
            if _viable_byte(guide, stack, lexer, bytes([b])) is not None:
                n_viable += 1
                if n_viable > 1:
                    break
        can_end = guide._eos_ok(stack, lexer)
        probed += 1
        if n_viable == 1 and not can_end:
            forced += 1
        nxt = _viable_byte(guide, stack, lexer, bytes([byte]))
        if nxt is None:  # replay byte not viable (should not happen on valid)
            return {"bytes_probed": probed, "bytes_forced": forced, "desync": True}
        stack, lexer = nxt
    return {"bytes_probed": probed, "bytes_forced": forced, "desync": False}


# ------------------------------------------------------------ token replay


def replay_instance(guide, adapter, token_ids: list[int]) -> dict | None:
    """Walk the guide along `token_ids`; per step record forced/free and the
    byte length of each token, so forced-run byte coverage is exact."""
    state = guide.initial_state
    eos = guide.eos_token_id
    steps: list[tuple[bool, int]] = []  # (forced, token_byte_len)
    for t in token_ids:
        ids, _ = guide._mask_ids(state)
        if not bool((ids == t).any()):
            return None  # replay desync (invalid-instance shape); skip
        is_forced = len(ids) == 1 and int(ids[0]) != eos
        if is_forced:
            assert int(ids[0]) == int(t), "singleton mask must force the replay token"
        steps.append((is_forced, len(adapter.token_bytes(int(t)))))
        state = guide.get_next_state(state, int(t))
    return {"steps": steps}


def reduce_runs(steps: list[tuple[bool, int]]) -> dict:
    """Forced-run lengths + byte coverage from one instance's step list."""
    runs: list[int] = []
    chain_bytes = 0
    cur = 0
    for forced, nbytes in steps:
        if forced:
            cur += 1
            chain_bytes += nbytes
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return {
        "n_steps": len(steps),
        "n_forced": sum(r for r in runs),
        "runs": runs,
        "token_chain_bytes": chain_bytes,
    }


def saved_fraction(runs: list[int], n_steps: int, j: int) -> float:
    """LOWER bound on steps saved with per-step draft budget j: a forced run
    of length L costs ceil(L/j) verification-carrying steps instead of L.
    Ignores bonus tokens, so an isolated forced token (L=1) saves nothing."""
    if n_steps == 0:
        return 0.0
    saved = sum(length - -(-length // j) for length in runs)
    return saved / n_steps


def saved_fraction_bonus(runs: list[int], n_steps: int, j: int) -> float:
    """UPPER bound: vLLM's rejection sampler emits a BONUS token when every
    draft position verifies (certain at forced positions), so a run of L
    forced tokens plus its free successor — L+1 tokens — costs
    ceil((L+1)/(j+1)) steps instead of L+1; an isolated forced token then
    saves one full step. Bonus mechanics are box-verified (S1 step 3)."""
    if n_steps == 0:
        return 0.0
    saved = sum(length + 1 - -(-(length + 1) // (j + 1)) for length in runs)
    return saved / n_steps


# ------------------------------------------------------------------- driver


def _split_of(file: str) -> str:
    """Split name: the parent directory (tmp/jsb-src/data/<split>/oNNN.json
    layout) unless the file sits directly in the data dir, then the
    maskbench flat-name convention."""
    import maskbench_grid

    parent = pathlib.Path(file).parent.name
    return parent if parent != "data" else maskbench_grid.split_of(file)


def _sample_files(data_dir: str, per_split: int, seed: int) -> list[str]:
    """Seeded per-split sample over either layout (nested split dirs, or the
    maskbench flat dir)."""
    import glob
    import random

    paths = sorted(glob.glob(str(pathlib.Path(data_dir) / "*" / "*.json"))) \
        or sorted(glob.glob(str(pathlib.Path(data_dir) / "*.json")))
    by_split: dict[str, list[str]] = {}
    for f in paths:
        by_split.setdefault(_split_of(f), []).append(f)
    rng = random.Random(seed)
    out: list[str] = []
    for split in sorted(by_split):
        files = by_split[split]
        out += files if len(files) <= per_split else rng.sample(files, per_split)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="jsonschemabench data dir")
    ap.add_argument("--sample", type=int, default=6, help="schemas per split (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer (box runs: the serving model's)")
    ap.add_argument("--time-limit", type=int, default=60, help="s per schema compile")
    ap.add_argument("--byte-cap", type=int, default=400,
                    help="max bytes probed per schema for the byte-run leg (256 "
                         "viability probes per byte; measurement-grade)")
    ap.add_argument("--max-instances", type=int, default=2, help="valid tests per schema")
    ap.add_argument("--no-bytes", action="store_true", help="skip the byte-run leg")
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    args = ap.parse_args()

    import signal

    from maskbench_grid import BenchTimeout, GridEngine

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    engine = GridEngine(tokenizer)

    def on_alarm(signum, frame):
        raise BenchTimeout()

    signal.signal(signal.SIGALRM, on_alarm)

    files = _sample_files(args.data, args.sample, args.seed)
    per_split: dict[str, dict] = {}
    t0 = time.monotonic()
    n_compiled = n_failed = 0
    for i, file in enumerate(files):
        with open(file) as f:
            data = json.load(f)
        split = _split_of(file)
        agg = per_split.setdefault(split, {
            "schemas": 0, "instances": 0, "steps": 0, "forced": 0, "runs": [],
            "token_chain_bytes": 0, "bytes_probed": 0, "bytes_forced": 0,
            "desyncs": 0,
        })
        signal.alarm(args.time_limit)
        try:
            engine.compile_grammar(data["schema"])
        except (BenchTimeout, Exception):  # noqa: BLE001 - density probe, not a gate
            n_failed += 1
            continue
        finally:
            signal.alarm(0)
        n_compiled += 1
        guide = engine.guide
        agg["schemas"] += 1
        done_bytes = False
        n_inst = 0
        for test in data.get("tests", []):
            if not test.get("valid") or n_inst >= args.max_instances:
                continue
            instance = json.dumps(test["data"], indent=None, ensure_ascii=False)
            token_ids = tokenizer.encode(instance, add_special_tokens=False)
            got = replay_instance(guide, engine.adapter, token_ids)
            if got is None:  # valid instance rejected mid-replay: report, never hide
                agg["desyncs"] += 1
                continue
            n_inst += 1
            red = reduce_runs(got["steps"])
            agg["instances"] += 1
            agg["steps"] += red["n_steps"]
            agg["forced"] += red["n_forced"]
            agg["runs"] += red["runs"]
            agg["token_chain_bytes"] += red["token_chain_bytes"]
            if not args.no_bytes and not done_bytes:
                done_bytes = True  # one byte-leg instance per schema (256x cost)
                bs = byte_run_stats(guide, instance.encode(), args.byte_cap)
                agg["bytes_probed"] += bs["bytes_probed"]
                agg["bytes_forced"] += bs["bytes_forced"]
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(files)}] {time.monotonic() - t0:.0f}s",
                  file=sys.stderr)

    report(per_split, args, n_compiled, n_failed, time.monotonic() - t0)


def report(per_split: dict, args, n_compiled: int, n_failed: int, wall: float) -> None:
    def hist(runs: list[int]) -> dict[str, int]:
        out = {}
        for b in RUN_BUCKETS:
            out[str(b)] = sum(1 for r in runs if r == b)
        out[f">{RUN_BUCKETS[-1]}"] = sum(1 for r in runs if r > RUN_BUCKETS[-1])
        return out

    rows = []
    total = {"schemas": 0, "instances": 0, "steps": 0, "forced": 0, "runs": [],
             "token_chain_bytes": 0, "bytes_probed": 0, "bytes_forced": 0,
             "desyncs": 0}
    for split in sorted(per_split):
        a = per_split[split]
        for k in total:
            total[k] += a[k]
        rows.append((split, a))
    rows.append(("TOTAL", total))

    summary = {"tokenizer": args.tokenizer, "sample": args.sample, "seed": args.seed,
               "compiled": n_compiled, "compile_failed": n_failed,
               "wall_s": round(wall, 1), "splits": {}}
    print(f"\njf-density: {n_compiled} schemas compiled, {n_failed} failed/timeout, "
          f"{total['desyncs']} replay desyncs, {wall:.0f}s wall "
          f"(tokenizer={args.tokenizer})")
    print(f"{'split':<18}{'inst':>5}{'steps':>8}{'forced':>8}{'pct':>7}"
          + "".join(f"{'sv@' + str(j):>7}" for j in (8,))
          + "".join(f"{'svb@' + str(j):>7}" for j in (4, 8))
          + f"{'byteF%':>8}{'cov%':>6}")
    for split, a in rows:
        pct = 100.0 * a["forced"] / a["steps"] if a["steps"] else 0.0
        saved = {j: saved_fraction(a["runs"], a["steps"], j) for j in J_CAPS}
        savedb = {j: saved_fraction_bonus(a["runs"], a["steps"], j) for j in J_CAPS}
        bpct = 100.0 * a["bytes_forced"] / a["bytes_probed"] if a["bytes_probed"] else 0.0
        # token-chain coverage of the byte-forced mass, scaled to probed bytes
        cov = (100.0 * a["token_chain_bytes"] / a["bytes_forced"]
               if a["bytes_forced"] else float("nan"))
        print(f"{split:<18}{a['instances']:>5}{a['steps']:>8}{a['forced']:>8}"
              f"{pct:>6.1f}%"
              + f"{100 * saved[8]:>6.1f}%"
              + "".join(f"{100 * savedb[j]:>6.1f}%" for j in (4, 8))
              + f"{bpct:>7.1f}%{cov:>5.0f}%")
        summary["splits"][split] = {
            **{k: a[k] for k in ("schemas", "instances", "steps", "forced",
                                 "token_chain_bytes", "bytes_probed",
                                 "bytes_forced", "desyncs")},
            "forced_pct": round(pct, 2),
            "saved_frac": {str(j): round(saved[j], 4) for j in J_CAPS},
            "saved_frac_bonus": {str(j): round(savedb[j], 4) for j in J_CAPS},
            "run_hist": hist(a["runs"]),
        }
    print("\nsv@j = LOWER bound on decode steps saved at draft budget j (run of "
          "L forced tokens -> ceil(L/j) steps; no bonus token, so L=1 saves 0). "
          "svb@j = UPPER bound with vLLM bonus-token mechanics (L+1 tokens in "
          "ceil((L+1)/(j+1)) steps; L=1 saves one step) — box-verified before "
          "any headline claim. byteF% = forced-byte density (the byte-level JF "
          "ceiling, deferred v2); cov% = token-chain bytes as % of forced bytes "
          "(chains can span free-byte boundaries, so >100% is possible; both "
          "sides capped by --byte-cap sampling).")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
        print(f"summary -> {args.out}")


if __name__ == "__main__":
    main()
