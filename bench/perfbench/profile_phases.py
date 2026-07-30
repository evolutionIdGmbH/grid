"""Per-phase TTFM attribution over perfbench task sets (DESIGN.md prerequisite).

Parent mode walks manifest sets and runs one child subprocess per schema
(timeout-guarded; a timeout still attributes, because the child streams each
phase record to disk the moment it finishes, so the phase in flight at kill
time is the last+1). Child mode runs the five compile phases:

    schema_compile -> spec_load -> projection -> lalr -> scanner

and records size stats alongside timings (terminals, grammar source bytes,
DFA states, peak RSS) so cost drivers can be correlated, not guessed.

--first-mask (E4) appends four flush-streamed phases:

    trie -> guide -> first_mask -> prefix_masks

`trie` (tokenizer + token trie + runtime imports, the per-engine constant
maskbench builds once in engine __init__) is EXCLUDED from TTFM; `guide` is
GridGuide construction (maskbench's compile_grammar includes it); `first_mask`
is one guide._mask_ids(initial_state) — the call maskbench's compute_mask
makes, and for a lazy factored scanner the first real payment of the deferred
product construction (pure-Python by the kernel/genN lazy gates);
`prefix_masks` walks mask+commit along the first valid test instance (<=64
tokens), catching lazy materialization that only triggers mid-instance. The
child writes stats.ttfm_compile_us (five compile phases + guide: maskbench
compile-only semantics) and stats.ttfm_first_us (+ first_mask) so summaries
can never recompute the two columns inconsistently.

Usage:
    python bench/perfbench/profile_phases.py \
        --group ttfm_capped:75:0:1 --group ttfm_tail_1pct:120:40:1 \
        --group stratified_200:30:0:7 --jobs 3 --out tmp/perfbench-profile

Group syntax: name:timeout_s:limit:stride (limit 0 = whole set).
--leg name:VAR=V,VAR=V (repeatable) runs every schema once per leg with those
env overrides, interleaved per schema in one session (the F2 protocol rule),
records under out_dir/<name>/. Attribution shares tolerate --jobs 3 on a
big-core host; publishable absolute numbers require --jobs 1.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "jsb-src", "data")

# measure the tree this script lives in: the venv's grid install points at the
# main checkout, so without this pin a worktree run silently profiles main
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PHASES = ("schema_compile", "spec_load", "projection", "lalr", "scanner")
FIRST_MASK_PHASES = ("trie", "guide", "first_mask", "prefix_masks")
ALL_PHASES = PHASES + FIRST_MASK_PHASES

DEFAULT_TOKENIZER = "unsloth/Meta-Llama-3.1-8B-Instruct"
PREFIX_TOKEN_CAP = 64


def child(schema_file: str, out_file: str, first_mask: bool = False,
          tokenizer_name: str = DEFAULT_TOKENIZER, tests_file: str | None = None) -> None:
    # flags snapshot: every leg is self-describing (the F1/F2 investigations
    # were blocked because leg env was unrecoverable from the records)
    rec: dict = {
        "file": schema_file,
        "flags": {k: v for k, v in os.environ.items() if k.startswith("GRID_PERF_")},
        "phases": {},
        "stats": {},
    }
    if first_mask:
        rec["mode"] = "first_mask"
        rec["tokenizer"] = tokenizer_name

    def flush() -> None:
        with open(out_file, "w") as f:
            f.write(json.dumps(rec, indent=1))

    def phase(name: str, fn):
        rec["running"] = name
        flush()
        t0 = time.monotonic()
        val = fn()
        rec["phases"][name] = round((time.monotonic() - t0) * 1e6)
        return val

    try:
        with open(schema_file) as f:
            schema = json.load(f)
        if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
            schema = schema["schema"]  # wrapped maskbench layout

        from grid.grammar import spec
        from grid.grammar.projection import RoleProjection
        from grid.jsonschema import compile_json_schema
        from grid.lalr.compile import compile_tables
        from grid.lexer.dfa import build_scanner

        src, recorded = phase("schema_compile", lambda: compile_json_schema(schema))
        rec["stats"]["src_bytes"] = len(src)
        rec["stats"]["recorded"] = len(recorded)
        grammar = phase("spec_load", lambda: spec.load(src))
        rec["stats"]["terminals"] = len(grammar.terminals)
        proj = phase("projection", lambda: RoleProjection.full(grammar).build())
        tables = phase("lalr", lambda: compile_tables(proj))
        rec["stats"]["lalr_states"] = len(getattr(tables, "action", ()) or ())
        dfa = phase("scanner", lambda: build_scanner(grammar.terminals, grammar.terminal_order))
        if getattr(dfa, "lazy", False):
            # GRID_PERF_FACTORED_SCANNER over-budget regime: LazyProductDFA has no
            # dense trans; report the product states materialized so far
            rec["stats"]["dfa_states"] = len(dfa._states)
            rec["stats"]["dfa_lazy"] = True
        else:
            rec["stats"]["dfa_states"] = len(dfa.trans)

        if first_mask:
            _first_mask_phases(rec, phase, schema_file, tests_file,
                               tokenizer_name, tables, dfa)
    except Exception as e:
        # declared outcomes (Unsupported, LALRConflictError, ...) become a
        # first-class record category instead of an anonymous nonzero exit
        rec["error"] = type(e).__name__
        rec["error_msg"] = str(e)[:200]
        flush()
        raise
    rec["running"] = None
    rec["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20))
    flush()


def _first_mask_phases(rec: dict, phase, schema_file: str, tests_file: str | None,
                       tokenizer_name: str, tables, dfa) -> None:
    """The four E4 phases. `trie` deliberately swallows the heavyweight
    imports (transformers, torch via grid.guide) alongside the tokenizer and
    trie builds: maskbench runs one process across all schemas, so its
    compile_grammar TTFM sees those imports amortized to ~zero — the `guide`
    phase here must measure construction only."""

    def _build_trie():
        from transformers import AutoTokenizer  # noqa: F401 (heavy, excluded)

        import grid.guide  # noqa: F401  pull torch/numpy chain outside `guide`
        from grid.models.hf_adapter import HFTokenizerAdapter
        from grid.trie.build import build_trie

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        adapter = HFTokenizerAdapter(tok)
        return tok, adapter, build_trie(adapter)

    tok, adapter, trie = phase("trie", _build_trie)

    from grid.guide import GridGuide

    guide = phase("guide", lambda: GridGuide(
        tables=tables, dfa=dfa, trie=trie, adapter=adapter))
    phase("first_mask", lambda: guide._mask_ids(guide.initial_state))
    # child-written columns (never recomputed by summaries): compile-only =
    # maskbench compile_grammar semantics (five phases + GridGuide);
    # first-mask-included adds the first compute_mask
    compile_us = sum(rec["phases"][p] for p in PHASES) + rec["phases"]["guide"]
    rec["stats"]["ttfm_compile_us"] = compile_us
    rec["stats"]["ttfm_first_us"] = compile_us + rec["phases"]["first_mask"]

    def _prefix_masks():
        tests = []
        source = "none"
        for cand, label in ((tests_file, "tests_file"), (schema_file, "schema_file")):
            if not cand:
                continue
            try:
                with open(cand) as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and isinstance(doc.get("tests"), list):
                tests, source = doc["tests"], label
                break
        rec["stats"]["prefix_source"] = source
        inst = next((t for t in tests if t.get("valid")), None)
        if inst is None:
            rec["stats"]["prefix_tokens"] = 0
            return
        text = json.dumps(inst["data"], indent=None, ensure_ascii=False)
        tokens = tok.encode(text, add_special_tokens=False)[:PREFIX_TOKEN_CAP]
        state = guide.initial_state
        n, mx = 0, 0
        for t in tokens:
            t0 = time.monotonic()  # maskbench TBM window: mask + commit
            ids, _ = guide._mask_ids(state)
            ok = bool((ids == t).any())
            if ok:
                state = guide.get_next_state(state, t)
            mx = max(mx, round((time.monotonic() - t0) * 1e6))
            n += 1
            if not ok:
                # a valid-instance rejection here is grammar drift, not noise —
                # keep the record ok (this phase is a probe) but make it visible
                rec["stats"]["prefix_rejected_at"] = n
                break
        rec["stats"]["prefix_tokens"] = n
        rec["stats"]["prefix_mask_max_us"] = mx

    phase("prefix_masks", _prefix_masks)


def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    # remaining ids (BFCL_*, JME_*, ...) are maskbench-layout files named by
    # their full id (same fallback as diff_hashcons/diff_scanner_digest)
    return os.path.join(DATA_DIR, "..", "maskbench", "data",
                        schema_id + ".json")


def tests_path(schema_id: str) -> str | None:
    """The maskbench-layout twin of a schema id: carries the `tests` key the
    prefix_masks phase walks. Exists for every corpus id; None when absent."""
    p = os.path.join(DATA_DIR, "..", "maskbench", "data", schema_id + ".json")
    return p if os.path.exists(p) else None


def _parse_leg(spec_str: str) -> tuple[str, dict[str, str]]:
    name, _, envs = spec_str.partition(":")
    over: dict[str, str] = {}
    for pair in filter(None, envs.split(",")):
        k, _, v = pair.partition("=")
        over[k] = v
    return name, over


def _check_tokenizer_cached(tokenizer_name: str, env: dict[str, str]) -> None:
    """Fail fast before queueing 2x45 children: with HF_HUB_OFFLINE=1 a cache
    miss would fail every --first-mask child identically."""
    r = subprocess.run(
        [sys.executable, "-c",
         "from transformers import AutoTokenizer; "
         f"AutoTokenizer.from_pretrained({tokenizer_name!r})"],
        env=env, capture_output=True, timeout=300,
    )
    if r.returncode != 0:
        sys.exit(f"tokenizer {tokenizer_name!r} not loadable offline "
                 f"(HF cache miss?):\n{r.stderr.decode()[-2000:]}")


def parent(groups: list[str], jobs: int, out_dir: str, first_mask: bool = False,
           tokenizer_name: str = DEFAULT_TOKENIZER,
           leg_specs: list[str] | None = None) -> None:
    with open(MANIFEST) as f:
        sets = json.load(f)["sets"]
    legs = [_parse_leg(s) for s in (leg_specs or [])] or [("", {})]
    assert len({n for n, _ in legs}) == len(legs), "duplicate leg names"
    for name, _ in legs:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    base_env = dict(os.environ)
    if first_mask:
        # offline: a mid-run HF hub probe would add network latency (or hang)
        # inside the measured child; cache presence is checked upfront
        base_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        TOKENIZERS_PARALLELISM="false")
        _check_tokenizer_cached(tokenizer_name, base_env)

    ids_order: list[tuple[str, int]] = []  # (schema_id, timeout_s)
    seen: set[str] = set()
    for g in groups:
        name, timeout_s, limit, stride = g.split(":")
        ids = sets[name][:: int(stride) or 1]
        if int(limit):
            ids = ids[: int(limit)]
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                ids_order.append((sid, int(timeout_s)))
    # legs innermost: ON/OFF pairs run back to back per schema (F2 protocol)
    work = [(sid, timeout_s, leg) for sid, timeout_s in ids_order for leg in legs]

    print(f"{len(ids_order)} schemas x {len(legs)} leg(s) queued, jobs={jobs}", flush=True)
    running: list[tuple[subprocess.Popen, str, str, float, int]] = []
    queue = list(reversed(work))
    done = 0

    def out_path(sid: str, leg_name: str) -> str:
        return os.path.join(out_dir, leg_name, sid + ".phases.json")

    while queue or running:
        while queue and len(running) < jobs:
            sid, timeout_s, (leg_name, leg_env) = queue.pop()
            path = out_path(sid, leg_name)
            if os.path.exists(path):  # resume support
                try:
                    with open(path) as f:
                        prev = json.load(f)
                except Exception:
                    prev = {"running": "?"}  # unreadable = treat as partial
                if prev.get("running") is None or "timeout_s" in prev or "rc" in prev:
                    done += 1
                    continue
                # unmarked partial from an interrupted parent: requeue, never
                # fossilize (partials read as ~0.0s legs in summaries)
                os.remove(path)
            argv = [sys.executable, __file__, "--child", schema_path(sid), path]
            if first_mask:
                argv += ["--first-mask", "--tokenizer", tokenizer_name]
                tp = tests_path(sid)
                if tp:
                    argv += ["--tests", tp]
            p = subprocess.Popen(
                argv,
                env={**base_env, **leg_env},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running.append((p, sid, leg_name, time.monotonic(), timeout_s))
        time.sleep(0.2)
        still = []
        for p, sid, leg_name, t0, timeout_s in running:
            tag = f"{leg_name}:{sid}" if leg_name else sid
            if p.poll() is not None:
                done += 1
                if p.returncode != 0:
                    # annotate exactly like the timeout branch: a crashed or
                    # killed child must never be mistakable for a completed one
                    try:
                        with open(out_path(sid, leg_name)) as f:
                            rec = json.load(f)
                    except Exception:
                        rec = {"phases": {}}
                    rec["rc"] = p.returncode
                    with open(out_path(sid, leg_name), "w") as f:
                        f.write(json.dumps(rec, indent=1))
                print(f"[{done}/{len(work)}] {tag} rc={p.returncode}", flush=True)
            elif time.monotonic() - t0 > timeout_s:
                p.kill()
                p.wait()
                done += 1
                # mark the timeout in the record the child left behind
                try:
                    with open(out_path(sid, leg_name)) as f:
                        rec = json.load(f)
                except Exception:
                    rec = {"phases": {}}
                rec["timeout_s"] = timeout_s
                with open(out_path(sid, leg_name), "w") as f:
                    f.write(json.dumps(rec, indent=1))
                print(f"[{done}/{len(work)}] {tag} TIMEOUT in {rec.get('running')}", flush=True)
            else:
                still.append((p, sid, leg_name, t0, timeout_s))
        running = still

    for leg_name, _ in legs:
        summarize(os.path.join(out_dir, leg_name))


def _pct(sorted_xs: list[int], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    return float(sorted_xs[min(len(sorted_xs) - 1, int(len(sorted_xs) * q))])


def summarize(out_dir: str) -> None:
    import glob as g

    files = g.glob(os.path.join(out_dir, "*.phases.json"))
    if not files:
        # leg layout: summarize each subdir that holds records
        subs = [d for d in sorted(g.glob(os.path.join(out_dir, "*")))
                if os.path.isdir(d) and g.glob(os.path.join(d, "*.phases.json"))]
        for d in subs:
            print(f"\n=== {d} ===")
            summarize(d)
        if not subs:
            print(f"{out_dir}: no records")
        return

    totals = dict.fromkeys(ALL_PHASES, 0)
    timeouts: dict[str, int] = {}
    incomplete: dict[str, int] = {}
    n_done = 0
    compile_col: list[int] = []
    first_col: list[int] = []
    prefix_max: list[int] = []
    for f in files:
        with open(f) as fh:
            rec = json.load(fh)
        if rec.get("running") is None and "error" not in rec and "rc" not in rec:
            # completed = child reached the final flush; a pre-first-phase
            # crash also lacks 'running' but carries error/rc
            n_done += 1
            stats = rec.get("stats", {})
            if "ttfm_compile_us" in stats:
                compile_col.append(stats["ttfm_compile_us"])
                first_col.append(stats["ttfm_first_us"])
            if stats.get("prefix_mask_max_us") is not None:
                prefix_max.append(stats["prefix_mask_max_us"])
        elif "timeout_s" in rec or "rc" in rec:
            ph = rec.get("running") or "?"
            timeouts[ph] = timeouts.get(ph, 0) + 1
        else:
            # unmarked partial (interrupted parent): its phase sums are not
            # attributions — exclude from totals entirely
            ph = rec.get("running") or "?"
            incomplete[ph] = incomplete.get(ph, 0) + 1
            continue
        for k, v in rec.get("phases", {}).items():
            totals[k] += v
    tot = sum(totals.values()) or 1
    print(f"\ncompleted: {n_done}; timeouts/crashed by in-flight phase: {timeouts}")
    fm_era = sum(n for ph, n in timeouts.items() if ph in FIRST_MASK_PHASES)
    if fm_era:
        # a timeout inside trie/guide/first_mask/prefix_masks is a
        # first-mask-era timeout — the deferred cost the compile-only metric
        # hid — NOT comparable to a v0.2.5 compile timeout
        print(f"  ({fm_era} of these are first-mask-era: in-flight phase past `scanner`)")
    if incomplete:
        print(f"incomplete (unmarked, will re-run on resume) by in-flight phase: {incomplete}")
    shown = PHASES + tuple(p for p in FIRST_MASK_PHASES if totals[p])
    for ph in shown:
        print(f"  {ph:15s} {totals[ph] / 1e6:9.1f}s  {100 * totals[ph] / tot:5.1f}%")
    if compile_col:
        compile_col.sort()
        first_col.sort()
        print(f"TTFM over {len(compile_col)} completed records (child-written stats), us:")
        for label, col in (("compile-only (five phases + guide)", compile_col),
                           ("first-mask-included", first_col)):
            print(f"  {label:34s} p50 {_pct(col, .5):>12,.0f}  "
                  f"p90 {_pct(col, .9):>12,.0f}  max {col[-1]:>12,}")
        if prefix_max:
            prefix_max.sort()
            print(f"  {'prefix-mask worst token (n=' + str(len(prefix_max)) + ')':34s} "
                  f"p50 {_pct(prefix_max, .5):>12,.0f}  "
                  f"p90 {_pct(prefix_max, .9):>12,.0f}  max {prefix_max[-1]:>12,}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        argv = sys.argv[2:]
        tok_name = (argv[argv.index("--tokenizer") + 1]
                    if "--tokenizer" in argv else DEFAULT_TOKENIZER)
        tf = argv[argv.index("--tests") + 1] if "--tests" in argv else None
        child(argv[0], argv[1], first_mask="--first-mask" in argv,
              tokenizer_name=tok_name, tests_file=tf)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--group", action="append", required=True, help="name:timeout_s:limit:stride")
        ap.add_argument("--jobs", type=int, default=1)
        ap.add_argument("--out", required=True)
        ap.add_argument("--first-mask", action="store_true",
                        help="append trie/guide/first_mask/prefix_masks phases per child")
        ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
        ap.add_argument("--leg", action="append", default=None,
                        help="name:VAR=V,VAR=V — run each schema once per leg, "
                             "interleaved; records under out/<name>/")
        ap.add_argument("--summarize-only", action="store_true")
        args = ap.parse_args()
        if args.summarize_only:
            summarize(args.out)
        else:
            parent(args.group, args.jobs, args.out, first_mask=args.first_mask,
                   tokenizer_name=args.tokenizer, leg_specs=args.leg)
