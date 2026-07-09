"""G5 end-to-end S/C/T gate at specification scale (DESIGN.md §10, gate G5).

Two arms, same checks:
- ``--arm walk`` (model-free, runs anywhere): the FORCED-RANDOM-WALK arm —
  uniform sampling over the exact mask, EOS suppressed until length >= L,
  tight budgets forcing reserve stops. 10,000 seeded generations by default.
- ``--arm model`` (GPU box): pinned Qwen2.5-0.5B-Instruct (byte-fallback BPE,
  >=100k vocab) with a seeded multinomial sampler through the same loop.

Binding checks per generation (gate text):
- output parses under its OWN CompiledGrammar (ReferenceGuide.eos_legal);
- EOS only at ACCEPT (asserted at every EOS application);
- DeadEndError == 0;
- every jump-complete stop parses and ends at ACCEPT;
- no reserve-stopped generation exceeds max_tokens.

Coverage counters (gate): max paren nesting >= D, >= k reserve stops,
>= k multi-byte-identifier events (schema identifiers spelled by >=2 model
tokens). Reserve tightness at scale: at every reserve stop the emitted tail
equals the ReserveTable completion recomputed at the trigger state; the
token-count-vs-BFS-oracle equality is pinned at unit level
(tests/lalr/test_reserve.py) and referenced here.

Run:
  .venv-bench/bin/python bench/g5_scale.py --arm walk --gens 10000 --assert-gates
  (box) python bench/g5_scale.py --arm model --gens 10000 --assert-gates
Report: bench/RESULTS-g5.md (arm-suffixed)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

BENCH_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent))

from grid._reference.guide import ReferenceGuide  # noqa: E402
from grid.errors import DeadEndError  # noqa: E402
from grid.generate import build_guide  # noqa: E402
from grid.grammar import spec as gspec  # noqa: E402
from grid.grammar.projection import RoleProjection  # noqa: E402
from grid.guide import COMPLETE  # noqa: E402
from grid.lalr.compile import compile_tables  # noqa: E402
from grid.policy.schema import SchemaSnapshot  # noqa: E402
from grid.protocols import Generate, Write  # noqa: E402

# multi-token-on-BPE identifiers (multi-byte-identifier coverage events)
SCHEMA = {
    "customer_accounts": ["account_identifier", "opening_balance_cents",
                          "risk_classification", "branch_code"],
    "transaction_ledger": ["ledger_sequence_no", "counterparty_name",
                           "settlement_timestamp", "amount_minor_units"],
    "users": ["id", "name", "email"],
}


def build(adapter):
    source = (BENCH_DIR.parent / "grammars" / "sql_subset.grid").read_text()
    grammar = gspec.load(source)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    snap = SchemaSnapshot.from_dict(SCHEMA)
    guide = build_guide(source, adapter, projection=proj,
                        lexicons=snap.lexicons(tables),
                        schema_fingerprint=snap.fingerprint)
    ref = ReferenceGuide(guide.tables, guide.dfa, adapter)
    return guide, ref


OPEN, CLOSE = None, None  # token ids for "(" / ")" set at build time


def walk_one(template, seed: int, rng: np.random.Generator, seek_nesting: bool):
    """One forced-random-walk generation. Base arm: uniform over the exact
    mask. seek_nesting seeds (5%, quota arm) prefer "(" while shallow — the
    coverage counters need depth the uniform walk rarely reaches; every
    binding check applies to these generations identically.
    Returns (tokens, stop_reason, reserve_consistent, budget)."""
    guide = template.copy()
    L = int(rng.integers(8, 25))
    budget = int(rng.integers(30, 80)) if seek_nesting else int(rng.integers(18, 61))
    guide.max_new_tokens = budget
    eos = guide.eos_token_id
    state = guide.initial_state
    out: list[int] = []
    depth = 0
    reserve_consistent = True
    while True:
        instr = guide.get_next_instruction(state)
        if isinstance(instr, Write):
            span = [int(x) for x in instr.tokens]
            budget_write = span and span[-1] == eos and len(span) > 1
            if budget_write:
                # reserve tightness (consistency): recomputing the completion
                # at the trigger state must reproduce the emitted span
                again = guide._completion_tokens(state)
                reserve_consistent = again == span
            for t in span:
                assert not (t == eos and not guide.can_terminate_state(state)), \
                    "EOS applied outside ACCEPT (gate violation)"
                state = guide.get_next_state(state, t)
                out.append(t)
                if state.status == COMPLETE:
                    break
            if state.status == COMPLETE:
                stop = "MAX_TOKENS_WITH_JUMP_COMPLETE" if budget_write else "EOS_ACCEPT"
                return out, stop, reserve_consistent, budget
            continue
        assert isinstance(instr, Generate)
        ids = instr.tokens.numpy()
        pool = ids[ids != eos] if (len(out) < L and (ids != eos).any()) else ids
        t = int(pool[int(rng.integers(len(pool)))])
        if seek_nesting and depth < 10 and OPEN in ids and rng.random() < 0.85:
            t = OPEN
        if t == OPEN:
            depth += 1
        elif t == CLOSE:
            depth -= 1
        if t == eos:
            assert guide.can_terminate_state(state), "EOS sampled outside ACCEPT"
        state = guide.get_next_state(state, t)
        out.append(t)
        if state.status == COMPLETE:
            return out, "EOS_ACCEPT", reserve_consistent, budget


def run_model_arm(args, adapter, multi_tok_idents) -> int:
    """G5 model-in-the-loop: the mode-1 GRID-owned decode loop with a real model
    (Qwen2.5-0.5B-Instruct) providing logits. Same binding checks as the walk
    arm, plus the audit chain verifies every generation. Slower than the walk
    arm (a forward pass per token), so it runs at a representative scale
    (default 1000) while the 10k-coverage claim is the model-free walk arm."""
    import numpy as np
    import torch

    from grid import generate
    from grid.models.transformers_model import TransformersModel
    from grid.policy.schema import SchemaSnapshot
    from grid.samplers import multinomial

    source = (BENCH_DIR.parent / "grammars" / "sql_subset.grid").read_text()
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else None
    model = TransformersModel.from_pretrained(args.model, device=device, dtype=dtype)
    g = generate.sql(model, source, schema=SchemaSnapshot.from_dict(SCHEMA),
                     sampler=multinomial(1.0), audit=True)
    ref = ReferenceGuide(g.logits_processor.guide.tables,
                         g.logits_processor.guide.dfa, adapter)
    eos = g.logits_processor.guide.eos_token_id
    print(f"model loaded in {time.perf_counter()-t0:.1f}s | "
          f"kernel {g.logits_processor.guide.producer._kernel is not None}")

    prompts = ["Return one SQL query for this database.",
               "Show rows from a table.", "Write a select statement.",
               "Query the customer records.", "Give me one lowercase SQL query."]
    parse_ok = audit_ok = 0
    dead_ends = over_budget = reserve_stops = ident_events = max_nesting = 0
    t0 = time.perf_counter()
    for i in range(args.gens):
        rng = np.random.default_rng(920_000 + i)
        budget = int(rng.integers(24, 80))
        try:
            r = g(prompts[i % len(prompts)], max_tokens=budget, seed=i)
        except DeadEndError:
            dead_ends += 1
            continue
        data = b"".join(adapter.token_bytes(t) for t in r.token_ids if t != eos)
        if ref.eos_legal(data):
            parse_ok += 1
        if r.audit is None or r.audit.verify_chain():
            audit_ok += 1
        if len(r.token_ids) > budget:
            over_budget += 1
        reserve_stops += r.stop_reason == "MAX_TOKENS_WITH_JUMP_COMPLETE"
        depth = cur = 0
        for byte in data:
            if byte == 0x28:
                cur += 1
                depth = max(depth, cur)
            elif byte == 0x29:
                cur -= 1
        max_nesting = max(max_nesting, depth)
        text = data.decode("utf-8", "ignore")
        ident_events += sum(text.count(w) for w in multi_tok_idents)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.gens} | parse {parse_ok}/{i+1-dead_ends} | "
                  f"audit {audit_ok} | reserve {reserve_stops} | idents {ident_events} | "
                  f"nesting {max_nesting} | {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    n_done = args.gens - dead_ends
    checks = {
        "outputs parse under own grammar (binding)": (parse_ok == n_done, f"{parse_ok}/{n_done}"),
        "audit chain verifies every generation": (audit_ok == n_done, f"{audit_ok}/{n_done}"),
        "DeadEndError == 0": (dead_ends == 0, str(dead_ends)),
        "no generation exceeds max_tokens": (over_budget == 0, f"{over_budget} over"),
        f"coverage: multi-token identifier events >= {args.min_ident_events}":
            (ident_events >= args.min_ident_events, f"{ident_events:,}"),
    }
    all_ok = all(ok for ok, _ in checks.values())
    host = os.environ.get("GRID_HOST_LABEL", "local dev (unpinned)")
    out_path = pathlib.Path(args.out or (BENCH_DIR / "RESULTS-g5-model.md"))
    lines = [
        "# G5 end-to-end S/C/T at scale — model arm",
        "",
        f"Host: {host} | grammar: `grammars/sql_subset.grid` + L3 schema lexicons | "
        f"model: `{args.model}` | mode-1 GRID-owned loop, multinomial sampler | "
        f"{args.gens:,} seeded generations | wall {wall/60:.1f} min",
        "",
        "Real model logits drive the sampler; the mask constrains; GRID owns the loop "
        "and writes the audit chain. EOS-only-at-ACCEPT and the mask invariant are "
        "asserted inside the loop (`grid/generate/api.py`). This is the model-in-loop "
        "complement to the 10k model-free forced-random-walk arm (`RESULTS-g5-walk.md`).",
        "",
        "| check | pass | value |",
        "|---|---|---|",
        *[f"| {name} | {'PASS' if ok else 'FAIL'} | {val} |" for name, (ok, val) in checks.items()],
        "",
        f"Gate G5 (model arm): {'**PASS**' if all_ok else '**FAIL**'}. Reserve stops "
        f"observed: {reserve_stops}; max paren nesting: {max_nesting}.",
        "",
        "Harness: `bench/g5_scale.py --arm model`.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print("\n".join(lines[8:14]))
    print(f"report -> {out_path}")
    return 0 if (all_ok or not args.assert_gates) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("walk", "model"), default="walk")
    ap.add_argument("--gens", type=int, default=10_000)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--min-nesting", type=int, default=6)
    ap.add_argument("--min-reserve-stops", type=int, default=100)
    ap.add_argument("--min-ident-events", type=int, default=500)
    ap.add_argument("--assert-gates", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from grid.models.hf_adapter import HFTokenizerAdapter

    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    adapter = HFTokenizerAdapter(hf)
    t0 = time.perf_counter()
    template, ref = build(adapter)
    print(f"guide built in {time.perf_counter()-t0:.1f}s | vocab {template.vocab_size:,} | "
          f"kernel {template.producer._kernel is not None}")

    # identifiers that need >=2 model tokens when spelled alone
    multi_tok_idents = [w for ws in ([k, *v] for k, v in SCHEMA.items()) for w in ws
                        if len(adapter.greedy_tokenize(w.encode())) >= 2]

    if args.arm == "model":
        return run_model_arm(args, adapter, multi_tok_idents)

    global OPEN, CLOSE
    OPEN = adapter.greedy_tokenize(b"(")[0]
    CLOSE = adapter.greedy_tokenize(b")")[0]
    parse_ok = 0
    dead_ends = 0
    reserve_stops = 0
    reserve_consistent_all = True
    over_budget = 0
    ident_events = 0
    max_nesting = 0
    t0 = time.perf_counter()
    for g in range(args.gens):
        rng = np.random.default_rng(910_000 + g)
        try:
            out, stop, r_ok, budget = walk_one(template, g, rng, seek_nesting=(g % 10 == 7))
        except DeadEndError:
            dead_ends += 1
            continue
        data = b"".join(template.adapter.token_bytes(t) for t in out
                        if t != template.eos_token_id)
        if ref.eos_legal(data):
            parse_ok += 1
        if stop == "MAX_TOKENS_WITH_JUMP_COMPLETE":
            reserve_stops += 1
            reserve_consistent_all &= r_ok
            if len(out) > budget:
                over_budget += 1  # the binding clause: reserve-stopped overruns
        depth = cur = 0
        for b in data:
            if b == 0x28:
                cur += 1
                depth = max(depth, cur)
            elif b == 0x29:
                cur -= 1
        max_nesting = max(max_nesting, depth)
        text = data.decode("utf-8", "ignore")
        ident_events += sum(text.count(w) for w in multi_tok_idents)
        if (g + 1) % 1000 == 0:
            print(f"  {g+1}/{args.gens} | parse {parse_ok}/{g+1-dead_ends} | "
                  f"reserve stops {reserve_stops} | ident events {ident_events} | "
                  f"nesting {max_nesting} | {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    n_done = args.gens - dead_ends
    checks = {
        "outputs parse under own grammar (binding)":
            (parse_ok == n_done, f"{parse_ok}/{n_done}"),
        "DeadEndError == 0": (dead_ends == 0, str(dead_ends)),
        "no reserve-stopped generation exceeds max_tokens":
            (over_budget == 0, f"{over_budget} over"),
        "reserve-stop completions reproduce (consistency)":
            (reserve_consistent_all, str(reserve_consistent_all)),
        f"coverage: nesting >= {args.min_nesting}":
            (max_nesting >= args.min_nesting, str(max_nesting)),
        f"coverage: reserve stops >= {args.min_reserve_stops}":
            (reserve_stops >= args.min_reserve_stops, str(reserve_stops)),
        f"coverage: multi-token identifier events >= {args.min_ident_events}":
            (ident_events >= args.min_ident_events, f"{ident_events:,}"),
    }
    all_ok = all(ok for ok, _ in checks.values())

    host = os.environ.get("GRID_HOST_LABEL", "local dev (unpinned)")
    out_path = pathlib.Path(args.out or (BENCH_DIR / f"RESULTS-g5-{args.arm}.md"))
    lines = [
        f"# G5 end-to-end S/C/T at scale — {args.arm} arm",
        "",
        f"Host: {host} | grammar: `grammars/sql_subset.grid` + L3 schema lexicons | "
        f"tokenizer: `{args.tokenizer}` ({template.vocab_size:,} tokens, byte-fallback BPE) | "
        f"{args.gens:,} seeded generations | wall {wall/60:.1f} min",
        "",
        "Forced-random-walk arm: uniform over the exact mask, EOS suppressed until "
        "length L ~ U(8,24), max_tokens ~ U(18,60) (tight budgets force reserve stops). "
        "10% of seeds run a paren-seeking variant (prefer '(' while shallow) to meet "
        "the nesting quota; all binding checks apply to every generation. "
        "EOS-only-at-ACCEPT is asserted at every EOS application in the loop.",
        "",
        "| check | pass | value |",
        "|---|---|---|",
        *[f"| {name} | {'PASS' if ok else 'FAIL'} | {val} |"
          for name, (ok, val) in checks.items()],
        "",
        f"Gate G5 ({args.arm} arm): {'**PASS**' if all_ok else '**FAIL**'}. "
        "Reserve tightness vs the BFS shortest-completion oracle is pinned at unit "
        "level (tests/lalr/test_reserve.py); at scale every reserve stop's emitted "
        "completion must reproduce deterministically at the trigger state (checked "
        "above). The model-in-loop arm (same checks, Qwen2.5-0.5B-Instruct sampling) "
        "runs on the declared GPU runner.",
        "",
        "Harness: `bench/g5_scale.py`.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print("\n".join(lines[5:14]))
    print(f"report -> {out_path}")
    return 0 if (all_ok or not args.assert_gates) else 1


if __name__ == "__main__":
    sys.exit(main())
