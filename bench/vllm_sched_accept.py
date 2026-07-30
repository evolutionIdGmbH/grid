"""Scheduler-side acceptance: GRID as a vLLM structured-output backend.

Applies bench/vllm_grid_patch.py (idempotent), then runs a constrained batch
with the DEFAULT scheduler — async scheduling allowed, the restriction that
mode 2 (the logits-processor route) carries — and verifies every output is a
viable prefix of the grammar (>= 1 complete) under the coverage oracle.

Accepted on GPU 2026-07-08 (Lambda 1x A10, vllm 0.24.0, Qwen2.5-0.5B-Instruct):
4/4 viable, 1 complete, zero desyncs.

Run (GPU host):  .venv/bin/python bench/vllm_sched_accept.py

--jump (S1 smoke, NOT yet run on a box — blocked-remote until the next GPU
session): adds a second leg with GRID_JUMP=1 (fresh engine; patch site 5
injects forced runs as draft tokens) and requires greedy outputs
token-identical to the jump-off leg — at forced positions the bitmask
admits exactly one token, so draft acceptance is certain and parity is the
lever's exactness gate. The coverage oracle runs on BOTH legs. Best-effort:
prints spec-decode acceptance telemetry when the engine exposes it (the
step-count evidence lives in vllm_serving_bench.py --jump-probe).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

GRAMMAR = (pathlib.Path(__file__).parent.parent / "grammars" / "sql_subset.grid").read_text()
SCHEMA = {"users": ["id", "name", "email"], "orders": ["id", "user_id", "total"]}
PROMPTS = [
    "Write one lowercase SQL query listing all user names: ",
    "Write one lowercase SQL query counting orders: ",
    "Write one lowercase SQL query, total per user: ",
    "Write one lowercase SQL query deleting old orders: ",
]


def _one_leg(jump: str | None) -> list:
    """One fresh engine + constrained greedy batch. GRID_JUMP is read at
    session construction inside the engine process, so it is pinned in the
    env before the LLM builds (spawned engine procs inherit it)."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    if jump is not None:
        os.environ["GRID_JUMP"] = jump
    envelope = json.dumps({"grammar": GRAMMAR, "schema": SCHEMA})
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.5,
              max_model_len=1024, enforce_eager=True,
              structured_outputs_config={"backend": "grid"})
    sp = SamplingParams(temperature=0.0, max_tokens=96,
                        structured_outputs=StructuredOutputsParams(grammar=envelope))
    outs = llm.generate(PROMPTS, sp)
    try:  # best-effort spec-decode telemetry (engine/version dependent)
        metrics = llm.llm_engine.get_metrics()
        for m in metrics:
            if "spec" in getattr(m, "name", ""):
                print(f"  [telemetry] {m.name}: {getattr(m, 'value', m)}")
    except Exception:
        pass
    del llm
    return outs


def main() -> None:
    import vllm_grid_patch

    vllm_grid_patch.main()

    jump_ab = "--jump" in sys.argv[1:]
    outs = _one_leg("0" if jump_ab else None)

    from spider_coverage import parse_ok

    from grid.grammar import spec as gspec
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.lexer.dfa import build_scanner
    from grid.trie.walk import Lexicons

    grammar = gspec.load(GRAMMAR)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    prio = {t: (0 if t in tables.literal_terminal_ids else 1, t)
            for t in range(tables.n_terminals)}
    t_id = tables.terminal_names.index("TABLE_NAME")
    c_id = tables.terminal_names.index("COLUMN_NAME")
    lex = Lexicons({t_id: {t.encode() for t in SCHEMA},
                    c_id: {c.encode() for cs in SCHEMA.values() for c in cs}})

    def coverage(outs, label):
        complete = viable = 0
        for o in outs:
            text = o.outputs[0].text.strip()
            good, why = parse_ok(tables, dfa, prio, text.encode(), lex)
            if good:
                complete += 1
                viable += 1
                print(f"PASS(complete)  {text[:80]!r}")
            elif why == "incomplete-at-end":
                viable += 1
                print(f"VIABLE(truncated)  {text[:80]!r}")
            else:
                print(f"FAIL({why})  {text[:80]!r}")
        print(f"SCHED-ACCEPT[{label}]: {viable}/{len(outs)} viable, {complete} complete")
        return viable == len(outs) and complete >= 1

    ok = coverage(outs, "jump-off" if jump_ab else "default")

    if jump_ab:
        outs_on = _one_leg("1")
        os.environ.pop("GRID_JUMP", None)
        ok &= coverage(outs_on, "jump-on")
        parity = all(
            list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids)
            for a, b in zip(outs, outs_on, strict=True)
        )
        # greedy + forced-position certainty => token identity is the gate
        print(f"JUMP-PARITY: {'OK' if parity else 'FAILED'} "
              f"({len(outs)} greedy requests, off vs on)")
        if not parity:
            for i, (a, b) in enumerate(zip(outs, outs_on, strict=True)):
                ta, tb = list(a.outputs[0].token_ids), list(b.outputs[0].token_ids)
                if ta != tb:
                    k = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                             min(len(ta), len(tb)))
                    print(f"  request {i}: first divergence at token {k}")
        ok &= parity

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
