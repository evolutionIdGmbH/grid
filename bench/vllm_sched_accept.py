"""M6 scheduler-side acceptance: GRID as a vLLM structured-output backend.

Applies bench/vllm_grid_patch.py (idempotent), then runs a constrained batch
with the DEFAULT scheduler — async scheduling allowed, the restriction that
mode 2 (the logits-processor route) carries — and verifies every output is a
viable prefix of the grammar (>= 1 complete) under the coverage oracle.

Accepted on GPU 2026-07-08 (Lambda 1x A10, vllm 0.24.0, Qwen2.5-0.5B-Instruct):
4/4 viable, 1 complete, zero desyncs.

Run (GPU host):  .venv/bin/python bench/vllm_sched_accept.py
"""

from __future__ import annotations

import json
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


def main() -> None:
    import vllm_grid_patch

    vllm_grid_patch.main()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    envelope = json.dumps({"grammar": GRAMMAR, "schema": SCHEMA})
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", gpu_memory_utilization=0.5,
              max_model_len=1024, enforce_eager=True,
              structured_outputs_config={"backend": "grid"})
    sp = SamplingParams(temperature=0.0, max_tokens=96,
                        structured_outputs=StructuredOutputsParams(grammar=envelope))
    outs = llm.generate(PROMPTS, sp)

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
    print(f"SCHED-ACCEPT: {viable}/{len(outs)} viable, {complete} complete")
    sys.exit(0 if viable == len(outs) and complete >= 1 else 1)


if __name__ == "__main__":
    main()
