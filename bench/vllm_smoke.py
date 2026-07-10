"""Mode-2 acceptance smoke: vLLM V1 + GridVLLMLogitsProcessor.

Batch of prompts constrained to the SQL-subset grammar with an L3 schema; every
constrained output must detokenize to a VIABLE PREFIX of the grammar (the mode-2
soundness guarantee), and at least one must be a complete statement. Truncated
outputs are expected under mode 2 — a logits processor cannot append the
reserve completion (DESIGN.md SS4.5); vLLM's max_tokens is the budget cut.

Requires the synchronous scheduler (see grid/models/vllm_processor.py).

Run (GPU host):  .venv/bin/python bench/vllm_smoke.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    from grid.models.vllm_processor import GridVLLMLogitsProcessor

    llm = LLM(
        model=args.model, logits_processors=[GridVLLMLogitsProcessor],
        gpu_memory_utilization=0.5, max_model_len=1024, enforce_eager=True,
        async_scheduling=False,  # sequence-stateful masking requires the sync scheduler
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        extra_args={"grid": {"grammar": GRAMMAR, "schema": SCHEMA}})
    outs = llm.generate(PROMPTS, sp)

    # oracle: the coverage-parse discipline, viable-prefix variant
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
            viable += 1  # mode-2 truncation: viable prefix, budget cut
            print(f"VIABLE(truncated)  {text[:80]!r}")
        else:
            print(f"FAIL({why})  {text[:80]!r}")
    print(f"SMOKE: {viable}/{len(outs)} viable prefixes, {complete} complete")
    sys.exit(0 if viable == len(outs) and complete >= 1 else 1)


if __name__ == "__main__":
    main()
