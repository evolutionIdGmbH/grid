# End-to-end soundness/completeness/termination at scale — walk arm

Host: local M-series dev (unpinned), kernel v4 | grammar: `grammars/sql_subset.grid` + L3 schema lexicons | tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` (151,665 tokens, byte-fallback BPE) | 10,000 seeded generations | wall 5.3 min

Forced-random-walk arm: uniform over the exact mask, EOS suppressed until length L ~ U(8,24), max_tokens ~ U(18,60) (tight budgets force reserve stops). 10% of seeds run a paren-seeking variant (prefer '(' while shallow) to meet the nesting quota; all binding checks apply to every generation. EOS-only-at-ACCEPT is asserted at every EOS application in the loop.

| check | measured |
|---|---|
| outputs parse under own grammar (binding) | 10000/10000 |
| DeadEndError == 0 | 0 |
| no reserve-stopped generation exceeds max_tokens | 0 over |
| reserve-stop completions reproduce (consistency) | True |
| coverage: nesting >= 6 | 10 |
| coverage: reserve stops >= 100 | 9896 |
| coverage: multi-token identifier events >= 500 | 2,933 |

Summary (walk arm): all 10,000 generations parse under their own grammar with zero dead-ends and no over-budget reserve stops. Reserve tightness vs the BFS shortest-completion oracle is pinned at unit level (tests/lalr/test_reserve.py); at scale every reserve stop's emitted completion must reproduce deterministically at the trigger state (checked above). The model-in-loop arm (same checks, Qwen2.5-0.5B-Instruct sampling) runs on the declared GPU runner.

Harness: `bench/g5_scale.py`.
