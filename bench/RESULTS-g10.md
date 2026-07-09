# G10 audit replay — full-scale run (E14)

Host: local M-series dev (unpinned), kernel v4 | grammar: `grammars/sql_subset.grid` | MockTokenizer (48 tokens) | mode-1 GRID-owned loop, max_tokens 40

- generations: **1000** (seeded multinomial over MockModel logits), 29,242 audited steps total (Write and EOS records included)
- namespace rollovers spanned: **1** (at generation 500; entries recompute content-addressed, replays of pre-rollover generations must still match)
- replay: **1000/1000 bit-identical record chains** (chain hash sequences compared record-by-record; 1.8s)
- tamper property: **1000/1000 detected** (random record x random field per trial)
- generation wall: 3.4s

Gate G10: **PASS** (criteria: every step of >=1,000 generations replayed bit-identical across >=1 namespace rollover; tamper detection 100% over >=10^3 trials).

Harness: `bench/g10_replay.py` (G10a smoke-scale versions of these properties run in CI: tests/audit/test_audit.py).
