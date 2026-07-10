# Audit-trail replay and tamper detection — full-scale run (E14)

Host: local dev (unpinned) | grammar: `grammars/sql_subset.grid` | MockTokenizer (48 tokens) | mode-1 GRID-owned loop, max_tokens 40 | key format: v2

- v1-log dual-key compat: **8/8 bit-identical** (legacy-key logs replayed through the genN producer; every consulted config byte-compared under both key forms)
- generations: **1000** (seeded multinomial over MockModel logits), 23,470 audited steps total (Write and EOS records included)
- namespace rollovers spanned: **1** (at generation 500; entries recompute content-addressed, replays of pre-rollover generations must still match)
- replay: **1000/1000 bit-identical record chains** (chain hash sequences compared record-by-record; 0.5s)
- tamper property: **1000/1000 detected** (random record x random field per trial)
- generation wall: 1.0s

Summary: every step of 1,000 generations replays bit-identical across a namespace rollover, tamper detection is 100% over 1,000 trials, and v1-format logs replay bit-identical via the dual-key path.

Harness: `bench/g10_replay.py` (smoke-scale versions of these properties run in CI: tests/audit/test_audit.py).
