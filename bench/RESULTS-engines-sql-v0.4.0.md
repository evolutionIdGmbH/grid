# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Measured 2026-07-31 at GRID 0.4.0 (grid_core 0.2.0, kernel v8), shipped defaults. Engine versions pinned in the table; produced by `bench/compare_engines.py` on the declared runner.

Tokenizer: `gpt2` | replays: 11 (491 steps total) | host: Lambda 1xH100 SXM 80GB HBM3, Ubuntu 24.04 (declared runner)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 391.1 ms | 3.7 us | 32.2 us | 2827.0 us | -4.991 | 0 |
| XGrammar 0.2.3 (EBNF) | 103.4 ms | 66.9 us | 7577.8 us | 25531.1 us | -43.702 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 301.9 ms | 8.5 us | 222.7 us | 357.2 us | -1.183 | 2 |

GRID cache split: hit p50 3.6 us | miss p50 2.7 ms | hit rate 92%

GRID warm-replay flat-cost check (120 steps): slope +0.005 us/pos; first-half p50 3 us vs second-half p50 4 us — per-token cost tracks grammar configuration, not absolute position (flat per-token cost).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
