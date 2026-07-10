# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `gpt2` | replays: 11 (491 steps total) | host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner), kernel v7

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 378.0 ms | 3.6 us | 80.0 us | 5347.4 us | -9.292 | 0 |
| XGrammar 0.2.3 (EBNF) | 94.1 ms | 72.2 us | 7503.3 us | 25586.7 us | -43.759 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 285.9 ms | 6.6 us | 223.9 us | 351.7 us | -1.176 | 2 |
| Outlines 1.3.1 (CFG backend = llguidance) | 22115.1 ms | 73.9 us | 431.5 us | 582.4 us | -2.152 | 2 |

GRID cache split: hit p50 3.5 us | miss p50 4.8 ms | hit rate 92%

GRID warm-replay R check (120 steps): slope +0.002 us/pos; first-half p50 3 us vs second-half p50 4 us — per-token cost tracks grammar configuration, not absolute position (requirement R).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
