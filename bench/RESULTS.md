# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `gpt2` | replays: 11 (491 steps total) | host: Lambda 1x H100 PCIe 80GB, Ubuntu 24.04 (declared runner; virtualized)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 596.7 ms | 25.6 us | 5935.4 us | 8624.4 us | -28.680 | 0 |
| XGrammar 0.2.3 (EBNF) | 170.6 ms | 63.2 us | 7216.1 us | 24304.9 us | -41.756 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 502.6 ms | 6.3 us | 221.2 us | 351.7 us | -1.197 | 2 |
| Outlines 1.3.1 (CFG backend = llguidance) | 28646.6 ms | 120.2 us | 559.0 us | 888.8 us | -2.864 | 2 |

GRID cache split: hit p50 25.2 us | miss p50 6.2 ms | hit rate 86%

GRID warm-replay R check (120 steps): slope -0.044 us/pos; first-half p50 39 us vs second-half p50 38 us — per-token cost tracks grammar configuration, not absolute position (requirement R).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
