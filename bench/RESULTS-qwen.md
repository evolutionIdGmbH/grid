# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` | replays: 11 (509 steps total) | host: Lambda 1x H100 PCIe 80GB, Ubuntu 24.04 (declared runner; virtualized)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 2130.8 ms | 91.2 us | 12065.9 us | 17149.7 us | -75.165 | 0 |
| XGrammar 0.2.3 (EBNF) | 553.9 ms | 565.9 us | 10487.3 us | 30245.8 us | -60.252 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 1772.4 ms | 13.2 us | 379.4 us | 1185.6 us | -1.826 | 1 |
| Outlines 1.3.1 (CFG backend = llguidance) | 14694.6 ms | 68.9 us | 560.4 us | 1447.8 us | -7.199 | 1 |

GRID cache split: hit p50 88.2 us | miss p50 15.6 ms | hit rate 87%

GRID warm-replay R check (120 steps): slope -0.265 us/pos; first-half p50 97 us vs second-half p50 82 us — per-token cost tracks grammar configuration, not absolute position (requirement R).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
