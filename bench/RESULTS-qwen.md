# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` | replays: 11 (509 steps total) | host: local dev (unpinned — G7/G9 bind on the declared cloud runner)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 1447.7 ms | 5.0 us | 112.9 us | 15795.6 us | -52.244 | 0 |
| XGrammar 0.2.3 (EBNF) | 324.6 ms | 586.8 us | 10004.5 us | 31607.8 us | -62.760 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 1017.5 ms | 14.9 us | 385.9 us | 1206.2 us | -1.834 | 1 |
| Outlines 1.3.1 (CFG backend = llguidance) | 14211.9 ms | 65.2 us | 461.9 us | 576.9 us | -2.264 | 1 |

GRID cache split: hit p50 4.9 us | miss p50 13.9 ms | hit rate 92%

GRID warm-replay R check (120 steps): slope -0.005 us/pos; first-half p50 5 us vs second-half p50 5 us — per-token cost tracks grammar configuration, not absolute position (requirement R).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
