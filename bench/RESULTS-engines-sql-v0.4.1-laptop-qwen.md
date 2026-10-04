# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` | replays: 11 (509 steps total) | host: local dev: Apple M3 Max, 36 GB (unpinned)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 757.8 ms | 3.8 us | 28.6 us | 4195.7 us | -7.500 | 0 |
| XGrammar 0.2.3 (EBNF) | 173.2 ms | 321.3 us | 5617.5 us | 17571.9 us | -34.842 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 562.6 ms | 16.1 us | 263.2 us | 1076.9 us | -1.324 | 1 |
| Outlines 1.3.1 (CFG backend = llguidance) | 1395.7 ms | 163.0 us | 369.5 us | 739.5 us | -1.088 | 1 |

GRID cache split: hit p50 3.8 us | miss p50 3.9 ms, p90 4.3 ms, p99 9.0 ms (the cold-walk distribution a one-shot workload sees) | hit rate 92%

Common-acceptance subset (10/11 replays that every engine accepts in full; percentiles over identical steps):

| engine | steps | p50 | p90 | p99 |
|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 389 | 3.9 us | 2714.4 us | 4291.1 us |
| XGrammar 0.2.3 (EBNF) | 389 | 321.1 us | 10046.1 us | 17901.9 us |
| llguidance 1.7.6 (lark, driven directly) | 389 | 16.1 us | 264.6 us | 1076.9 us |
| Outlines 1.3.1 (CFG backend = llguidance) | 389 | 162.5 us | 369.6 us | 739.5 us |

GRID warm-replay flat-cost check (120 steps): slope -0.001 us/pos; first-half p50 4 us vs second-half p50 4 us — per-token cost tracks grammar configuration, not absolute position (flat per-token cost).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
