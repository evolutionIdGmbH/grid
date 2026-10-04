# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `gpt2` | replays: 11 (491 steps total) | host: local dev: Apple M3 Max, 36 GB (unpinned)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 223.3 ms | 3.0 us | 19.1 us | 2443.7 us | -4.185 | 0 |
| XGrammar 0.2.3 (EBNF) | 53.9 ms | 39.3 us | 4153.6 us | 14591.2 us | -24.475 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 181.6 ms | 6.2 us | 146.3 us | 215.9 us | -0.920 | 2 |
| Outlines 1.3.1 (CFG backend = llguidance) | 7759.1 ms | 142.8 us | 280.1 us | 373.7 us | -1.041 | 2 |

GRID cache split: hit p50 3.0 us | miss p50 2.3 ms, p90 2.5 ms, p99 2.7 ms (the cold-walk distribution a one-shot workload sees) | hit rate 92%

Common-acceptance subset (9/11 replays that every engine accepts in full; percentiles over identical steps):

| engine | steps | p50 | p90 | p99 |
|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 251 | 3.3 us | 1649.0 us | 2458.6 us |
| XGrammar 0.2.3 (EBNF) | 251 | 39.5 us | 8233.5 us | 14783.1 us |
| llguidance 1.7.6 (lark, driven directly) | 251 | 6.0 us | 143.7 us | 215.3 us |
| Outlines 1.3.1 (CFG backend = llguidance) | 251 | 140.0 us | 278.5 us | 373.7 us |

GRID warm-replay flat-cost check (120 steps): slope -0.002 us/pos; first-half p50 3 us vs second-half p50 3 us — per-token cost tracks grammar configuration, not absolute position (flat per-token cost).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
