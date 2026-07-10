# GRID vs XGrammar vs llguidance vs Outlines — SQL-subset constrained decoding

Tokenizer: `gpt2` | replays: 11 (491 steps total) | host: local dev (unpinned — G7/G9 bind on the declared cloud runner)

GRID's hot path runs in grid_core Rust kernels: the trie walk (in-kernel CD grouping + alias expansion) and the per-step CD-group verdicts + LALR simulate; masks stay in i32 buffers end-to-end. Cold misses pay the full walk (see the cache split). Outlines' CFG path delegates to llguidance (CFG_DEFAULT_BACKEND='llguidance'), so the Outlines and raw-llguidance arms share the same core matcher — the Outlines row adds outlines' logits-processor wrapper (consume + bitmask fill + apply).

| engine | compile | p50 | p90 | p99 | slope (us/pos) | rejected replays |
|---|---|---|---|---|---|---|
| GRID (grid_core Rust kernels: walk + CD verdicts + LALR) | 427.8 ms | 3.7 us | 82.2 us | 5147.6 us | -9.288 | 0 |
| XGrammar 0.2.3 (EBNF) | 103.0 ms | 70.4 us | 7514.9 us | 25474.9 us | -43.668 | 0 |
| llguidance 1.7.6 (lark, driven directly) | 288.1 ms | 7.8 us | 219.5 us | 342.4 us | -1.173 | 2 |
| Outlines 1.3.1 (CFG backend = llguidance) | 22729.4 ms | 73.2 us | 431.1 us | 668.3 us | -2.190 | 2 |

GRID cache split: hit p50 3.6 us | miss p50 4.9 ms | hit rate 92%

GRID warm-replay R check (120 steps): slope -0.000 us/pos; first-half p50 3 us vs second-half p50 3 us — per-token cost tracks grammar configuration, not absolute position (requirement R).

Notes:
- Rejected replays count language-parity corners between the grammar encodings
  (maximal-munch vs explicit-whitespace), not correctness bugs.
- Outlines has no independent CFG engine: `outlines.types.CFG` routes to a
  backend, default llguidance (`CFG_DEFAULT_BACKEND`), so its row tracks
  llguidance plus wrapper overhead (JSON-schema/regex default to outlines_core).
