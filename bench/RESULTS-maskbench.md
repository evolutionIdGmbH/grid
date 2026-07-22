# MaskBench (guidance-ai/jsonschemabench) - GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 15 schemas/split, seed 0 (315 schemas, 21 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 562 | 20 | 101 |
| TBM p25 | 14 | 5 | 3 |
| TBM p50 | 28 | 10 | 10 |
| TBM p75 | 34 | 21 | 29 |
| TBM p90 | 75 | 31 | 57 |
| TBM p95 | 7,371 | 57 | 326 |
| TBM p99 | 7,665 | 179 | 2,658 |
| TBM p99.9 | 7,963 | 1,055 | 7,472 |
| TBM max | 11,671 | 2,013 | 12,049 |
| TTFM avg | 14,473 | 595 | 689,160 |
| TTFM p25 | 4,539 | 227 | 770 |
| TTFM p50 | 5,975 | 313 | 2,393 |
| TTFM p75 | 8,079 | 436 | 209,317 |
| TTFM p90 | 15,073 | 821 | 1,170,668 |
| TTFM p95 | 31,098 | 1,704 | 2,964,222 |
| TTFM p99 | 301,510 | 7,721 | 13,010,521 |
| tokens | 58,188 | 62,311 | 70,275 |
| schemas | 315 | 315 | 315 |
| passing | 206 | 251 | 283 |
| compile error | 79 | 62 | 0 |
| timeout | 0 | 0 | 0 |
| validation error | 0 | 3 | 27 |
| invalidation error | 68 | 0 | 37 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once - the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut first 9.3x by the kernel v5.1 verdict-equivalence grouping (TBM p90 27.8 ms -> 208 us vs the v3-era run), then a further ~2.8x by the kernel v7 fused walk->blob->register path that eliminated the Python-side per-cold-entry materialization/glue cost (6.8-8.7 ms -> 0.003 ms per boundary entry) and its gen-2 GC pauses (this record; TBM p90 208 us -> 75 us).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.0.7, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): minimum (33), maximum (24), pattern (21), oneOf-exclusivity (18), minLength (13), maxLength (13), format (12), minItems (9), uniqueItems (9), minProperties (6), maxItems (5), multipleOf (1).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): Unsupported: allOf (30), Unsupported: unsupported keys ['patternProperties'] (14), LALRConflictError (5), Unsupported: unsupported keys ['not'] (4), Unsupported: unsupported keys ['additionalItems'] (3), Unsupported: $ref with sibling keys ['type'] (3), Unsupported: unsupported keys ['dependencies'] (3), Unsupported: unsupported keys ['else', 'if', 'then'] (3), Unsupported: oneOf with sibling keys ['additionalProperties' (3), Unsupported: anyOf with sibling keys ['additionalProperties' (2).
