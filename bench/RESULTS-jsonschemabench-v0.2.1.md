# MaskBench (guidance-ai/jsonschemabench) — GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 15 schemas/split, seed 0 (315 schemas, 21 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 528 | 20 | 101 |
| TBM p25 | 11 | 5 | 3 |
| TBM p50 | 29 | 10 | 10 |
| TBM p75 | 37 | 21 | 29 |
| TBM p90 | 82 | 31 | 57 |
| TBM p95 | 7,637 | 57 | 326 |
| TBM p99 | 7,927 | 179 | 2,658 |
| TBM p99.9 | 8,732 | 1,055 | 7,472 |
| TBM max | 24,315 | 2,013 | 12,049 |
| TTFM avg | 96,952 | 595 | 689,160 |
| TTFM p25 | 4,998 | 227 | 770 |
| TTFM p50 | 7,096 | 313 | 2,393 |
| TTFM p75 | 11,276 | 436 | 209,317 |
| TTFM p90 | 54,208 | 821 | 1,170,668 |
| TTFM p95 | 319,683 | 1,704 | 2,964,222 |
| TTFM p99 | 3,468,279 | 7,721 | 13,010,521 |
| tokens | 66,069 | 62,311 | 70,275 |
| schemas | 315 | 315 | 315 |
| passing | 268 | 251 | 283 |
| compile error | 41 | 62 | 0 |
| timeout | 0 | 0 | 0 |
| validation error | 0 | 3 | 27 |
| invalidation error | 6 | 0 | 37 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once — the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.2.0, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention — these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (8), property-order-assumed (required-set beyond cap) (7), scanner-budget: constrained string degraded (6), length (length window (0,1024) beyond cap) (4), scanner-budget: length window degraded (4), length (length window (0,4096) beyond cap) (4), maxLength-with-pattern (3), uniqueItems (3), length (length window (0,500) beyond cap) (3), length (length window (0,256) beyond cap) (3), length (length window (0,375) beyond cap) (2), length (length window (0,65535) beyond cap) (2).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (13), Unsupported: allOf (merge failed) (5), Unsupported: unsupported keys ['not'] (5), Unsupported: $ref with sibling keys ['maxItems'] (4), Unsupported: patternProperties overlaps declared key (merge) (3), Unsupported: anyOf with sibling keys ['additionalProperties' (2), Unsupported: patternProperties with propertyNames/forbid (2), Unsupported: patternProperties complement '^[a-zA-Z0-9]([a-z (1), Unsupported: oneOf with sibling keys ['required'] (1), Unsupported: rule budget exceeded (size cap) (1).
