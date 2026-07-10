# MaskBench (guidance-ai/jsonschemabench) — GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 15 schemas/split, seed 0 (315 schemas, 21 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 683 | 20 | 103 |
| TBM p25 | 15 | 5 | 3 |
| TBM p50 | 31 | 10 | 11 |
| TBM p75 | 38 | 21 | 30 |
| TBM p90 | 208 | 32 | 60 |
| TBM p95 | 7,760 | 59 | 338 |
| TBM p99 | 8,278 | 185 | 2,680 |
| TBM p99.9 | 46,028 | 1,079 | 7,790 |
| TBM max | 115,824 | 2,015 | 12,455 |
| TTFM avg | 17,488 | 622 | 705,140 |
| TTFM p25 | 5,325 | 229 | 848 |
| TTFM p50 | 7,188 | 320 | 2,408 |
| TTFM p75 | 9,940 | 441 | 211,992 |
| TTFM p90 | 19,389 | 952 | 1,219,010 |
| TTFM p95 | 81,170 | 1,788 | 3,025,603 |
| TTFM p99 | 320,163 | 7,956 | 13,383,740 |
| tokens | 58,188 | 62,311 | 70,275 |
| schemas | 315 | 315 | 315 |
| passing | 206 | 251 | 283 |
| compile error | 79 | 62 | 0 |
| timeout | 0 | 0 | 0 |
| validation error | 0 | 3 | 27 |
| invalidation error | 68 | 0 | 37 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once — the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.0.7, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention — these surface as invalidation errors when an invalid instance hinges on them): minimum (33), maximum (24), pattern (21), oneOf-exclusivity (18), minLength (13), maxLength (13), format (12), minItems (9), uniqueItems (9), minProperties (6), maxItems (5), multipleOf (1).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): Unsupported: allOf (30), Unsupported: unsupported keys ['patternProperties'] (14), LALRConflictError (5), Unsupported: unsupported keys ['not'] (4), Unsupported: unsupported keys ['additionalItems'] (3), Unsupported: $ref with sibling keys ['type'] (3), Unsupported: unsupported keys ['dependencies'] (3), Unsupported: unsupported keys ['else', 'if', 'then'] (3), Unsupported: oneOf with sibling keys ['additionalProperties' (3), Unsupported: anyOf with sibling keys ['additionalProperties' (2).
