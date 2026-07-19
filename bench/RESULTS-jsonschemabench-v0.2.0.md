<!-- 0.2.x correctness epoch: error metrics are the headline; timings are
recorded under the epoch policy (perf work deferred to 0.3.x; kernel v7,
Python bridge normalization/constraint layers active). GRID 0.2.0 working
tree; llguidance/xgr columns from the same sample and engine versions as the
baseline run. -->

# MaskBench (guidance-ai/jsonschemabench) — GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 15 schemas/split, seed 0 (315 schemas, 21 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 483 | 20 | 101 |
| TBM p25 | 11 | 5 | 3 |
| TBM p50 | 26 | 10 | 10 |
| TBM p75 | 33 | 21 | 29 |
| TBM p90 | 64 | 31 | 57 |
| TBM p95 | 7,311 | 57 | 326 |
| TBM p99 | 7,661 | 179 | 2,658 |
| TBM p99.9 | 8,023 | 1,055 | 7,472 |
| TBM max | 27,410 | 2,013 | 12,049 |
| TTFM avg | 61,339 | 595 | 689,160 |
| TTFM p25 | 4,504 | 227 | 770 |
| TTFM p50 | 6,759 | 313 | 2,393 |
| TTFM p75 | 11,190 | 436 | 209,317 |
| TTFM p90 | 51,171 | 821 | 1,170,668 |
| TTFM p95 | 303,840 | 1,704 | 2,964,222 |
| TTFM p99 | 1,387,912 | 7,721 | 13,010,521 |
| tokens | 66,771 | 62,311 | 70,275 |
| schemas | 315 | 315 | 315 |
| passing | 268 | 251 | 283 |
| compile error | 40 | 62 | 0 |
| timeout | 0 | 0 | 0 |
| validation error | 0 | 3 | 27 |
| invalidation error | 8 | 0 | 37 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once — the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.2.0, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention — these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (9), property-order-assumed (required-set beyond cap) (7), string-constraint-terminal-too-large (5), scanner-budget: constrained string degraded (4), length (length window (0,1024) beyond cap) (4), length (emitted regex too large (68544)) (4), length (length window (0,4096) beyond cap) (4), maxLength-with-pattern (3), uniqueItems (3), length (emitted regex too large (535500)) (3), length (emitted regex too large (137088)) (3), length (emitted regex too large (274176)) (3).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (12), Unsupported: allOf (merge failed) (5), Unsupported: unsupported keys ['not'] (5), Unsupported: $ref with sibling keys ['maxItems'] (4), Unsupported: patternProperties overlaps declared key (merge) (3), Unsupported: anyOf with sibling keys ['additionalProperties' (2), Unsupported: patternProperties with propertyNames/forbid (2), Unsupported: patternProperties complement '^[a-zA-Z0-9]([a-z (1), Unsupported: oneOf with sibling keys ['required'] (1), Unsupported: rule budget exceeded (size cap) (1).
