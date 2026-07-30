# MaskBench (guidance-ai/jsonschemabench) - GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 15 schemas/split, seed 0 (315 schemas, 21 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 487 | 19 | 100 |
| TBM p25 | 10 | 5 | 3 |
| TBM p50 | 26 | 10 | 10 |
| TBM p75 | 33 | 21 | 29 |
| TBM p90 | 64 | 31 | 55 |
| TBM p95 | 7,369 | 56 | 335 |
| TBM p99 | 8,186 | 177 | 2,628 |
| TBM p99.9 | 8,596 | 1,040 | 7,501 |
| TBM max | 40,086 | 2,047 | 11,755 |
| TTFM avg | 46,725 | 590 | 683,195 |
| TTFM p25 | 4,220 | 213 | 754 |
| TTFM p50 | 5,795 | 304 | 2,143 |
| TTFM p75 | 9,135 | 422 | 207,151 |
| TTFM p90 | 52,695 | 829 | 1,154,529 |
| TTFM p95 | 210,265 | 1,719 | 2,929,719 |
| TTFM p99 | 1,441,582 | 7,642 | 12,976,596 |
| tokens | 72,099 | 62,311 | 70,275 |
| schemas | 315 | 315 | 315 |
| passing | 282 | 251 | 283 |
| compile error | 26 | 62 | 0 |
| timeout | 0 | 0 | 0 |
| validation error | 0 | 3 | 27 |
| invalidation error | 7 | 0 | 37 |

Sample refresh at GRID 0.4.0 (llguidance 1.7.6, XGrammar 0.2.3; same 15-per-split
seed-0 sample as the historical run). Vs the GRID 0.0.7 record this file
previously held: passing 206 -> 282, invalidation errors 68 -> 7, validation
errors 0 -> 0. GRID's TTFM average RISES vs 0.0.7 (14.5ms -> 46.7ms on this
sample) because 0.0.7 only ever compiled the easy 65% and declined the rest -
the average is now taken over a strictly harder compiled population. The
authoritative full-corpus record (11,306 schemas) is
RESULTS-jsonschemabench-v0.4.0rc1.md.

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once - the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.4.0, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (14), required-not-enforced (required-set beyond cap) (7), pp-extras-complement-unavailable (5), branch-string-values-unified (5), string-constraint-terminal-too-large (5), scanner-budget: length window degraded (5), maxLength-with-pattern (4), minLength-with-pattern (4), not-unenforced (4), length (length window (0,1024) beyond cap) (4), scanner-budget: constrained string degraded (4), pp-overlap-merge-unenforced (3).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (19), Unsupported: anyOf with sibling keys ['additionalProperties' (2), Unsupported: oneOf with sibling keys ['required'] (1), Unsupported: rule budget exceeded (size cap) (1), Unsupported: anyOf with sibling keys ['minProperties'] (1), RxUnsupported (1), Unsupported: $ref with sibling keys ['type'] (1).
