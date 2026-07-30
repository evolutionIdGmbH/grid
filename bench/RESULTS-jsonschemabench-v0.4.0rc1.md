# MaskBench (guidance-ai/jsonschemabench) - GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 1000000 schemas/split, seed 0 (11306 schemas, 1 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | GRID | GRID |
|:---|---:|---:|---:|
| TBM avg | 485 | 559 | 513 |
| TBM p25 | 9 | 9 | 11 |
| TBM p50 | 25 | 25 | 26 |
| TBM p75 | 32 | 32 | 35 |
| TBM p90 | 78 | 78 | 99 |
| TBM p95 | 7,232 | 7,171 | 7,332 |
| TBM p99 | 7,596 | 7,435 | 8,228 |
| TBM p99.9 | 8,090 | 7,814 | 8,980 |
| TBM max | 1,676,539 | 513,902 | 1,049,484 |
| TTFM avg | 418,844 | 146,984 | 65,684 |
| TTFM p25 | 5,682 | 4,841 | 5,198 |
| TTFM p50 | 8,608 | 7,133 | 7,438 |
| TTFM p75 | 27,794 | 18,989 | 19,875 |
| TTFM p90 | 216,742 | 78,733 | 83,851 |
| TTFM p95 | 671,923 | 198,600 | 216,349 |
| TTFM p99 | 4,366,059 | 1,173,082 | 1,157,520 |
| tokens | 3,461,146 | 3,489,325 | 3,491,288 |
| schemas | 11,306 | 11,306 | 11,306 |
| passing | 10,117 | 10,154 | 10,159 |
| compile error | 668 | 635 | 637 |
| timeout | 16 | 7 | 0 |
| validation error | 5 | 5 | 5 |
| invalidation error | 870 | 876 | 876 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once - the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.2.0, GRID 0.2.0, GRID 0.2.5.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (471), required-not-enforced (required-set beyond cap) (415), scanner-budget: constrained string degraded (278), maxLength-with-pattern (207), scanner-budget: length window degraded (173), minLength-with-pattern (171), length (length window (0,255) beyond cap) (163), length (length window (1,255) beyond cap) (154), uniqueItems (105), string-constraint-terminal-too-large (104), not-unenforced (99), length (length window (0,32767) beyond cap) (84).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (542), Unsupported: allOf (merge failed) (29), TypeError (14), Unsupported: terminal budget exceeded (size cap) (9), RxUnsupported (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: rule budget exceeded (size cap) (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), Unsupported: anyOf with sibling keys ['properties'] (2).

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (494), required-not-enforced (required-set beyond cap) (411), scanner-budget: constrained string degraded (277), maxLength-with-pattern (210), scanner-budget: length window degraded (193), minLength-with-pattern (172), length (length window (0,255) beyond cap) (163), length (length window (1,255) beyond cap) (154), string-constraint-terminal-too-large (128), uniqueItems (105), pp-extras-complement-unavailable (99), not-unenforced (97).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (519), Unsupported: allOf (merge failed) (30), RxUnsupported (9), Unsupported: terminal budget exceeded (size cap) (9), Unsupported: rule budget exceeded (size cap) (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), Unsupported: anyOf with sibling keys ['properties'] (2), Unsupported: oneOf with sibling keys ['additionalProperties' (2).

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (433), required-not-enforced (required-set beyond cap) (350), scanner-budget: constrained string degraded (240), maxLength-with-pattern (170), minLength-with-pattern (162), length (length window (0,255) beyond cap) (158), scanner-budget: length window degraded (157), length (length window (1,255) beyond cap) (142), string-constraint-terminal-too-large (97), length (length window (0,32767) beyond cap) (78), length (length window (0,1024) beyond cap) (75), minimum (73).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (519), Unsupported: allOf (merge failed) (30), RxUnsupported (9), Unsupported: terminal budget exceeded (size cap) (9), Unsupported: rule budget exceeded (size cap) (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), LALRBudgetExceeded (2), Unsupported: anyOf with sibling keys ['properties'] (2).

## Waves A+B+C vs v0.3.0 vs v0.2.5 (columns above: v0.2.5 | v0.3.0rc2 | this run)

Configuration: main @ 0d57bed, SHIPPED DEFAULTS (factored scanner, DP LALR,
hashcons, kernel-lazy on grid_core v8, LALR construction budget, component
budgets, LALR-conflict retry; counting/slicer/jump-forward/store present
but default-off). One machine, one process, 120s wall.

HEADLINE: zero timeouts. All 11,306 schemas terminate deterministically -
compile, declare, or budget-decline. Full-corpus outcome enumeration vs
v0.3.0rc2 (bench/perfbench/outcomes.py): 7 improved, 11,299 unchanged,
zero regressions. The 7: the five substring-union schemas
(o5195/o47656/o47657/o48423/o48427) timeout -> ok via component budgets
(P3); helm-testsuite and o27148 timeout -> declared:LALRBudgetExceeded
in <1s (P5).

TTFM avg 419ms (v0.2.5) -> 147ms (v0.3.0) -> 66ms; p99 4,366 -> 1,158ms.
Median ~7.4ms stable across the epoch by design; the p90/p95 upticks vs
rc2 (79 -> 84ms, 199 -> 216ms) and the TBM p90+ upticks are within the
single-run noise band on unpinned hardware and carry the standing load
caveat. TBM warm p50 25-26us throughout.

Two-column TTFM (E4) and warm-store numbers are recorded per-set in
bench/perfbench/BAKEOFF.md postscripts; this table remains compile-only
TTFM for cross-version comparability with the v0.2.5 baseline.
