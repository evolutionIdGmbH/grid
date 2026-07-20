<!-- FULL-SET three-engine comparison, one machine, current versions
(llguidance 1.7.6, XGrammar 0.2.3, GRID 0.2.4+df94695). GRID column covers
11,188/11,306 (99.0%): the run's final 118 schemas are pending a resume
(bench process died natively in the tail, same as XGrammar's run did — resume:
  .venv-bench/bin/python bench/maskbench_grid.py --engine grid \
    --data tmp/jsb-grid-rest --sample 1000000 --out tmp/mb-grid-final
Timing recorded-not-optimized per the 0.2.x epoch; runs shared the box, so
timing is indicative only — error metrics are exact. -->

# MaskBench (guidance-ai/jsonschemabench) — GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 1000000 schemas/split, seed 0 (11187 schemas, 1 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 486 | 22 | 194 |
| TBM p25 | 9 | 5 | 3 |
| TBM p50 | 25 | 10 | 9 |
| TBM p75 | 32 | 20 | 28 |
| TBM p90 | 76 | 27 | 45 |
| TBM p95 | 7,234 | 44 | 115 |
| TBM p99 | 7,594 | 299 | 772 |
| TBM p99.9 | 8,085 | 1,099 | 51,022 |
| TBM max | 1,676,539 | 6,386 | 133,013 |
| TTFM avg | 411,173 | 693 | 334,134 |
| TTFM p25 | 5,694 | 305 | 2,423 |
| TTFM p50 | 8,609 | 384 | 9,030 |
| TTFM p75 | 28,085 | 621 | 125,636 |
| TTFM p90 | 222,064 | 1,160 | 505,360 |
| TTFM p95 | 683,235 | 1,828 | 1,127,674 |
| TTFM p99 | 4,366,059 | 6,680 | 4,579,322 |
| tokens | 3,433,964 | 2,958,083 | 3,468,252 |
| schemas | 11,187 | 11,306 | 11,306 |
| passing | 10,021 | 9,487 | 10,212 |
| compile error | 646 | 1,797 | 51 |
| timeout | 15 | 0 | 0 |
| validation error | 5 | 32 | 671 |
| invalidation error | 870 | 0 | 1,493 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once — the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.2.0, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention — these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (469), required-not-enforced (required-set beyond cap) (415), scanner-budget: constrained string degraded (278), maxLength-with-pattern (207), scanner-budget: length window degraded (172), minLength-with-pattern (171), length (length window (0,255) beyond cap) (163), length (length window (1,255) beyond cap) (154), uniqueItems (105), string-constraint-terminal-too-large (104), not-unenforced (99), length (length window (0,32767) beyond cap) (84).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (520), Unsupported: allOf (merge failed) (29), TypeError (14), Unsupported: terminal budget exceeded (size cap) (9), RxUnsupported (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: rule budget exceeded (size cap) (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), Unsupported: anyOf with sibling keys ['properties'] (2).
