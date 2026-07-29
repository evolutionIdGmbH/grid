# MaskBench (guidance-ai/jsonschemabench) - GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 1000000 schemas/split, seed 0 (11306 schemas, 1 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | GRID |
|:---|---:|---:|
| TBM avg | 485 | 559 |
| TBM p25 | 9 | 9 |
| TBM p50 | 25 | 25 |
| TBM p75 | 32 | 32 |
| TBM p90 | 78 | 78 |
| TBM p95 | 7,232 | 7,171 |
| TBM p99 | 7,596 | 7,435 |
| TBM p99.9 | 8,090 | 7,814 |
| TBM max | 1,676,539 | 513,902 |
| TTFM avg | 418,844 | 146,984 |
| TTFM p25 | 5,682 | 4,841 |
| TTFM p50 | 8,608 | 7,133 |
| TTFM p75 | 27,794 | 18,989 |
| TTFM p90 | 216,742 | 78,733 |
| TTFM p95 | 671,923 | 198,600 |
| TTFM p99 | 4,366,059 | 1,173,082 |
| tokens | 3,461,146 | 3,489,325 |
| schemas | 11,306 | 11,306 |
| passing | 10,117 | 10,154 |
| compile error | 668 | 635 |
| timeout | 16 | 7 |
| validation error | 5 | 5 |
| invalidation error | 870 | 876 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once - the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.2.0, GRID 0.2.0.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (471), required-not-enforced (required-set beyond cap) (415), scanner-budget: constrained string degraded (278), maxLength-with-pattern (207), scanner-budget: length window degraded (173), minLength-with-pattern (171), length (length window (0,255) beyond cap) (163), length (length window (1,255) beyond cap) (154), uniqueItems (105), string-constraint-terminal-too-large (104), not-unenforced (99), length (length window (0,32767) beyond cap) (84).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (542), Unsupported: allOf (merge failed) (29), TypeError (14), Unsupported: terminal budget exceeded (size cap) (9), RxUnsupported (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: rule budget exceeded (size cap) (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), Unsupported: anyOf with sibling keys ['properties'] (2).

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (494), required-not-enforced (required-set beyond cap) (411), scanner-budget: constrained string degraded (277), maxLength-with-pattern (210), scanner-budget: length window degraded (193), minLength-with-pattern (172), length (length window (0,255) beyond cap) (163), length (length window (1,255) beyond cap) (154), string-constraint-terminal-too-large (128), uniqueItems (105), pp-extras-complement-unavailable (99), not-unenforced (97).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (519), Unsupported: allOf (merge failed) (30), RxUnsupported (9), Unsupported: terminal budget exceeded (size cap) (9), Unsupported: rule budget exceeded (size cap) (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), Unsupported: anyOf with sibling keys ['properties'] (2), Unsupported: oneOf with sibling keys ['additionalProperties' (2).

## rc2 vs v0.2.5 (same machine, same protocol; column 1 = v0.2.5, column 2 = rc2)

Configuration: integration/0.3.x @ 76130d0, GRID_PERF_HASHCONS=norm,dedupe
GRID_PERF_LALR_DP=1 GRID_PERF_FACTORED_SCANNER=1, NFA live sets default-on,
LALR-conflict retry active in the runner. No verify-mode inflation (the
11.3k zero-divergence verify gate passed separately on rc1).

Accuracy: passing 10,117 -> 10,154 (+37: 11 latent-TypeError fixes, 7
recursive-cap fixes, 26 LALR-conflict retry converts, minus offsets);
compile errors 668 -> 635; timeouts 16 -> 7; false-reject class unchanged;
invalid-accepts 870 -> 876 (+6 from converts, every one recorded).

Timing: TTFM avg 419ms -> 147ms (2.9x), p90 217 -> 79ms, p95 672 -> 199ms,
p99 4,366 -> 1,173ms (3.7x). TTFM p99 sits slightly above rc1's
verify-inflated 1,054ms because the 26 retry converts (1.2-2.5s compiles
each) joined the completing distribution - coverage growth shifts the
all-schemas lens by construction; the previously-passing lens is unchanged.
TBM: warm p50 25us unchanged; p99.9 8.1 -> 7.8ms; max 1.68s -> 0.51s.

Known remaining (mapped, not hidden): 542-class LALR conflicts now 518
(mechanisms documented: unsatisfiable-leaf duplication, member-chain twins);
substring-union scanner family (5 timeouts + slow compiles) and helm-class
LALR divergence tracked in bench/perfbench/ROADMAP.md waves.
