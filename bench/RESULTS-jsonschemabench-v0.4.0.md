# MaskBench (guidance-ai/jsonschemabench) - GRID vs llguidance vs XGrammar

Tokenizer: `unsloth/Meta-Llama-3.1-8B-Instruct` | sample: 999999 schemas/split, seed 0 (11306 schemas, 6 splits) | time limit 120s/schema

Protocol: maskbench's runner semantics reproduced verbatim (TTFM = schema compile; TBM = per-token compute_mask+commit window, pooled; valid instances must be fully accepted, invalid ones rejected mid-stream). Times in microseconds. Host: local dev (unpinned).

| metric | GRID | llguidance | XGrammar (compliant) |
|:---|---:|---:|---:|
| TBM avg | 486 | 21 | 191 |
| TBM p25 | 9 | 5 | 2 |
| TBM p50 | 25 | 9 | 9 |
| TBM p75 | 32 | 19 | 28 |
| TBM p90 | 80 | 25 | 44 |
| TBM p95 | 7,268 | 41 | 113 |
| TBM p99 | 7,741 | 294 | 757 |
| TBM p99.9 | 8,638 | 1,063 | 50,665 |
| TBM max | 543,078 | 6,038 | 128,756 |
| TTFM avg | 60,905 | 624 | 362,186 |
| TTFM p25 | 4,715 | 284 | 2,768 |
| TTFM p50 | 6,850 | 352 | 10,292 |
| TTFM p75 | 18,220 | 532 | 138,196 |
| TTFM p90 | 74,224 | 1,025 | 603,374 |
| TTFM p95 | 192,855 | 1,604 | 1,253,768 |
| TTFM p99 | 1,142,255 | 6,189 | 5,037,974 |
| tokens | 3,491,288 | 2,958,083 | 3,468,252 |
| schemas | 11,306 | 11,306 | 11,306 |
| passing | 10,159 | 9,487 | 10,212 |
| compile error | 637 | 1,797 | 51 |
| timeout | 0 | 0 | 0 |
| validation error | 5 | 32 | 671 |
| invalidation error | 876 | 0 | 1,493 |

Reading the table:
- The three engines sit at different points of the coverage/upfrontness/latency trade-off: compile errors are *declared* non-support (visible, safe); validation errors (valid instance rejected) and invalidation errors (invalid instance accepted) are silent correctness gaps.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512 terminals run in-kernel); the p90+ tail is cold-miss trie walks over the 128k vocabulary. MaskBench runs each schema once - the write-back cache that amortizes GRID's misses across requests in serving never warms here; the cold walk was cut 9.3x by the kernel v5.1 verdict-equivalence grouping (this record; TBM p90 27.8 ms -> 208 us vs the v3-era run).
- GRID's TTFM is the Python table build per schema (scanner subset construction is alphabet-compressed with per-state eps closures; further kernel work possible).
- GRID counts zero validation errors: every valid instance of every schema it compiled was accepted (definition-order properties, spec-default additionalProperties incl. typed extras).

Engine versions: GRID 0.4.0, llguidance 1.7.6, XGrammar (compliant) 0.2.3.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default convention - these surface as invalidation errors when an invalid instance hinges on them): oneOf-exclusivity (433), required-not-enforced (required-set beyond cap) (350), scanner-budget: constrained string degraded (240), maxLength-with-pattern (170), minLength-with-pattern (162), length (length window (0,255) beyond cap) (158), scanner-budget: length window degraded (157), length (length window (1,255) beyond cap) (142), string-constraint-terminal-too-large (97), length (length window (0,32767) beyond cap) (78), length (length window (0,1024) beyond cap) (75), minimum (73).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront): LALRConflictError (519), Unsupported: allOf (merge failed) (30), RxUnsupported (9), Unsupported: terminal budget exceeded (size cap) (9), Unsupported: rule budget exceeded (size cap) (8), Unsupported: anyOf with sibling keys ['additionalProperties' (7), Unsupported: oneOf with sibling keys ['required'] (7), Unsupported: $ref with sibling keys ['type'] (4), LALRBudgetExceeded (2), Unsupported: anyOf with sibling keys ['properties'] (2).

## Session provenance (columns above: GRID 0.4.0 | llguidance 1.7.6 | XGrammar 0.2.3)

One sitting, one machine, sequential legs (llguidance, then XGrammar, then
GRID), 2026-07-30. Versions pinned and smoke-tested before the session:
llguidance 1.7.6 (current), XGrammar 0.2.3 + apache-tvm-ffi 0.1.12 (XGrammar
0.2.4/0.2.5 wheels fail at import on macOS arm64 - tvm-ffi 0.1.13 ABI break -
so the newest importable release ran; note stands until upstream ships a
working wheel), GRID 0.4.0 with grid_core 0.2.0 (kernel v8), shipped
defaults. Outcome counts are bit-identical to the previous GRID run and to
the v0.2.5-era llguidance/XGrammar runs: outcomes are deterministic;
timings are what the session refreshes.

## Schema-level vs instance-level reconciliation

The summary rows above count instances; schema-level counts from the same
status files: GRID valid-rejected 3 schemas / 5 instances, invalid-accepted
507 schemas / 876 instances (every one carrying the names of its unenforced
constraints); llguidance 22 / 32 and 0 / 0; XGrammar 427 / 671 and 627 /
1,493 (all silent). Any table elsewhere in this repo states which unit it
uses.

## TBM, read correctly (the two-lens rule)

GRID's TBM average and tail vs earlier epochs shifted for the same reason
its TTFM p99 did: v0.4.0 compiles schemas the older engine declined or
timed out on, and their (cold-walk-heavy) masks joined the distribution.
The attribution probe (60 mask-heaviest schemas, three configs): disabling
the v8 kernel-lazy path explodes TBM p99 from 8ms to 240ms - the new
default is a large win, not a cost; the legacy configuration "looks faster"
only by surviving on the easy subset (28/60 passing vs 38/60). GRID's
bounded cold walk keeps TBM p99.9 at 8.6ms vs XGrammar's 50.7ms; warm-path
p50 is 25us and amortizes in serving (the benchmark runs each schema once).
