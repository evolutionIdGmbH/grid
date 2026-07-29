# Changelog

Versions in the 0.2.x line are **correctness-only** (the coverage epoch,
DESIGN-JSON-COVERAGE.md): error metrics are the headline; timings are
recorded, not optimized (kernel frozen at v7). Speed work is the 0.3.x epoch.

## Unreleased

0.3.x flag disposition (E3): the epoch's measured winners become the
shipped defaults, gated on the v0.3.0 full-corpus run
(bench/RESULTS-jsonschemabench-v0.3.0rc2.md; outcome movement vs v0.2.5
fully adjudicated there).

- Defaults flipped ON, each with an env kill switch restoring the legacy
  path: `GRID_PERF_FACTORED_SCANNER` (`=0` restores the eager union
  builder, kept as the exactness oracle), `GRID_PERF_LALR_DP` (`=0`
  restores canonical `lr1_merge`, kept as the construction-independent
  oracle), `GRID_PERF_HASHCONS` (unset now means `norm,dedupe` — the exact
  measured configuration; `=0` disables, comma lists still select
  components). Value grammars are unchanged; only unset defaults moved.
- Deleted: `GRID_PERF_NFA_LIVE` and the legacy live-set implementations it
  selected — `_live_fixpoint` + verify branch (dfa.py), `_graph_co_acc` /
  `_live_mode` + verify branch (factored.py) — sanctioned by the 11.3k
  zero-divergence verify pass on v0.3.0rc1. Live sets now have exactly one
  implementation (NFA terminal-reach); the factored component memo key
  drops its mode dimension `(pattern, is_literal, live_mode)` ->
  `(pattern, is_literal)`. Independent gates kept: forward-BFS oracle in
  tests/lexer/test_live_sets.py + the eager-vs-factored byte-identical
  differential.
- `GRID_PERF_ARTIFACT_STORE` stays default-off by design: BAKEOFF F2
  measured +5-7ms cold schema_compile per fast build; default-on is
  deferred to a serving-epoch warm-hit measurement.
- Default-visible outcome changes (all adjudicated in the rc2 results):
  5 former 120s-caps now compile, the 4-schema frontend family declares
  Unsupported in 0.03-0.5s, wp_105 compiles via the LALR-conflict retry;
  residual known tails: substring-union scanner family x5, helm-testsuite,
  o27148 (retry pushes it over the limit).

Scanner-build dedup (E2, structural, zero behavior change): the
subset-construction core that build_scanner and factored._build_component
ran as duplicated code now lives once in grid/lexer/subset.py (eps-closure
memoization, byte-class refinement, per-class edge index, FIFO subset loop);
the regex parser and NFA layers split into grid/lexer/rx.py + nfa.py, with
dfa.py kept as the stable import facade (ScannerDFA, build_scanner, and the
historically-imported privates re-exported — importers unchanged). Gated by
the new bench/perfbench/diff_scanner_digest.py byte-identity harness:
241-unit corpus (stratified_200 + ttfm_capped + in-repo floor), digests of
both flag arms plus every per-terminal component bit-equal pre/post,
GrammarInvalid message text included. factored.py gains the type-only
ScannerComponent Protocol — the seam a COUNTING_WINDOWS component type
plugs into beside TerminalDFA without touching the product.

## 0.3.0 - 2026-07-30

The performance epoch: compile-time (TTFM) tail work, selected by measured
bake-off from a 20-candidate pool (bench/perfbench/), correctness-gated
throughout (flag-off byte-identical; full-corpus zero-divergence verify).

- Factored scanner: per-terminal DFA library with lazy product combination
  (`GRID_PERF_FACTORED_SCANNER`); the measured scanner tail collapses
  (per-state-cost family 45-84s -> under 1s).
- Structural hash-consing in normalization (`GRID_PERF_HASHCONS`): the
  exponential $ref re-normalization family now terminates deterministically.
- NFA-derived live sets (default on): the global live fixpoint replaced by
  per-accept reachability, proven equivalent over all 11,306 corpus schemas.
- DeRemer-Pennello LALR lookaheads (`GRID_PERF_LALR_DP`): table-identical to
  the canonical construction, modestly faster.
- On-disk compile-artifact store (`GRID_PERF_ARTIFACT_STORE`, default off):
  insertion-order-preserving keys, epoch invalidation, atomic writes.
- Recursive-additionalProperties conjuncts are detected and recorded
  (`x-grid-cap-dropped`) instead of failing the whole allOf merge.
- LALR-conflict retry: `compile_schema(unify_string_values=True)` unifies
  overlapping per-branch string values; callers retry once on
  LALRConflictError (26 previously-conflicting schemas now compile).
- Fixed: latent TypeError in two-pattern merge (11 schemas now compile);
  unconditional serving import on the compile path.
- Full-set (11,306, one machine): 10,154 passing (89.8%), 3 false-rejects,
  every unenforced constraint recorded; TTFM avg 419 -> 147ms, p99
  4.37s -> 1.17s; warm mask p50 unchanged at 25us.
- bench/perfbench/: attribution profiler, candidate pool, selection record,
  bake-off results, post-0.3.0 ROADMAP.

## 0.2.5 - 2026-07-21
- `grid.jsonschema` package: JSON Schema -> grammar compilation promoted out
  of `bench/` with public API `compile_json_schema(schema, strict=False)`.
- Official JSON-Schema-Test-Suite (draft-07 + 2020-12) in CI under the
  honesty contract (valid never rejected; invalid accepted only if recorded).
  Found and fixed: integer-type zero-fraction floats, typeless multipleOf,
  one-sided properties vs additionalProperties under merge, items/prefixItems
  cross-level scoping.
- `grid/jsonschema/SUPPORT.md` keyword matrix; upstream PR kit
  (`bench/upstream/`) verified under jsonschemabench's own runner.
- Full-set (11,306): 10,117 passing (89.5%), 3 false-rejects, all
  unenforced constraints recorded per schema.

## 0.2.4 - 2026-07-20
- patternProperties overlap families -> recorded fallbacks; general object
  negation; record-and-drop for unrewritable narrowing keywords; branch
  string-value unification (kills a token-capture false-reject class);
  required-through-patternProperties satisfiability fix.

## 0.2.2 / 0.2.3 - 2026-07-20
- Full-set validation-error hunt: composite-enum maximal-munch capture,
  untyped type-sniffing, anyOf const harmonization, draft-<=07 $ref-replaces-
  siblings, order-free object machine beyond the required cap, routing-
  terminal degradation exemption, unsatisfiable-schemas -> never-grammar.

## 0.2.1 - 2026-07-19
- Dialect `{m,n}` bounded repetition (parse-time expansion; kernel untouched);
  length windows <= 64 chars enforced via a compact counting form.

## 0.2.0 - 2026-07-19
- Coverage sprint 1: schema normalization (allOf merge, dependencies,
  if/then/else, not), constrained terminals (pattern/format/length/bounds),
  order-free objects, hash-consed rules. Sample passing 206 -> 268.

## 0.0.7 - 2023-11 (baseline)
- SQL-first engine: configuration-keyed viable-prefix masks, byte-level
  token<->terminal bridge, Rust kernels, RBAC/schema projections, audit
  chain, checker-guided repair.
