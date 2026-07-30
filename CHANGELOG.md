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

Per-component state budget (P3, `GRID_PERF_COMPONENT_BUDGET`, default
16384): the substring-union terminal family (13-17 unanchored keywords in
one JSON-string terminal, BAKEOFF.md F1) keeps a per-keyword matched bit
alive, so its eager per-terminal subset construction discovers ~2^k states
(o83132: 268,803 / ~87s / 2.2GB; o5195: >200k with the frontier open) while
a walk demands at most one new subset per scanned byte. Components that
breach the budget come back as demand-interned LazyTerminalDFAs — exact
same subsets/annotations as the eager build (accepting = accept-in-subset,
co_acc = NFA terminal-reach over the subset), consumed through a
step(state, cls) facade on ScannerComponent — and the scanner skips product
materialization (the union DFA is at least component-sized, so the product
budget would abort it anyway); the lazy product keeps its existing
kernel/genN/T2/reserve gates. Default fixed by a manifest-set sweep (853
schemas, 21,223 unique patterns: largest terminating non-family component
7,210; largest inside a dense-today build 15,865 — strmprivacy Stream —
which must stay byte-identical; every 2^k member breaches in 1.8-11.9s).
Outcome changes, all in the family: 5 compile timeouts (o5195, o48423,
o47656, o47657, o48427) become lazy compiles; o83132 (~87s), o83133,
o33033, and the >16384-state components of strmprivacy
BatchJob/DataConnector build lazily (same lazy-product outcome class as
before, seconds instead of tens of seconds). `=0` restores the eager
component builds byte-identically (digest-gated), family hang included.

Honest metrics (E4, bench/ only — no runtime change): TTFM is now published
as two labeled columns, *compile-only* (maskbench compile_grammar
semantics; the historical definition, name unchanged) and
*first-mask-included* (+ the first compute_mask, which for lazy factored
scanners is the first real payment of the deferred product construction —
previously measured nowhere). profile_phases.py grows `--first-mask`
(trie/guide/first_mask/prefix_masks phases, child-written
ttfm_compile_us/ttfm_first_us stats) and `--leg` (interleaved env legs);
new bench/perfbench/outcomes.py classifies records
(ok/declared/timeout@phase/crash/incomplete/malformed; incomplete is never
ok — the F1-retraction guard) and gates cross-leg compares by the oracle
rule. Regenerated capped+fast legs at wave-A HEAD (BAKEOFF.md E4
postscript): capped-16 accounting is 10 compiled / 5 declared / 1 timeout
(replaces the withdrawn "10/16"), OFF-leg strict identity 45/45 vs v0.2.5;
measured deferred cost on lazy schemas: initial mask 1.6-8.0ms but
mid-instance cold tokens 239-298ms worst (dense band: ~8ms), 64-token cold
prefix 1.6-11.8s — the recorded P1 go/no-go input. maskbench_grid.py now
clears engine extras per schema (stale-write fix, informational fields
only); frozen status dirs keep the artifact and outcomes.extras() defends
them at read time.

Direct grammar-object emission (P2, `GRID_PERF_DIRECT_EMIT`, default on):
the schema compiler now produces a `GrammarParts` manifest
(`grid/grammar/parts.py`); `compile_json_schema_grammar` builds the
`DialectGrammar` straight from it (`DialectGrammar.from_parts` +
`RoleProjection.full_built`), skipping the `.grid` render and the regex
re-parse that made spec_load ~49% of front-end compile time. `render_text`
over the same manifest IS the legacy text emitter (byte-identical over all
11,306 corpus schemas, PYTHONHASHSEED pinned) and stays the debug/audit
path; validate()/freeze() run unchanged on the object path, so
GrammarInvalid outcomes (the unproductive-recursion family), L-REC01
warnings, and terminal numbering are shared code. Gates:
bench/perfbench/diff_direct_emit.py corpus differential (11,306 schemas,
zero flips; terminal_order tuple equality is the primary assertion —
fingerprint hashes sorted names and cannot see a numbering bug), --tables
leg (role_shape_hash + LALRTables.fingerprint equal over 309 set schemas),
sampled mask-walk differential (25 stratified schemas x 2,000 steps,
equal), reduction-worklist set-equality (307 grammars + 6,140 random
projections + 2,000 synthetic), and a live render+reload oracle
(`GRID_PERF_DIRECT_EMIT_CHECK=1`, CI leg). Measured (profile sets,
ttfm_capped + stratified_200): spec_load+projection totals 1.01s -> 0.54s;
front-end on the 12 most expensive grammars -50.3% (127 -> 61ms on
o87865); full-pipeline p50 8.03 -> 7.68ms (p90 scanner-bound, unchanged).
`=0` restores text -> `spec.load`. The reduction primitives
(`grid/grammar/reduction.py`) are now linear worklists in ALL
configurations (the 30k-rule chain: 139.8s -> 0.019s); serving is
untouched (`vllm_processor` receives grammar TEXT in the request spec).
Artifact-store schema_src entries remain text; store hits load text in
both flag states.

Tokenizer slicer (S2, `GRID_PERF_SLICER`, default OFF pending the H100
serving stamp): the flat ~7ms string-interior cold-walk mode (76.3% of
pooled mask time in the 1-8ms band) is a full 275,348-node trie DFS whose
answer is knowable upfront for 96.24% of the Llama-3.1 vocab. With the flag
on, `build_trie` partitions the vocabulary by the JSON-string-safe byte
class (`[^"\\\x00-\x1f]`, exactly STRING_RX's body class) into a sorted
alias-complete slice-id array plus a 10,415-node rest-trie (`TrieSlices`);
at walk time the kernel (and the Python spec walk, mirrored) proves slice
containment structurally from the seeded state — closure BFS over the class
bytes, cap 64 states, every transition non-DEAD (no emission can fire inside
a sliced token), every reachable state live for A|ignored and disjoint from
lexicon-constrained terminals (the audited genN lexicon-inertness guard) —
and on success walks only the rest-trie, sorted-merging the slice ids into
ci. Output is BYTE-IDENTICAL to the full walk (ci bytes, CD group order,
v7 blob, entry_id) because both tries enumerate tokens in the same byte-lex
DFS order and the proof is all-or-nothing; proof failure (maxLength/pattern
windows, boundary states, identifier positions, lazy factored DFAs) is
today's full walk byte-for-byte. Measured on the real trie: string-interior
cold walks 6.83ms -> 0.269ms (25.3x); warm paths untouched. Differential +
containment-refusal suite in tests/trie/test_slicer.py; real-config fuzz
byte-identity over corpus schemas (kernel + >512-terminal spec paths).

Serving jump-forward (S1, `GRID_JUMP`, default OFF — parity-gated, not yet
box-validated): forced token runs (singleton-mask chains, the §4.5
mechanism mode 1 already jumps) delivered to the vLLM scheduler-side
backend as draft tokens. `GridGuide.forced_run` (public pure-query span
surface) + `GridGrammarSession.jump_tokens()` — state-neutral, v5 guide
path and v6 kernel path (warm-only chaining via session_fill/accept +
rollback; cold successors end the jump, never walk) — + patch site 5
(spec_token_ids injection after accept_tokens; drafter proposals keep
winning) + the upstream-shaped vllm_upstream_jump_tokens.patch. At a
forced position the bitmask admits exactly one token, so draft acceptance
is certain under any sampler: greedy token parity off-vs-on is the flip
gate (bench/vllm_serving_bench.py --jump-probe, vllm_sched_accept.py
--jump — both pending the next GPU session, as is the step-3 probe of
0.24's SO+spec interplay/bonus-token mechanics). Measured stage-0 density
(bench/RESULTS-jf-density.md, JSB replay, gpt2): forced steps 2.3%
overall, 8-13% on rigid function-call-style splits, runs ~all length-1 —
the realistic saving rides on bonus tokens; byte-level JF (~10x the mass)
stays the recorded v2 deferral. Mode-2 logits-processor route untouched
(§4.5 rule 3: singleton-degrade, never a unioned span mask).

Kernel-resident lazy scanner (P1, `GRID_PERF_KERNEL_LAZY`, grid_core v8):
lazy factored DFAs (the over-budget LazyProductDFA regime P3 created) now
serve trie walks through the Rust kernel instead of pure-Python `_walk_py`.
The walker's scanner becomes a backend enum behind the four accessor
touchpoints (tr/accepting/accepts_all/live): `Dense` keeps the v7 arenas
verbatim; `Lazy` is the in-kernel lazy product — sparse (tid, comp-state)
tuples and per-component subset bitsets interned on demand under one build
mutex (reads lock-free: append-only OnceLock arenas, AtomicI32 rows),
annotations folded from per-component flags at intern time, so masks are
pure functions of state VALUES and instance-local demand-order numbering
never crosses the FFI (kernel payloads carry token/terminal ids only; lazy
schemas already use raw schema-scoped T1/T2 keys). Components ship as
compact blobs from `factored.kernel_lazy_payload` — dense arenas for eager
components, NFA arenas (byte classes, eps-CLOSED per-class edge lists,
accept id, reach words) for capped ones — so the kernel does no regex/NFA
work; the Python facade stays the executable specification and the
fallback (intern-cap breach surfaces as ValueError -> `_walk_py`, masks
exact either way; the 262,144-state cap is ~36x the worst measured need).
Gates: per-token full-vocab mask digests identical kernel-vs-spec across
the nine-schema substring-union family (3,095 instance steps) plus
eager-leg cross-checks; forced-all-lazy parity legs (toy/wide/sql-lexicon);
id-independence under opposite-order interning and the rayon pool;
recorded degradation sets untouched by construction (the flag's only
consumer is walk dispatch). Measured (family AB, interleaved legs, jobs 1):
64-token cold-prefix worst token p50 276ms -> 9.2ms, max 313 -> 15.9ms
(~30x; full-instance worst tokens up to ~221ms remain on the two heaviest
schemas — Python-side CD re-checks, the recorded phase-2 item), pooled
prefix walk 64.6s -> 2.6s, TTFM columns and RSS unchanged, dense schemas
byte-identical (flag consumed only for lazy DFAs); MaskBench family arms
(outcomes.py --strict): 14/14 outcomes unchanged, pooled TBM p99
257 -> 8.6ms over 10,965 masks, warm p50 untouched, stratified-29
interleaved p50 gate at ratio 1.003 (the gate RUST_SCANNER failed). RustVerdicts stays
Python-side for lazy schemas this phase. The artifact store now never
persists the lazy facade (deterministic component artifacts are the S3
follow-on). RUST_SCANNER (held since the bake-off) is subsumed: v8 ships
blobs once at walker construction and never rehydrates arenas into Python,
so the +18-22ms p50 FFI floor that sank it is structurally avoided; the
eager `build_scanner_arena` port is retired unharvested (the v8 payload
pre-computes eps-folded NFA artifacts Python-side, leaving nothing for
in-kernel regex/NFA machinery to do).

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
