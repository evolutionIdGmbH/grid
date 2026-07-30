# GRID post-0.3.0 roadmap

Synthesized 2026-07-29 from twelve item plans (eight carried from the 0.3.x
epoch journals, four fresh from this session's probes). Every number below is
measured, with its source leg named; every claim that is not yet measured is
labeled an unknown. Companion docs: BAKEOFF.md (0.3.x A/B evidence),
SELECTION.md (candidate adjudication), DESIGN.md (protocol ground rules),
CANDIDATES.md (raw candidate sketches).

## 1. Where 0.3.0 stands, and what this roadmap buys

0.3.0 (integration/0.3.x at rc) merged FACTORED_SCANNER, NFA_LIVE, HASHCONS,
LALR_DP, and the default-off ARTIFACT_STORE, cutting the measured >5s tail
mass from ~1016s to ~156s (~6.5x) on the bake-off sets. Measured state on the
11.3k JSONSchemaBench corpus: TTFM p50 7.3ms / p90 77ms / p99 1.05s
(compile-only semantics, verify-inflated at p99); TBM warm p50 25us with a
bounded ~8ms cold-miss tail; on mb-grid-v030rc1 (3.47M masks) the 1-8ms
cold-walk band is 6.4% of masks but 76.3% of pooled mask time. Outcome
accounting on the 16 formerly-capped schemas: 5 compiled, 5 terminate
deterministically, 6 still time out (the old "10/16 fixed" union number is
withdrawn; E4 has since published the replacement at wave-A HEAD:
10 compiled / 5 declared / 1 timeout, classifier-gated, both TTFM columns —
BAKEOFF.md E4 postscript).
Two structural tail families remain: the substring-union scanner family
(5 timeouts: o5195, o48423, o48427, o47656, o47657; plus slow compiles
o83132 89.3s, o83133 5.8s, DataConnector 13.4s, BatchJob 11.9s; root cause
reproduced this session: 2^k eager subset blowup from per-keyword accept
bits) and the helm-testsuite LALR family (LR(0) core itself diverges,
62.75M items at 60s and still growing; o27148 detects its 3,636 conflicts
only after 48.17M LR(0) items). The llguidance gap is now precisely scoped:
its compile-only TTFM p50 0.38ms vs our 7.3ms is a front-end and laziness
gap (items P2, P3, P1), and its mask p99 6.7ms vs our ~7.2-7.7ms
string-interior cold-walk mode is a slicing gap (item S2); our warm p50 25us
is already ahead. What each area buys: P items (performance) kill the two
remaining tail families asymptotically, cut the front end, and put lazy
schemas back on the kernel path; S items (serving) collapse the cold-walk
mode, skip forced decode steps, and make redeploy cold-start approximate an
unpickle; E items (health) centralize flags, unify the scanner core, ship
the epoch's defaults, and settle metric definitions before the one-shot
full-corpus republish.

## 2. Measurement discipline (applies to every item)

- Measure first. Every item's step 1 is a probe that can kill or rescope it
  (P3 budget sweep, P1 touched-state bound, S1 forced-span density, P4
  step-0 coverage count, S2 H100 transfer check, S3 tail attribution, E4
  lazy census). Negative results are published, not buried.
- Outcome parity is the default. Every intentional recorded-outcome change
  is enumerated per schema before merge: the E3 rc2 allowlist, P3's family
  flip (timeout -> compiled), P5's two changes, P4's phase-3 manifest.
  Anything outside the enumeration fails the item.
- Flag-off is byte-identical to today. Every default flip retains a kill
  switch. New flags are born as grid/perf_flags.py readers (E1) with
  grammar-oracle tests.
- The full 11.3k corpus runs once per epoch (DESIGN.md ground rule). E4's
  TTFM definition and outcome classifier land before that run. Publishable
  numbers are --jobs 1 with the flag snapshot recorded in-child.
- Cold and warm are never blended: fresh-process harnesses only
  (store_coldwarm.py protocol); verify-inflated numbers are labeled;
  TTFM is published in two columns (compile-only and first-mask-included)
  from E4 onward, permanently.
- The p50 gate is interleaved flag-ON/OFF fast-schema runs (BAKEOFF F2
  protocol). RUST_SCANNER's +18-22ms FFI floor failing exactly this gate is
  the standing precedent; no item ships a median regression for a tail win.

## 3. Dependency graph

"A -> B" means A must land before B starts (or B forfeits value / rebases).

- E1 -> every new-flag item: P3 (GRID_PERF_COMPONENT_BUDGET), P5
  (GRID_LALR_BUDGET), P2 (GRID_PERF_DIRECT_EMIT), S1 (GRID_JUMP), S2
  (GRID_SLICER), S3 (GRID_PERF_STORE_* sub-flags), P1
  (GRID_PERF_KERNEL_LAZY, GRID_PERF_LAZY_COMPONENT_CAP), P4 (counting
  flags). New flags are born as perf_flags readers, not inline reads.
- E2 -> P3, P1, P4-revival, and the RUST_SCANNER un-hold decision. All four
  edit the subset-construction core E2 deduplicates; landing E2 after any of
  them forces double edits and forfeits the single-reference-implementation
  guarantee. E2's Protocol seam is what P4's CountingTerminalDFA plugs into.
- rc2 full-corpus run -> E3 scope B (default flips + flag disposition).
  No flips on bake-off subsets alone. E3's deletion also lands before new
  scanner work so P3/P1/P4 never thread live_mode into new code.
- E3 (LALR_DP default flip) -> P5 (budget is calibrated against the dp path
  as the shipped construction).
- E4 -> the one-shot full-corpus republish. The TTFM definition
  (first-mask-included) and the outcome classifier must be settled before
  the corpus runs once. P5's fire-set enumeration rides that same run.
- E4 (lazy first-mask distribution) -> P1 go/no-go. The measured deferred
  first-mask cost of lazy scanners is the direct input to the
  kernel-residence bet and the recorded RUST_SCANNER disposition.
- P3 -> P1. P3's LazyTerminalDFA + step facade is P1's executable Python
  spec (P1 steps 2-4 are P3's scope). P3 also rescopes S3: lazy components
  are not persisted (they rebuild in ms), only eager ones.
- P1 resolves the RUST_SCANNER hold: subsume (harvest scanner_build.rs
  modules, retire the eager build_scanner_arena entry point; size-gated
  dispatch retained only as the fallback if v8 slips).
- P1/RUST decision -> S3 kernel-arena namespace. S3 reserves the key shape
  in Wave B; arena payloads are implemented only after the v8 backend
  exists.
- Factored component library (merged 0.3.x) -> P4. P4 is unimplementable
  without it; P4's kernel counts register rides the same v8 bump as P1's
  lazy backend and S2's slice tables (one version bump carries all).
- P2 -> memo/fingerprint follow-ons: stable grammar fingerprints feed the
  memo-tier candidate (CANDIDATES 16), the optional artifact-store grammar
  namespace (S3 follow-on), and the int-id LALR handoff (candidate 6).
- S2 kernel ABI coordinates with P1 v8 (RustWalker constructor signature);
  S1 is dependency-free (GPU box only); E4 is dependency-free.

## 4. Waves

Each wave is independently shippable: it ends at a releasable state with its
own gates green and its outcome changes enumerated.

### Wave A: ship the epoch, kill the scanner tail (target 0.3.1)

Order: E1 (S) -> E3 (S, rc2-gated) -> E2 (M) -> P3 (M).

Health-heavy and cheap: two S items, two M items. E1 centralizes flags
before anything mints new ones. E3 turns the epoch's measured wins into
shipped defaults (the 6.5x tail cut is currently opt-in) and deletes the
legacy live-set path, so E2 dedupes a two-mode codebase, not three. P3 then
lands the asymptotic substring-union fix on the deduped substrate. Outcome
changes in this wave: the adjudicated rc2 allowlist (E3) and the
substring-union family flipping timeout/87s -> compiled (P3). After Wave A
the only known cap family is helm LALR.

### Wave B: honest numbers, faster front end, warm serving (target 0.3.2)

Items: E4 (M), P5 (M), P2 (M), S1 (M), S3 (M, arena keying reserved only).

Five M items, all parity-gated or enumerated-scope. E4 lands first (metric
definitions), then the wave closes with the epoch's one-shot full-corpus
republish: P5's budget calibration and fire set are enumerated on that run,
P3's family flip is confirmed corpus-wide, and BAKEOFF gets dual-column
TTFM tables. P2 (front end), S1 (jump-forward), and S3 (store warm set) are
mutually independent within the wave. Outcome changes: helm-testsuite
120s cap -> declared decline in ~4-6s, o27148 error-class change
(LALRConflictError@132-290s -> LALRBudgetExceeded@~10-15s), both enumerated
(P5). E4's lazy first-mask distribution, published here, is the go/no-go
input for Wave C's P1.

### Wave C: kernel v8 (target 0.4.0)

Items: P1 (L), S2 (M), P4 (L), plus the S3 arena-namespace follow-on.

All kernel work, deliberately batched: one coordinated grid_core v8 version
bump carries the LazyProduct scanner backend (P1), the slice tables (S2),
the counts register + guard rows (P4), and the reserved arena ops (S3).
Every piece ships behind its own default-off flag with the v7 path as kill
switch; defaults flip only after the H100 serving stamp (GRID_V7 flip
precedent). P1 records the RUST_SCANNER disposition (subsumed). This wave
is the riskiest by nature, so it is smallest in item count and gated
hardest: bit-identical mask parity kernel-vs-spec everywhere, and the
interleaved p50 gate that RUST_SCANNER failed.

## 5. Item plans (condensed)

### E1-perf-flags (health, S, Wave A)

Mechanism: one leaf module grid/perf_flags.py with seven call-time readers
replacing nine inline env-read sites across dfa.py, factored.py,
lalr/compile.py, jsonschema/normalize.py + __init__.py, artifact_store.py.
Each reader preserves its site's exact value grammar (the empty string
enables ARTIFACT_STORE via != "0" but disables FACTORED_SCANNER via == "1";
deliberately not one generic parser). The module imports stdlib os only, so
the jsonschema fast-path pre-check keeps skipping the ~3-7ms grid.serving
import chain. Back-compat aliases keep normalize.py importers unchanged.

Steps: inventory freeze (grep gate); readers + oracle tests as commit 1;
baseline byte-identity dumps across the full flag matrix; then one
mechanical commit per family (scanner, lalr, hashcons, store) each rerunning
its Gate B slice; final grep gate: environ.get("GRID_PERF only in
perf_flags.py.

Gates: (A) grammar oracle, helper output equals the verbatim original
expression on every raw value including "", "true", "2"; (B) per-site byte
identity of compiled grammars / ScannerDFA / LALR tables across the matrix;
(C) all existing differential suites green (they monkeypatch env between
calls, proving the call-time-read contract); (D) import-cost invariant:
flag-off compile leaves grid.serving out of sys.modules. 100% outcome
parity; any deviation is a plan violation.

Gain: zero runtime delta by construction. Kills the duplicated inverted
ARTIFACT_STORE grammar (the highest-value drift hazard in the flag surface),
unit-tests the NFA_LIVE tri-state and HASHCONS list grammars directly, and
gives every Wave A-C flag a reviewed landing spot.

Effort: S. Risks: caching temptation (env-flip test kills any lru_cache);
leaf-module erosion re-adding the serving import cost (sys.modules test);
review-time grammar unification (a semantic change, forbidden by oracles);
premature alias removal breaking compiler.py imports.

### E3-legacy-deletion (health, S, Wave A, rc2-gated)

Mechanism: (A) delete the legacy live-set path sanctioned by the 11.3k
zero-divergence verify pass: _live_fixpoint, the GRID_PERF_NFA_LIVE read and
verify branches in dfa.py, _live_mode/_graph_co_acc in factored.py; the
component memo key shrinks to (pattern, is_literal), removing the mode-flip
coherence caveat before S3 freezes a persistence key. (B) flag disposition
per the BAKEOFF merge verdicts, gated on the rc2 full-corpus run:
FACTORED_SCANNER default on (flag kept, the eager union builder stays the
factored path's exactness oracle); LALR_DP default on (lr1_merge kept as the
construction-independent oracle); HASHCONS default all (kill switch kept);
ARTIFACT_STORE stays default-off (BAKEOFF F2 p50 floor); NFA_LIVE deleted
outright.

Steps: verify/audit the rc2 run (run it with NFA_LIVE=verify if missing, one
final free corpus-scale cross-check); outcome-parity audit vs v0.2.5 against
the adjudicated allowlist; one flip commit per flag with that flag's
differential suite run in both positions and a delenv fixture audit
(test_genn_keys.py:38, test_factored_walk_parity.py:121 currently read the
old default); serving smoke of the LazyProductDFA facade at default-on (the
one regime the bake-off never exercised); the mechanical deletion commit;
grep gate; CHANGELOG/ONBOARDING flag-disposition tables; stratified p50 spot
run.

Gates: the enumerated allowlist is the only permitted outcome movement
(5 cap -> compile, 4 cap -> declared Unsupported, wp_105 -> fast
LALRConflictError, 6 known residual timeouts); post-deletion, live sets stay
pinned by two surviving independent oracles (test-local _bfs_live and the
eager-vs-factored byte-identical differential); per-commit rollback via
retained kill switches.

Gain: health, not milliseconds, but this is the item that makes the measured
tail cut (1016s -> ~156s) and the fast declines the default configuration;
~60 lines of triple-implemented live-set logic deleted; every later
scanner-touching item writes one co_acc implementation instead of three.

Effort: S (machine hours if rc2 must be run). Risks: rc2 existence and
flags-snapshot unverified (step 1 measures first); fixture inversions
silently testing the wrong path; lazy facade at default-on in long-lived
producers (smoke covers it; documented as residual if issues surface);
default-visible outcome changes must be release-noted and the baseline
re-snapshotted.

### E2-scanner-dedup (health, M, Wave A)

Mechanism: build_scanner (dfa.py:389-568) and _build_component
(factored.py:119-234) share three verbatim blocks plus the FIFO subset-loop
skeleton (diff-confirmed). Extract into grid/lexer/subset.py pure helpers:
eps-closure memo, byte-class partition, edges_by_class, subset_construct.
build_scanner keeps three provably byte-identical post-passes (256-wide
expansion, accepts_all, live_masks over the same FIFO order). Phase 2 splits
rx.py (regex parser) and nfa.py out of dfa.py, with dfa.py as a permanent
facade re-exporting the ~6 public-in-practice names so ~15 importers stay
unmodified, and adds the typing.Protocol seam (trans, class_of, accepting,
co_acc, matches_empty) that P4's counting component and P3/P1's lazy
component conform to.

Steps: digest gate harness first, at the base commit
(diff_scanner_digest.py, modeled on diff_hashcons.py; goldens across the
FACTORED x NFA_LIVE matrix, in-repo corpora as the floor, jsb-src sets when
synced); helper extraction; rewrite both call sites; full gate run; phase-2
split + Protocol; importer smoke over the step-0 inventory.

Gates: pure-refactor discipline, no scoped changes: byte-identical digests
per grammar per matrix cell including GrammarInvalid message-text parity;
full pytest per cell; NFA_LIVE=verify pass while it still exists;
scanner-phase timings within +-3% on profile_phases (build_scanner is on the
cold-TTFM path, indirection must be measured not assumed); no new flag, the
digest gate is why none is needed.

Gain: zero perf claim. Drift-proofs the scanner core before P3, P1, and P4
all edit exactly this code; removes ~85 duplicated lines; shrinks the
factored->dfa import cycle to a type-only edge.

Effort: M. Risks: cold-path regression from indirection (gate d);
"harmless" reordering breaking state numbering that genN keys and the store
depend on (digest gate catches, corpus sync widens coverage); phase-2
re-export omissions (permanent facade policy); merge churn with concurrent
scanner worktrees (single small phase-1 commit, sequence first).

### P3-component-budget (performance, M, Wave A)

Mechanism: root cause reproduced this session: substring-union terminals
(13-17 unanchored keywords) carry a persistent per-keyword accept bit, so
eager subsets scale ~2^k (o83132: 268,803 states / 89.3s / 2.2GB RSS;
o5195: 200k states at 86.5s with the frontier still open). Demand-driven the
automaton is tiny: 61 distinct subsets in 4.6ms for 541 walked bytes. Fix in
factored.py, two layers: (L1) per-component state cap in _build_component
(GRID_PERF_COMPONENT_BUDGET, provisional 4096, fixed by a corpus sweep;
under cap the TerminalDFA is bit-identical to today); (L2) on breach return
a LazyTerminalDFA (locked on-demand subset interning, LazyProductDFA idiom;
accepting/co_acc/matches_empty computed exactly from the subset and NFA
terminal-reach, so the product's recorded sets are preserved by
construction, no approximation). Consumers adapt via a comp.step(state, cls)
facade; materialize() is unchanged and gate-1 exact; over-budget products
already carry lazy=True and stay gated off kernel/genN/T2. Applies in
default nfa live-mode only.

Steps: corpus component-size sweep to fix the default (expect bimodal,
<=~150 vs >=200k); cap in _build_component; LazyTerminalDFA + facade;
consumer adaptation (shortest-word BFS gains a states-visited counter);
differential extensions (breached-component-within-budget-product case,
budget=0 all-lazy leg over corpora + zombie patterns + a scaled k=2..6
union generator); mask-parity extension; full-corpus differential +
MaskBench; family bench re-run recording TBM warm p50, cold tail, and
lazy-memo growth slope per 1k generated bytes.

Gates: exact field-for-field ScannerDFA equality (numbering included) for
everything under budget; per-prefix observable equality + EmissionEvent
stream + shortest_lexemes equality at budget 0; full-corpus differential
and MaskBench where the only permitted changes are the family flipping
timeout/87s -> compiled; TBM warm p50 unchanged; lazy DFAs stay off
kernel/genN/T2. Flag: GRID_PERF_COMPONENT_BUDGET (0 disables), default on
at the sweep-confirmed threshold.

Gain: asymptotic (2^k build -> O(bytes walked)), so it covers future family
members at any k: the 5 hangs become compiles, o83132's scanner phase
89.3s -> sub-second, o83133/BatchJob/DataConnector drop to ms-scale, family
RSS 2.2GB -> tens of MB. Removes the entire substring-union family from the
TTFM tail; helm LALR becomes the only known cap family.

Effort: M. Risks: reserve BFS on a lazy component with a long shortest word
(counter + decline-lazy fallback keeps exactness); unbounded lazy-memo
growth in long-lived servers (slope measured in step 8; deferred degradation
lever = memo cap + fresh-product swap between generations); shared-memo
thread safety (reuse the _intern lock idiom, threaded probe in gate 2);
legacy/verify live-modes keep the eager build and therefore the hang
(documented; E3 deletes them).

### E4-honest-metrics (health, M, Wave B)

Mechanism: two verified dishonesty sources. (1) TTFM blind spot: under the
factored default, over-budget schemas return a lazy facade whose product
construction is deferred to the first mask, which is guaranteed pure-Python
(kernel walker asserts not lazy), and profile_phases stops timing at the
scanner phase, so that cost is measured nowhere. Fix: child gains
trie/guide/first_mask/prefix_masks phases; stats.ttfm_first_us (compile
phases + guide + first mask) computed in-child; prefix_masks walks the first
valid test instance (capped 64 tokens) to catch mid-instance lazy
materialization. (2) Outcome classifier bench/perfbench/outcomes.py:
ok / declared:<class> / timeout@<phase> / crash / incomplete, where
incomplete is never ok (the F1-retraction failure mode) and extras are never
trusted on non-ok records (verified stale: timeout records inherit the
previous schema's n_terminals/kernel); compare mode applies the oracle rule
(timeout -> terminating sanctioned; ok -> anything is a gate failure).

Steps: census which leg schemas actually go lazy (sizes the delta up
front); classifier + unit gates against the real 11,306 records
(668 compile_error / 16 timeout / 10,622 ok exactly, plus synthetic
malformed fixtures); child extension with unchanged compile-phase ordering;
dual-column summarize; maskbench stale-extras clear (scoped: informational
fields only); three-schema smoke; regenerate capped + fast legs
(--jobs 1, interleaved ON/OFF, --first-mask); compare vs mb-grid-final;
republish BAKEOFF with dual columns, definition footnotes, and the
replacement for the withdrawn 10/16 number.

Gates: exact bucket counts on the real corpus; eager-leg outcome invariance
per schema (zero tolerance); lazy-leg changes only in the sanctioned
direction; a --first-mask record missing the first_mask phase classifies as
malformed, not ok; no grid/ runtime changes beyond the declared extras
clear.

Gain: metric integrity, no runtime delta: a defensible cap-fix number; the
first measurement of the lazy first-mask cost (the direct go/no-go input
for P1 and the recorded RUST disposition); a classifier that keeps cap
schemas out of ok buckets before the one-shot 11.3k republish; the
apples-to-apples basis before any cross-engine claim (llguidance's 0.38ms
p50 is also compile-only, and stays labeled as such).

Effort: M. Risks: definition churn (both columns always published, old
metric never renamed); cross-engine asymmetry (columns labeled; a
same-definition llguidance leg is follow-on, not smuggled in); the census
may find zero lazy builds in these legs (item still lands, stated up
front); prefix_masks can itself hit the cap on a pathological schema (the
honest cost, attributed as a first-mask-era timeout distinct from v0.2.5
compile timeouts).

### P5-lazy-lalr (performance, M, Wave B)

Mechanism: measured this session: helm-testsuite's blowup is the LR(0) core
itself (2,610 productions -> 447,471 states / 62.75M items at 60s, still
diverging), so it hangs under both constructions by necessity; o27148 needs
the completed LR(0) automaton (48.17M items, 66.9s) plus DP lookaheads to
report its 3,636 conflicts at 132.5s. Two alternatives evaluated and
rejected on fresh measurement: true lazy rows (the cost is the substrate;
deferred conflict detection changes outcome buckets; partial tables cannot
persist) and class-preserving early fail-fast (lookaheads need the finished
automaton; saves only the 4.6s fill stage). Winner: a deterministic
items-materialized budget (sum of closure sizes at state creation:
input-derived, machine-independent, memory-proportional) plus a state cap in
both constructions, raising a declared LALRBudgetExceeded(GridError),
mirroring the frontend's "rule budget exceeded (size cap)" discipline.
ITEM_BUDGET=8M (4.4x the largest measured completer, tmlanguage at 1.83M
items / 5.0s), STATE_BUDGET=1M, GRID_LALR_BUDGET=0 as the audit escape
hatch. Declines are never cached (store puts only on success); serving
propagates declared exceptions as-is.

Steps: calibrate first (extend diff_lalr.py to record per-schema
lr0_states/lr0_items/lr1_items over the profiled sets, confirm the
completer maximum); add the error class; instrument both constructions
(two int compares per new state); wire the env knob; determinism tests
(identical fire counts warm/cold and across repeats; under-budget tables
field-identical; over-budget raises in both algorithms); family bench
re-run; full-corpus verify enumerating the exact fire set for BAKEOFF.

Gates: zero budget fires on any v0.2.5 completer corpus-wide, >=4x headroom
rule (raise the budget if a bigger completer appears); exactly two
enumerated outcome changes: helm 120s cap -> declared decline in ~4-6s
(measured 8M-item crossing at 4.1s), and o27148
LALRConflictError@132-290s -> LALRBudgetExceeded@~10-15s (decline ->
decline, same compile_error bucket, class change declared); dp/lr1_merge
differential scoped (table equality under budget, class equality over);
stratified p50 within noise.

Gain: eliminates the last LALR-family timeout deterministically; combined
with P3, residual 120s caps drop to zero known families; median untouched
by construction; no TTFM p50/p90 movement claimed (both schemas are
declines, not compiles).

Effort: M. Risks: a legitimate >8M-item completer outside the profiled
sets (full-corpus zero-fires gate + headroom rule + kill switch); review
may refuse o27148's class change (fallback ITEM_BUDGET=50M keeps its class
at ~120s but with only 4% margin; the scoped 8M variant is strongly
preferred); ~1GB peak before fire at 8M CPython items; counter-placement
bugs pinned by the determinism tests.

### P2-direct-emission (performance, M, Wave B)

Mechanism: today: compile_json_schema -> .grid text -> spec.load(text) ->
projection.build(). Probed: spec_load is 48% of front-end time (64% of that
is regex parsing, 32% validate); mock object-build is 3.8-5.4x cheaper
(11.7 -> 2.2ms, 8.9 -> 1.9ms, 6.9 -> 1.8ms); a trusted full-projection fast
path replaces build()'s 6.8-9.5ms with 0.4-0.5ms, hash-equal verified.
Plan: SchemaCompiler.compile_parts() -> GrammarParts manifest +
render_text(parts) (compile() = render_text(compile_parts()), byte
identical); DialectGrammar.from_parts replays only the _parse_source
contract (single decl counter, literal first-use ordering which determines
terminal_order and therefore mask numbering, unescaping) and then runs the
existing validate() and freeze() verbatim, because GrammarInvalid outcomes
(unproductive recursion really occurs) are load-bearing recorded outcomes;
RoleProjection.full_built fast path; linear worklist rewrite of the
reduction fixpoints. Text emission is demoted to the debug/audit path and
permanent CI oracle. Serving is unchanged (vllm_processor receives grammar
text); gains land in the schema -> mask compile pipeline.

Steps: freeze the phase-share baseline; manifest refactor (Gate A before
proceeding); from_parts + unit tests (decl assignment, epsilon alts,
first-use order, GrammarInvalid and L-REC01 parity); wire behind
GRID_PERF_DIRECT_EMIT (default 0; store schema_src hits keep the text
path); full_built fast path; worklist reduction (set-equality gated);
diff_direct_emit.py over all 11.3k + a CI subset + GRID_DIRECT_EMIT_CHECK=1
render/reparse oracle mode; TTFM A/B, flip only after the bake.

Gates: Gate A byte-identical emitted text over 11.3k; Gate B flag
differential where terminal_order tuple equality is the primary assertion
(the fingerprint hashes sorted names and cannot catch a numbering bug that
would silently ship wrong masks under the same kernel key), plus
fingerprint/productions/outcome-bucket/L-REC01 parity; Gate C reduction
set-equality; Gate D role_shape_hash + LALRTables.fingerprint byte-equality
plus sampled mask-level walk differential and median no-regression.

Gain: tens of ms in the p90 band (77ms) on 10-40k-rule grammars
(CANDIDATES id-10 estimate 10-30% of the p90-p95 band); sub-ms absolute at
p50; p99 unchanged (scanner/LALR-bound). Honest unknown: the corpus-wide
aggregate rests on a 26-schema stratified probe until the step-8 A/B on the
standard bench host.

Effort: M. Risks: terminal-numbering drift is the top hazard and the gates
are built around it; silent renderer/builder divergence over time (single
manifest feeds both + CI oracle); statechart lifecycle parity
(compile_tables asserts proj.state); store epoch invalidation on merge is a
bench-comparability trap (re-warm before A/B); do not oversell in-serving
TTFM.

### S1-jump-forward (serving, M, Wave B)

Mechanism: the guide already emits Write spans for singleton-mask chains
(j_max=8), and mode 1 consumes them; serving delivery is the gap. Physics
correction: forced tokens still need KV entries, so this collapses k decode
steps into one multi-token verification step, exactly the spec-decoding
compute shape vLLM already has; at a singleton position the bitmask puts
probability 1 on the forced token, so a forced-span draft is accepted with
certainty under any sampler: parity-exact, not approximate. Layers: public
GridGuide.forced_run + GridGrammarSession.jump_tokens (v5 via guide states;
v6/v7 via session_fill scratch-row popcount + session_accept, chaining
warm-only, ~0.2ms per j=8 chain at 25us TBM); patch site 5 injects the span
as the request's draft tokens post-accept_tokens (idempotent-anchor
discipline); the logits-processor route stays singleton-degrade untouched.
Byte-level jump-forward stays a recorded v2 deferral.

Steps: Stage-0 density probe first (fraction of singleton non-eos steps,
span-length histogram, byte-run coverage gap; proceed only if forced steps
>= ~5% on a target workload, publish the histogram either way); session API
+ unit parity tests; GPU-box probe of vLLM 0.24 SO+spec composition and the
draft-injection point (decides the patch shape; hard block -> fallback is a
scheduler multi-token schedule, re-estimate at L); patch site 5 + dry-run
test; acceptance smoke (rate 1.0 at forced positions, token-identical
greedy output); serving bake across batch 1/8/32 before any default flip;
kernel session_forced_run op only if the Python chain shows up at batch 32.

Gates: seeded generations token-identical GRID_JUMP on/off, greedy and
multinomial, v5/v6 both ways, CI-pinned; jump_tokens sequence ==
_forced_span sequence; flag-off byte-identical no-op; mode-1 loop and audit
replay unchanged; processor route untouched.

Gain: no TTFM/TBM impact (decode-step lever): one saved GPU step
(~10-40ms at 0.5-7B scale) per jumped token for ~25-50us of chain work.
Saved-step fraction is honestly unknown until Stage 0: plausible 5-25% of
decode steps on SQL, higher on rigid JSON-schema runs (XGrammar/SGLang
report 1.5-2.5x with the stronger byte-level variant; token-level captures
a subset). <5% density = documented negative result; only the cheap session
API lands.

Effort: M. Risks: vLLM 0.24 may block SO+spec or lack a drafter-free
injection point (probe first); low density on SQL choice points; draft
verification edge cases (bonus tokens, partial-acceptance rollback);
scheduler-thread chain cost at batch 32 (warm-only + j_max bound, kernel op
escape hatch); byte-level retokenization must never be bundled in (its gate
is corpus validity + EX delta, not parity).

### S3-store-wiring (serving, M, Wave B; arena payloads Wave C)

Mechanism: the merged store already persists schema_src / scanner / lalr.
Verified gaps: (1) component namespace, keyed blake2b(is_literal||pattern),
cross-schema by construction, consulted inside factored._component behind
the in-memory memo; lazy facades explicitly skip put (unpicklable lock,
probe-confirmed; post-P3 they rebuild in ms, so only eager components are
worth persisting); (2) trie namespace keyed by tokenizer fingerprint
(build_trie measured 178ms on gpt2/50k, fingerprint pass 11ms, unpickle
sub-ms); (3) epoch fix: add grid.lexer.factored and grid.trie.build to
_EPOCH_MODULES (wholesale invalidation, no FORMAT bump); (4) journal
namespace (CANDIDATES 20a): ContextJournal snapshot/restore of keys and
contexts only (never masks), keyed blake2b(grammar_src), restored before
admission_warmup under GRID_ADMIT_WARM=1; genN (p,q) keys are valid
cross-process because scanner numbering is deterministic per (code epoch,
grammar) and the epoch is in the store path; (5) persisted T2 masks
(20b) are an explicit non-goal this epoch (wrong-key hit = served wrong
mask); keying reserved via kernel_fingerprint(); (6) kernel-arena
namespace: key shape reserved only, implementation blocked on the P1/RUST
decision.

Steps: measure first (fresh-process phase attribution on the tail family;
post-P3 this population shrinks, so the namespace's tail claim is
re-scoped honestly); epoch fix + test; component namespace with
verify-mode read bypass and lazy put-skip; poison-builder + roundtrip +
GrammarInvalid-ordering + differential gates; trie namespace + entry_id
parity; journal snapshot/restore with cap enforcement, entry_id-identical
fuzz replay, and a cross-process genN numbering test; reserved-key note in
DESIGN.md; store_coldwarm.py harness (fresh process per measurement,
scenarios cold / redeploy-warm / flag-off reported separately, never
blended; production and verify flags reported separately; store size on
disk); sampled corpus differential CI gate.

Gates: outcome parity everywhere, zero scoped changes: warm hits proven by
builder poisoning, never timing; roundtrip equality; error parity (failed
builds never put); verify-mode always cross-checks; journal is timing-only
by construction with OBL-KEY1 hard-fail on divergence; epoch tests; all
namespaces behind the GRID_PERF_ARTIFACT_STORE master (default-off) with
per-namespace kill switches.

Gain: median-class redeploy stays at the measured ~2.3ms residual under
factored defaults; the once-per-process trie build leaves the first-request
path; journal restore moves the recurring-population cold-walk set off the
token path (first-request TBM toward the warm 25us kernel path vs the ~8ms
cold tail); no help on never-seen schemas, reported as such; redeploy TTFM
for seen populations moves toward the unpickle+load floor, re-baselined
with production flags (the 1.05s p99 is verify-inflated).

Effort: M. Risks: genN key coherence rests on deterministic subset
numbering (cross-process tripwire test); journal flush point unresolved
(implementation-time decision, exit hooks rejected); pickle surface grows
(store law: self-produced artifacts under a user-owned dir); epoch-module
addition invalidates existing stores once on rollout; file-count growth
reported by the harness.

### P1-kernel-lazy (performance, L, Wave C)

Mechanism: the one remaining scanner-family gap after P3: over-budget
schemas serve every mask via pure-Python _walk_py because lazy DFAs are
gated off grid_core. grid_core v8 introduces a Scanner backend enum behind
the existing accessor seam (lib.rs:208-259; tr()/accept/accepts_all/live
are the only scanner touchpoints in walk_raw): Dense keeps v7 verbatim;
LazyProduct holds per-component blobs (dense arenas for eager components,
compact NFA arenas for capped ones), interns component subset-bitsets and
sparse (tid, comp_state) product tuples on demand, and fills byte-class
rows + annotation words at first touch, so states materialize only along
trie paths (token-length-bounded, ~16-30 bytes). P3's LazyTerminalDFA is
the executable Python spec. Instance-local demand-order numbering is sound
as-is: lazy schemas already use raw schema-scoped T1/T2 keys and no kernel
output embeds scanner state ids (deviation from the CANDIDATES id-1 sketch,
whose coherent-numbering phase is unnecessary and whose NFA-simulation
bridge is obsolete). RUST_SCANNER disposition: subsume, not size-gate:
harvest scanner_build.rs's CharSet/NFA/eps_star modules as in-kernel
machinery, retire the eager build_scanner_arena entry point; its +18-22ms
p50 FFI floor is structurally avoided because v8 ships compact blobs once
at walker construction and never rehydrates arenas into Python. Size-gated
dispatch is retained only as the fallback if v8 slips a milestone.

Steps: measure the touched-state bound first on the family + full-vocab
walks under a throwaway prototype (expect 10^3-10^4 distinct product
states vs 268,803 eager; abort or rescope if unbounded); Python spec =
P3's substrate + the enumeration consumers (subset-BFS shortest-lexemes
leg, audit for silent full-build forcing); spec gates (lazy
full-materialization == eager per component, corpus mask differential);
v8 backend (subset interner, sparse tuple interner, memoized rows,
mutex-guarded creation, intern cap with wholesale reset, harvested RUST
modules); dispatch behind GRID_PERF_KERNEL_LAZY with the v7 assert and
fallbacks preserved, RustVerdicts staying Python-side for lazy schemas
this phase; kernel parity leg incl. a repeated-parallel-walk
id-independence test; recorded-set and interleaved p50 parity; record the
RUST disposition and the component-artifact persistence rule (deterministic
component artifacts only, never interner state); bench and flip.

Gates: mask-outcome parity, not numbering parity: full-vocab differential
kernel-lazy vs Python-lazy vs eager (where buildable) over the parity
corpus + the nine family schemas, state ids explicitly excluded; component
equality vs eager; recorded degradation sets and error text byte-identical
(budget predicates read pattern text, not build success); shortest_lexemes
bytes identical; interleaved fast-schema ON/OFF p50/p90 within noise (the
exact gate RUST_SCANNER failed). Flags: GRID_PERF_KERNEL_LAZY default off
until gates pass; GRID_NO_RUST and the factored budget preserved as kill
switches.

Gain: kills the entire remaining scanner-family serving gap: mask serving
for over-budget schemas moves from _walk_py to the v7-speed kernel path
(warm ~25us p50, bounded ~8ms cold-miss parity restored for the lazy
regime); the family's compile cost is already sub-second from P3, and v8
bounds its materialized states at ~10^3-10^4; post-change TTFM p99 becomes
verify- and LALR-dominated instead of scanner-dominated; llguidance's
lazy-DFA serving at p99 6.7ms is the existence proof for the target
regime. Median untouched by construction.

Effort: L. Risks: the touched-state bound could fail on some vocab/schema
combination (measured before any Rust work; intern cap + reset bounds
production damage either way); interning nondeterminism under the rayon
pool (no kernel output embeds state ids, test-enforced; sequential-walk
fallback); Rust/Python subset-semantics divergence (Python lazy leg stays
the executable spec, parity gates bound it); RustVerdicts staying
Python-side makes the win partial if CD re-checks dominate (measured,
phase-2 follow-on); per-byte NFA stepping vs dense lookup (memoized rows,
cap tuning).

### S2-slicer (serving, M, Wave C)

Mechanism: measured: the flat ~7.2-7.7ms grammar-independent mode (1-8ms
band, 6.4% of masks, 76.3% of pooled mask time) is string-interior giant
masks: 84/121 cold walks, 97% of cold-walk time, each a full 275,348-node
trie DFS. llguidance-style but proof-carrying: once per tokenizer, partition
the vocab by one JSON-string-safe slice class (96.2% of Llama-3.1 tokens)
into a sorted alias-complete slice-id array plus a rest-trie (10,415 nodes).
At walk time, prove slice containment structurally from the seeded state
(BFS closure over the slice byte-class, cap 64 states, every transition
non-DEAD, live/ignored condition, lexicon-inertness guard mirroring the
audited genN normalization); proof succeeds -> ci = sorted-merge(slice_ids,
walk(rest-trie)), byte-identical to the unsliced walk including the v7 blob
and entry_id because both tries enumerate in the same byte-lex DFS order;
proof fails -> today's full walk byte-for-byte (all-or-nothing v1;
maxLength windows simply fail closure and fall back). Warm paths untouched;
lazy DFAs gated off in v1.

Steps: re-verify the macOS probe numbers on the H100 box first; TrieSlices
in build_trie behind GRID_SLICER; kernel slice tables + slice_contained +
merged walk (rayon path guarded); mirror the identical logic in the Python
spec so parity stays bit-for-bit; new differential + containment unit tests
(quote/backslash/control bytes force failure; maxLength chains fall back;
lexicon-live states refuse the proof); maskbench rerun on the same
corpus/sample/seed (zero validation-count delta, then the TBM comparison);
H100 serving stamp with co-batch and adversarial max-step checks before the
default flip; per-slice length-capped slices and lazy-DFA slicing are
explicitly follow-on.

Gates: outcome parity, not timing parity: sliced output byte-identical
(ci_bytes + blob + entry_id) to unsliced on every configuration, kernel and
spec paths, both key regimes; all existing parity/digest suites green both
flag values; maskbench zero delta in validation/invalidation counts; TBM
p95 <=1ms, p99 <=1.5ms, warm p50 within 10% of 25us; default stays off
until the H100 stamp passes.

Gain: TBM p95 7.20ms -> ~0.3-0.5ms and p99 7.46ms -> ~1ms (walked nodes
275,348 -> 10,415 plus a ~0.5MB memcpy); pooled mask time drops ~70-75%;
GRID's cold tail moves below llguidance's measured p99 6.7ms; warm p50
unchanged; the >=8ms scanner-family tail is out of scope (P3/P1's job).

Effort: M. Risks: false-ACCEPT from an unsound containment proof is the
forbidden class (proof-or-walk rule, closure-derived byte exclusions, the
byte-identity differential turns divergence into a test failure, never a
served mask); per-slice partial containment later breaks blob parity and is
a scope-creep hazard (M -> L); containment memo growth bounded by the
existing cap pattern, degrading to full walks, never wrong masks;
low-coverage tokenizers (non-JSON grammars) get little benefit, coverage
logged at build time with a skip threshold.

### P4-counting-component (performance, L, Wave C)

Mechanism: revive the held COUNTING_WINDOWS surface (worktree
wf_12480d7a-e5d-1) as a factored component type, discarding its eager
combined construction (BAKEOFF: standalone subsumed by FACTORED).
CountingTerminalDFA per terminal (guarded-eps loops, annotated-closure
determinization, single-terminal fallback replaces the cross-terminal
rebuild loops); LazyProductDFA gains a global counter table with per
component cid offsets and a guarded step (variants keyed per (sid, gclass),
never a memoized single successor); materialize emits the held worktree's
verified ScannerDFA format so every ported runtime dispatcher works
unchanged. Kernel v8: counts register ([u16;8]) + CSR guard rows; the
walk_payload blob and entry-id hashing must fold counter values or T2
sharing serves wrong masks. Phase 3 (separately flagged, outcome-changing):
lift LENGTH_CAP and the window-degradation predicates so counting-eligible
windows are enforced instead of aliased to STRING, with closed-form loop
completion for guide/reserve at large m.

Steps: step 0 counts the coverage win from the full-run status JSONs
(schemas carrying window-degradation records) and the lazy-facade
window-bearing population, and measures per-component counting build times;
this number decides whether phase 3 and the kernel leg are worth L effort,
decided at step 0 and not after. Then: port the runtime surface verbatim
where possible; CountingTerminalDFA with the geps-reachability fix in
_terminal_reach (a known outcome-changing gap: guarded eps edges must count
as reachable or co_acc under-reports and forced emissions fire a byte
early); product extension with lazy+counting gated off genN exactly as
lazy-plain is; phase-1 gates with budget predicates frozen; kernel v8 leg
only if justified; phase 3 behind its own flag with a per-schema manifest;
full-corpus re-run.

Gates: phase 1: default-off byte-identity; flag-on outcome parity (emitted
grammars byte-identical, full-corpus compile-outcome diff zero,
recorded/degraded sets frozen, boundary lexemes at exactly m-1/m/n/n+1
including escapes and multi-byte UTF-8 straddling the count, MaskBench
parity, genN/T2 fuzz across counting x lazy). Phase 2: bit-identical masks
kernel-vs-spec; counts proven folded into entry ids by cross-instance T2
adoption tests. Phase 3: enumerated outcome manifest (exactly the step-0
set), one-sided mask containment (enforcement only tightens), round-trip
generation validation, no stratified p50 regression >10%.

Gain: window components drop from O(n) to O(1) control states, so
multi-window schemas that breach the product budget into the lazy facade
re-materialize eagerly and return to the 25us kernel warm path; phase 3 is
the item's real point: every "length window degraded / beyond cap" record
becomes an enforced window up to 8192 at O(1) build cost. Honest limits:
does not touch the substring-union family (not counting-eligible) nor helm
LALR; headline p99 barely moves; p50 unaffected.

Effort: L. Risks: memoizing a count-dependent successor is the forbidden
wrong-mask class (variants only; adversarial counter-boundary tests); the
geps-blind reach bug is subtle and outcome-changing (fix + verify
cross-check mandatory); entry ids omitting counts alias masks
cross-instance; large-m BFS blowup without the closed-form completion;
step-0 may show the coverage win is small (then only a reduced scope
ships); BAKEOFF honesty: composition value must be demonstrated by the
lazy-facade population actually shrinking.

## 6. Non-goals (this roadmap)

- Coverage and error-count work is excluded per user decision: no items
  here target reducing the declared Unsupported / GrammarInvalid /
  conflict-family counts or expanding schema-feature support. When that
  program is picked up, its inputs already exist: the lalr_conflict_family
  schema set and the retry-on-conflict cost protocol are prepared artifacts;
  it will need its own epoch with its own outcome-change adjudication.
  (P4 phase 3 window enforcement is the one scoped exception in this
  roadmap: it tightens enforcement on schemas that already compile, is
  decided by its own step-0 count, and ships with a per-schema manifest.)
- Cross-engine reruns (regenerating llguidance / XGrammar reference legs,
  including a same-definition first-mask-included llguidance leg) happen
  only on explicit user request. Until then, published llguidance numbers
  are quoted as-published and labeled compile-only (E4 discipline).
- Byte-level jump-forward: recorded v2 deferral (tokenization changes audit
  records and guide-state keys); token-level S1 never bundles it.
- Persisted T2 masks (CANDIDATES 20b): deferred until grid_core exports a
  version/blob-format constant and a served-mask-parity gate exists; only
  the key shape is reserved (S3).
- ARTIFACT_STORE default-on: deferred to a serving-epoch decision with a
  warm-hit p50 measurement (the BAKEOFF F2 cold-build floor is the reason).
- Kernel/serving flag disposition (GRID_V7, GRID_GENN_KEYS, GRID_NO_RUST,
  GRID_DEFER*, GRID_ADMIT_WARM*): out of E3's scope; those flags carry
  their own byte-for-byte replay disciplines and belong to a kernel-epoch
  decision.
