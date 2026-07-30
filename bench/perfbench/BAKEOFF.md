# 0.3.x bake-off: per-candidate A/B against the v0.2.5 baseline

One machine, strictly sequential (one child process at a time), flag-ON runs
inside each candidate's worktree vs the flag-OFF attribution baseline
(tmp/perfbench-profile, byte-identical code path). Sets: the 16 compile-capped
schemas, the top-40 of the tail-1%, stratified stride-7 (29). Timeout 120s.
Caveat: the RUST_SCANNER leg ran under its worktree's own venv (rebuilt
wheel); interpreter differences may inflate its small-schema constant
slightly.

## Headline table

| candidate | caps fixed (of 16) | tail (>5s) total | fast-schema ON/OFF p50 / p90 | decision-rule fit |
|---|---|---|---|---|
| FACTORED_SCANNER | 1 (o79409, 1.36s); 4 more improved from >120s cap to 5.8-89s (unmarked partial records had counted them as fixed) | ~1016s -> ~156s (~6.5x); the 27.9x excluded four killed children's real times | 0.94 / 1.01 | PASS (b,c,d) |
| HASHCONS | 5 decline-latency wins (120s cap -> <=0.5s declared outcome: 4 Unsupported + wp_105 conflict), not compiles | (its wins are all caps) | 0.97 / 0.99 | PASS (b,d) |
| COUNTING_WINDOWS | 0 | 1016s -> 454s (2.2x) | 0.99 / 1.01 | subsumed by FACTORED standalone |
| NFA_LIVE | 0 | 1016s -> 980s (1.0x) | 0.89 / 0.96 | PASS (d); prerequisite value only |
| LALR_DP | 0 (helm still hangs) | ~1.1-1.27x LALR phase | 0.94 / 1.01 | PASS (d); family half-covered |
| RUST_SCANNER | 6 (but to 5-99s, not <1s) | 1016s -> 66s (15.4x) | **2.53 / 4.32** | **FAILS (d)**: +18-22ms FFI floor per build |
| ARTIFACT_STORE | n/a | no standalone win | n/a | standalone caches schema_src only (0.2% of tail mass) |

## Per-cap outcome matrix

Outcome accounting at HEAD (see the Combined run section): 5 compiled,
5 terminate deterministically, 6 still time out. The former "union of caps
fixed to sub-second: 10/16" figure is withdrawn — it counted unmarked
partial records as sub-second fixes — and no replacement union number is
published: it is PENDING the outcome-aware re-run (the HASHCONS leg has not
been audited for the same artifact). RUST drags o47656/o47657 under the 120s
cap (74s/99s) by constant factor but not near 1s.
[Resolved: the replacement accounting is published in the E4 postscript
below — 10 compiled / 5 declared / 1 timeout at wave-A HEAD, classifier-
gated, with both TTFM definitions.]

Never fixed by any candidate: `o5195, o48423, o48427` (substring-union
scanner family, same asymptotics under the lazy product), `helm-testsuite`
(LALR hang; the DP construction also times out). These 4 (plus RUST's two
marginals) are the honest remaining tail for a follow-on wave.

## Verdicts for integration into 0.3.0

MERGE (default-on after full-corpus verify): HASHCONS, FACTORED_SCANNER,
NFA_LIVE (exactness prerequisite; ~1.06x standalone, median 0.89 here),
LALR_DP (small real win, zero regression, 273-schema zero-mismatch
differential).

MERGE default-off (infrastructure, no perf credit claimed):
ARTIFACT_STORE - verified + fixed (insertion-order keying, FORMAT=2);
its real payoff (persisting factored components / arenas) is integration
work, measured then.

HOLD: COUNTING_WINDOWS - standalone subsumed by FACTORED (its 15x wins are
FACTORED's 100x wins); revisit as a component type under the factored
library, where its counter automata also unlock the <=64 window budget.
RUST_SCANNER - real 15x tail carrier but fails the p50 gate (+18-22ms FFI
floor on every small build); integration requires size-gated dispatch
(route to Rust only above an NFA-size threshold); revisit with FACTORED
composition.

## Expected v0.3.0 shape (to be confirmed by the combined run)

TTFM p99: 4.37s -> sub-second everywhere except ~6 known schemas
(vs llguidance p99 6.7ms: same order of magnitude for p95, honest gap
remains at p99+). Median: unchanged (by design this epoch). The 6 residual
schemas and the spec_load/direct-emission median work define the follow-on
wave (CANDIDATES.md ids 1, 10, 13, 16, 19, 20).

## Combined run (integration/0.3.x @ a06b672: HASHCONS + LALR_DP + FACTORED, NFA default)

The 16 baseline caps, three outcome classes: **5 compiled** (o79409 1.36s,
o83133 5.8s, o83132 89s, strmprivacy BatchJob 11.9s / DataConnector 13.4s),
**5 terminate deterministically** (frontend family: 4 declared-Unsupported
in 0.03-0.5s; wp_105 LALRConflictError in 0.4s — a genuine anyOf-overlap
ambiguity manufactured by `_harmonize_string_consts`' consts path, NOT an
identical-RHS dedupe gap; hashcons converted its flag-off >150s
schema_compile hang into this fast honest decline), **6 still time out**
(substring-union scanner family x5 + helm LALR).

**F1 RETRACTED**: the "o83132 0.0s FACTORED-alone vs 88.9s COMBINED"
regression never existed. The FACTORED-leg records for
o83132/o83133/DataConnector/BatchJob are unmarked partial records from
killed children (running=scanner, no timeout_s) that read as ~0.0s. Emitted
grammar is byte-identical hashcons on/off (verified, md5-equal across 4
independent probes), and FACTORED-alone == COMBINED within noise on all
five schemas (89.0 vs 90.1s; 5.75 vs 5.76; 13.42 vs 13.43; 11.91 vs 11.87;
1.01 vs 1.00). The real cost: one substring-union terminal (a 13-branch
`.*trace.*|...` JSON-string union) builds a 268,803-state component DFA in
~86-89s — the component stage is the only unbudgeted stage in the factored
path. o83132/o83133/DataConnector/BatchJob therefore move into the
substring-union residual family alongside o5195/o48423/o48427.

**F2 attributed**: the fast-schema p50 1.12 / p90 1.46-1.48 was NOT flag
composition. Root cause: compile_json_schema's unconditional grid.serving
import (uniform ~+3.2ms per process on the integration branch vs main's
passthrough), now fixed with the env pre-check. Per-phase decomposition of
the raw records self-contradicts (all regression sits in schema_compile
while HASHCONS-alone measures that phase negative), confirming
baseline-session drift on top. Protocol rule going forward: interleave
flag-ON/OFF pairs per schema in one session; records now carry a
GRID_PERF_* flags snapshot. Recorded separately: GRID_PERF_ARTIFACT_STORE=1
adds ~5-7ms schema_compile per cold fast build (canon + round-trip +
blake2 + get/put) — the exact p50-gate failure mode that sank RUST_SCANNER;
must be re-measured warm-hit before any default-on decision.

**F3 rescoped**: wp_105 raises exactly 2 reduce-reduce + 2 shift-reduce
conflicts (not 4 RR); the twin rules are NOT identical-RHS (r279_v: `E6` vs
r2048_v: `E16 | E6`), and an identical-RHS merge would be language-changing
(it would admit "custom_embed" where only "reference" is legal) — refuted,
do not implement. Both LALR constructions agree on the conflict set. A
normalize-level fix (unify the consts path of `_harmonize_string_consts`
across all branches) is prototyped: WashingtonPost 24 -> 3 conflicts, 528
schemas swept, 22 fixed, zero regressions — a follow-on wave item behind
the full corpus-differential + MaskBench gate. Residual conflict families
after it: 3 WP member-chain twins (wp_17/35/98) + 18 Snowplow STRING value
twins.

Footnote: records with running != null and no timeout_s/rc marker are
incomplete, not fast; profile_phases.py now marks crashed children (rc),
requeues unmarked partials on resume, and reports incomplete records
explicitly.

## Postscript: flag disposition (E3, post-v0.3.0, wave A)

Executed against the rc2 full-corpus run (tmp/mb-grid-v030rc2, 11,306
schemas, GRID_PERF_HASHCONS=norm,dedupe GRID_PERF_LALR_DP=1
GRID_PERF_FACTORED_SCANNER=1; adjudicated in
bench/RESULTS-jsonschemabench-v0.3.0rc2.md). This section records outcomes;
the tables above are history and stand as written.

| flag | verdict executed |
|---|---|
| NFA_LIVE | DELETED (flag + `_live_fixpoint` + `_graph_co_acc`/`_live_mode` + verify branches); sanction: the 11.3k zero-divergence verify pass on rc1. Surviving gates: test-local forward-BFS oracle (test_live_sets.py), eager-vs-factored byte-identical differential. Component memo key loses its mode dimension. |
| FACTORED_SCANNER | default ON; `=0` kill switch kept — the eager union builder remains the factored path's exactness oracle |
| LALR_DP | default ON; `=0` kill switch kept — lr1_merge remains the construction-independent oracle (helm residual work wants the A/B) |
| HASHCONS | default `norm,dedupe` (the measured configuration, pinned by name — a future component needs its own default decision); `=0`/comma-list grammar kept |
| ARTIFACT_STORE | stays default-off per F2 (+5-7ms cold schema_compile per fast build); revisit with a warm-hit measurement in the serving epoch |
| FACTORED_BUDGET | unchanged: documented tuning knob (20k), not a flag |

Serving smoke at the new defaults (replaying valid corpus instances
byte-token-wise to COMPLETE): dense regime OK; naturally-over-budget
schema (Github_medium o33033) served 1091 tokens through the
LazyProductDFA facade with the kernel- and genN-exclusion gates asserted
live, forced-lazy == natural trajectories.

## Postscript: P3 component budget (post-v0.3.0, wave A)

The substring-union residual family is closed. Root cause held up under a
direct probe (the F1 anatomy above): the persistent per-keyword matched bit
gives the eager per-terminal subset construction ~2^k reachable subsets
(o83132 S2 reproduced at 50k states/9.9s still climbing toward the recorded
268,803; o5195 S0 breached a 30k probe cap at 8s with the frontier open),
while a demand-driven walk of the same automaton materializes at most one
new subset per scanned byte (o83132 post-fix: 9,028 walked test-corpus
bytes -> 478 product states / 58 S2 subsets, 0.75us/byte cold, 0.35us/byte
warm, zero growth on re-walk — the deferred memo-cap/product-swap lever
stays unneeded).

Fix (grid/lexer/factored.py + subset.py): GRID_PERF_COMPONENT_BUDGET caps
the eager component build; over the cap the terminal comes back as a
demand-interned LazyTerminalDFA (annotations are pure functions of the
subset value, so the product's recorded sets are exact by construction) and
the scanner skips straight to the lazy product — the union DFA is at least
component-sized, so the product budget would abort materialization anyway.
Default 16384 from the manifest-set sweep (853 schemas, 21,223 unique
patterns): largest terminating NON-family component 7,210, but strmprivacy
Stream carries a 15,865-state terminating component inside a product that
compiles DENSE today (its eager and factored digests are equal), so 16384
is the smallest power of two that keeps every dense-today build
byte-identical; every 2^k member still breaches (certifying "component >
budget" inherently costs budget-many subset states: 1.8-11.9s across the
family, o48423 worst).

Family outcomes, one process per schema (profile_phases p3_family set,
jobs 1), vs the rc2/F1 baselines:

| schema | before | after (scanner phase) |
|---|---|---|
| o5195 / o48423 / o47656 / o47657 / o48427 | compile timeout (>=120s) | 5.35 / 12.38 / 2.94 / 2.94 / 5.13s |
| o83132 | 87-89s, 2.2GB RSS | 3.15s, 136MB (family RSS max 237MB) |
| o83133 / o33033 | 5.8 / 6.2s | 2.32 / 2.22s |
| BatchJob / DataConnector | 11.9 / 13.4s | 10.29 / 11.70s (their 8.7-15.9k-state terminating components stay eager under 16384 by the Stream/DataContract byte-identity constraint; an 8192 budget would halve these two at the cost of flipping Stream/DataContract off the dense path) |
| DataContract / Stream | 7.4 / 0.7s (warm-memo) | byte-identical path (6.58 / 3.14s cold; Stream dense, 16,229 states) |
| o83677 / io-package | LALRConflictError before scanner | unchanged |

Gates run: (a) scanner-digest over all five manifest sets + builtin (878
units, tmp/scanner-digest-p3-{pre,post,off}): kill-switch leg
(--flag GRID_PERF_COMPONENT_BUDGET=0) 878/878 bit-identical to pre; default
leg diverges on exactly one unit — o33033, the only family schema whose
eager arm finishes inside the tool timeout — and a 200-state
FIFO-order product probe of both variants shows identical rows/accept/
accepts_all/live/h_max (the divergence is the component-representation
encoding in the digest, not behavior). (b) MaskBench over the 14-schema
family vs rc2: all 9 previously-completing schemas outcome-IDENTICAL
(test verdicts, token counts, and TBM p50 unchanged: o83132 165->165us,
BatchJob 180->181us, Stream dense 27->30us; p99 tails unchanged), and the 5
timeouts flip to runs with ZERO validation/invalidation errors. (c) reserve
BFS on lazy components: 10-30ms, 65-87 states materialized (visits each
state once, bounded by shortest-word depth — no cap needed, and none could
help: capping would trade this for the strictly larger eager cost).

Residual: the certification cost itself (seconds, linear in the budget) —
any exact cap pays it once per breaching pattern per process; and the
(16384, 20000] corner (a hypothetical out-of-corpus terminating component
whose product fits the product budget) degrades gracefully to the lazy
product with identical masks, restorable via GRID_PERF_COMPONENT_BUDGET=0.
helm-testsuite LALR is now the only known compile-cap family.

## Postscript: E4 honest metrics (post-v0.3.0, wave B) — dual-column TTFM + outcome-aware republish

Two dishonesty sources closed. (1) TTFM blind spot: under the wave-A
defaults, over-budget schemas return a LazyProductDFA facade whose product
construction is deferred past every compile phase; the kernel walker and
genN cache exclude lazy DFAs, so the deferred cost is paid as pure-Python
cold trie walks — previously measured NOWHERE. (2) Published counts had no
classifier: the F1 retraction (unmarked partial records read as sub-second
fixes) was a tooling failure mode, not a one-off.

**Definitions (permanent; DESIGN.md ground rule).** *TTFM compile-only* =
the five compile phases + GridGuide construction — maskbench
compile_grammar semantics, the definition behind every published
RESULTS-*.md number; name and meaning unchanged. *TTFM
first-mask-included* = compile-only + the first compute_mask at the
initial state. Every table labels its column; publishing either number
unlabeled is a publication error from here on. The per-child tokenizer +
trie build (~2.4-2.5s, maskbench's per-engine constant) is its own
excluded phase in both columns.

**Tooling.** profile_phases.py --first-mask appends trie / guide /
first_mask / prefix_masks phases (flush-streamed, so timeouts attribute);
the child writes stats.ttfm_compile_us / stats.ttfm_first_us.
bench/perfbench/outcomes.py is the classifier behind every count here:
ok | declared:<class> | timeout@<phase> | crash | incomplete | malformed,
where incomplete (running != null, unmarked) is never ok, and a completed
--first-mask record missing its phases is malformed, not ok. Unit-gated
against the full real corpora: mb-grid-final = 668 declared / 16 timeout /
10,622 ok of 11,306 (ok+timeout == the manifest's 10,638 ranked);
mb-grid-v030rc2 = 635 / 7 / 10,664.

**Census (step 1).** Under wave-A defaults, 9 of the 45 headline-leg
schemas build LAZY scanners: the substring-union five (o5195, o48423,
o48427, o47656, o47657 — component-budget breaches, P3), o83132/o83133
(same family), and strmprivacy BatchJob/DataConnector (their breaching
substring-union terminals skip the product to lazy). The plan's zero-lazy
risk did not materialize: the deferred cost is live in exactly the
headline schemas.

**Protocol.** One session, --jobs 1, 120s cap, flag-ON/OFF interleaved per
schema (the F2 rule), --first-mask, tokenizer
unsloth/Meta-Llama-3.1-8B-Instruct offline, flags snapshot in every
record. ON = wave-A defaults; OFF = kill switches
(GRID_PERF_FACTORED_SCANNER=0, GRID_PERF_LALR_DP=0, GRID_PERF_HASHCONS=0).
Records: tmp/e4-legs/{on,off}; canonical prefix = first valid test
instance, capped 64 tokens.

**Parity gates.** OFF leg vs v0.2.5 (mb-grid-final), strict identity:
45/45 unchanged — all 16 caps reproduce as timeouts in their historical
phases (10 scanner, 5 schema_compile, 1 lalr), all 29 stratified ok. ON
leg vs v0.2.5, oracle rule: 30 unchanged + 15 improved, every improvement
in the sanctioned timeout -> terminating direction, zero gate failures,
zero incomplete/malformed records, zero prefix-walk rejections on either
leg (the grammar-drift tripwire).

### capped-16 at wave-A HEAD (ON leg): the replacement accounting

**10 compiled / 5 declared / 1 timeout** — supersedes both the withdrawn
"10/16 fixed" union number and the pre-P3 "5 compiled / 5 declared / 6
timeout" Combined-run accounting (P3 moved the five family caps into
compiles). The first-mask-included definition flips NO bucket: every
compiling cap also served its first mask and 64-token prefix inside the
cap.

| schema | scanner | TTFM compile-only | TTFM first-mask-incl | prefix walk (n tok, worst) |
|---|---|---|---|---|
| o79409 | dense | 1.43s | 1.44s | 64, 9.6ms |
| o83133 | lazy | 2.48s | 2.49s | (no tests) |
| o47656 | lazy | 3.00s | 3.01s | (no tests) |
| o47657 | lazy | 3.08s | 3.09s | 64, 251ms |
| o83132 | lazy | 3.30s | 3.31s | 64, 249ms |
| o48427 | lazy | 5.33s | 5.33s | 64, 252ms |
| o5195 | lazy | 5.35s | 5.36s | 31, 239ms |
| BatchJob | lazy | 10.72s | 10.72s | 64, 298ms |
| DataConnector | lazy | 11.66s | 11.66s | 64, 295ms |
| o48423 | lazy | 13.12s | 13.12s | 64, 288ms |
| o12175, o11667, o39217, cloudify | — | declared Unsupported <=0.1s | — | — |
| wp_105 | — | declared LALRConflictError 0.6s | — | (single-shot: profile_phases has no conflict retry; the maskbench runner's retry compiles it, rc2) |
| helm-testsuite | — | timeout@lalr (both legs, both constructions) | — | — |

OFF leg: all 16 time out at the 120s cap (same in-flight phases as
v0.2.5). The honest speedup statement is therefore ">=120s ->
1.4-13.1s or a fast declared outcome", not a sub-second claim.

### The measured deferred cost (the number that was asserted-not-measured)

The first mask at the INITIAL state is cheap even on lazy products
(1.6-8.0ms across the nine: the JSON-structure start position wakes few
component states) — which is why the two TTFM columns differ by only
~1-10ms here and the blind spot never showed in aggregate columns. The
deferred product construction is actually paid MID-INSTANCE, when the
pathological terminal goes live inside a constrained string: on the seven
lazy schemas with test instances, the worst prefix token costs
239-298ms (a tight band; dense-path worst on the same legs: 7.7-9.6ms,
the known cold-walk band — the lazy pure-Python regime is ~30x that), and
the full 64-token cold prefix costs 1.6-11.8s. On the worst two the cold
prefix is the same order as the entire compile (DataConnector 11.77s walk
vs 11.66s compile; BatchJob 9.78s vs 10.72s). MaskBench's pooled TBM p50
for these schemas (165-181us, P3 gate) is warm-dominated and hides this
window; the prefix lens is the cold-start truth. This distribution is the
recorded go/no-go input for P1 (kernel-lazy residence): ~0.25s/cold-token
x instance-length exposure is the cost of NOT doing P1, and the
RUST_SCANNER size-gated-dispatch fallback stays unneeded for compile (the
certification cost dominates) but unresolved for serving until P1.

Caveat: o83133 and o47656 carry no test instances, so their mid-instance
cost is extrapolated from the family band, not measured; and prefix_masks
walks the FIRST instance only — a 64-token cap on a lens, not a bound on
the cost.

### stratified-29 (fast set), both columns, interleaved

| leg | compile-only p50 / p90 / max | first-mask-incl p50 / p90 / max |
|---|---|---|
| ON (defaults) | 9.1ms / 184.5ms / 449.6ms | 10.2ms / 185.6ms / 479.6ms |
| OFF (kill switches) | 8.2ms / 210.2ms / 1947.2ms | 9.3ms / 221.4ms / 1976.3ms |

Per-schema ON/OFF ratio (compile-only): p50 1.09, p90 1.29 — the first
interleaved same-session measurement of the shipped-defaults median cost
(what F2's protocol rule was written for). The defaults pay ~0.9ms at the
fast-set median and win the set's own tail (max 1.95s -> 0.45s); within
the decision rule's 10% p50 envelope, and the first-mask increment on
dense schemas is ~1.1ms (one cold walk).

### Cross-engine caveat (standing)

llguidance's published TTFM p50 0.38ms is compile-only semantics on a
fully lazy engine; its first-mask-included number does not exist in our
records. Any future cross-engine TTFM claim uses first-mask-included on
BOTH sides or is not made; a same-definition llguidance reference leg is
an explicit follow-on (ROADMAP non-goal until requested), never smuggled
into a GRID-only republish.

## Postscript: P2 direct emission (post-v0.3.0, wave B)

The spec_load re-parse is gone from the default schema->mask compile
pipeline. Baseline held up under the step-1 freeze (ttfm_capped +
stratified_200, 216 schemas, jobs 4, .venv-bench dev box): spec_load median
49.1% of front-end (schema_compile+spec_load+projection), spec_load +
projection 83.2% — the CANDIDATES id-10 probe shares reproduced on the full
profile sets.

Change (grid/grammar/parts.py, spec.py, projection.py, reduction.py,
grid/jsonschema/): compile_parts() emits a GrammarParts manifest; the text
emitter is render_text over that same manifest (compile() unchanged,
byte-identical over all 11,306 corpus schemas with PYTHONHASHSEED pinned —
the sweep surfaced pre-existing seed-sensitive emission on 8/11,306
schemas, out of P2 scope, noted in the step-2 commit);
DialectGrammar.from_parts replays only the _parse_source contract and runs
validate()/freeze() verbatim; RoleProjection.full_built registers the full
projection without the compose/reduce/verify rebuild; the reduction
fixpoints are linear worklists in all configurations (30k-rule chain:
139.8s -> 0.019s; corpus/projection/synthetic set-equality gated).
GRID_PERF_DIRECT_EMIT default ON after the gates; =0 restores text ->
spec.load; GRID_PERF_DIRECT_EMIT_CHECK=1 is the permanent render+reload
oracle (CI leg).

Gates run: (a) Gate A byte-identity 11,306/11,306 (114 status-equal
declared outcomes). (b) diff_direct_emit full corpus: 0 flips — 11,191
equal grammars (terminal_order tuple primary), 106 equal_unsupported, 8
equal_RxUnsupported, 1 equal_grammar_invalid (the unproductive-recursion
family occurs once in corpus and matches). (c) --tables leg over
ttfm_capped+ttfm_tail_1pct+stratified_200+tbm_tail_100 (309): 303 equal
incl. role_shape_hash + LALRTables.fingerprint + action/goto, 1
equal_LALRConflictError, 4 equal_unsupported, helm both-arms-timeout (no
oracle). (d) mask-level walk differential: 25 stratified schemas x 2 seeds
x 40 steps over one MockTokenizer trie, 2,000 allowed-id sets equal.
(e) full pytest on three legs: defaults, kill-switch, check-oracle.

Measured (same profile sets, flag-off vs flag-on on the post-change tree):

| metric | flag-off | flag-on |
|---|---|---|
| spec_load p50 / p90 / total | 1.30ms / 3.22ms / 0.65s | 1.12ms / 1.95ms / 0.39s |
| projection p50 / p90 / total | 0.83ms / 1.82ms / 0.36s | 0.66ms / 0.77ms / 0.15s |
| front-end, 12 costliest grammars | 693ms | 344ms (-50.3%; o87865 127->61ms) |
| terminals>=60 band spec_load+projection p50 | 3.83ms | 2.32ms |
| full pipeline p50 / p90 | 8.03ms / 239ms | 7.68ms / 238ms |

p90+ of the whole pipeline stays scanner-bound (P3's residual analysis
unchanged); completion/timeout structure identical across legs (210
complete; 4 declared-slow schema_compile + 2 helm-family lalr). Absolute
numbers are dev-box; shares are the durable claim (the plan's step-8
standard-host rerun remains open alongside the S1 GPU-box items).

Residual: store schema_src remains text (hits re-parse; the optional
'grammar' pickle namespace is the recorded follow-on), and the DP-LALR
int-id handoff (CANDIDATES id 6) now has a clean seam via GrammarParts.

## Postscript: S3 artifact store (post-v0.3.0, wave C)

The redeploy warm set shipped — component/trie/journal namespaces beside
schema_src/scanner/lalr, all under the default-off GRID_PERF_ARTIFACT_STORE
master with per-namespace kill switches (the CHANGELOG Unreleased S3 entry
carries the full design: cross-schema TerminalDFA payloads + breach
markers, slicer-variant trie keys, journal snapshot/restore under
GRID_ADMIT_WARM). Two reconciliations landed on review + the wave-b merge:

- **Failed-build law, scoped.** The old blanket "failed builds never put"
  predates the component namespace and broke against it: /x*/'s
  TerminalDFA is validly persisted before the scanner-level empty-match
  law rejects the grammar. Scoped law: grammar-keyed namespaces
  (schema_src/scanner/lalr) never persist from a failed build; the
  component namespace keeps VALIDLY-built components — its
  (kind, budget, pattern) identity is grammar-independent, and the failing
  check re-runs and re-raises identically over exactly such partial warm
  stores (GrammarInvalid ordering + message-parity tests construct them
  deliberately). The alternative (conditioning component puts on
  parent-build success) would strand cross-schema warmth precisely for the
  substring-union family the namespace was built for.
- **code_epoch() executes nothing.** import_module semantics charged
  grid.trie.build — and with it numpy, ~20ms — to the FIRST store access:
  a pure import tax on the warm-hit p50 this postscript quotes. find_spec
  fixed that but still executed parent-package inits
  (grid.jsonschema's pulls the whole compiler chain, ~31ms fully cold —
  real in grid-source-only serving processes). Final semantics: sys.modules,
  else a PathFinder walk over parent search locations — specs located,
  zero modules executed, fully-cold ~10ms / mid-compile ~7ms, epoch value
  unchanged. Pinned by a subprocess probe (no new grid.* module, no
  numpy).

Gates at this HEAD (post-merge, post-scoping, post-epoch-fix): full pytest
shipped defaults exit 0; full pytest legacy kill-switch leg
(FACTORED_SCANNER=0 LALR_DP=0 HASHCONS=0 DIRECT_EMIT=0) exit 0; the store
differential test file 8/8 under both flag states; diff_store_warm over
stratified_200+p3_family+ttfm_capped (221 units x off/cold/warm legs):
off-vs-cold OK 221 identical, off-vs-warm OK 221 identical — src, tables,
scanner, drive entry-ids, error text; timeout (helm) and declared-error
(x7) units agree across legs.

Measured (store_coldwarm.py protocol: fresh child per scenario, sequential;
C = flag-off, A = cold+persist on empty store, B = redeploy warm hit;
ms/schema pipeline totals; LOAD CAVEAT: dev box under concurrent wave-C
agent load, absolutes are upper bounds — scenario STRUCTURE is the durable
claim; a quiet-machine retake is warranted before quoting absolutes
anywhere else):

| group (n) | C p50/p99 | A p50/p99 | B p50/p99 |
|---|---|---|---|
| p3_family (14) | 5245 / 12947 | 5249 / 12815 | **17.1** / 919 |
| ttfm_capped (7) | 1039 / 1392 | 1207 / 1438 | **22.7** / 155 |
| stratified_200 (50, stride 4) | 7.1 / 999 | 18.4 / 1129 | 9.0 / 178 |

timeouts 1/1/1 (helm-testsuite, all scenarios — parity); declared errors
6/6/6. Per-schema store p50 139 KiB, max 15.8 MiB; deduplicated deployment
footprint 65.6 MiB over 1,616 unique entries (scanner 28.8M, component
17.7M, lalr 16.1M, schema_src 3.0M). Journal TBM legs (spider dialect,
compile artifacts warm in both): redeploy-warm first-request mask p50
14us / p99 48us vs redeploy-cold 52us / 64us — the journal moves the
recurring cold-walk set off the first-request path.

Reading: the warm set erases exactly the tails it was built for —
p3_family redeploy p50 307x under flag-off (breach markers skip the capped
eager attempt), ttfm_capped 46x — while fast schemas PAY at p50 (B 9.0 vs
C 7.1ms: consult + unpickle beats a trivial rebuild only when the build is
nontrivial; same shape F2 recorded cold, +5-7ms/A-leg). Disposition
unchanged: GRID_PERF_ARTIFACT_STORE stays default-off this epoch; the
default-on decision belongs to a serving-box warm-hit measurement over a
real redeploy population, where the B-leg tails and the journal legs are
the operative numbers.

## Postscript: P5 LALR construction budget (post-v0.3.0, wave C)

The last LALR-family hangs terminate declared. Both constructions now
count items materialized (sum of closure sizes at state creation —
input-derived, machine-independent, memory-proportional) plus states,
checked at every new state (two int compares); crossing either cap raises
the declared LALRBudgetExceeded carrying (states, items, item_budget,
state_budget), phrased in the frontend's "budget exceeded (size cap)"
family. GRID_LALR_BUDGET overrides the item budget ("0" disables both
caps — the audit/oracle escape hatch); the state cap rides the same knob
at budget // 8. Declines are never cached (store puts only on success;
serving's SingleFlight negative-caches 30s, then rebuilds to the identical
fire point), so budget outcomes reproduce exactly warm and cold.

### Calibration: the planned 8M default was refuted by the full corpus

The roadmap's anchor ("largest completer = tmlanguage, 1.83M items; 8M =
4.4x headroom") was profiled-sets-only. The calibration sweep this item
demanded ran the SHIPPED maskbench build sequence (initial build + LALR-
conflict retry) over all 11,306 corpus schemas, dp path, recording per-leg
construction counters (diff_lalr.py --counts --ids-from-dir
tmp/mb-grid-v030rc2; out: tmp/perfbench-lalr-counts-pre). Census
reconciles with rc2 exactly — ok 10,669 (rc2's 10,664 + the 5 P3-fixed
substring-union schemas, whose LALR is small and which timed out in the
scanner phase this child never runs), conflict 519 (== rc2's declared
LALRConflictError), skip 116 (== 106 Unsupported + 9 RxUnsupported + 1
GrammarInvalid), killed 2 (helm-testsuite, o27148 — rc2's LALR-family
timeouts). 11,734 completed constructions (conflict-retry legs included):

| completer (top by lr0_items) | items | states | s/leg | rc2 outcome |
|---|---|---|---|---|
| o21112 | 20,094,330 | 926,457 | ~54 | LALRConflictError |
| o21108 | 8,881,971 | 430,362 | ~24 | LALRConflictError |
| io-package | 5,792,920 | 207,610 | ~29 | LALRConflictError |
| o43189 | 4,534,113 | 240,116 | ~16 | LALRConflictError |
| meta | 2,852,780 | 112,688 | ~7 | LALRConflictError |
| pkg_schema | 1,950,817 | 189,477 | ~5 | ok |
| tmlanguage | 1,831,000 | 171,509 | ~8 | ok |

(completer p50 642 items, p99 227,972.) The conflict family's heavy tail
was invisible to the profiled sets: conflicts are a fill-stage outcome,
reported only AFTER the full construction, so a conflict schema is a
completing BUILD the budget must respect. o21112 at 20.09M items sits 2.5x
ABOVE the planned 8M default (which is therefore rejected — it would have
changed two completers' declared class), and only 2.4x BELOW the smallest
hang the budget exists to declare (o27148, 48.17M items). The plan's
">=4x headroom" rule is unsatisfiable on the items axis; the shipped
default is the log-midpoint of the measured gap:

    ITEM_BUDGET = 32M (1.59x above the largest completer, 1.51x below
    o27148), STATE_BUDGET = 32M // 8 = 4M (4.3x the largest completer's
    926k states — >=4x holds on the states axis).

Two structural arguments cover the thin items headroom: the completer
distribution is not near a continuum edge (p99 is 88x below the max; the
>2M population is five schemas, all measured exactly), and completer size
is cap-bounded — a build near 32M items costs ~90s+ per leg, two legs per
compile, which could not have finished inside the 120s corpus cap that
defines today's completer set. Deployments that want o27148's actual
conflict report back can raise GRID_LALR_BUDGET to 50M (~132s build).

### Measured fire behavior and gates

- helm-testsuite (LR(0) core diverges; 62.75M items and climbing at 60s
  pre-budget): 120s timeout -> declared in 28.3-29.7s at 32,000,099 items
  / 289,811 states, peak RSS 4.36-4.48GB across runs — bounded, and
  comparable to what the o21112 completer legitimately materializes today;
  identical fire counts across four independent runs (idle and
  10-job-loaded).
- o27148 (conflict detection needs the full 48.17M-item automaton + DP
  lookaheads, ~132s): 120s timeout -> declared in 39.3-42.6s at
  32,000,035 items / 1,096,970 states, peak RSS 4.65GB; identical fire
  counts across three independent runs.
- Corpus verify (tmp/perfbench-lalr-counts-on vs -pre, 11,306 records):
  fire set is EXACTLY {helm-testsuite, o27148}, both timeout ->
  declared:LALRBudgetExceeded — the outcomes.py-sanctioned direction
  (baseline timeouts have no oracle), and vs rc2 NO error-class changes
  at all (o27148 was a timeout in rc2, not a conflict; the plan's
  "conflict->budget class change" scoping dissolved on measurement).
  Zero fires on the other 11,304 schemas; construction counters
  byte-identical on every unchanged record.
- Overhead: o948 (928 states / 2,627 items) in-process compile_tables
  median, 200 reps, idle box: 4.933ms (pre-P5 b39c326) -> 4.973ms
  (counters + budget, default ON), +0.8%; interleaved
  default-vs-kill-switch under full sweep load: ratio 1.015.
  Independently reproduced in the verification pass: 4.897ms -> 4.945ms
  (+1.0%), interleaved ratio 1.006. The per-schema paired diff-run
  comparison (269 schemas equal in both differential sweeps) has ON/OFF
  dp-time ratio p50 0.97 (>=100ms builds, n=25: aggregate 0.91 — the ON
  sweep ran on a lighter-loaded box), i.e. at corpus scale the
  counter+budget cost is indistinguishable from run-to-run load noise;
  the controlled number is the in-process interleaved pair. All within
  the <10% stratified-p50 gate.
- dp/lr1_merge differential (ttfm_tail_1pct + stratified_200 +
  tbm_tail_100, budget ON): zero mismatches; helm classifies as the
  scoped over-budget outcome (class equality only — LR(1) materializes
  >= LR(0) items, so fire counts differ by construction).

Residual: with P3's substring-union fix, no known compile-cap family
remains corpus-wide — pending confirmation by the epoch's one-shot full
maskbench run (DESIGN.md ground rule), where the two fires will appear as
compile_error:LALRBudgetExceeded and outcomes.py compare must show exactly
two "improved" rows vs rc2. Peak-RSS-at-fire scales with the budget
(4.4-4.7GB measured at 32M CPython items); a deployment that needs a
tighter memory envelope lowers GRID_LALR_BUDGET at the documented
completer tradeoff.

## Postscript: P1 kernel-resident lazy scanner (post-v0.3.0, wave C) — grid_core v8

The lazy serving gap P3 left is closed: over-budget schemas (LazyProductDFA)
now walk the kernel under `GRID_PERF_KERNEL_LAZY` (default ON post-gates,
`=0` the kill switch; == "1" value grammar per the E3 convention) with a
v8+ grid_core, instead of pure-Python `_walk_py`. The scanner behind the walker is a
backend enum at the four accessor touchpoints — `Dense` is v7 verbatim,
`Lazy` interns sparse (tid, comp-state) product tuples and per-component
subset bitsets on demand (one build mutex for mutation, lock-free reads via
append-only OnceLock arenas + AtomicI32 rows; annotations folded from
component flags at intern time, so masks are functions of state VALUES and
demand-order ids never cross the FFI). Components arrive as compact blobs
(`factored.kernel_lazy_payload`): dense arenas for eager components, NFA
arenas with eps-CLOSED per-class edge lists for capped ones — closure
distributes over union, so the kernel unions bitsets and never eps-walks;
no regex/NFA machinery was added to the crate. Intern cap 262,144 states;
breach = ValueError at the walk FFI -> `_walk_py` fallback (masks exact
either way); dense walks keep no sanctioned failure mode.

Step-1 bound (the bet, measured before Rust work; full valid instances,
full-vocab walks, Python lazy leg): touched product states 71-7,265 and
lazy-component subsets 26-2,140 across the nine substring-union schemas —
the predicted 10^2-10^4 vs 268,803 eager subset states for o83132's single
component; cap headroom ~36x.

Gates (all local macOS, jobs 1, worktree venv wheel):

- Mask parity (the outcome gate): per-token full-vocab mask digests
  kernel-ON vs spec-OFF identical across all nine family schemas + the two
  no-instance members' initial masks — 3,095 instance steps, zero
  divergence; eager third leg (component budget 0) identical on the two
  schemas whose uncapped build terminates locally (o83133 15.4s, o33033).
- Parity suite: forced-all-lazy legs (toy / wide W=2 / sql+lexicons),
  opposite-order interning and rayon-pool id-independence, intern-cap
  fallback (lazy degrades to spec, dense re-raises), payload-shape pins;
  full suite 809+ passed / 0 failed on the v8 wheel, flag on and off.
- Recorded sets: by construction — the flag's only consumer is walk
  dispatch (grep-audited); `_apply_scanner_budget` reads pattern text at
  schema-compile time; GrammarInvalid text and reserve shortest-lexemes
  equality stay pinned by the existing P3 differential suite.
- Family AB (p3_family, --first-mask, interleaved ON/OFF, jobs 1):

  | metric (12 completed; o83677/io-package declare LALR conflicts both legs) | OFF | ON |
  |---|---|---|
  | 64-token cold-prefix worst token p50 / p90 / max | 276 / 313.5 / 313.5ms | 9.2 / 15.9 / 15.9ms |
  | pooled prefix_masks phase | 64.6s | 2.6s |
  | TTFM compile-only p50 / p90 | 5.357 / 12.333s | 5.337 / 12.391s |
  | TTFM first-mask-included p50 / p90 | 5.358 / 12.336s | 5.343 / 12.588s |

  Dense control: Stream (16,229-state dense DFA) worst token 8.4ms OFF /
  8.6ms ON — the flag is unobservable off the lazy path. DataContract (lazy
  product over all-eager components) 255 -> 9.8ms: the all-dense-component
  lazy product is covered, not just the NFA-component case. RSS unchanged
  (family max ~1.0GB OFF -> 843MB ON, tokenizer/trie-dominated).
- p50 gate (the one RUST_SCANNER failed; stratified-29 fast set,
  interleaved, jobs 1): compile-only p50 8.85ms ON vs 9.07ms OFF,
  first-mask-included 10.34 vs 10.67ms; per-schema ON/OFF ratio p50 1.003
  (compile-only) / 1.001 (first-incl), spread 0.78-1.40 both directions —
  session noise on a sub-10ms set, zero timeouts/crashes on either leg.
  Structurally expected: dense schemas execute one extra getattr.
- MaskBench protocol (maskbench_grid.py, family 14, ON/OFF arms,
  outcomes.py --strict): 14/14 unchanged (12 ok incl. every test verdict
  and token count, 2 compile_error:LALRConflictError both legs); zero
  validation/invalidation errors both legs. Pooled TBM over 10,965 masks:
  p50 196us ON vs 215 OFF (warm path untouched), p90 7.8ms vs 233ms,
  p99 8.6ms vs 257ms (~30x — the plan's "bounded ~8ms cold-miss parity",
  llguidance's lazy-DFA p99 6.7ms being the cross-engine reference regime),
  max 185ms vs 373ms (the CD-re-check residue below). Stream (dense
  control) p50 31us both arms.

RUST_SCANNER disposition (held worktree wf_6c21bc07-902-4): SUBSUMED and
retired. v8 ships blobs once at walker construction and never rehydrates
arenas into Python, so the +18-22ms per-build FFI floor that failed gate
(d) cannot recur; the eager `build_scanner_arena` entry point is not
ported, and nothing was harvested — the payload pre-computes eps-folded
NFA artifacts Python-side, leaving no work for in-kernel CharSet/NFA/
eps_star modules. scanner_build.rs remains in its held worktree as
reference. Size-gated dispatch, the recorded fallback, is unneeded.

Store law (the #1-vs-#8 conflict, resolved): the artifact store never
persists product-interner state — `load_or_build_scanner` skips the put
for lazy facades (instance-local ids, unpicklable locks; post-P3 lazy
builds are seconds). Deterministic component artifacts (eager TerminalDFA
arenas + component NFA arenas) are the recorded S3 follow-on payload.

Residue, recorded: full-instance worst tokens reach ~185/221ms ON
(BatchJob/DataConnector, vs 297/416ms OFF) at steps whose cost is
Python-side CD re-checks + registration — RustVerdicts stays non-lazy this
phase and is the declared phase-2 item; one-time payload serialization on
first walk costs ~40-200ms on schemas carrying 8-16k-state eager
components (tuple-table -> np conversion; revisit only if it surfaces
outside the family). Full-corpus (11.3k) republish with the flag default
is owed on the next GPU-box session, as with every wave flip.
