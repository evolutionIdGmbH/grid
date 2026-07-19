# DESIGN-JSON-COVERAGE — the 0.2.x correctness epoch

**Goal:** zero error across all JSONSchemaBench/MaskBench metrics — compile errors,
validation errors, invalidation errors, timeouts — on the **full** schema set
(~9.5k schemas; guidance-ai/jsonschemabench), not the 15/split sample.

**Non-goal (explicit):** speed. Every 0.2.x release records TTFM/TBM but does not
optimize them; performance work is the 0.3.x epoch. The Rust kernel stays frozen at
**kernel v7** throughout 0.2.x so that any timing movement is attributable to coverage
mechanics alone. Versioned bench reports keep the two dimensions separable in all docs.

**Companions:** `bench/json_schema_to_grid.py` (the v1 bridge this epoch replaces),
`bench/RESULTS-maskbench.md` (baseline numbers), `DESIGN.md` (engine),
`GUARDRAIL-REDESIGN.md` (design rationale; §4.6 post-parse layer).

---

## 1. Baseline (GRID 0.0.7, sample: 15/split × 21 splits = 315 schemas, seed 0)

| metric | GRID | llguidance 1.7.6 | XGrammar 0.2.3 (compliant) |
|---|---:|---:|---:|
| passing | 206 | 251 | 283 |
| compile error (declared) | 79 | 62 | 0 |
| validation error (silent: valid rejected) | 0 | 3 | 27 |
| invalidation error (silent: invalid accepted) | 68 | 0 | 37 |
| timeout | 0 | 0 | 0 |

Failure causes (from `RESULTS-maskbench.md`):

- **Invalidation hinges** (ignored-but-accepted, per schema): minimum 33, maximum 24,
  pattern 21, oneOf-exclusivity 18, minLength 13, maxLength 13, format 12, minItems 9,
  uniqueItems 9, minProperties 6, maxItems 5, multipleOf 1.
- **Compile-error reasons**: allOf 30, patternProperties 14, LALRConflictError 5,
  not 4, additionalItems 3, $ref-with-siblings 3, dependencies 3, if/then/else 3,
  oneOf+additionalProperties siblings 3, anyOf+additionalProperties siblings 2,
  remainder = size caps (`MAX_PROPERTIES` 120 / `MAX_NAMED_TERMINALS` 300 /
  `MAX_RULES` 3000) and long tail.

No engine measures zero on every class: llguidance declares 62 and over-rejects 3;
XGrammar compiles everything and leaks 64 silently. **Zero-across-all-metrics is
unclaimed territory.**

## 2. Architecture: why zero is reachable

Three enforcement mechanisms plus a universal fallback. The design rule: **no schema
is ever refused, and nothing accepted is ever unenforced.**

- **M1 — grammar-native enforcement.** Everything CFG/DFA-expressible compiles into
  the grammar: structure, enums/consts, local `$ref` recursion, bounded repetition
  (min/maxItems, min/maxProperties), tuple arrays (prefixItems/additionalItems),
  string constraints compiled into the lexer DFA (pattern/length/format via
  escape-aware product construction), numeric bounds as digit-range DFAs.
- **M2 — schema normalization.** A pre-compilation rewrite pass: `allOf` merge
  (object/constraint conjunction — the llguidance approach), `$ref`-with-siblings
  merge, `if/then/else` → discriminator-guarded alternation when `if` is
  const/enum/required-shaped, `dependencies`/`dependentRequired`/`dependentSchemas`
  expansion, `oneOf` → `anyOf` where branches are **provably disjoint**.
- **M3 — semantic mask refinement.** The non-CFG residue (uniqueItems, contains
  counting, general multipleOf, residual oneOf-exclusivity, residual `not`,
  unevaluated\*) is enforced *in-mask* by a validator hook consulted only at
  **boundary tokens** (string close, number end, item/object close): a candidate
  token that would *complete* a constraint-violating unit is vetoed from the mask;
  tokens that keep the unit extensible pass. This is the SemanticChecker DNA moved
  from post-parse to decode time, satisfying MaskBench's rejected-mid-stream
  semantics. Every veto is recorded in the audit chain (a capability no other engine
  has — refinement decisions become replayable).
- **M0 — universal fallback.** If M1+M2 compilation fails for any reason (size caps,
  LALR conflict, exotic composition), fall back to the generic JSON value grammar
  (always LALR-clean, always compiles) with the *entire* schema enforced through M3.
  Slower, never wrong. **This is what drives compile errors to zero by construction** —
  the caps and conflicts become performance events, not coverage events.

Validation errors stay zero by policy: a constraint is enforced only when its
implementation is verified against a reference validator (§5); anything not yet
verified stays in M3-off/strict-declared mode rather than guessing (over-strict
`format` regexes are exactly how llguidance picked up its 3 validation errors).

## 3. Version ladder

Package versions (pyproject `grid-guardrail`); kernel lineage frozen at v7. Each
release ships `bench/RESULTS-jsonschemabench-v0.2.N.md` — full-bench error metrics
(headline) + recorded-not-optimized TTFM/TBM (footnote), engine versions pinned.

| ver | scope | expected metric movement | gate to ship |
|---|---|---|---|
| **0.2.0** | Infra + honesty: full-bench runner (all ~9.5k schemas, not sample); per-keyword error attribution; **strict mode** (ignored→declared, XGrammar-style default/compliant mode pair); CHANGELOG.md; external-`$ref` audit of full bench | invalidation 0 in strict mode (compile errors rise — disclosed) | full run completes; attribution table per split |
| **0.2.1** | M1 strings: pattern + min/maxLength + format into lexer DFA (escape-aware; format regex set reference-validated) | −pattern(21) −length(26) −format(12) hinges | JSON-Schema-Test-Suite string sections green; validation errors stay 0 |
| **0.2.2** | M1 numerics: min/max/exclusive digit-range DFAs (int exact; decimal incl. exponent forms); multipleOf decimal-places special case | −minimum(33) −maximum(24) hinges | test-suite numeric sections green |
| **0.2.3** | M1 counting: min/maxItems, min/maxProperties as bounded repetition (incl. optional-property interplay that caps llguidance at 90%) | −minItems(9) −maxItems(5) −minProperties(6) | test-suite array/object sections green |
| **0.2.4** | M2 normalization: allOf merge, $ref-siblings, dependencies expansion, provable-disjoint oneOf→anyOf | −allOf(30) −$ref-sib(3) −dependencies(3); compile errors ≈ halved | **llguidance-parity checkpoint**: passing ≥ their 251-equivalent on full bench → jsonschemabench upstream PR goes out here |
| **0.2.5** | M1/M2 dynamic keys: patternProperties (disjoint + overlap-priority), additionalProperties-as-schema alongside properties, propertyNames | −patternProperties(14) −AP-sibling(5) | no new LALRConflictError on full bench |
| **0.2.6** | M2: prefixItems/additionalItems; if/then/else discriminator rewrite | −additionalItems(3) −if/then/else(3) | test-suite conditional sections (discriminator subset) green |
| **0.2.7** | **M3 semantic refinement layer** + audit integration: uniqueItems, contains/min/maxContains, general multipleOf, residual oneOf-exclusivity, small-scope `not` (finite/regular complement) | −uniqueItems(9) −oneOf-excl(18) −not(4) −multipleOf(1) | mid-stream rejection verified against MaskBench runner semantics; audit replay reproduces vetoes bit-for-bit |
| **0.2.8** | **M0 universal fallback**: generic-JSON grammar + full-M3 path for anything M1/M2 can't compile (size caps, LALR conflicts, unevaluated\*, exotic compositions); caps become fallback triggers, not errors | **compile errors → 0** by construction | every full-bench schema compiles via M1/M2 or M0; timeouts stay 0 |
| **0.2.9** | Zero-error hardening: fix every residual full-bench discrepancy; LALR conflict factoring for generated grammars (shrinks M0 usage — correctness-neutral, perf-relevant, still recorded-only) | **all four metrics = 0 on the full bench** | 3 consecutive full runs at zero; test-suite (draft-07 + 2020-12 applicable sections) green; differential fuzz clean |

Estimated effort: ~2.5–4 months solo alongside the campaign. The upstream PR
(IMPACT plan, Workstream B) does **not** wait for zero — it goes at 0.2.4 parity;
0.2.9 is its own announcement moment.

## 4. Runner-semantics fidelity

Zero is only meaningful if measured exactly the way MaskBench measures: valid
instances fully accepted; invalid instances rejected **mid-stream** (some token
masked before completion). M3 must therefore veto at the earliest completing token,
not at end-of-sequence validation. `maskbench_grid.py`'s verbatim-semantics
reproduction is the harness; any ambiguity gets resolved by reading their runner,
not by convenient interpretation — discrepancies in our favor are bugs.

## 5. Verification (how validation errors stay zero while coverage grows)

1. **Official JSON-Schema-Test-Suite** (draft-07 + 2020-12, applicable sections) in CI
   from 0.2.1 — each M1/M2 feature lands with its suite section green.
2. **Differential fuzzing** against a reference validator (`jsonschema`): sample
   instances by walking the compiled grammar (valid side) and by mutating valid
   instances (invalid side); GRID's accept/reject must agree with the reference on
   every sampled instance. Disagreement = release blocker.
3. **Full-bench nightly** with per-keyword attribution — regressions caught per
   feature, per split.
4. **Audit invariant**: every M3 veto replayable; `g10_replay`-class tests extended
   to refinement decisions.

## 6. Performance policy for 0.2.x

- Record TTFM/TBM (full distribution) in every versioned report; never optimize
  during this epoch. Timing tables are labeled *"recorded under the correctness
  epoch — perf work deferred to 0.3.x"*.
- Movement expectations to disclose, not hide: lexer-DFA products (0.2.1–0.2.2) grow
  automata (TTFM up); M3 boundary checks add per-boundary cost (TBM tail); M0
  fallback schemas run semantic-heavy (slowest class — report their count and share
  per release; shrinking M0 usage is 0.3.x's roadmap).
- Hard constraint carried through the epoch: **timeouts stay 0** — M3 stays
  boundary-local (no unbounded lookahead), M0 stays linear in instance length.
- 0.3.x (perf epoch, later): migrate hot M3 constraints into kernels/grammar, TTFM
  kernelization, cold-walk tail — measured against the 0.2.9 zero-error baseline,
  which becomes the correctness regression floor no perf change may break.

## 7. Risks

| risk | mitigation |
|---|---|
| Escape-aware pattern∩JSON-string DFA products blow up | size-budget the product; over-budget → M0 fallback (correct, slower) |
| Wrong format regexes flip errors to the validation column | reference-validated regex set only; unverified formats stay declared until verified |
| LALR conflicts in generated dynamic-key grammars | disjointness restriction + lexer priority + factoring; residue → M0 |
| External `$ref` in the full bench | 0.2.0 audit; offline bundling at bench-prep if present; match the runner's convention for unresolvables |
| oneOf exclusivity semantics debate | provable-disjoint conversion (M2) + completion-time veto (M3); document the exact semantics enforced |
| Full-bench runtime in CI | coverage runs are CPU/tokenizer-only; nightly full run, per-PR sampled run |

## 8. Implementation status (first coverage sprint, July 19 2026)

Shipped in `bench/` (kernel v7 untouched): `jsonschema_normalize.py` (M2:
allOf keyword-merge with $defs preservation, shallow-$ref sibling merge,
anyOf/oneOf sibling distribution incl. type, dependencies→forbid-key variants,
if/then/else via negate(), not→markers for required/type/enum/pattern/
minItems/anyOf/allOf shapes, draft-04 exclusive booleans, bottom-up recursion,
depth-guarded merge algebra), `jsonschema_rx.py` (M1: ECMA-subset patterns →
escape/UTF-8-aware serialized-byte regexes, int digit-range construction,
number bounds over canonical float forms, length windows, format table
(9 formats), NOT-literals, pattern complements for prefix/fixed/window/
head+star+end shapes, class-window-minus-literals, compact byte-ANY), and the
compiler rewrite (constrained terminals with dedupe + hash-consed rules,
**order-free object machine** over required-subsets (2^R, R≤10; ordered
fallback recorded beyond), tuple arrays, counted sequences, enum sibling
filtering + numeric float-twins, patternProperties incl. declared-key
subtraction and extras-complement, propertyNames, forbid-keys extras,
oneOf-disjointness prover, strict mode, scanner-budget degradation,
false-schema never-grammar).

Sample (15/split, 315 schemas, seed 0), all error classes:

| stage | passing | compile | validation | invalidation | timeout |
|---|---:|---:|---:|---:|---:|
| baseline (0.0.7) | 206 | 79 | 0 | 30 | 0 |
| + M1/M2 core     | 253 | 46 | 1 | 9 | 6 |
| + structural/order-free | 260 | 47 | 0 | 8 | 0 |
| **v0.2.0** (residual sweep) | **268** | **40** | **0** | **7** | **0** |
| **v0.2.1** (dialect {m,n}, windows <=64 enforced) | 268 | 41 | 0 | **6** | 0 |
| **v0.2.2** (full-set validation-error hunt: 5 root bugs) | 270 | 39 | 0 | 6 | 0 |

Full set (11,306): v0.2.0 measured passing 9,551 (84.5%) / compile 1,082 /
validation 31 / invalidation 629 / timeout 13 — all BFCL splits 100%.
v0.2.2 fixes: composite-enum max-munch capture, untyped type-sniffing,
token-level anyOf branch capture (const harmonization), legacy draft $ref
semantics, ordered-fallback false-rejects, emptied-enum never-grammar,
reachability pruning.

Reference points, same sample: llguidance 251/62/3/0, XGrammar 283/0/27/37.
GRID 0.2.0 passes the most schemas among honest-declaring engines with zero
false-rejects; full table in `bench/RESULTS-jsonschemabench-v0.2.0.md`.

Remaining invalidation hinges: big length windows (needs dialect `{m,n}` —
engine-adjacent, scheduled), uniqueItems/contains (M3), oneOf-exclusivity
residue, multipleOf. Remaining compile errors: adversarial Handwritten shapes
(not-existentials, allOf-not webs), pp+propertyNames combos, one $ref web.

## 9. Doc & versioning policy

- **0.2.x = fewer errors only; 0.3.x = speed.** No release mixes the two intents.
- Every release: CHANGELOG.md entry, `bench/RESULTS-jsonschemabench-v0.2.N.md`,
  LESSONS.md entry (what changed, why, measured result, next — repo convention),
  README coverage table updated **with the version pinned next to every number**.
- Cross-engine comparisons always pin all engine versions (existing convention) and
  credit guidance-ai/jsonschemabench as the benchmark's source.
