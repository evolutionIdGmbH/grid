# GRID — Lessons Learned

What changed from the initial design, why we changed it, what the result was, and
what we did next. Ordered by project phase; each entry follows
**Change → Why → Result → Next**.

Lineage, for orientation: **v0.0.5** (authored Aug 2023)
captured an early, partial snapshot of this design. Planned,
iterative engineering carried it forward: **v0.0.6** already outperformed
guidance (its July-2023 release) on the scaling benchmarks of that era;
**v0.0.7** — this repository — is the line the final patent application
followed; and the work has kept improving since the patent. The design drew
inspiration from published work (notably the Outlines paper); the design and
implementation are our own throughout.

---

## Phase 1 — Evolving the v0.0.5 design snapshot (authored Aug 2023)

The v0.0.5 snapshot — an early, partial capture of the design: precompute an
in-memory table tagging LLM **token sequences** as accepted/rejected against a
pushdown automaton, then mask logits per step from table lookups. The design
review planned on our roadmap examined it through six lenses (complexity,
scalability, formal soundness, domain coverage, state of the art, serving),
stress-testing each conclusion before adopting it, and produced four
load-bearing changes.

### 1.1 Sequence-keyed table → configuration-keyed masks
- **Change:** key validity information on parser **configurations** (lexer state ×
  LALR stack), not token sequences.
- **Why:** a full sequence-keyed table scales combinatorially — |T|^m entries;
  at a 32k vocabulary, length 3 is already ~3.3×10¹³ (~400 TB), length 18 exceeds
  the atoms in the observable universe. The "representative subset" variant was
  defined by outcome, not construction, so it offered no build recipe.
- **Result:** grammar-sized key spaces (10⁴–10⁶) with identical guarantees; the
  viable-prefix property of LR automata (Knuth 1965) carries exactly the
  information the table was trying to enumerate.
- **Next:** the write-back mask cache kept the *spirit* of the original table
  (precomputed-where-cached, compute-on-miss) as a differentiator vs
  compile-time-only caches.

### 1.2 Complete-string acceptance → viable-prefix oracle
- **Change:** tag semantics moved from "the PDA accepts this sequence" to "this
  sequence is a viable prefix"; the incrementally-advanced LALR parser *is* the
  oracle (correct-prefix property).
- **Why:** under acceptance semantics every intermediate generation state is
  "rejected" — the mask is empty at step 1 and generation deadlocks. The early
  snapshot had described prefix-viability in prose without yet
  formalizing it; this revision settled the tag semantics on prefix-viability
  throughout.
- **Result:** masking works at every step; EOS gating became a separate,
  explicit membership check (ACCEPT reachable via the reduce chain of $end).
- **Next:** the mid-lexeme EOS rule (1.4) refined this further.

### 1.3 Dropped: whole-sequence rejection sampling and 1-token backtracking
- **Change:** per-step hard masking (logits = −∞) is the only enforcement mode;
  the original's regenerate-on-reject and single-token-backtrack variants were
  cut.
- **Why:** rejection sampling needs ~169 expected full regenerations for a
  100-token statement at 95% per-token validity (and loops forever under greedy
  decoding); one-token backtracking cannot escape dead ends committed earlier.
  Soft down-weighting ("preferably lowering probability") voids the guarantee
  outright.
- **Result:** single-pass guaranteed-valid output; the multi-token pruning idea
  survived as jump-forward `Write` spans.
- **Next:** distribution-faithfulness (masking ≠ P(x | x ∈ L)) recorded as a
  known trade-off, deferred post-v1 with EX-delta measurement in the bench plan.

### 1.4 The token↔terminal bridge became a first-class component
- **Change:** added the byte-level trie walk with maximal-munch lexer state — the
  early snapshot had illustrated the subword-misalignment problem (the
  "sel"/"ect" example) and left the mechanism as later work; this revision
  built it.
- **Why:** LLM tokens don't align with grammar lexemes; schema identifiers are
  never single tokens; masking out prefix fragments destroys completeness.
- **Result:** exact masks over real vocabularies (verified byte-complete on
  GPT-2 and Qwen tokenizers).
- **Next:** this component became the performance centerpiece — see 4.1/4.2.

### 1.5 Honest scope boundaries replaced blanket guarantees
- **Change:** the guarantee statement narrowed to *syntax + verb/table-level
  RBAC*; column-to-table binding moved to a post-parse `SemanticChecker`;
  complexity restated as **amortized** O(1)/token (per-step worst case bounded
  by nesting depth — SQL's prefix operators are inherently right-recursive).
- **Why:** column RBAC is provably not left-to-right CFG-enforceable (SELECT
  list precedes FROM; alias binding is context-sensitive). "Grammar-constant
  per step" is false under reduce cascades. The early draft's blanket wording
  ("does not break target system functionality") was narrowed to what is
  provable.
- **Result:** every stated guarantee in DESIGN.md §4 has explicit preconditions
  and passes its gate; `SemanticChecker` catches exactly the cross-table cases
  the mask provably admits (tested).
- **Next:** the standard-benchmark phase should measure execution accuracy, not
  only syntactic validity, to keep this boundary visible.

## Phase 2 — Design-document review (before any code)

The planned four-lens design review of DESIGN.md (interface fidelity against the
pinned protocol sources, state-machine completeness, implementability, gate
adequacy) produced 47 findings. The ones that changed the architecture:

### 2.1 Reserve exhaustion: bare EOS → jump-complete
- **Change:** on budget exhaustion, emit `Write(shortest_completion + EOS)`,
  never a bare EOS.
- **Why:** the draft returned `Generate([eos])` at the reserve trigger — EOS is
  grammatically illegal away from ACCEPT, so the design's own G5 gate would have
  failed against its own pseudocode, and the runtime would have crashed on the
  first budget-tight generation.
- **Result:** budget-bound generations end with grammatical statements
  (observed throughout mini-G5 and the stress walks: `... or ( id = 0 ) ;`).
- **Next:** trigger threshold got a safety slack (completion length can grow by
  one token's inflation), recorded as `RESERVE_SAFETY`.

### 2.2 The interface convention was deeper than its surface
- **Change:** the review traced our internal protocol convention to its actual
  definition site (the instruction/guide shapes live one dependency deeper than
  the surface package) and documented the de facto tensor-typed `tokens`
  contract that annotations alone did not capture.
- **Why:** conformance tests written against the wrong source would have been
  unimplementable, and list-typed instructions crash the processor path.
- **Result:** `grid/protocols.py` became the normative definition; a dedicated
  test asserts every guide instruction satisfies the tensor contract.
- **Next:** the conformance tests were later made fully self-contained (expected
  signatures stated in the tests themselves), removing all third-party source
  fixtures from the repo — GRID's implementation is original throughout, and the
  repo now carries no external constrained-decoding code outside `bench/`
  comparisons.

### 2.3 Jump-forward needed a home: the GRID-owned decode loop
- **Change:** two execution modes — a GRID-owned step loop (appends `Write`
  spans without forward passes, registers intermediate states, audits per token)
  and a processor-only mode where `Write` degrades to a singleton mask.
- **Why:** a logits processor can only mask, never append; the pinned adapter
  delegates the loop to the model provider. Worse, the pinned processor unions
  a whole `Write` span into one step's mask — which admits out-of-order span
  tokens and breaks soundness; we documented it as a must-not-replicate.
- **Result:** jump-forward works with per-token audit records in mode 1; mode 2
  stays sound with no model-call savings.
- **Next:** vLLM backend lands in mode 2 first.

### 2.4 FORCED and RESERVE_EXHAUSTED are not states
- **Change:** GridState statuses reduced to {ACTIVE, ACCEPTING, GRAMMAR_END,
  COMPLETE}, all O(1)-derivable; "forced" is an instruction-level outcome;
  reserve exhaustion is a session-level trigger.
- **Why:** FORCED requires mask cardinality (a trie walk) — deriving it in
  `get_next_state` would double the work or desync; budget is not grammar state.
- **Result:** status is a pure function of (stack, lexer, eos_consumed); an
  entire desync bug class removed; statechart tests generate from YAML.
- **Next:** none — closed.

### 2.5 Multi-terminal tokens forced the CI/CD split *by construction*
- **Change:** the cache stores only context-independent verdicts; tokens whose
  bytes cross a terminal boundary and continue are checked per step against the
  live stack, never cached.
- **Why:** a boundary-crossing token's viability depends on the post-shift
  allowed set — i.e., on the parser stack — which no (lexer, A)-keyed cache
  entry can capture. Without the split, the cache-key soundness obligation is
  violated *by construction*, not merely unverified.
- **Result:** cache-on ≡ cache-off holds in G4 differentials; the residue check
  became the main hot-path cost (see 4.2).
- **Next:** per-depth residue-size telemetry is a first-class bench output.

### 2.6 Reserve is denominated in model tokens, keyed by tokenizer
- **Change:** completion costs count vocabulary tokens (one identifier terminal
  can cost many), and the ReserveTable is a separate artifact keyed
  (grammar, tokenizer).
- **Why:** a terminal-denominated reserve under-reserves → silent truncation,
  violating the termination guarantee; token costs are tokenizer-specific but
  grammar identity must stay tokenizer-independent.
- **Result:** G5 asserts no reserve-stopped generation exceeds max_tokens; holds
  across mock, GPT-2, and Qwen tokenizers.
- **Next:** none — closed.

## Phase 3 — What implementation taught us (bugs the design didn't predict)

The dual-implementation strategy (a slow, obviously-correct reference guide as
the executable spec; differential tests binding the fast path to it) caught
every one of these before any debugging session:

### 3.1 Byte-identical token aliases (completeness bug)
- **Change:** the trie carries an alias table; masks expand to *all* token ids
  sharing a byte spelling.
- **Why:** vocabularies contain distinct ids with identical bytes; a
  one-id-per-node trie silently dropped the duplicates from masks. The very
  first differential run caught it (`fast-only=[] oracle-only=[261, 262, ...]`).
- **Result:** masks are complete over ids, not spellings.
- **Next:** none — closed.

### 3.2 Lexicon-blind reserve completions (soundness bug)
- **Change:** identifier terminals render completions from the *allowed*
  identifier set, never the BFS-shortest lexeme.
- **Why:** the reserve rendered `COLUMN_NAME` as `_` (shortest regex match) —
  which the identifier composition rule correctly rejects, crashing the
  jump-complete path. The design said "shortest allowed identifier"; the code
  skipped it; the e2e run caught it within minutes.
- **Result:** completions pass their own masks by construction; regression test
  added.
- **Next:** none — closed.

### 3.3 Same-regex terminals forced the contextual lexer
- **Change:** emission events carry candidate **sets**; the parser-viability
  choice picks the terminal (keyword-vs-identifier, TABLE_NAME-vs-COLUMN_NAME).
- **Why:** TABLE_NAME and COLUMN_NAME share a regex — terminal identity is
  parser-context-dependent, which a winner-only lexer cannot express. This was a
  mid-implementation design change (documented in the walk/lexer module docs
  with the known maximal-munch caveat).
- **Result:** per-category L3 lexicons work (forbidden tables unreachable even
  spelled byte-by-byte); the G3 differential validates the discipline.
- **Next:** the caveat (a viable terminal accepting strictly shorter than the
  union-longest match is not chosen) stands; no grammar has hit it — the
  differential will surface any that does.

### 3.4 Audit-kind bookkeeping for Write spans
- **Change:** tokens applied to states that never received an instruction
  default to WRITE-kind records (no mask entry id).
- **Why:** intermediate span tokens defaulted to GENERATE-with-no-entry, tripping
  the E14 invariant assertion (entry id iff GENERATE) — the invariant did its
  job and caught the modeling gap.
- **Result:** every step audits, including span interiors and the EOS tail;
  tamper detection 200/200; seed-replay reproducibility holds.
- **Next:** none — closed.

## Phase 4 — Real vocabularies and the benchmark

### 4.1 Scan-from-scratch walk → incremental-state walk
- **Change:** the DFS carries an O(1)-updatable scan state per trie frame
  (dfa state, current segment, last-accept, emission cascade via a pending-byte
  queue) instead of rescanning `remainder + path` per node.
- **Why:** O(depth²) per node was fine on 300-token mock vocabularies and
  unusable on 50k–151k real ones.
- **Result:** identical semantics (G3 differential unchanged); this *is* the
  Rust kernel algorithm, so nothing is throwaway.
- **Next:** port to `grid_core` (M4).

### 4.2 The bottleneck was never the walk — it was the CD residue
- **Change:** two rounds. First, per-step memoization of allowed-sets/shifts
  (189 ms → 10.6 ms hit path). Then grouping CD entries at *publish time* by
  (candidate sequence, lexicon segments, tail live set) so each step evaluates
  ~hundreds of group verdicts instead of ~25k entries (10.6 ms → **290 µs**).
- **Why:** profiling the first real benchmark run showed cache *hits* costing
  189 ms — the walk was cached, but the uncached-by-design residue check
  re-derived `allowed_terminals` per entry.
- **Result:** GRID warm-hit p50 290 µs @ 84% hit rate (GPT-2), 1.0 ms (Qwen
  151k). The measurement that mattered most came from running the real
  benchmark early, not from the unit suite.
- **Next:** the remaining hit-path cost splits across group verdicts and
  parser-simulate calls — both Rust-kernel targets.

### 4.3 Requirement R is demonstrated, not just proven
- **Change:** the harness gained a warm-replay arm: replay the longest walk
  twice, fit latency-vs-position on the warm pass.
- **Why:** raw slopes across mixed arms were artifacts (cold misses concentrate
  at identifier-heavy regions, not at late positions).
- **Result:** slope **+0.076 µs/position** on GPT-2 (first-half p50 284 µs vs
  second-half 279 µs), +2.2 µs/pos on Qwen — per-token cost tracks grammar
  configuration, not sequence position. This is the property the entire
  v0.0.7 design was built around, now measured on real vocabularies.
- **Next:** re-run on the pinned dedicated runner with the G7 acceptance
  criteria once `grid_core` lands.

### 4.4 The competitive landscape shifted under us — as predicted
- **Change:** the benchmark's "Outlines arm" is llguidance, driven directly.
- **Why:** Outlines 1.3 removed its own CFG engine; `outlines.types.CFG`
  delegates to a backend (default llguidance). The Phase-1 state-of-the-art
  review had concluded the field would converge on state-keyed engines — it
  did, to the point that the original comparison target no longer exists as an
  engine.
- **Result:** the real bar is llguidance (p50 11 µs on GPT-2, 22 µs on Qwen);
  XGrammar sits at 39–316 µs p50 — but with p90/p99 of 4.5–18 ms on our
  recursive SQL grammar. The companion's §8.1 risk (SQL's context-dependent
  token set ≫ JSON's) is **engine-universal**, not a GRID weakness — our CD
  split and per-depth telemetry were designed for exactly this.
- **Next:** the user-provided standard benchmark becomes the fixed yardstick;
  language-parity corners between grammar encodings (maximal munch vs explicit
  whitespace; llguidance's stricter tokenization discipline rejected 2 of our
  byte-fallback stress walks on GPT-2) are documented in the harness.

### 4.4a The second kernel: the cost had moved to the FFI boundary
- **Change:** two moves. (1) `RustVerdicts`: the per-step CD-group verdict batch
  and the LALR virtual-stack machinery behind it (simulate/allowed/shift) moved
  into `grid_core`; groups register once per cache entry (content-addressed by
  entry_id) with everything stack-independent precomputed at registration
  (lexeme-passing candidate masks, ignored-fallback picks, remainder-scan tail
  masks) — the per-step call is pure table lookups over an arena of stack nodes,
  memoized exactly like `_StepMemo`. (2) Masks became numpy end-to-end: the
  kernel returns passing ids as an i32 buffer (`np.frombuffer`, zero-copy), ci
  ids are cached per entry as an int32 array, and `_mask_ids` concatenates
  arrays instead of building Python lists.
- **Why:** after the first verdict-kernel drop, per-position timing showed warm
  hits *alternating* 4–34 µs and 225–440 µs. The slow steps were identifier
  positions passing 20–32k CD ids — the verdicts were fast, but materializing
  tens of thousands of Python ints per step across the FFI (plus tuple→list
  copies) dominated. The cosmetic `sorted()` in `Generate` assembly fell out for
  free — every mask consumer is order-free (set semantics / scatter indices),
  which a grep of the tests confirmed before the change.
- **Result:** warm-hit p50 276 µs → **10.9 µs** (gpt2) and 992 µs → **31 µs**
  (Qwen 151k), 86–87% hit rate; GRID's p50 now beats XGrammar on both tokenizers
  and sits within ~2× of llguidance. Warm-replay R slope: −0.006 µs/pos (gpt2).
  The G9 tracked KPI (within 2× XGrammar p50) is exceeded, pending the pinned
  runner. Parity: a dedicated suite binds cd_pass/allowed/eos_ok to the Python
  executable spec (order-exact, not just set-equal), and the existing
  differential-vs-reference tests run through the kernel path unchanged.
- **Next:** the remaining tail is cold-miss walk cost (6–12 ms p50 at 25k-entry
  identifier positions) — amortized by the cache (misses cluster early), gated
  by G7's per-depth telemetry on the pinned runner. Bitmask-native instructions
  (kernel #4, `apply_token_bitmask`) remain for the vLLM backend, where the
  fixed-size bitmask is the natural interface.

### 4.5 The Rust kernel: port the algorithm, chase the profile
- **Change:** `grid_core` (Rust/pyo3, abi3) implements the incremental-state walk
  — a line-for-line transcription of the Python kernel — then two profile-driven
  moves *into* the kernel: CD grouping (one representative per verdict class
  crosses the FFI boundary instead of ~25k entries) and alias expansion.
- **Why:** the first Rust drop barely moved the miss latency (74 -> 60 ms):
  profiling showed the cost had migrated to Python-side marshalling and grouping,
  not the walk. Two kernel iterations later: 74 -> 9.4 -> 6.1 ms (GPT-2),
  131 -> 12.5 ms (Qwen). A strict parity test (group partitions + verdict-relevant
  representative fields) caught one real divergence: the Rust group key included
  event lengths that the spec keys only under lexicons — the executable-spec
  discipline works across languages.
- **Result:** GRID's tail is now competitive on the recursive SQL grammar
  (p99 8.3 ms vs XGrammar's 14.9 ms on GPT-2); warm hits unchanged at ~290 us
  (that path is Python: per-step CD-group verdicts + parser simulation).
- **Next:** move the per-step group verdict + LALR simulate into the kernel
  (targets the 290 us hit path), then the pinned-runner G7/G8/G9 numbers.

## Phase 5 — Standard benchmarks begin: MaskBench

### 5.1 A second grammar domain in a day — and what it flushed out
- **Change:** GRID entered MaskBench (guidance-ai/jsonschemabench) via a
  JSON-Schema→`.grid` compiler (definition-order properties with skippable
  optionals, spec-default `additionalProperties` including typed extras,
  enum/const as exact serialized literals, anyOf/oneOf alternation, recursive
  local $refs) plus a protocol-exact reimplementation of maskbench's runner
  (TTFM/TBM semantics verbatim), with llguidance and XGrammar arms on the same
  llama-3.1 tokenizer and sample.
- **Why:** the design claims "extensible to any LALR(1)-parsable language"; a
  10k-schema public corpus is the cheapest way to find out where that's true.
  The keyword-vs-identifier machinery (LESSONS 3.3) turned out to be exactly
  what JSON needs — property-key terminals and STRING share spellings, and the
  parser-viability pick resolves them with no new mechanism.
- **Result** (315-schema stratified sample, local host): TBM p50 32 µs — the
  kernel hit path holds the median against llguidance/XGrammar's 10 µs — with
  **zero validation errors** on every compiled schema; 79 declared compile
  errors (allOf, patternProperties, if/then/else — the honest llguidance-style
  boundary); 66 invalidation errors, all traceable to deliberately ignored
  value constraints (pattern/min/max — the XGrammar-default convention).
  The corpus also flushed out a real latent core bug: the LALR conflict
  reporter's eager format dict indexed `prod_names[state_id]` for SHIFT
  actions — IndexError instead of LALRConflictError; first grammar in the
  project's life to hit a reportable conflict with a large state id.
- **Next:** the measured tail assigns the next kernel work: (a) schemas past
  the 64-terminal bound run the pure-Python walk (6% of compiled — widen the
  kernel's terminal masks past u64); (b) TTFM is the pure-Python LALR+scanner
  build (28 ms p50 vs llguidance 0.3 ms — table construction into grid_core);
  (c) MaskBench's one-shot-per-schema protocol never warms the write-back
  cache — the serving bench (G8) is where that design choice pays.

### 5.2 Spider EX: the first real-data soundness find — a theorem's unvalidated precondition
- **Change:** the Spider dialect grammar (100% dev-gold coverage,
  `bench/spider_coverage.py` as its committed oracle) plus the EX harness
  (`bench/spider_ex.py`: GRID-constrained vs unconstrained arms, per-database
  L3 lexicons, sqlite execution, G9 ablation flags). One generation in the
  first 100-question run died with `DeadEndError: empty mask (bug by
  theorem)` — the error class the architecture promises cannot happen.
- **Why:** the orchestra database has a column named
  `Official_ratings_(millions)`. Parentheses are outside COLUMN_NAME's regex
  language, so the L3 allow-list contained a word the scanner DFA can never
  accept: every prefix of it passes `prefix_ok` (it IS a lexicon-word prefix),
  but no token can ever complete the lexeme. The completeness proof is
  conditional on lexicon ⊆ terminal language — the fixtures all satisfied it,
  so no gate ever checked it; the first real-world schema violated it within
  100 generations.
- **Result:** the precondition is now VALIDATED, not assumed:
  `MaskProducer._validate_lexicons` raises `GrammarInvalid` at guide build for
  any allow-list word the DFA cannot scan to an accepting state of its
  terminal (regression-tested in `tests/mask/test_lexicon_language.py`); the
  Spider harness filters schema names to the identifier language, which is
  sound — names outside the grammar's language were never generatable anyway.
  The failing generation now completes grammatically.
- **Next:** audit other theorem preconditions for the same pattern (assumed on
  fixtures, unvalidated on input): tokenizer byte-completeness is already
  asserted (`greedy_tokenize`); reserve-table assumptions on lexicons are
  covered by the same validation; grammar reducedness is checked at load.

### 5.3 Spider at scale: the EX delta is a function of model size — and that's the honest pitch
- **Change:** the EX harness ran the full 1034-question dev set with Qwen2.5-7B
  on the H100 runner, after the 0.5B gate run reproduced the local preview.
- **Why:** G9 requires EX-delta vs unconstrained on the reference-class model,
  not just the bring-up model.
- **Result:** at 0.5B, constraint is worth **+13 EX points** (29% vs 16%) and
  +26 syntax points; at 7B the deltas nearly vanish (EX **53.7% vs 52.9%**,
  syntax 91.3% vs 91.0%) — a capable model rarely commits the syntax-error
  class the mask eliminates. The honest product claim at scale is therefore the
  *floor and the guarantees*: 100% parse-by-construction, schema-valid
  identifiers, RBAC projection, replayable audit, grammatical budget stops
  (0.9% truncation) — not raw accuracy. Ablations (EX-invariant by
  construction) put numbers on two design decisions: **cache-off costs 32%
  generation throughput** (2.5→1.7 tok/s — the write-back cache's serving
  value), audit-off and jump-forward-off move tok/s within noise at n=20.
- **Next:** the G9 report should always show both scales side by side; an
  eventual `SemanticChecker`-guided check-and-regenerate arm targets the
  residual alias↔column failures that EXPLAIN exposes at both scales.

### 5.4 vLLM mode 2: the async scheduler hands you the past one step late
- **Change:** the M6 first slice — `GridVLLMLogitsProcessor` (vLLM V1 plugin)
  over a vllm-free `GridRequestTracker` core, accepted by a GPU smoke on the
  runner (4/4 viable prefixes, ≥1 complete statement, zero desyncs).
- **Why:** written against the introspected 0.24 interface; the smoke
  immediately caught two integration realities the docs don't advertise:
  (a) the live `output_tok_ids` lists contain **-1 placeholders** for
  not-yet-sampled slots, and (b) under **async scheduling** the previous
  token is *still* -1 when `update_state` runs (the CPU prepares step k+1
  during GPU step k) — any sequence-stateful mask is one step stale, which
  surfaced as a mid-identifier desync precisely where the legal set tightens.
- **Result:** the tracker pauses at placeholders and resumes when they are
  overwritten (unit-tested); mode 2 requires `async_scheduling=False`,
  documented as part of the integration contract. Mode-2 semantics held
  exactly as designed elsewhere: Write spans degrade to singletons, truncation
  replaces reserve completion (no appends from a processor), soundness intact.
- **Next:** scheduler-integrated masking (the route vLLM's native structured
  output takes) lifts the sync-scheduler requirement and is where kernel #4
  (bitmask instructions) naturally lands; then G8's batch/overlap gates.

### 5.4a Repair-arm interim (0.5B): the checker works, the model can't use it
- **Change/measurement:** the SemanticChecker-guided `grid-repair` arm ran 55
  dev questions at 0.5B before the host slept (greedy, one retry with the
  violations quoted back; 15 retries engaged and were kept by the checker).
- **Result:** metrics identical to plain `grid` (syntax 54.5%, EX 29.1% both) —
  the 0.5B repairs its alias-binding mistakes into different-but-equally-wrong
  queries. A clean negative at this scale, and the inverse of the EX-delta
  finding: weak models need the mask but cannot exploit feedback; capable
  models need less mask but can. The repair-arm value claim is therefore a
  **7B claim** and is measured there (grid's 7B failures are ~pure binding
  errors, exactly what the feedback names).
- **Next:** the 7B `grid,grid-repair,unconstrained` run on the next declared-
  runner session decides whether "guarantees + repair" clears the value bar.

### 5.4b Repair-arm verdict (7B, full dev set): the claim survives at scale
- **Measurement:** `grid` vs `grid-repair`, Qwen2.5-7B, all 1034 dev questions
  on the declared A10 runner (`bench/RESULTS-spider-repair.md`). Plain grid
  reproduced its H100 numbers exactly (91.3% executes, 53.7% EX) — cross-host,
  cross-GPU-generation reproducibility of the whole harness.
- **Result:** one SemanticChecker-guided constrained retry converts a third of
  the residual binding-failure floor: executes 91.3→**94.5%**, EX
  53.7→**55.2%** (**+2.3 over unconstrained**, vs +0.8 for grid alone),
  truncation halved, at +5 tok/query (+14%) with 75 retries kept and a 21%
  newly-correct conversion rate. Deltas were stable from n≈500 onward.
- **The capability symmetry, now measured end to end:** at 0.5B the mask is
  worth +13 EX but feedback is worthless (5.4a's clean negative); at 7B the
  mask alone is worth ~+1 but feedback converts — because grid's failures at
  7B are *precisely named* by the alias-aware checker, while unconstrained
  failures are unrepairable prose. Constraint quality determines feedback
  quality: the checker can only name violations because the mask already
  guaranteed everything else.
- **Next:** a second retry round shows diminishing returns by construction
  (best-by-checker keeps round 1 unless improved) — measure before adding;
  fold `grid-repair` into the standard G9 arm set.

### 5.5 The MaskBench tail had two names, and both were cheap once measured
- **Change:** (a) kernel masks widened from a single u64 to `[u64; W]`,
  W ∈ {1,2,4,8} (512 terminals), monomorphized per width so W=1 compiles back
  to the original scalar ops; (b) the scanner build gained alphabet compression
  (subset construction over byte-equivalence classes instead of 256 raw bytes)
  and per-state eps-closures (closure distributes over union — the per-call
  fixpoint was 67% of the build).
- **Why:** the user's challenge — "p50 is only median; at p90/p99 we're too
  slow" — decomposed into exactly two named costs: 19/315 MaskBench schemas
  past the 64-terminal bound owned the 200 ms step class, and `build_scanner`
  was 81.8% of TTFM.
- **Result** (315-schema re-run): fallback schemas 19 → **0**; TBM p75
  459 → **39 µs**, p99 201.6 → **30.2 ms**, max 315 → 128 ms, validation errors
  still 0; TTFM p50 27.9 → **13.2 ms**, p99 854 → 359 ms; W=1 hot path
  unregressed (9.7 µs warm hit, flat R). Pipeline profile: build_scanner
  576 → 116 ms over 60 schemas (5×).
- **Next:** the remaining TBM p90 (~28 ms) is the cold trie walk itself —
  vocabulary-sized, already in Rust; needs walk-level pruning, not width.
  TTFM's remaining split is now even (~48% scanner, ~52% everything else) —
  further gains want a table-build kernel, priced against demand.

## Phase 6 — Closing the gates: kernel v4, the serving contract, scale runs

### 6.1 The warm hit was three Python calls and a concat — kernel v4 folded them into one
- **Change:** `RustVerdicts` gained a persistent, structurally-interned stack
  arena — nodes deduplicated by `(parent kidx, LALR state)` — with cross-token
  memos for allowed-sets, EOS, shifts, and the whole per-entry CD batch, plus a
  one-call `hit_pass` that assembles `ci ++ cd-pass ++ eos` kernel-side. Python
  addresses nodes by intern index (`StackNode.kidx`/`kgen`); a generation
  counter invalidates all kidx on `reset_interning`, a cache-epoch counter drops
  the `(kidx, remainder)→entry` lookaside on namespace rollover.
- **Why:** profiling the v3 warm hit (11.6 µs) showed the *cost had left the
  compute*: `cd_pass` FFI 8.6 µs (re-simulating the stack every call because the
  arena was rebuilt per call from a fresh chain), plus a Python `concatenate`
  1.6 µs. Parser configurations recur massively across token positions, so
  interning + memoizing turns almost every warm verdict into a dict lookup.
- **Result:** warm-hit p50 **11.6 → 2.9 µs** (bench mean) / **12.9 → 3.5 µs**
  (G7 harness p50) — the `hit p50 < 10 µs` gate met on the M-series dev host for
  the first time (still officially gated on the declared runner). Slope stayed
  ~0 (requirement R holds); a `(handle, kidx)` result memo took the CD batch
  itself off the hot path. Parity: the kidx-addressed APIs and `hit_pass` are
  bit- and order-exact vs the Python spec, and `advance_frames` (the fourth §2
  symbol, `lalr_advance`) matches `shift_terminal` including config_hash.
- **Lesson:** once a hot path is "in Rust," the next win is usually not faster
  Rust — it's *not calling it*. The interned arena is memoization at the FFI
  boundary; the biggest single drop came from never recomputing a configuration
  the engine had already seen.

### 6.2 "In Rust" is not "off the GIL" — and the cold walk needed the latter
- **Change:** the cold trie walk (`walk_raw`) now runs under `py.detach` (GIL
  released); the ms-scale build happens on a worker thread while the scheduler
  thread keeps moving (`grid/serving/prefetch.py`).
- **Why:** a first overlap measurement showed **9%** main-thread liveness during
  a cold replay on a worker — the walk held the GIL the whole time, so "overlap"
  was a lie. Releasing it lifted liveness to **88%**. (A red herring en route:
  passing the kernel's own walk payload straight to `register` instead of
  reconverting frozensets shaved ~30% of the *Python* cold cost but barely moved
  liveness — because the walk, not the glue, held the GIL.)
- **Lesson:** overlap claims must be *measured with a busy other thread*, not
  asserted from "it's a background submit." The GIL turns a threaded prefetch
  into a sequential one silently.

### 6.3 The reserve safety margin was tokenizer-calibrated — and byte-BPE broke it
- **Change:** `RESERVE_SAFETY` 8 → 16.
- **Why:** the G5 10k forced-random-walk arm surfaced exactly one budget
  overrun (seed 910102): a Qwen byte-BPE whitespace blob inflated the
  greedy-tokenized completion by 9+ tokens in a single step, past the slack the
  gpt2-era constant assumed. The gate ("no reserve-stopped generation exceeds
  max_tokens") is the kind of property a 25-seed slice never hits and a
  10k-generation run finds once.
- **Lesson:** safety margins tuned on one tokenizer are silent liabilities on
  another; scale runs are where single-in-ten-thousand budget edges show up.

### 6.4 Gate runs are cheap to author once the properties are unit-tested
- Four gate harnesses landed this phase (`g10_replay`, `g5_scale`,
  `g6_adversarial`, `vllm_serving_bench`), each a scaled version of a property
  already pinned at unit level (audit chain, S/C/T, RBAC mask, metric math). The
  G6(b) arm's **positive controls** (the same adversarial speller must *reach*
  an allowed identifier) were the load-bearing addition — without them "0
  bypasses" could mean "the driver never reached an identifier position," which
  an early greedy version in fact did for table names.
- **Lesson:** a soundness gate that can pass vacuously is worse than no gate;
  every "nothing forbidden happened" needs a paired "something allowed did."

### 6.5 The G8 batched-TPOT explosion was three stacked serving-only defects
- **Symptom:** G8 first real run (H100, Qwen2.5-7B): batch 1 at **+1.26%** TPOT
  overhead — batch 8 at **+3984%**, batch 32 at **+5151%**. Fine solo,
  catastrophic batched; classic "warm path not engaging under load."
- **Diagnosis** (instrumented in-engine probe, not microbenchmarks — the
  microharness numbers were all still true): three independent defects, each
  invisible at batch 1: (1) `GridGuide.fill_bitmask` skipped the kernel warm
  path entirely — `producer.masks()` + a per-call `np.bitwise_or.at` repack of
  10k+-id entries, ~12 ms per request per step; (2) the prefetcher scheduled
  EVERY successor (warm or not) onto ONE worker, so fills queued behind a
  serialized build queue inflated by GIL ping-pong (209 ms/step at batch 8
  with ZERO cache misses); (3) each request-copy built its own Rust kernel and
  re-REGISTERED every entry it touched — literal-interior entries carry tens
  of thousands of token ids, 20-100 ms *per registration*, ~one per fill.
- **Fix, in dependency order:** kernel v5 `fill_bits` (pre-packed per-entry ci
  bit words + CD bits + EOS written into vLLM's row in one FFI call);
  schedule-only-cold prefetch (`GridGuide.is_mask_warm` gate, 4 workers);
  copies share the template's `MaskProducer` (one kernel, one registration
  space, one T1 cache). Warm steady state after: batch 32 fills p50 **3 µs**,
  sum 4 ms across 1056 fills; 16 ms/step vs 708 before.
- **Lesson:** a warm path that exists is not a warm path that is *on* — every
  serving-layer hop (fill, schedule, copy) must be audited for "does this
  re-do per-request what the design amortizes per-template?" Batch-1 benches
  structurally cannot see this class of bug; the in-engine probe with
  per-call component splits found in an hour what the mean-only report
  obscured.

### 6.6 Measure, don't model: the investigation that rewrote two "known" problems
- **Change:** kernel v5.1 (verdict-equivalence CD grouping, shared DFS segment
  buffer, in-kernel packed-row fill memo, bytes-path FFI + vectorized
  `adaptive_encode`) and kernel v6 (session-in-kernel accept+fill, serving-gated,
  audit paths stay v5).
- **Why:** two beliefs from the G8 runs — "152k-vocab walks are superlinear"
  and "the warm gap is Python dispatch" — both failed under a measured,
  independently-verified investigation. The walk was linear per class; the real
  cold cost was CD-group keys embedding raw lexeme/remainder bytes whenever a
  lexicon exists (≈55k singleton groups vs ≈1.4k verdict classes — 86–91% of
  the cost), and the warm cost was dominated by `fill_bits` re-packing ~47k
  CD ids per call, not dispatch. The pre-implementation review of v6 also
  surfaced a real v5 bug (post-COMPLETE resurrection under speculative decode)
  that the differential suite now pins.
- **Result:** cold CD-heavy mask builds 124 → 13.3 ms (9.3×); warm fill p90
  38 → 3.8 µs; warm serving step 7.39 → 1.33 µs/request (accept 11×) — under
  the 6 µs G8 per-request budget with margin — confirmed on the declared H100
  SXM5 runner: TPOT overhead +1.02% @batch 32 (<2% gate PASS; +0.12% @1).
  Zero divergence in 200-seed lockstep fuzzing of v6 vs v5; the pure-Python
  spec path and every parity/entry-id invariant unchanged.
- **Lesson:** name a suspect only after a component-level measurement under the
  real engine; both prior theories were plausible, load-bearing, and wrong.

### 6.7 The cold-schema fix — and the fuzz "failure" that was a soundness catch
- **Change:** the cold-schema-into-hot-batch stack, all levers kill-switched:
  genN key normalization (`("genN", p, q, v, A, schema_fp)` — remainders the
  walk provably cannot distinguish share one entry), a per-dialect
  ContextJournal + admission-time warmup inside `compile_grammar` (the request
  absorbs its own warmup while WAITING), the §6 skip-a-round realized as a
  scheduler mask-readiness defer (patch site 4 in our own patch file + a
  drafted upstream vLLM PR), rayon-parallel walks (bit-identical, 2.05× at 8
  threads locally), and the ratified adversarial **metric v2** (per-request
  TPOT of the warm co-batched requests; the fresh request reported, not gated).
- **Why:** the measured anatomy said 73% of the stall was schema-specific
  ident-boundary walks — unshareable by E11 — so no cache trick alone closes
  the gate; the request had to be warmed at admission or deferred out of the
  round, with exact masks always (the defer only changes WHEN they compute).
- **Result:** during validation the 50-seed shared-registry fuzz "failed" —
  and the root cause was a **latent soundness bug in the original T2 tier**,
  not the new code: unscoped cross-schema generic sharing could serve one
  schema's continuations to another (walk-time CD filtering embeds schema
  words). Isolated-process ground truth matched the NEW side. Generic keys are
  now schema-scoped; the unsound legacy behavior survives only behind
  `GRID_GENN_KEYS=0` for old-log replay. Honest local payoff: with sharing
  soundly scoped, admission warmup recovers ~20% of stall (44% of the ident
  class) in a harness that cannot exercise the defer — the defer is the
  primary co-batch protection and only the H100 W10 matrix can measure it.
- **Lesson:** a differential fuzz that fails against the OLD behavior is not
  automatically a regression — sometimes the fuzz has finally been given a
  correct oracle. Root-cause with an isolated ground truth before "fixing"
  parity, or the fix would have re-introduced the poisoning.

### 6.8 The W10 runs: three real fixes, one exogenous ghost, one conservative pass
- **Change:** admission warmup default-off (`GRID_ADMIT_WARM=0`; opt-in with a
  per-fingerprint gate and `GRID_ADMIT_WARM_THREADS`); the v2 step-loop warm
  pass mirrors the timed passes exactly (full length + a fresh-schema request
  — Triton JIT compiles off the clock); driver GC hygiene in
  `_step_loop_batch` (disable during timed legs, the G7 pattern) and a
  one-shot `gc.freeze()` in the backend (engine-core resident).
- **Why (measured on the H100 matrix):** warmup as shipped starved the live
  engine via GIL-bound tier work — fresh TTFT 10× worse AND multi-second
  batch stalls, so the "WAITING absorbs the cost" premise died on contact
  with the GIL; the step loop JIT-compiled kernels inside timed windows;
  and a once-per-leg 0.7–2 s frozen engine step remained that we exonerated
  five ways — it fires in ALL-WARM baselines with zero grid work, is
  invariant to defer/warmup/walk-threads/JIT-warming and to child- AND
  driver-side GC control, and vanishes entirely with the engine in-process.
  It is a vLLM-0.24 multiprocess-topology artifact that lands randomly
  inside either metric's window (both metrics read +270–370% on the runs it
  poisons, near-clean on the runs it misses).
- **Result:** warm gates PASS solidly (batch-32 TPOT overhead +0.64–0.99%
  across four record runs; TTFT 26–27 ms cold / 1.3 ms warm). In
  artifact-free windows the adversarial pair PASSES with the full stack —
  defer + genN + rayon threads=8: **−1.75% co-batched degradation, 23.9 ms
  max step** — on the legacy lockstep leg, which is biased AGAINST the
  design (it charges the fresh request's tail to the batch), making that a
  conservative pass. Defer attribution (same leg, artifact-free windows):
  +8.2%/24.5 ms with defer on vs +373%/65.7 ms off. Fresh-request UX:
  TTFT ~145 ms, effective TPOT 1.00× warm.
- **Lesson:** when a benchmark number refuses to move under any lever you
  own, stop optimizing and start exonerating — the five-way elimination
  (grid levers, JIT, child GC, driver GC, process topology) cost one hour
  and prevented shipping "fixes" for an artifact that was never ours.
- **Closure (2026-07-10):** the artifact is reported upstream
  (vllm-project/vllm#48229) and the gate now uses artifact-robust
  estimators (owner-ratified): median-over-legs degradation, min-over-legs
  max step, raw per-leg maxima always printed. Clean-window max step reads
  36–38 ms (vs 30 budget) and is defer-cap-INVARIANT (100/250/400 ms
  identical) — the residual is OUR GIL-bound entry materialization
  (make_entry/publish/register in Python) during the fresh request's ~0.9 s
  window, the same mechanism class that killed admission warmup: defer-off
  spreads it (+64% degradation, 65 ms max), defer-on concentrates it
  (+115–125%, 36 ms max). The real fix is kernel-v7 territory (entry
  materialization in Rust); until then the cost is bounded, transient
  (once per never-seen schema), and documented rather than hidden.

### 6.9 Kernel v7: the fused walk→blob→register path, and where the wall really was
- **Change:** the cold mask miss now stays in Rust end to end — `walk_payload`
  returns (ci bytes, an opaque group blob), `register_blob` parses+builds+
  encodes+hashes+registers all inside one GIL-released call; Python gets a
  handle and a thin `MaskEntryV7` shell. Plus walk/pool thread niceness
  (`GRID_WALK_NICE`/`GRID_POOL_NICE`). `GRID_V7` default ON; `GRID_V7=0` is
  byte-identical (digest-fuzz-verified, both key regimes; entry_id byte-equal
  across 486 entries so G10 replay is stable).
- **Why:** the red-team's premise check overturned the sketch — the ~15–25 ms
  GIL hold was NOT `make_entry`/`adaptive_encode`/blake2b (those are µs). It
  was (a) the `WalkResult` glue building per-group `CDEntry` reps in Python
  (5–7 ms/boundary entry) and (b) the ~30–60k gc-tracked objects per cold miss
  triggering 60–190 ms gen-2 pauses *inside* the walk burst (drive: 1655 ms
  gc-on vs 743 ms gc-off). v7 removes both by never creating the objects.
- **Result (H100 stamp):** G8 adversarial **max step 50.3 → 15.3 ms PASS**
  (thread-invariant 15–18 ms), co-batched degradation **114.7% → 33.8%**, warm
  gates unmoved (+1.51% @32). G8 5/7 → **6/7**. Also refreshed at v7:
  MaskBench TBM p90 208 → 75 µs (the p90 knee, −64%) and TTFM p50 −17%;
  G7 + engine-comparison unchanged (v7 touches neither the warm-hit nor the
  walk path — the win is localized to the serving cold-entry path, which is
  the honest scope claim).
- **The last red, and what it now is:** the +33.8% co-batch degradation is no
  longer software — it is genuine host CPU/memory-bandwidth contention between
  the cold walk and the engine forward loop during a fresh schema's ~0.66 s
  window (confirmed: MORE walk threads → LESS degradation, because the window
  shrinks; niceness helps the residual). Closing it fully is a
  compute-isolation tradeoff (throttle the walk, or accept the number), not a
  bug — recorded honestly rather than chased.
- **Lesson:** twice now (6.6, and here) the measured cause was not the modeled
  one; the red-team's mandatory premise-measurement before coding paid for
  itself both times. And a gate can legitimately reduce from "software defect"
  to "physical resource tradeoff" — name the transition, don't paper over it.

## Meta-lessons

1. **Systematic, planned review at every phase paid for itself.** The Phase-1
   design pass moved us off the sequence-keyed table before a line of code was
   written; the design review caught a runtime crash (2.1) and an
   unimplementable test plan (2.2) before code; the differential suite caught
   three soundness/completeness bugs (3.x) before any debugging session.
2. **An executable specification beats a prose one.** The reference guide is
   ~150 lines nobody optimizes, and every fast-path change since M2 (incremental
   walk, memoization, grouping) shipped same-day because "bit-identical to the
   oracle" is a one-command check.
3. **Run the real benchmark earlier than feels ready.** The unit suite was green
   at 82 tests while the hit path cost 189 ms; one benchmark run reordered the
   optimization priorities completely (4.2).
4. **State machines checked at boundaries convert debugging into type errors.**
   The E14 audit invariant and the statechart engine each caught a real modeling
   gap as an assertion at the exact faulty step, not as corrupted output later.
5. **Design documents should record what they exclude.** Every "deferred /
   out-of-scope" line in DESIGN.md §13 (deadline fallbacks, GAD/CRANE, Earley
   fallback, BIRD) was asked about later; having the decision written with its
   reason avoided re-litigating each one.

## Where this leaves us

| Layer | Status |
|---|---|
| Guarantees (sound/complete/terminating masks, RBAC verb+table, audit replay) | implemented, gate-tested, differential-bound |
| Real tokenizers (byte-level BPE, SentencePiece) | done; byte-complete verified on GPT-2 + Qwen |
| Requirement R (flat per-token cost vs position) | measured: slope ~0 all depths (−4e-6 to −1.4e-5 µs/pos) on warm replays; cumulative R² > 0.99 |
| Competitive constants | grid_core kernel v4 (walk + CD verdicts + LALR advance + bitmask fill): warm-hit p50 **3.5 µs** (GPT-2, G7 harness) — the `<10 µs` criterion now met on the dev host — vs llguidance 6–19 µs, XGrammar 39–325 µs; G9's 2×-XGrammar KPI exceeded (unpinned host) |
| Serving contract (M6, §6) | overlap (GIL-released cold walk + worker prefetch, 88% main-thread liveness) and E17 single-flight implemented in `grid/serving/`, wired into the scheduler-side vLLM backend |
| Gates green locally this phase | G7 (hit p50 met), G10 (1k-gen replay + 100% tamper), G5 walk arm (10k, 0 failures, quotas met), G6(b) model-free (0 bypasses, controls non-vacuous) |
| Remaining for R0 | GPU-box numbers (G8 TPOT/TTFT/throughput, G5 model arm, G6(b) prompt suite), T2 cache tier, SynCode/GBNF G9 arms |
