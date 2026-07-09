# GRID — Onboarding and Milestone Documentation

**Audience:** a new researcher joining the project. This document is a guided tour, not a
new source of truth. The normative documents remain:

| Document | Role |
|---|---|
| `GUARDRAIL-REDESIGN.md` | The *why*: the v0.0.5→v0.0.7 design evolution, chosen methods, proofs, cost model, benchmark plan |
| `DESIGN.md` | The *what/how* (normative spec): architecture §2, entity catalog §5, per-token hot path §6, gates §10, decision log §13 |
| `LESSONS.md` | The retrospective: every phase's changes, the bugs found, measured results |
| `bench/RESULTS*.md` | Committed benchmark reports (every number in this file is sourced from one of them or from `LESSONS.md`) |

Where this document says "measured", the host is local dev (unpinned) unless stated;
binding G7/G8/G9 numbers run on the declared cloud runner — a named provider
instance type + image recorded in each report's host label (`DESIGN.md` §10;
bare-metal pinning was dropped from the plan).

---

## Part 1 — What GRID is, why it exists, and how it works

### 1.1 In one paragraph

GRID (Grammar-Railed Decoding) is a grammar-constrained decoding engine for LLM code
generation, SQL-first, built for enterprise CRUD/RBAC settings. Per decode step it
computes the exact set of vocabulary tokens whose bytes keep the output a *viable
prefix* of a role- and schema-projected grammar, hard-masks everything else
(`logits = -inf`), gates EOS on actual sentence membership, reserves token budget so
generations end grammatically instead of truncating, and appends a hash-chained,
replayable audit record for every emitted token. Four guarantees — soundness,
completeness, termination, and near-linear cost (requirement R) — are stated with
explicit preconditions (`DESIGN.md` §1, `GUARDRAIL-REDESIGN.md` §4) and each is bound
to a verification gate rather than asserted.

### 1.2 Origin: from v0.0.5 to v0.0.7

GRID is the result of a versioned design program, evolved by planned iteration.
**v0.0.5** — designed August 2023 — captured an early,
partial snapshot of the design. The design review planned on our roadmap
revised it on two load-bearing points before implementation
(`GUARDRAIL-REDESIGN.md` §1, `LESSONS.md` Phase 1); **v0.0.6** already
outperformed guidance (its July-2023 release) on the scaling benchmarks of
that era; and the step-by-step, test-validated iteration continued to
**v0.0.7** — this repository, the line the final patent application followed,
improved further since. (The work drew inspiration from published systems,
notably the Outlines paper; the design and implementation are our own.) The
two load-bearing revisions:

1. **Token sequences → parser configurations as the key.** v0.0.5 precomputes
   an in-memory table tagging LLM *token sequences* as accepted/rejected. A full
   table of that shape has `|T|^m` entries: at a 32k vocabulary, sequence length 3
   is already ~3.3×10¹³ entries (~400 TB); length 18 exceeds the atoms in the
   observable universe — and the "representative subset" variant was defined by
   outcome, not construction, so it offered no build recipe (`LESSONS.md` 1.1).
2. **Acceptance → prefix-viability as the tag semantics.** v0.0.5's PDA tags
   "does this sequence belong to the language" — under which every intermediate
   generation state is *rejected*, the mask is empty at step 1, and generation
   deadlocks. The snapshot had described prefix-viability in prose
   without yet formalizing it; the revision settled the semantics on
   prefix-viability throughout (`LESSONS.md` 1.2).

The replacement keys validity on **parser configurations** — (lexer state × LALR stack)
— instead of token sequences, exploiting the viable-prefix property of LR automata
(Knuth 1965; Aho & Johnson 1974: an incrementally advanced LR parser *is* a recognizer
of `Prefix(L)`). Configuration key spaces are grammar-sized (10⁴–10⁶), not
vocabulary-exponential, and carry exactly the information the table tried to enumerate.

The revision also kept what was right (`GUARDRAIL-REDESIGN.md` §2): decode-time logit
masking as primary enforcement (K1), the table idea reframed as a **write-back
configuration-keyed mask cache** (K2), persistent accepted/rejected lists reframed as a
**per-token audit trail** (K3), policy-derived grammars (K4), multi-token pruning
reframed as **jump-forward `Write` spans** (K5), and sampling-within-the-mask (K6).

### 1.3 End to end: what happens when you generate

**Offline / per deployment** (`DESIGN.md` §2 architecture diagram):

- The **L1 dialect grammar** (`grammars/*.grid`, loaded by `grid/grammar/spec.py`) is
  parsed, validated, and frozen with a fingerprint. Terminals are regexes; whitespace
  and comments are first-class `%ignore` terminals.
- The **L2 role projection** (`grid/grammar/projection.py`) subsets productions per
  role (verb subsets, clause bans), then `grid/grammar/reduction.py` runs mandatory
  useless-symbol elimination and verifies `L(G_role) ≠ ∅` — without reducedness,
  "non-empty action set ⇒ viable prefix" fails and dead ends return
  (`GUARDRAIL-REDESIGN.md` §3).
- `grid/lalr/compile.py` builds **LALR(1) tables** (canonical LR(1) item sets, merged
  by core; conflicts raise `LALRConflictError` with a report). `grid/lexer/dfa.py`
  builds the combined **scanner DFA**. `grid/trie/build.py` builds the **TokenTrie**:
  one numpy `uint64` array of DFS-contiguous 8-byte nodes
  (edge byte | token_id+1 | subtree size) from the adapter's canonical
  `token_bytes`, plus an alias table for byte-identical token spellings — exactly the
  buffer `grid_core` consumes zero-copy.
- The **L3 schema lexicon** (identifier allow-lists per terminal category, e.g.
  `TABLE_NAME`, `COLUMN_NAME`) is wrapped by `grid.trie.walk.Lexicons`, and — since
  `LESSONS.md` 5.2 — **validated at guide build**: every allow-list word must be
  scannable to an accepting state of its terminal (`MaskProducer._validate_lexicons`,
  `grid/mask/producer.py`), else `GrammarInvalid`.
- The **ReserveTable** (`grid/lalr/reserve.py`) precomputes token-denominated
  min-completion costs (shortest lexeme per terminal — for L3 categories the shortest
  *allowed* identifier — greedy-tokenized), keyed by (grammar, tokenizer).

**Per request:** `grid.generate.build_guide` assembles a `GridGuide`
(`grid/guide.py`) whose `initial_state` is `GridState(stack=root LALR node,
lexer=empty LexerRun, n_generated=0, prev_token=None, status=ACTIVE)`, wrapped by a
`GridLogitsProcessor` (`grid/processors.py`) that anchors at the first call
(constrained span only — the guide never sees prompt ids) and keeps an incremental
per-row state registry (no Θ(n) prefix hashing; `DESIGN.md` §4.3).

**Per token — the hot path** (`DESIGN.md` §6 is normative; the implementation is
`GridGuide.get_next_instruction` / `get_next_state` + `MaskProducer.masks`):

1. **Allowed terminals.** `A = { t : simulate(stack, t) reaches SHIFT }`
   (`grid/lalr/stack.py::simulate`): LALR default reductions make raw action-row reads
   over-approximate (spurious reduces), so viability runs the reduce chain on a
   *virtual overlay stack* until shift/accept or error. Memoized per stack node;
   in-kernel as `RustVerdicts.allowed_mask`.
2. **EOS legality, mid-lexeme aware.** `_eos_ok` finalizes the pending remainder into
   its winning maximal-munch segmentation (`LexerRun.finalize`), virtually shifts it on
   a scratch chain (`_finalized_node`), then simulates the reduce chain of `$end` to
   ACCEPT. After `... FROM t`, `t` is a complete IDENT awaiting finalization — the
   stack alone would wrongly say EOS is illegal.
3. **Reserve trigger** (session budget, not grammar state). If
   `remaining ≤ len(concrete completion) + RESERVE_SAFETY`, return
   `Write(shortest_completion + EOS)` — never a bare EOS away from ACCEPT.
   `ReserveTable.completion(node)` synthesizes the exact minimal completion from the
   stack configuration; stop reason `MAX_TOKENS_WITH_JUMP_COMPLETE`.
4. **Cache key.** `("ident"|"generic", remainder bytes, sorted(A), schema_fp?)`
   (`MaskProducer.cache_key`) — identifier positions get a *type-distinct* key carrying
   the schema fingerprint; consulting a generic entry there raises
   `IdentifierMaskBypassError` in all builds.
5. **Cache lookup** (`grid/mask/cache.py`, T1 per-grammar tier; hits are the common
   case — 86–87% on the SQL bench, 97–98% steady-state in `bench/RESULTS-r.md`).
6. **Miss → trie walk with the CI/CD split** (`grid/trie/walk.py`, kernel
   `RustWalker.walk`). A DFS over the token trie carries an O(1)-updatable scan state
   per frame; each token is classified **CI** (context-independent: resolvable from
   (remainder, A, lexicons) alone — cacheable), **CD** (context-dependent: its bytes
   cross a terminal boundary *and continue*, so viability depends on the post-shift
   stack — never cached), or rejected-subtree (monotone prune). CD entries are grouped
   at publish time (one representative per verdict class); the entry is published
   idempotently (content-hashed `entry_id`).
7. **Per-step CD verdicts.** Each cached entry's CD *groups* are checked against the
   live stack (`RustVerdicts.cd_pass`: arena LALR simulate/shift with `_StepMemo`-style
   memoization, returning passing token ids as an i32 buffer consumed zero-copy). The
   mask is `ci ∪ cd_pass ∪ {EOS if step 2 said legal}`.
8. **Empty mask ⇒ `DeadEndError`** — unreachable by theorem; always a bug (G5 asserts
   zero occurrences; `LESSONS.md` 5.2 is the one real-world time this fired, on a
   violated theorem precondition, and it produced a validation, not a workaround).
9. **Singleton mask ⇒ jump-forward.** The maximal chain of singleton masks (bounded by
   `J_max = 8`) is returned as one `Write` span. In the GRID-owned decode loop
   (`grid/generate/api.py`, mode 1) span tokens are appended **without forward
   passes**, each advancing the guide and appending an audit record; in processor-only
   mode (vLLM, mode 2) `Write` degrades to a singleton mask per step — sound, no
   model-call savings (`DESIGN.md` §4.5).
10. **Otherwise `Generate(ids)`** — the *full* exact mask (never a first-legal-token
    shortcut), applied as an in-place hard `masked_fill_(-inf)`; the sampler samples
    within it.
11. **Advance** (`get_next_state`): `token_bytes(id)` → `LexerRun.advance` (maximal
    munch; forced emission events carry candidate *sets*) → `pick_viable` chooses the
    parser-viable terminal per event (contextual lexer) → `shift_terminal` pushes
    persistent stack nodes (each carrying a rolling `config_hash`) → partial-lexeme
    viability check on the new remainder → **audit append**: every step, including
    each Write-span interior token and the EOS tail record
    (`grid/audit/log.py`; `mask_entry_id` iff the step was a GENERATE).

Termination: EOS is only in the mask when the output is a complete sentence; consuming
it moves the state to COMPLETE and seals the audit chain with the stop reason and
artifact fingerprints.

### 1.4 Repo map

| Path | Role |
|---|---|
| `grid/protocols.py` | Normative Guide / Write / Generate / tokenizer / sampler protocol shapes (tensor-typed tokens); conformance tests in `tests/protocols/` |
| `grid/guide.py` | `GridGuide` + `GridState`: the §6 hot path behind the Guide protocol |
| `grid/processors.py` | `GridLogitsProcessor`: anchoring, incremental state registry, hard masking, lifecycle |
| `grid/generate/` | `sql()` / `cfg()` entry points; `api.py` = GRID-owned decode loop (mode 1, jump-forward's home) |
| `grid/grammar/` | `spec.py` (L1 load/validate/freeze), `projection.py` (L2), `reduction.py` (useless-symbol elimination) |
| `grid/lalr/` | `compile.py` (LALR(1) tables, conflict reports), `stack.py` (persistent stack, virtual-stack simulate, config hash), `reserve.py` (E4a token-denominated reserve + completion synthesis) |
| `grid/lexer/` | `dfa.py` (scanner DFA), `run.py` (immutable `LexerRun`, maximal-munch scan, emission events with candidate sets) |
| `grid/trie/` | `build.py` (TokenTrie artifact format + aliases), `walk.py` (the walk: executable spec + kernel dispatch, CI/CD classification, `Lexicons`) |
| `grid/mask/` | `producer.py` (steps 1–8 orchestration, cache keys, CD residue check, lexicon validation), `cache.py` (T1 cache, adaptive encoding, CD grouping, idempotent publish) |
| `grid/audit/` | `log.py` (hash-chained records, seal, verify) |
| `grid/policy/` | `bundle.py` (RBAC store → role shapes), `schema.py` (schema → lexicons), `semantic.py` (post-parse `SemanticChecker`) |
| `grid/models/` | Tokenizer/model adapters (HF, transformers, mock); canonical `token_bytes` lives here |
| `grid/_reference/` | `ReferenceGuide`: the brute-force trial-parse oracle (the executable specification) |
| `grid/_statecharts/` | Statechart engine; §5 state machines are YAML-checked at boundaries |
| `grid_core/` | Rust crate (pyo3): `RustWalker` (trie walk) + `RustVerdicts` (CD verdicts + LALR simulate) |
| `grammars/` | `sql_subset.grid` (bench grammar), `sql_spider.grid` (Spider dialect, 100% dev-gold coverage), `toy_expr.grid` |
| `tests/` | Gate suites; note `tests/trie/test_rust_parity.py` and `tests/mask/test_kernel_parity.py` (kernel ≡ spec) |
| `bench/` | `compare_engines.py`, `r_microharness.py`, `maskbench_grid.py` + `json_schema_to_grid.py`, `spider_ex.py` + `spider_coverage.py`, `RESULTS*.md` |

Spec-vs-code deltas a new reader should know (drift is documented, not hidden):
`DESIGN.md` §3's planned `grammar/registry.py`, `grammar/lexicon.py`, `mask/keys.py`,
`audit/replay.py` are not yet split out (their responsibilities live in
`mask/cache.py`, `trie/walk.py::Lexicons`, `mask/producer.py`, `audit/log.py`); the T2
cross-family cache tier and namespace rollover are deferred to the serving work
(`grid/mask/cache.py` docstring; `bench/RESULTS-r.md` reports the cross-role hit factor
as N/A); kernel #4 (`apply_token_bitmask`) lands with the vLLM backend
(`LESSONS.md` 4.4a); the implemented `StackNode` carries no `reserve_sum`/`refcount`
fields — the reserve trigger computes an exact concrete completion per step instead,
and Python GC replaces refcounts.

### 1.5 The dual-implementation discipline

Two implementations of one semantics, bound by tests (`DESIGN.md` §2):

- **`grid/_reference/guide.py`** is the executable specification: for every candidate
  token it re-derives viability from scratch on the full byte string — no trie, no
  cache, no incremental state, no CI/CD split. ~140 lines nobody optimizes.
- The **fast path** (Python orchestration + `grid_core` Rust kernels) is bound to it by
  differential tests, and the kernels are bound to the *Python* fast path bit-identically
  by parity tests (`tests/trie/test_rust_parity.py`, `tests/mask/test_kernel_parity.py`
  — order-exact, not just set-equal). `GRID_NO_RUST=1` forces the spec path everywhere.

This discipline caught, before any debugging session: byte-identical token aliases
dropped from masks (completeness, `LESSONS.md` 3.1), lexicon-blind reserve completions
(soundness, 3.2), Write-span audit-kind gaps (3.4), and a real Rust/Python divergence in
the CD group key (4.5). The meta-lesson stands: every fast-path change since M2 shipped
same-day because "bit-identical to the oracle" is a one-command check.

### 1.6 Critical decisions, trade-offs, consequences

1. **Viable-prefix semantics, not acceptance semantics.** The parser recognizes
   `Prefix(L)`; membership (EOS) is a separate explicit check via the `$end` reduce
   chain. *Consequence:* masking works at every step; the price is the machinery of
   step 2's mid-lexeme EOS rule, which acceptance semantics never needs
   (`LESSONS.md` 1.2, 1.4-adjacent).

2. **Per-step hard masking only; rejection sampling and 1-token backtracking cut.**
   Rejection sampling needs ~169 expected full regenerations for a 100-token statement
   at 95% per-token validity (and loops forever under greedy); 1-token backtracking
   cannot escape deep dead ends; soft down-weighting voids the guarantee
   (`LESSONS.md` 1.3). *Consequence:* single-pass guaranteed-valid output — but
   per-step masking ≠ sampling from `P(x | x ∈ L)` (GAD, arXiv:2405.21047). The
   distribution-faithfulness gap is a recorded, deferred trade-off; the bench plan
   measures EX-delta instead of pretending it away (`DESIGN.md` §13).

3. **LALR(1), not Earley.** Production SQL grammars (PostgreSQL, MySQL, SQLite) are
   LALR(1), so deterministic-PDA coverage suffices (`GUARDRAIL-REDESIGN.md` §3).
   *Given up:* (a) arbitrary CFGs — MaskBench shows the boundary concretely: 79/315
   schemas are declared compile errors, 5 of them `LALRConflictError`
   (`bench/RESULTS-maskbench.md`); (b) the flattest possible tails — llguidance's
   Earley/derivative core keeps p99 at 224–867 µs where GRID's cold misses cost
   milliseconds (`bench/RESULTS.md`, `bench/RESULTS-qwen.md`). *Gained:* small
   deterministic tables, the virtual-stack simulate that makes allowed-terminals and
   EOS exact under table compression, and a persistent stack with O(1) rollback. An
   Earley fallback stays a recorded option until a non-LALR dialect construct forces
   it (`DESIGN.md` §13).

4. **The CI/CD cache split.** A token whose bytes cross a terminal boundary and
   continue (`'),'`, `'1;'`) has viability depending on the *post-shift* allowed set —
   on the stack — which no (lexer, A)-keyed entry can capture; caching it would violate
   key soundness *by construction* (`LESSONS.md` 2.5). So the walk returns a cacheable
   CI mask plus a CD list re-checked against the live stack every step. *Consequence:*
   correctness held (G4 cache-on ≡ cache-off), but the uncached-by-design residue became
   the dominant hot-path cost — 189 ms hits on the first real benchmark — and drove
   two optimization rounds (memoization, publish-time grouping) and ultimately the
   second Rust kernel (`LESSONS.md` 4.2, 4.4a).

5. **Write-back cross-request cache, not compile-time-only precompute.** XGrammar
   precomputes per compiled grammar; GRID computes-on-miss and publishes, exploiting
   cross-request locality across a per-role/per-tenant grammar family — the actual
   enterprise deployment shape (`GUARDRAIL-REDESIGN.md` K2, §7.1). *Consequence:* the
   design is only rewarded where requests repeat configurations. One-shot protocols
   never warm it: MaskBench runs each schema once, so GRID's p90+ TBM tail there is
   all cold misses (`bench/RESULTS-maskbench.md` reading notes). The serving bench
   (G8) is where this choice is designed to pay.

6. **The kernel bitmask bound: 64 → 512 terminals.** `grid_core` originally held
   candidate terminal sets as single `u64` bitmasks; on MaskBench 6% of compiled
   schemas exceeded 64 terminals, fell off the kernel path, and owned the 200 ms TBM
   tail class. Kernel v3 widened masks to `[u64; W]`, W ∈ {1,2,4,8} (512 terminals),
   monomorphized per width so W=1 compiles back to the scalar ops (`LESSONS.md` 5.5):
   fallback schemas went 19→0, TBM p75 459→39 µs, p99 202→30 ms. *Consequence:*
   grammars past 512 terminals still fall back to the spec walk; the remaining TBM
   p90 (~28 ms) is cold trie-walk cost, the current named target.

7. **Byte-level trie + maximal-munch contextual lexer.** LLM tokens do not align with
   grammar lexemes (the misalignment the v0.0.5 spec illustrates: `sel`+`ect` spells
   `SELECT`); the byte-level trie walk from the current lexer state is what makes masks
   exact over real vocabularies (`LESSONS.md` 1.4). Same-regex terminals
   (keyword-vs-identifier; `TABLE_NAME` vs `COLUMN_NAME` share `[a-z_][a-z0-9_]*` in
   `grammars/sql_spider.grid`) forced the contextual discipline: emission events carry
   candidate *sets* and the parser-viability choice picks the terminal
   (`LESSONS.md` 3.3). *Consequence:* per-category L3 lexicons work, and the same
   mechanism resolved JSON's property-key-vs-STRING overlap with no new code
   (`LESSONS.md` 5.1). Known recorded caveat: a viable terminal accepting only at a
   strictly shorter position than the union-longest match is not chosen; no grammar
   has hit it, and the G3 differential will surface any that does
   (`grid/lexer/run.py` docstring).

8. **L3 lexicons, and a theorem precondition that had to become a VALIDATION.** The
   identifier composition rule: at identifier positions the mask comes from L3
   allow-list intersection, never from unioned generic-IDENT entries (a generic verdict
   admits tokens spelling *forbidden* identifiers, and the parser will not reject them
   later — silent RBAC violation; `GUARDRAIL-REDESIGN.md` §3.4). Its completeness
   precondition — **lexicon ⊆ terminal language** — was satisfied by every fixture and
   validated by no gate, until Spider's `orchestra` database supplied a column named
   `Official_ratings_(millions)`: parentheses are outside `COLUMN_NAME`'s regex, every
   prefix passes `prefix_ok` (it *is* a lexicon-word prefix), no token can ever
   complete the lexeme, and the mask went empty — `DeadEndError`, the error class the
   architecture promises cannot happen (`LESSONS.md` 5.2). The precondition is now
   checked at guide build (`MaskProducer._validate_lexicons` raises `GrammarInvalid`;
   regression test `tests/mask/test_lexicon_language.py`), and the harness filters
   schema names to the identifier language — sound, because names outside the grammar's
   language were never generatable anyway.

9. **Column RBAC is provably not mask-enforceable → `SemanticChecker`.** In SQL the
   SELECT list precedes FROM, so alias→table binding is unknown at column-mask time
   (and alias binding is context-sensitive); no left-to-right CFG mask can enforce
   per-table column policy (`GUARDRAIL-REDESIGN.md` §4.6). The mask guarantees
   **verb- and table-level** policy; `grid/policy/semantic.py` re-parses the completed
   statement's terminal stream and flags unknown tables and columns not belonging to
   any referenced table (G6(d): 100% of column-violation fixtures flagged).

10. **Honest scope boundaries.** Out of mask scope by proof: column RBAC, semantic
    validity beyond the grammar. Out of v1 by decision, with reasons recorded
    (`DESIGN.md` §13): distribution-faithful sampling (GAD/CRANE), BIRD, byte-level
    jump-forward with re-tokenization, Earley fallback, approximate deadline fallbacks
    (a deadline over-approximation reopens the soundness hole, so v1 has none — the
    serving contract is skip-a-round, `DESIGN.md` §6).

### 1.7 Working on the repo

```
.venv/bin/pytest tests/ -q                        # gate suite (G0–G6 slices, G10a)
GRID_NO_RUST=1 .venv/bin/pytest tests/ -q         # force the executable-spec path
(cd grid_core && maturin develop --release)       # rebuild kernels
.venv-bench/bin/python bench/compare_engines.py   # engine comparison (SQL)
.venv-bench/bin/python bench/r_microharness.py --quick   # requirement-R harness
```

---

## Part 2 — Formal and technical design

### 2.1 Objects

Let `Σ` be the byte alphabet and `V` the tokenizer vocabulary with the canonical
spelling function `bytes: V → Σ*` (`token_bytes`, one definition used by the trie
build, the fast path, and the reference oracle — `DESIGN.md` E6).

- **L1 (dialect grammar).** A CFG `G = (N, T, P, S)` whose terminals `τ ∈ T` carry
  regular lexeme languages `R_τ ⊆ Σ*`, plus an ignored subset `I ⊆ T`
  (whitespace/comments). Lexing is **maximal-munch over the union automaton with
  contextual resolution**: a forced emission event carries the full candidate set at
  the longest match, and the consumer picks the highest-priority *parser-viable*
  candidate (literals before named terminals), else an ignored candidate, else rejects
  (`grid/lexer/run.py`, `grid/trie/walk.py::pick_viable`).
- **L2 (role projection).** A production subset `P_role ⊆ P`, closed under mandatory
  useless-symbol elimination, with `L(G_role) ≠ ∅` verified. Only reduced, verified
  projections reach the LALR compiler (`DESIGN.md` E2).
- **L3 (schema lexicon).** For identifier categories `C ⊆ T`, finite allow-lists
  `W_c ⊆ Σ*`. **Precondition (validated at build since `LESSONS.md` 5.2):**
  `W_c ⊆ R_c` for every category — each allowed word is scannable to an accepting
  state of its terminal.
- **The constrained language.** `L = L(G_role, schema)` = byte strings that lex and
  parse under `G_role` (with the discipline above) such that every category-`c` lexeme
  lies in `W_c`.
- **Parser configuration.** `κ = (σ, r)`: the persistent LALR stack chain `σ`
  (`grid/lalr/stack.py::StackNode`) and the lexer state — concretely the remainder
  bytes `r` of the single in-progress partial lexeme, which determine the DFA state
  deterministically (`grid/lexer/run.py::LexerRun`).
- **Viable prefix.** `w ∈ Prefix(L) ⇔ ∃ w′ ∈ Σ* : w·w′ ∈ L`. Operationally: the
  incremental scan+shift of `w` succeeds and the trailing partial lexeme is live for
  some allowed-or-ignored terminal passing `prefix_ok` — by the correct-prefix
  property this recognizer accepts exactly `Prefix(L)` *under the preconditions in
  §2.2* (reduced grammar, validated lexicons).
- **The mask.** For configuration `κ` with current output `w`:

  `M(κ) = { t ∈ V∖{EOS} : bytes(t) extends κ to a viable prefix } ∪ { EOS iff w ∈ L }`

  EOS enters the mask only through the explicit membership check (§6 step 7 of
  `DESIGN.md`); special tokens are excluded from the trie and permanently masked.

### 2.2 The four guarantees, with preconditions

Stated as in `GUARDRAIL-REDESIGN.md` §4 / `DESIGN.md` §1; every clause has a gate.

1. **Soundness.** Every emitted token keeps the detokenized output in
   `Prefix(L(G_role, schema))`. *Preconditions:* exact masks (no lookahead
   approximation); hard `-inf` masking (soft down-weighting voids it); the identifier
   composition rule (L3 intersection at identifier positions, structurally enforced by
   type-distinct cache keys + `IdentifierMaskBypassError`). *Gates:* G3 (mask
   exactness vs the trial-parse oracle), G5, G6(a).

2. **Completeness.** No token is blocked whose byte string can extend the current
   viable prefix toward a member of `L`. *Preconditions:* (a) **byte-complete
   vocabulary** — all byte values reachable via `token_bytes`; verified per adapter
   (E6 `verify()`; degradation is an explicit warning `W-COMPLETENESS01` that formally
   voids completeness, never soundness); (b) reduced grammar; (c) exact trie walk with
   alias expansion (masks over token *ids*, not spellings — `LESSONS.md` 3.1);
   (d) **lexicon ⊆ terminal language — now VALIDATED at build, not assumed**
   (`LESSONS.md` 5.2: the one real-data violation produced an empty mask at a viable
   state within 100 Spider generations; `MaskProducer._validate_lexicons` closes it).
   Under (a)–(d), every viable prefix has ≥ 1 legal token: dead-end freedom
   (`GUARDRAIL-REDESIGN.md` §3, dead-end theorem). *Gate:* G3's byte-fallback arm;
   G5 `DeadEndError = 0`.

3. **Termination.** Output ∈ `L` on every non-error stop. EOS is legal **iff** the
   current output is a complete sentence — computed by simulating the reduce chain of
   `$end` to ACCEPT on the (mid-lexeme-aware, virtually finalized) stack, never by a
   raw row read (LALR spurious reduces). The **token-denominated reserve** prevents
   budget truncation: completion costs count *model tokens* (a terminal-denominated
   reserve under-reserves — one identifier terminal can span many tokens;
   `LESSONS.md` 2.6), the trigger fires at
   `budget_remaining ≤ |completion| + RESERVE_SAFETY`, and the response is
   `Write(shortest legal completion + EOS)` — never a bare EOS away from ACCEPT
   (`LESSONS.md` 2.1: the draft's bare-EOS reserve would have crashed its own gate).
   *Gate:* G5 — EOS only at ACCEPT, every jump-complete stop parses, no
   reserve-stopped generation exceeds `max_tokens`.

4. **Requirement R (near-linear cost).** Amortized O(1) guard-rail cost per token,
   total O(n); **per-step worst case bounded by nesting depth** (a single terminal can
   trigger a reduce cascade proportional to stack depth; SQL prefix operators are
   inherently right-recursive, so cascades cannot be linted away), **never by output
   position n**. Nothing in the hot loop reads state proportional to `n`: the
   processor keys states incrementally (splitmix64 chaining, `DESIGN.md` §4.3), the
   config hash is a rolling O(1)-per-push mix, and mask cost is a function of the
   configuration, not the position. *Measured evidence:*
   - `bench/RESULTS-r.md` (G7 R-microharness, gpt2, n = 16,000 tokens/stream, 20
     seeded runs per depth, warm-pass OLS): slope mean ± 95% CI of
     **−0.000013 ± 0.000016 µs/pos** (depth 0), −0.000010 ± 0.000014 (4),
     −0.000011 ± 0.000018 (8), **−0.000003 ± 0.000009 (16)** — CI upper bounds two
     orders of magnitude under ε = 1e-4 µs/pos at every depth; cumulative-cost
     R² ≥ 0.99934; steady-state hit rate 97.2–98.2%; per-depth CD-residue telemetry
     (mean 143–150 groups/step, mean 9.3k–14.1k passing ids).
   - `bench/RESULTS.md` / `bench/RESULTS-qwen.md` warm-replay checks: slope
     **−0.006 µs/pos** (gpt2) and **−0.012 µs/pos** (Qwen), with first-half p50 equal
     to second-half p50 (10/10 µs and 29/29 µs). The negative *mixed-arm* table slopes
     (−24 to −56) are an artifact of cold misses clustering early in replays — the
     warm-pass slope is the R statistic (`LESSONS.md` 4.3).

### 2.3 The cache key and OBL-KEY1

**Key as implemented** (`MaskProducer.cache_key`; the grammar fingerprint scopes the
whole cache instance):

```
( kind ∈ {"ident","generic"},   # type-distinct identifier keys (E11)
  remainder bytes r,            # determines the lexer DFA state deterministically
  tuple(sorted(A)),             # allowed-terminal signature from the live stack
  schema_fingerprint | None )   # REQUIRED iff kind == "ident"
```

`DESIGN.md` E11 words the first component as `lexer_product_state`; keying on the
remainder bytes refines that (the state is a function of `r`), so the implemented key
is finer-or-equal — sound, recorded here as deliberate drift.

**OBL-KEY1 (soundness obligation).** Any two configurations sharing a key must produce
**byte-identical context-independent masks** — the key must refine the Myhill–Nerode
classes of the (lexer product-DFA × allowed-terminal set × identifier-lexicon) product.
The CD residue is exempt *because it is never cached* — that is the entire point of the
split (`LESSONS.md` 2.5: without it the obligation is violated by construction, not
merely unverified). Verified two ways: G4 differentials (cache-on ≡ cache-off), and a
runtime tripwire — publish is content-addressed, so a racing writer of the same key
with a different mask trips `assert cur.entry_id == entry.entry_id`
("OBL-KEY1 violation") in `MaskCache.publish`.

**Entry encoding** (E10, deterministic across implementations for G10 replay): payload
chosen among accept-list / reject-list / bitset by byte size
(`4·|accept|` vs `4·(V−|accept|)` vs `⌈V/8⌉`, ties accept < reject < bitset), ids
ascending; `entry_id = BLAKE2b-128(canonical key ‖ tag ‖ payload)` — racing writers
produce the same id, so publish is idempotent by construction.

### 2.4 CD groups: publish-time construction

The verdict of `check_context_dependent(e, σ)` depends on the entry `e` only through:
its per-event candidate sets; its segments and trailing remainder **only via the
lexicon predicates** (`lexeme_ok`/`prefix_ok`); and the live set of its remainder's
scan state. Therefore entries are partitioned **once, at publish time**, by the group
key `(candidate-set sequence, segments·remainder if lexicon-sensitive, tail live set)`
(`grid/mask/cache.py::make_entry`; in-kernel in `RustWalker.walk` with byte-serialized
keys), and each step evaluates **one stack-dependent verdict per group**, extending the
mask with the group's alias-expanded token ids. This is what turned ~25k per-step
entry checks into ~150 group verdicts (`LESSONS.md` 4.2; `bench/RESULTS-r.md` reports
143–152 groups/step). The strict parity suite caught one real divergence here — the
Rust key over-included event lengths that the spec keys only under lexicons — which is
the executable-spec discipline working across languages (`LESSONS.md` 4.5).

### 2.5 The kernels (`grid_core/src/lib.rs`, kernel version 3)

Both kernels are optional accelerators bound bit-identical to the Python executable
specification; `GRID_NO_RUST=1` forces the spec path, and grammars past the
512-terminal bound fall back automatically.

**`RustWalker` — the incremental-state trie walk** (transcription of
`grid/trie/walk.py::_walk_py`). DFS over the 8-byte packed node array; each frame
carries `(subtree end, dfa_state, current segment, last-accept length/state,
events-stack length, n_real, cd_flag)`. A byte that kills the DFA triggers the
maximal-munch forced-emission cascade via a pending-byte queue (emit the last-accept
lexeme, requeue the unconsumed tail plus the dead byte, restart the DFA) — exactly
`lexer.run.scan`'s restart semantics. Rejection is monotone (live sets shrink under
extension), so rejected subtrees are skipped wholesale via the packed subtree size.
Terminal candidate sets are `[u64; W]` bitmask arrays, W ∈ {1,2,4,8} chosen from
`n_terminals` and monomorphized (the source of the 512-terminal bound; masks cross
the FFI as little-endian word lists). Aliases are expanded and CD groups built
in-kernel; only group representatives cross the FFI boundary.

**`RustVerdicts` — per-step CD verdicts + the LALR virtual stack** (transcription of
`producer.check_context_dependent`/`_StepMemo` and
`stack.py::simulate`/`allowed_terminals`/`shift_terminal`/`eos_ok_stack`). Two-phase:

- `register(groups)` — once per cache entry, content-addressed by `entry_id`:
  precomputes everything stack-independent — per event the lexeme-passing candidate
  mask and the ignored-fallback pick; per group the remainder's scan state and tail
  classification (`Empty` / `Dead` / `Live{ign_ok, allow-mask}`); token ids serialized
  to i32-le bytes ready to memcpy.
- `cd_pass(handle, stack-chain)` — per step: builds an **arena** of
  `(parent index, state)` nodes seeded from the live stack chain; runs each group's
  event sequence with `allowed`/`shift` **memoized per arena node exactly like
  `_StepMemo`**; `simulate` runs reduces on a virtual overlay over the arena until
  shift/accept/error. Passing groups' token bytes are concatenated and returned as one
  i32-le buffer that `np.frombuffer` consumes zero-copy — materializing tens of
  thousands of Python ints per step across the FFI was the measured dominant cost the
  buffer design removed (`LESSONS.md` 4.4a). `allowed_mask` and `eos_ok` expose the
  same simulate for §6 steps 1–2.

### 2.6 The audit chain

- **Configuration rolling hash** (`grid/lalr/stack.py`, pinned normatively for
  cross-implementation replay): `H(node) = low 64 bits of
  BLAKE2b-128( H(parent) ‖ u32le(lalr_state) ‖ u32le(goto_symbol) )`, `H(root) = 0`.
  O(1) per push — hashing the stack from scratch would be a hidden Θ(depth)/token
  n-dependence (`GUARDRAIL-REDESIGN.md` §3.6). Audit-only; never used for cache
  equality (2⁻⁶⁴ collision policy accepted).
- **Records** (`grid/audit/log.py`): `(step, config_hash, mask_entry_id,
  chosen_token, blocked_count, instruction_kind ∈ {GENERATE, WRITE, EOS},
  record_hash)`, chained by
  `record_hash = BLAKE2b-128(prev_hash ‖ canonical fields)` from the genesis constant.
  **Every step appends** — each interior token of a `Write` span and the EOS tail
  record included — with the invariant `mask_entry_id ≠ None ⇔ kind = GENERATE`
  (this invariant caught the Write-span modeling gap as an assertion at the faulty
  step, `LESSONS.md` 3.4).
- **Seal:** stop reason, chain head, artifact fingerprints, mode flags
  (processor-only downgrades, `stop_at` exclusions). `verify_chain()` recomputes the
  chain; tamper detection measured 200/200 in the G10a slice (`LESSONS.md` 3.4).
- **Replayability:** entry ids are content hashes over versioned, immutable cache
  entries, and the config hash pins the parser trajectory — so a log replays against
  archived grammar artifacts to bit-identical masks (G10: ≥1,000 generations across a
  namespace rollover; G10a — chain integrity + replay smoke — gates M3 exit and is
  green at the current slice).

---

## Part 3 — Where GRID beats Outlines, and which decisions did it

Everything below cites committed numbers only: `bench/RESULTS.md` (gpt2, 11 replays /
491 steps), `bench/RESULTS-qwen.md` (Qwen2.5-0.5B, 151k vocab, 509 steps),
`bench/RESULTS-maskbench.md` (llama-3.1 tokenizer, 315-schema stratified sample),
`bench/RESULTS-spider.md` (the FULL 1034-question Spider dev set, Qwen2.5-7B, on the
declared H100 runner) and `bench/RESULTS-spider-05b.md` (Qwen2.5-0.5B, 100 questions,
same runner). Latency numbers are local dev (unpinned) unless the report says otherwise.

### 3.1 The Outlines comparison is really an llguidance comparison

Outlines ≥ 1.x has **no CFG engine of its own**: `outlines.types.CFG` routes to a
backend, default llguidance (`CFG_DEFAULT_BACKEND`); JSON-schema and regex default to
`outlines_core`. The bench's Outlines arm drives Outlines' own
`LLGuidanceLogitsProcessor`, so it measures llguidance plus Outlines' Python wrapper
(consume + bitmask fill + apply) — identical rejects and slope to the raw llguidance
arm by construction (`bench/compare_engines.py`, `bench/RESULTS.md` notes;
`LESSONS.md` 4.4: the field converged on state-keyed engines to the point that the
original comparison target no longer exists as an engine).

Measured (gpt2): Outlines p50 **129.2 µs** vs raw llguidance **5.9 µs** — ~123 µs of
pure wrapper — and compile 1431 ms vs 174 ms. On Qwen: 132.1 µs vs 18.5 µs, compile
1378 ms vs 550 ms. GRID's p50 (11.5 µs gpt2 / 32.6 µs Qwen) beats Outlines-as-a-product
by ~11×/4× at the median with a fraction of the compile time. That comparison is
real for anyone constraining *through* Outlines, but it is won against the wrapper.
The substantive engine comparison is against llguidance itself (§3.4) and XGrammar
(§3.3), where the differentiators are architectural:

### 3.2 What GRID has that neither Outlines nor the engines it wraps have

1. **Mask-level RBAC and schema enforcement.** Per-role grammar projections (L2) and
   per-schema identifier lexicons (L3) make policy part of the *language*: identifiers
   are schema-valid **by construction**, forbidden tables are unreachable even spelled
   byte-by-byte (G6 property tests, `LESSONS.md` 3.3). Evidence that the mechanism is
   exact where it applies: **zero validation errors** on all 315 MaskBench schemas GRID
   compiles (every valid instance accepted — `bench/RESULTS-maskbench.md`), and on
   Spider the constrained arm's outputs all parse with schema-valid identifiers by
   construction. The Spider EX story is scale-dependent and worth stating precisely:
   at **0.5B** (`bench/RESULTS-spider-05b.md`, 100 questions, H100) constraint is worth
   **+13 EX points** (29% vs 16%) and +26 points of syntax validity (57% vs 31%) —
   the mask erases the syntax-error class a weak model commits constantly. At **7B**
   over the **full 1034-question dev set** (`bench/RESULTS-spider.md`) the raw deltas
   nearly vanish (EX 53.7% vs 52.9%, syntax 91.3% vs 91.0%): a capable model rarely
   errs syntactically. The resolution is **checker-guided repair**
   (`bench/RESULTS-spider-repair.md`): GRID's residual failures are alias↔column
   binding — provably out of mask scope, but *precisely named* by the alias-aware
   `SemanticChecker` — and one constrained retry with the violations quoted back
   takes executes to **94.5%** and EX to **55.2% (+2.3 over unconstrained)** at
   +14% tokens on the ~7% of queries that engage. The capability symmetry is
   measured in both directions: the 0.5B cannot exploit the same feedback at all
   (`LESSONS.md` 5.4a/5.4b). GRID's value at scale = the ~94.5% execution floor +
   schema-valid identifiers by construction + RBAC projection + replayable audit +
   repairability. No mainstream engine offers per-role/per-schema projection as a
   mask-level feature.
2. **The hash-chained per-token audit trail** (K3, `GUARDRAIL-REDESIGN.md` §2/§7):
   every permit/block decision bound to a versioned, content-addressed mask entry and a
   rolling configuration hash, replayable against archived artifacts, tamper-evident.
   No mainstream system (XGrammar, llguidance, SynCode, Outlines) has a replayable
   per-token audit; for the RBAC/compliance setting this is a first-class feature, and
   the Spider ablation shows its runtime cost is in the noise (`grid-audit-off` moves
   tok/s, not EX — `bench/RESULTS-spider-ablations.md`).
3. **The write-back cross-request cache** (K2): compute-on-miss, publish, and reuse
   across requests of a grammar *family* — vs compile-time-only precompute (XGrammar)
   or per-request computation. Honest corollary in both directions: one-shot protocols
   like MaskBench never reward it (each schema runs once; GRID's p90+ TBM tail there is
   cold misses by design — `bench/RESULTS-maskbench.md` notes), while in replay-shaped
   workloads it produces the 86–87% hit rates and 97–98% steady-state hit rates behind
   every GRID median in `bench/RESULTS.md` / `bench/RESULTS-r.md`.
4. **Requirement R as a designed and measured contract.** Competitors do not state a
   per-token-cost-vs-position guarantee; GRID ships a microharness and committed slope
   CIs for it (§2.2.4). The property is architectural (configuration-keyed cost), not
   an emergent benchmark artifact.

### 3.3 Vs XGrammar (0.2.3)

XGrammar's adaptive token-mask precompute gives it small medians on simple grammars,
but its context-dependent token handling is exercised hard by recursive SQL and complex
schemas — the companion's §8.1 risk, which `LESSONS.md` 4.4 found to be
engine-universal:

- **SQL tails** (`bench/RESULTS.md`, `bench/RESULTS-qwen.md`): XGrammar p99
  **14.5 ms** (gpt2) and **17.9 ms** (Qwen) vs GRID's 8.8 / 15.3 ms — same cost class,
  GRID ahead; llguidance is in a different class entirely (§3.4).
- **Compile-time blowup on complex JSON** (`bench/RESULTS-maskbench.md`): XGrammar TTFM
  p75 **207 ms**, p90 1.17 s, p99 **13.0 s** (avg 688 ms) vs GRID's p75 46 ms, p99
  **865 ms** (avg 63 ms) — GRID's TTFM is a pure-Python LALR+scanner build and still
  sits an order of magnitude under XGrammar's tail. (llguidance's 0.3 ms TTFM p50 beats
  both; that is the next named kernel target.)
- **Median mask latency — GRID now beats XGrammar on both tokenizers:** p50
  **11.5 µs vs 39.0 µs** (gpt2) and **32.6 µs vs 324.5 µs** (Qwen). The decisions that
  caused this, each traceable in `LESSONS.md`:
  - the **CI/CD split** (2.5) — the cacheable majority of each mask is a hit;
  - **publish-time CD grouping** (4.2) — ~150 group verdicts per step instead of ~25k
    entry checks;
  - **i32-buffer masks end-to-end** (4.4a) — the profiled bottleneck was never the
    verdicts but materializing tens of thousands of Python ints across the FFI;
    fixing it took warm hits 276 µs → 10.9 µs (gpt2), 992 µs → 31 µs (Qwen);
  - the **arena-based LALR simulate inside the kernel** (4.4a/4.5) — allowed-terminals
    and shifts became memoized table lookups;
  - **kernel v4's persistent interned-stack arena + one-call `hit_pass`** (6.1) —
    memoizing verdicts across token positions (parser configurations recur) and
    assembling the whole allowed-id buffer kernel-side took the warm hit
    10.9 → 3.5 µs (gpt2), meeting the G7 `< 10 µs` gate on the dev host.
- **Correctness posture** (`bench/RESULTS-maskbench.md`): XGrammar declares zero
  compile errors but shows **27 validation errors** (valid instances rejected) and 37
  invalidation errors — silent gaps; GRID declares 79 compile errors upfront
  (llguidance-style honesty) and has **0 validation errors**, with its 66 invalidation
  errors all traceable to deliberately ignored value constraints (the XGrammar-default
  convention, itemized in the report).
- Fairness notes: XGrammar compiles this SQL grammar faster than GRID (52 ms vs
  227 ms on gpt2), and its GBNF-style grammar is a different encoding of the language
  (parity corners are counted, not asserted away — `bench/grammars.py`).

### 3.4 Where llguidance wins — honestly — and GRID's answer

llguidance (1.7.6) is the strongest engine in these comparisons and the bar GRID
measures itself against (`LESSONS.md` 4.4):

- **Flattest tails, best raw latency.** Its Earley/derivative core keeps SQL p99 at
  **223.8 µs** (gpt2) and **867.1 µs** (Qwen) with p50 **5.9 / 18.5 µs**
  (`bench/RESULTS.md`, `bench/RESULTS-qwen.md`); GRID's p50 is within ~2× but its
  cold-miss tail is milliseconds. On MaskBench, after the W1/W2 kernel round, GRID
  reaches p75 parity (27/39 µs vs 10/21) but llguidance's TBM max is 2.2 ms vs
  GRID's 128 ms — the cold trie walk remains the gap.
- **Compile speed and JSON maturity.** TTFM p50 **0.3 ms** vs GRID's 13.2 ms (was
  27.9 before the scanner-build fixes, `LESSONS.md` 5.5); 62 compile errors vs
  GRID's 79 (broader schema coverage), though with 3 validation errors where GRID
  has 0 (`bench/RESULTS-maskbench.md`).

GRID's answer, in measured-priority order (each item named by a benchmark, not by
aspiration):

1. **Kernel roadmap:** ~~widen the terminal masks past 64~~ (done, v3: 512 terminals,
   fallback schemas 0, TBM p99 202→30 ms) and ~~the scanner-build dominator~~ (done:
   alphabet compression + per-state ε-closures, TTFM p50 2.1× better); remaining:
   the cold trie walk (TBM p90 ~28 ms) and, if demand prices it in, a table-build
   kernel for the residual TTFM gap.
2. **Serving-cache amortization:** GRID's misses cluster at first-seen configurations
   and the write-back cache carries them across requests — the G8 serving benchmark
   (batching, CPU/GPU overlap, skip-a-round contract) is where the design choice pays,
   and no one-shot protocol can show it. The Spider ablation prices the cache at
   **32% of generation throughput** (`bench/RESULTS-spider-ablations.md`).
3. **The feature set llguidance does not offer:** mask-level role/schema projection,
   the replayable audit chain, guarantee statements with preconditions and gates, and
   requirement R as a committed, CI-checked contract.

### 3.5 State of the milestone

| Claim | Evidence | Host / conditions | Status |
|---|---|---|---|
| Guarantees implemented and gate-tested (masks sound/complete/terminating, RBAC verb+table, audit chain) | gate suite `tests/` (G0–G6 slices, G10a); `LESSONS.md` "Where this leaves us" | local CI + declared-runner scale arms | proven at slice AND at scale (G5 both arms, G6/G6(b), G10 — see the full-scale row below) |
| Kernels bit-identical to executable spec | `tests/trie/test_rust_parity.py`, `tests/mask/test_kernel_parity.py` (order-exact) | any (GRID_NO_RUST toggles) | proven |
| SQL mask latency vs engines (p50 beats XGrammar both tokenizers; within ~2× llguidance) | `bench/RESULTS.md`, `bench/RESULTS-qwen.md` | local dev, unpinned | measured on laptop + declared cloud runner; G9 KPI exceeded (ratios host-invariant) |
| Requirement R (flat cost vs position) | `bench/RESULTS-r.md` (slope CIs, 4 depths, n=16k, 20 seeds); warm-replay lines in `RESULTS*.md` | **H100 declared runner, kernel v4** | **Gate G7 MET on the binding host**: hit p50 7.6–8.9 µs (< 10 µs criterion; was 20.9–23.5 µs at v3), slope ≈ 0 all depths, R² ≥ 0.9989 |
| Kernel v4 (all four §2 symbols) | `grid_core/src/lib.rs` (persistent interned-stack arena + memos + `hit_pass`; `advance_frames` = `lalr_advance`), `tests/mask/test_kernel_parity.py` | any (GRID_NO_RUST toggles) | warm hit 11 → 3.5 µs; parity order-exact incl. config_hash |
| Kernel v5 (`fill_bits`: scheduler-side bitmask row fill) | `grid_core/src/lib.rs` (pre-packed per-entry ci bit words ++ live CD-pass bits ++ EOS written into vLLM's row in ONE FFI call); parity + poisoned-row-overwrite tests in `tests/mask/test_kernel_parity.py`; serving wiring: cold-only prefetch + shared per-template producer (LESSONS 6.5) | H100, vLLM 0.24 in-engine probe | in-engine warm fills p50 3–13 µs, batch-32 step 708 → 16 ms; parity bit-exact |
| Kernel v5.1 (verdict-equivalence CD grouping + shared DFS seg buffer + packed-row memo + bytes FFI) | `grid_core/src/lib.rs`, `grid/mask/cache.py`; soundness theorem in `tests/mask/test_verdict_equivalence.py` (group members verdict-indistinguishable at every configuration); `tests/mask/test_adaptive_encode.py` (numpy encode byte-identical, entry_id/G10-stable) | local M-series, Qwen 151k trie | cold CD-heavy mask build 124 → 13.3 ms (9.3×; ~55k singleton groups → ~1.4k verdict classes); warm fill p90 38 → 3.8 µs |
| Kernel v6 (session-in-kernel accept+fill) | `grid_core/src/lib.rs` sessions (own root→top chain, gen-tagged kidx cache, rollback deltas, ported maximal-munch lexer, memoized status/EOS derivation, (kidx,remainder)→handle bindings); serving gate: kernel present ∧ `audit is None` (audit/processor paths stay v5; `GRID_NO_V6=1` forces v5); differential suite `tests/models/test_v6_session.py` + 200-seed×3-grammar lockstep fuzz (zero divergence); also fixes the v5 post-COMPLETE resurrection (a non-eos token after eos revived a terminated request under spec decode) | local M-series, Qwen 151k trie | warm serving step 7.39 → **1.33 µs**/request (accept 6.16 → 0.56, fill 1.22 → 0.77); confirmed on the declared H100 SXM5 runner: TPOT overhead +0.12/+0.23/**+1.02%** @ batch 1/8/32 — the <2% gate PASSES |
| JSON-Schema domain (MaskBench) | `bench/RESULTS-maskbench.md` (post-W1/W2): TBM p50/p75 27/39 µs (p75 parity with llg/xgr), p99 30 ms, kernels on 100% of schemas, 0 validation errors, 79 declared compile errors; TTFM p50 13 ms | local dev, llama-3.1 tokenizer, 315 schemas | measured; remaining targets: cold trie walk (TBM p90), residual TTFM vs llguidance |
| Spider execution accuracy + repair | 7B full dev (`RESULTS-spider.md`, `RESULTS-spider-repair.md`): grid 91.3%/53.7 EX (reproduced exactly across H100→A10); **grid-repair 94.5% executes / 55.2 EX (+2.3 over unconstrained)**, 21% retry conversion, +14% tokens; 0.5B (`RESULTS-spider-05b.md`): mask +13 EX but feedback worthless (LESSONS 5.4a/b — the capability symmetry, measured both directions). Ablations EX-invariant; cache-off −32% throughput | greedy, H100 + A10 declared runners | measured; grid-repair joins the standard G9 arm set |
| Full-scale S/C/T + RBAC (G5/G6/G10) | `bench/RESULTS-g5-walk.md` (10k forced-random-walks: 10000/10000 parse, 0 dead-ends, 0 budget overruns, coverage quotas met), `bench/RESULTS-g10.md` (1k generations across a namespace rollover, bit-identical replay + 100% tamper detection), `bench/RESULTS-g6.md` (model-free adversarial speller: 0 RBAC bypasses, positive controls non-vacuous), `bench/RESULTS-g5-model.md` (model-in-loop, 1000/1000 parse + audit, 0 dead-ends), `bench/RESULTS-g6b.md` (12 injection prompts, 0 forbidden lexemes) | model-free: local dev; model-in-loop: H100 declared runner, Qwen2.5-0.5B | all arms green — Gate G5 (both arms), G6 + G6(b), G10 PASS |
| Serving: vLLM V1 backends + §6 contract (M6) | Mode 2 (logits processor, requires `async_scheduling=False`) and the **scheduler-side backend** (`grid/models/vllm_structured.py` + kernel v5 `fill_bits`) accepted on GPU (H100 + A10). Serving contract realized (`grid/serving/`): cold-only prefetch + E17 single-flight; G8 harness `bench/vllm_serving_bench.py` (warm-through cells, adversarial cold-miss, single-flight; unit-tested metrics); first-run +5151% batched-TPOT pathology diagnosed and fixed (LESSONS 6.5) | vllm 0.24, H100 declared runner, Qwen2.5-7B | `bench/RESULTS-serving.md` (kernel v6, H100 SXM5, 5-repeat, warm-through): G8 **5/7** — TPOT overhead **+1.02%** @batch 32 (+0.12% @1, +0.23% @8) **PASS** vs the <2% gate; TTFT cold 24.2 ms + warm 1.36 ms PASS; single-flight PASS; remaining red: the adversarial cold-miss pair (max step 50.3 ms vs 30 ms budget; co-batched degradation +233% over a 6.3 ms base) — a fresh schema's cold literal-interior walks (~13 ms residual each after v5.1's 9.3× cut) stall the batch because vLLM 0.24 exposes no RUNNING-request defer hook; open decision: re-scope the criterion to the measured walk envelope, or subtree-memoize cold walks / upstream a defer hook |
| Deferred by decision | `DESIGN.md` §13: GAD/CRANE, BIRD, byte-level jump-forward, Earley fallback, deadline fallbacks; T2 cache tier; SynCode/GBNF G9 arms | — | recorded with reasons |

### Further reading, in order

1. `GUARDRAIL-REDESIGN.md` — read §1's table first; every row is one snapshot
   element and its v0.0.7 replacement.
2. `DESIGN.md` §6 (the hot path) side-by-side with `grid/guide.py` and
   `grid/mask/producer.py`.
3. `LESSONS.md` end to end — it is the shortest honest history of why the code looks
   the way it does.
4. `grid/_reference/guide.py` — 140 lines that define what "correct" means here.
