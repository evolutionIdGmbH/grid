# 0.3.x candidate pool: 20 ideas against the measured TTFM tail

Generated from a 7-lens ideation pass (automata theory, systems, caching,
competitor forensics, grammar shape, staged compilation, pipeline) merged and
deduplicated; grounded in the bench/perfbench attribution run (scanner 98.6%
of tail compile time in two regimes; exponential $ref re-normalization in the
frontend on 5 schemas; one LALR hang). Selection of the implementation slate
is recorded in SELECTION.md.

## 1. Kernel-resident lazy scanner (lazy determinization in grid_core v8)  `[L]`

**Target:** scanner (build eliminated; cost moved to bounded on-demand walks) + runtime kernel

Move determinization into grid_core: construct the kernel from a compact NFA/derivative arena (Glushkov or Thompson bitsets plus per-NFA-state term-reach and accept masks, a few KB) instead of the dense trans table RustWalker ingests today (trie/walk.py:108-158), memoizing subset-bitset -> state-id in a hash map and materializing byte-class rows on first touch during trie walks. live[s] and accepts_all[s] are served exactly from NFA-local masks (candidate 11 is the prerequisite); h_max becomes |live[start]| since live is monotone non-increasing along transitions. State ids are insertion-ordered within one kernel instance so configuration-keyed masks and T1/T2 stay coherent (one numbering per template, never a mid-request swap); reserve.py's shortest_lexemes switches to the same BFS over NFA edges so nothing forces full materialization. Decisive bound: states materialize only along trie paths and emitted lexemes, so window-product states deeper than the longest vocab token (~16-30 bytes) — exactly the blowup states — are never built. Low-risk phase 1 (merged NFA-simulation bridge): serve first masks by pure-Python Thompson bitmask simulation over the token trie while the eager DFA builds in background, per-request pinning so cache keys never mix; keep _apply_scanner_budget firing byte-identically for recorded-set parity.

**Expected gain:** TTFM p99 4.37s and the 16x120s caps collapse toward p50 (~10ms): first-mask cost becomes O(states touched by the first trie walk), token-length-bounded even for the 196-terminal/145s case; TBM cold walks in the same schemas shrink too. llguidance p99 6.7ms is the existence proof.

**Correctness risk:** medium-high — same Myhill-Nerode states by construction, but: (a) h_max/INV-LEX1 must switch to the |live[start]| formulation; (b) reserve/audit paths that enumerate the DFA must be ported or they silently force full builds; (c) deterministic state ids under concurrent kernel walks need a lock or fixed work order; (d) recorded scanner-budget degradations must keep firing identically. Mask outcomes unchanged, so the v0.2.5 gate holds.

**Composes with:** NFA-derived live sets (delete the global live fixpoint); Counting-set scanner states for {m,n} windows; Bit-parallel bitset subset construction + interval mintermization; Rust scanner build behind a single-call arena FFI; Versioned on-disk compile-artifact store; Direct grammar-object emission + front-end fusion; AST-based scanner cost model with bounded-path routing
**Conflicts with:** Per-terminal DFA library with lazy product combination (alternative lazy-state representation for the kernel — pick one)
**Prior art:** RE2's lazy DFA with bounded state cache (Cox 2010); llguidance/derivre — lazily constructed lexer DFA serving grammar-constrained decoding at p99 6.7ms; GNU grep and Hyperscan hybrid engines; Thompson 1968 simulation for the bridge.

## 2. Counting-set scanner states for {m,n} windows (existing #2 concretized)  `[L]`

**Target:** scanner (NFA size + subset-state count for bounded repetition)

Replace _expand_repeat (dfa.py:32-43) for counted repetitions of single character-class atoms — the only shape length_body/int_range_rx emits — with a counted-loop NFA node carrying (m, n, counter-id), and determinize to a counting-set automaton: a scanner state becomes (control bitset, one interval [lo,hi] per active counter), since for letter-uniform loops the reachable counter sets are provably contiguous intervals, collapsing O(n) window positions to O(1) control states. Transitions carry counter ops (INC, saturate at n); accepts carry guards (exit iff hi >= m), so maximal munch and priorities are unchanged — accept-at-byte-i remains a per-state predicate, now guard-conditional. Runtime: scan_with_last_accept and the kernel walker gain counter registers (kernel v8 op stream), with a mirrored Python fallback; assert the interval invariant at build so any non-uniform shape falls back to expansion. Scoping for the gate: _apply_scanner_budget and rx_costs predicates stay byte-identical in phase 1 — the same terminals degrade, recorded sets frozen — so the gain applies to windows currently KEPT; lifting caps is a separately flagged coverage change.

**Expected gain:** 10-100x on window-bearing tail schemas: per-window NFA drops from ~20n to ~20 states and the window-x-keys subset product loses its dominant dimension — the measured (0,64) 0.45s build goes to low-ms, and the 2.3s/145s product cases collapse. Directly attacks the shared TTFM+TBM tail, where window schemas dominate both.

**Correctness risk:** medium — language equality for the counted-loop shape needs a differential harness (boundary lexemes at exactly m, m-1, n, n+1 chars; escapes; multi-byte UTF-8 straddling the count) against the expanded automaton; recorded-set drift is the gate hazard, avoided by freezing the budget predicates.

**Composes with:** Kernel-resident lazy scanner (lazy determinization in grid_core v8); Per-terminal DFA library with lazy product combination (counter automaton slots in as a component type); Bit-parallel bitset subset construction + interval mintermization; Rust scanner build behind a single-call arena FFI; AST-based scanner cost model with bounded-path routing
**Prior art:** Turonova et al., 'Regex Matching with Counting-Set Automata' (OOPSLA 2020); .NET NonBacktracking engine (Moseley et al., PLDI 2023) symbolic bounded counters; Hyperscan bounded-repeat subengines.

## 3. Runtime required-key bitset (linear object grammar, kill 2^R)  `[L]`

**Target:** schema_compile + lalr + runtime kernel (rule/state count; TBM cold-config collapse)

Replace the member_chain(S) subset machine (compiler.py:1048, capped at compiler.py:1027 by R<=10 and (1<<R)*(props+generics+1)<=40_000) with ONE linear member machine (members: member | members ',' member) of O(U) rules, plus a side table on LALRTables tagging each required pair's key terminal with bit i. grid_core keeps an R-bit seen-set stack in lockstep with the parse stack (push/pop at tracked object open/close); consuming key terminal K_k ORs its bit, and '}'-plus-EOS-through-'}' legality is admitted only when bits==FULL — provably the same token-level language as today's machine, which also lets seen keys repeat (S|b==S) and gates only the close. Mask factoring keeps caching sound without widening keys: entries for a member-position configuration differ across bit-values ONLY in '}'-crossing tokens, so the kernel stores one base entry plus a close-gate bit, and '}'-crossing tokens become context-dependent entries GRID's CD machinery already re-checks per step. Deploy only where 2^R is enforced today; R>10 schemas keep the 'required-not-enforced' record byte-for-byte, and ReserveTable/completion synthesis (grid/lalr/reserve.py, guide._completion_tokens) must become bits-aware so budget-stop outcomes don't change.

**Expected gain:** 10-100x rule-count reduction on required-heavy tail schemas (up to ~40k chain rules -> ~P+G), shrinking spec_load text, LALR item sets, and the up-to-2^R distinct cold configurations to one base entry + gate — plausibly the biggest single lever on the p99/120s bucket if attribution confirms 2^R dominance in LALR.

**Correctness risk:** high — (1) any token whose walk crosses '}' must be CD-checked against the register or masks go stale (false-accept, the forbidden class); (2) honesty contract: required-ness moves from grammar-enforced to runtime-enforced and must be declared as such; (3) register-stack/LALR-stack alignment through error paths; (4) completion synthesis must emit missing required pairs. Gate with full-corpus differential masks against the compiled 2^R machine for every R<=10 schema.

**Composes with:** LR(0) + DeRemer-Pennello LALR lookaheads (smaller input grammar); Kernel-resident lazy scanner (lazy determinization in grid_core v8); Literal-class set terminals (keys and enums); Direct grammar-object emission + front-end fusion
**Conflicts with:** Optional-run factoring of the 2^R member chain (superseded if this lands; keep as the low-risk fallback); Lazy LALR row materialization with background completion (not incompatible, but removes its main payoff)
**Prior art:** XGrammar's context-dependent tokens checked at runtime against persistent stack state (arXiv:2411.15100); XGrammar-2 repetition state compression; ANTLR semantic predicates; RELAX NG derivative-based interleave. Note: llguidance/XGrammar's forced-declaration-order transplant is forbidden here (measured false-rejects, compiler.py comment).

## 4. Per-terminal DFA library with lazy product combination (subsumes existing #4)  `[L]`

**Target:** scanner (replaces build_scanner's combined subset construction) + cross-schema amortization

Never determinize the union NFA: build one small trimmed (optionally Hopcroft/Moore-minimized) DFA per terminal, memoized process-wide keyed by exact pattern source — the identity the compiler already uses (rx_terms/key/lit dicts) — so STRING_RX, NUMBER_RX, WS, the 9 FORMAT_PATTERNS, and every repeated key/enum literal are built once per process ever; wheel-ship the enumerable common family (formats, structural literals, the budget-admitted (m,n<=64) window family) as package data regenerated-and-verified in CI. The combined scanner state becomes a sparse tuple of per-component states (dead components omitted), memoized tuple->id lazily on demand: accept = priority-min over accepting components (Terminal.priority is total, spec.py:55-60), accepts_all = the accepting set, and live[s] = {t : comp_t != DEAD} on trimmed components — the global fixpoint at dfa.py:440-455 is deleted. Assert per-component empty-match rejection (mirroring dfa.py:434-436) and canonicalize sparse tuples deterministically for stable ids; the kernel consumes the factored representation (v8) or the product stays Python-side. Blowup product states materialize only when an input prefix reaches them (token-length-bounded), which also delivers the live-set shrink that motivated per-subtree lexical modes without changing segmentation.

**Expected gain:** Cold scanner cost drops from 'product of everything' to 'sum of unseen components' (seconds -> ~10ms on the p99, same order as candidate 1), plus fleet-level amortization llguidance doesn't have: repeated formats/keys across the 11.3k corpus become near-free after first sight.

**Correctness risk:** medium — product semantics must reproduce priority ties and empty-match rejection exactly; Moore minimization must keep distinct accept-futures apart; deterministic sparse-tuple canonicalization needed for stable cache keys. Masks identical, so the outcome gate holds; verify by corpus-wide differential scan behavior (states renumber).

**Composes with:** NFA-derived live sets (delete the global live fixpoint); Counting-set scanner states for {m,n} windows; Bit-parallel bitset subset construction + interval mintermization; Rust scanner build behind a single-call arena FFI (rayon per-terminal parallelism); Versioned on-disk compile-artifact store; Hierarchical in-process memo tiers (substructure/signature caches)
**Conflicts with:** Kernel-resident lazy scanner (lazy determinization in grid_core v8) — alternative lazy-state representation; pick one
**Prior art:** RE2::Set and Hyperscan (per-pattern determinization + combined multi-pattern matching); on-the-fly product construction from model checking; Hopcroft & Ullman union-via-product; regex crate precompiled tables for the shipped artifacts.

## 5. Rust scanner build behind a single-call arena FFI (existing #3 pinned)  `[L]`

**Target:** scanner (whole build_scanner behind one FFI call)

Pin existing #3's port boundary so NOTHING per-state or per-set crosses PyO3: Python encodes one buffer of (pattern bytes, is_literal, priority) per terminal — patterns are already latin-1 byte regexes (dfa.py:330-333) — and grid_core returns three zero-copy buffers: flat i32 class-compressed trans arena, i32 accept array, and u64 terminal-bitmask word arrays for accepts_all/live (the exact format walk.py:115-116 hand-builds today). Inside Rust: port the ~140-line grid regex parser (dfa.py:53-190) bug-for-bug, u64-block bitset subset construction with hashbrown dedupe, and — combined with candidate 4 — rayon-parallel per-terminal determinization (independent builds; the only place 'parallel build' works, since CPython threads can't under the GIL). Keep the pure-Python builder as the differential oracle and gate on the full 11.3k corpus producing bit-identical arenas.

**Expected gain:** 50-200x on the construction loop vs CPython for this pointer-chasing set workload: the 0.45s single-window case goes to low-ms; the largest single constant-factor lever on a scanner-dominated p99, with rayon scaling many-terminal pathologies across cores.

**Correctness risk:** medium — reimplementation divergence (regex edge cases, equal-length priority tie-breaks, deterministic state numbering), bounded by the corpus-wide bit-identical differential gate; runtime hazard low because the boundary is data-only, no callbacks.

**Composes with:** Per-terminal DFA library with lazy product combination; Class-compressed flat-arena DFA end-to-end (the return format); Counting-set scanner states for {m,n} windows; Versioned on-disk compile-artifact store (cache arena bytes directly); Kernel-resident lazy scanner (lazy determinization in grid_core v8) — this is its eager sibling; shared Rust regex/NFA code; Prefill-overlap async compile (subprocess pool + phase fork) — releases the GIL, making overlap real
**Conflicts with:** Bit-parallel bitset subset construction + interval mintermization (redundant once this lands; keep as interim and oracle fast path)
**Prior art:** rust regex-automata dense DFA construction (bitset subset construction, byte classes, one flat table); XGrammar's C++ core; grid_core v7 already consumes the same blob shape (trans.tobytes() at trie/walk.py:158), so the boundary style is proven in-repo.

## 6. LR(0) + DeRemer-Pennello LALR lookaheads (drop canonical-LR(1)-then-merge)  `[M]`

**Target:** lalr (compile_tables)

compile_tables builds the full canonical LR(1) machine — items are (prod, dot, la) triples, closure() re-derives first_seq(rhs[d+1:], la) per item per call (compile.py:106-119), states are keyed by hashing whole closed frozensets (compile.py:127-144) — then merges by core anyway (compile.py:146-164), discarding the lookahead-multiplicity excess. Replace with the standard route: build the LR(0) automaton directly over (prod, dot) cores (these ARE the final LALR states, no merge pass), then compute exact LALR(1) lookaheads via DeRemer-Pennello DR/reads/includes/lookback relations closed with one Tarjan-SCC digraph traversal, near-linear in LR(0) transitions. Emit REDUCE entries only under computed lookaheads (no Bison-style default reductions — viable-prefix masks derive from action-row keys) and keep state_items as full closures since reserve (E4a) and completion synthesis need them. The 2^R member-chain grammars are the worst case for canonical splitting (shared cores, wildly different key-terminal lookaheads), so this lands exactly on the tail. Absorbed follow-up (json-skeleton fragment splice): only if generic-heavy schemas still dominate post-DP, precompile the ~12-production json_value skeleton's LR(0) fragment once per process and splice with terminal-id remap.

**Expected gain:** 5-20x on the LALR phase (believed #2 contributor; factor conditional on the attribution run), largest as R approaches the 10-key cap; item counts fall from core-x-lookahead splits to exactly the LALR state count and giant-frozenset hashing disappears.

**Correctness risk:** medium — LALR(1) is uniquely defined so action/goto CONTENT and the conflict SET are construction-independent, but: (a) state renumbering must be deterministic (kernel caches are per-instance, safe, yet audit anything persisted on state ids); (b) LALRConflictError reports embed state ids (compile.py:182,200), so the error-outcome comparator must key on error class, not text; (c) reads/includes nullable-handling is the classic implementation bug. Gate: differential run asserting isomorphic action/goto and identical conflict membership over all 11.3k schemas.

**Composes with:** Runtime required-key bitset (linear object grammar, kill 2^R); Lazy LALR row materialization with background completion (shares the digraph machinery); Direct grammar-object emission + front-end fusion (int-id productions are its natural input); Versioned on-disk compile-artifact store; Rust scanner build behind a single-call arena FFI (port DP, not LR(1))
**Prior art:** DeRemer & Pennello, TOPLAS 1982; Bison's production implementation for 40 years; PLY as a proven pure-Python implementation; Bravenboer & Visser parse-table composition for the splice follow-up.

## 7. Bit-parallel bitset subset construction + interval mintermization  `[M]`

**Target:** scanner (subset construction inner loop)

Representation-only rewrite of build_scanner's inner loop, semantics untouched: (a) eps_star[s] becomes an int bitmask and eps-closure is pre-folded into the move table (move_star[st][cl] = OR of eps_star[dst]), deleting the per-DFA-state eps_closure(frozenset(dsts)) call at dfa.py:424; (b) the DFA dedupe dict keys on the int itself (word-parallel hash/eq vs O(|set|) frozenset hashing at dfa.py:406,425); (c) accepts_all extracted by ANDing a precomputed accept mask and iterating set bits. (d) Memoize charset->class-list by id() of the shared chars frozenset when building edge_by_class (dfa.py:389-403) — _expand_repeat copies share the same chars object, so today each of the m window copies re-iterates ~250 chars per class edge — and compute alphabet equivalence classes from sorted byte-interval boundary points instead of per-byte set partition refinement (dfa.py:368-379). Optionally swap Thompson for a Glushkov position automaton (no eps edges at all); preserve BFS discovery order and min-byte class sort (dfa.py:380) so state numbering and emitted tables stay byte-identical.

**Expected gain:** 3-20x constant factor in CPython on the believed-dominant phase, applying across the whole p90-p99 scanner tail including the 0.45s window and 145s pathological cases (faster but still super-linear — pair with candidate 2 for the asymptotics); also cuts peak memory. This is the S/M-effort hedge that pays off regardless of what the attribution run says.

**Correctness risk:** low — identical algorithm and visit order, only set representation changes; hazards are bit-iteration order (rows/accepts/live are order-insensitive) and state-numbering determinism, both fixed by keeping insertion order. Differential gate: identical trans/accept/accepts_all/live tuples on the full bench.

**Composes with:** NFA-derived live sets (delete the global live fixpoint); Kernel-resident lazy scanner (lazy determinization in grid_core v8); Per-terminal DFA library with lazy product combination; Counting-set scanner states for {m,n} windows; Class-compressed flat-arena DFA end-to-end; Versioned on-disk compile-artifact store
**Conflicts with:** Rust scanner build behind a single-call arena FFI (this becomes the interim and the oracle's fast path once the port lands)
**Prior art:** Bitset subset construction (Aho/Sethi/Ullman); Hyperscan LimEx bit-parallel Glushkov; Navarro & Raffinot bit-parallel automata; interval mintermization from symbolic automata (D'Antoni & Veanes; brics/MONA); bigint-as-bitset is the established CPython idiom.

## 8. Versioned on-disk compile-artifact store (existing #5 concretized)  `[M]`

**Target:** whole pipeline (all phases, warm deployments)

Two-tier store using keys the code already computes. Tier 1: blake2b(canonical schema JSON + strict flag + compiler epoch) -> (.grid source or grammar objects, recorded-unenforced set) from compile_schema; Tier 2: DialectGrammar.fingerprint (spec.py:231) -> pickled ScannerDFA, and LALRTables.fingerprint ('g.fingerprint:role_shape_hash', compile.py:216) -> pickled LALRTables — all frozen dataclasses of tuples/frozensets/dicts, picklable as-is. Store under ~/.cache/grid/<code_epoch>/<key>.bin with atomic tmp+rename; code_epoch = hash of package version + content hashes of dfa.py/compile.py/compiler.py/normalize.py so any engine change invalidates wholesale. Wire into _GuideRegistry._build (vllm_processor.py) and grid/generate/__init__.py before spec.load/build_scanner/compile_tables; SingleFlight already dedupes concurrent cold builds; store source text alongside tier-2 blobs and verify fingerprint on load.

**Expected gain:** Repeat-deployment TTFM p99 collapses from 4.37s to unpickle time (~5-50ms); the 16 pinned schemas become one-time costs per deployment. Zero help on true first compile — bench honesty requires reporting cold-run numbers separately; this amortizes production, it does not fix the cold algorithm.

**Correctness risk:** low-medium — the hazard is stale/mismatched artifacts (a missed key dependency serves a wrong grammar silently, the forbidden class); mitigated by the code-epoch key covering every phase's source, fingerprint verification on load, and self-produced pickles in a user-owned dir.

**Composes with:** Hierarchical in-process memo tiers (substructure/signature caches) — same keys, persisted; Per-terminal DFA library with lazy product combination (persists the component library); Prefill-overlap async compile (subprocess pool + phase fork) — async path checks disk first; Persistent T2/ContextJournal warm-set restore (same store, different namespace); Rust scanner build behind a single-call arena FFI (key on epoch; cache arenas)
**Conflicts with:** Kernel-resident lazy scanner (a partially-built lazy DFA is not a persistable artifact unless the explored frontier is snapshotted — needs an explicit merge strategy)
**Prior art:** outlines' on-disk FSM cache (regex+vocab keys); XGrammar CachedGrammarCompiler; ccache/bazel content-addressed action caches; pip wheel caches.

## 9. AST-based scanner cost model with bounded-path routing  `[M]`

**Target:** schema_compile budget policy -> scanner construction strategy

Replace the proxies in _apply_scanner_budget (len(src), cost=40*max_l, blanket n_terms>80, _SCANNER_BIG=600) with a real estimator computed at emission: parse each rx body once, take per-terminal NFA position counts p_i and window widths w_i, and bound subset states as (product of (w_i+1) over co-resident length windows) x (sum of p_i over terminals sharing the '"' first-byte class) — only shared-first-byte position machines multiply; numeric/structural terminals never product with string terminals. Stage A (gate-safe): keep every v0.2.5 degradation decision bit-identical and use the estimate only to ROUTE kept-but-expensive terminals to a bounded construction path — counting-set windows (candidate 2), lazy determinization (candidate 1), or per-terminal product (candidate 4) — instead of eager expansion; records and routing_terms untouched. Stage B (explicitly flagged, outcome-changing): degrade minimally by greedy largest-marginal-cost to a calibrated ~50ms budget, recovering enforcement lost to the blanket n_terms>80 rule — ships only as a declared coverage change outside the gate.

**Expected gain:** Stage A deterministically converts the residual pathological keepers — the 0.45s-per-window and 145s-class cases today's thresholds happen to keep — into bounded builds: this is the policy layer that guarantees the 16x120s cap pins are actually eliminated rather than accidentally retained. Tail-cap improvement, not a median improvement.

**Correctness risk:** high for stage B (ANY change to which terminals degrade changes recorded sets, pinned exactly by the gate — cannot ship inside it); low for stage A (identical decisions and records; only construction strategy changes), inheriting the correctness burden of whichever bounded path it routes to.

**Composes with:** Counting-set scanner states for {m,n} windows; Kernel-resident lazy scanner (lazy determinization in grid_core v8); Per-terminal DFA library with lazy product combination; Rust scanner build behind a single-call arena FFI
**Prior art:** RE2's max_mem budget with automatic fallback strategy selection; Hyperscan compile-time resource models choosing an engine per pattern class.

## 10. Direct grammar-object emission + front-end fusion (existing #6 sharpened)  `[M]`

**Target:** schema_compile emit + spec_load (eliminated) + projection (fast-pathed)

Kill the text->reparse round trip and the two provably-redundant passes behind it. Compiler keeps rules as token TUPLES internally (also stops _dedupe_rules re-splitting alternative strings every fixpoint pass) and a new DialectGrammar.from_parts constructs Terminal/Production objects directly, replaying freeze()'s numbering exactly (named terminals by decl_index then literals by first-use order, spec.py:110-117) and matching _parse_rhs escaping so Terminal.pattern is identical; fingerprint is unaffected because _fingerprint hashes terminals/productions/start/ignored, not source (spec.py:231). validate() still runs (GrammarInvalid outcomes are load-bearing recorded outcomes), but RoleProjection.full().build()'s four redundant _prod_key formatting passes and reduce/verify fixpoints (projection.py:43-82) collapse to a trusted full_built fast path — reducedness is already guaranteed by the compiler's prune plus spec._validate — keeping role_shape_hash byte-identical since it feeds LALRTables.fingerprint (compile.py:216) which keys kernel configurations and T1/T2. Remaining analyses (validate's useless_symbols; non-full roles) switch from while-changed rescans (reduction.py:25,39) to linear counter-worklist productivity + BFS reachability. Hand compile_tables int-id productions to delete the per-RHS-symbol string dispatch (compile.py:63-64); keep the text emitter as a lazy property and CI differential oracle (emit + reparse + assert fingerprint/terminal_order equality over all 11.3k schemas).

**Expected gain:** Removes the observed spec_load re-parse and the projection phase outright: an estimated 10-30% (hundreds of ms) on the p90-p95 band where 10-40k-rule grammars hit MB-scale text; negligible on subset-construction-bound p99 pins; also cuts peak RSS.

**Correctness risk:** low-medium — the hazard is silent object-path/text-path divergence (literal decl_index, escaping, terminal_order drift changes terminal ids and hence masks), pinned by the fingerprint-equality differential gate; validation must keep running so GrammarInvalid outcomes stay in the recorded set.

**Composes with:** LR(0) + DeRemer-Pennello LALR lookaheads; Hierarchical in-process memo tiers (substructure/signature caches) — stable fingerprints are its keys; Versioned on-disk compile-artifact store; Structural hash-consing + worklist dedupe in schema_compile; Runtime required-key bitset (linear object grammar, kill 2^R)
**Prior art:** Standard compiler practice — build the IR, never reparse your own pretty-printer output (LLVM IRBuilder); llguidance loads grammars into flat 32-bit-int arrays with 2ms startup; textbook linear-time CFG usefulness (Hopcroft & Ullman; Grune & Jacobs).

## 11. NFA-derived live sets (delete the global live fixpoint)  `[S]`

**Target:** scanner (live-set computation); prerequisite for candidates 1 and 4

The live[] computation at dfa.py:440-455 claims 'reverse-topological' in its comment but actually iterates states in FORWARD discovery order (for s in range(n)) while liveness flows backward from accept states, so deep window/chain DFAs need O(depth) full while-changed passes of Python frozenset unions. Replace it: one reverse DFS over the combined NFA (reversed eps + byte edges) computes term_reach[q] = int bitmask of terminals whose accept state is reachable from q — O(NFA edges x words) — then live(S) for any subset state S is the OR of term_reach[q] over q in S, computable AT STATE CREATION with zero global knowledge. Provably identical: the subset state after word w is exactly the union of NFA states reachable via w, so terminal-accept reachability distributes over the union; accepts_all is already local, the empty-match check (dfa.py:434) needs only the start state, and h_max = |live[start]| since live is monotone non-increasing. Verify by asserting live/accepts_all/h_max equality against the old fixpoint over the full 11.3k bench before deleting it; cheapest fallback if deferred: SCC condensation + one true reverse-topological pass with int masks.

**Expected gain:** Standalone 1.2-2x on the scanner phase for deep window DFAs (share pending attribution); strategically it is the exactness prerequisite that lets every lazy/on-demand scanner serve viable-prefix hypothesis sets without a global pass — candidates 1 and 4 cannot be exact without it.

**Correctness risk:** low — the equivalence is a two-line subset-construction theorem; output representation (frozensets in ScannerDFA) unchanged; only subtlety is h_max availability timing for partial DFAs (interim bound = terminal count; internal invariant only).

**Composes with:** Kernel-resident lazy scanner (lazy determinization in grid_core v8); Per-terminal DFA library with lazy product combination; Bit-parallel bitset subset construction + interval mintermization; Rust scanner build behind a single-call arena FFI
**Prior art:** Rabin-Scott subset-construction invariant (language of a subset state = union of member languages); RE2/rust regex-automata derive per-state match/live flags from NFA-local data during lazy determinization (Cox 2010); Tarjan 1972 / dataflow evaluation ordering for the fallback.

## 12. Lazy LALR row materialization with background completion  `[L]`

**Target:** lalr (+ kernel incremental state registration)

Keep an eager, cheap LR(0) core map (shared substrate with candidate 6; cores are what state_items at compile.py:214 stores, so reserve/E4a keeps working), then materialize action/goto ROWS only when decode or the trie walk first enters a state: closure plus DeRemer-Pennello lookaheads computed on demand, with lookback/includes digraph edges pulled lazily but TRANSITIVELY COMPLETELY per queried row (an includes edge from a never-visited state still contributes) and memoized. Rows containing a completable item BLOCK until the global lookahead propagation finishes in a background worker whose output equals the eager tables; masks served early are exact because shift structure is final at row creation. Conflict-timing parity is the crux: first prove on the v0.2.5 bench records that compiler-emitted JSON grammars raise ZERO LALRConflictError (the _dedupe_rules pass exists to kill the reduce-reduce class) and gate lazy mode to that family, or withhold outcome classification until background completion — otherwise a late conflict lands in a different outcome bucket, violating the gate. Coarser-grained variant absorbed: per-property-value modular submachines compiled on first parser entry (uniform follow {',', '}'} makes module linking tractable), keeping the single global union scanner untouched.

**Expected gain:** LALR share of TTFM becomes O(rows visited before the first token) — near-zero upfront, unbounded factor on 2^R-heavy schemas if candidate 3 is not adopted; the only LALR idea that changes p99 asymptotics rather than constants. XGrammar-2's JIT mask cache (same defer-to-decode principle) bought 8.1x preprocessing reduction.

**Correctness risk:** high — error-TIMING parity (late LALRConflictError = different v0.2.5 outcome bucket; must be proven vacuous or classification withheld) and lookahead-restriction bugs (per-query digraph pulls must be transitively complete); reserve/audit modes need an eager fallback flag.

**Composes with:** LR(0) + DeRemer-Pennello LALR lookaheads (same digraph machinery; DP is the eager fallback); Kernel-resident lazy scanner (lazy determinization in grid_core v8) — same lazy runtime shape; Versioned on-disk compile-artifact store (completed artifacts persist)
**Conflicts with:** Runtime required-key bitset (linear object grammar, kill 2^R) — removes this idea's main payoff; pick one as primary for the 2^R tail
**Prior art:** Heering, Klint & Rekers, 'Incremental Generation of Parsers' (IEEE TSE 1990); XGrammar-2 JIT compilation of the token mask cache (arXiv:2601.04426); ANTLR ALL(*) memoized parse-time analysis; llguidance's lazy construction (p99 6.7ms) as the existence proof.

## 13. Prefill-overlap async compile (subprocess pool + phase fork)  `[M]`

**Target:** serving integration (latency hiding + tail isolation; artifacts unchanged)

Move the whole compile pipeline off the request path: at request admission, submit the schema to a pre-warmed ProcessPoolExecutor (subprocess, not thread — all phases are pure-Python CPU-bound and fight the vLLM V1 scheduler for the GIL; with the Rust port a GIL-releasing thread suffices); the request sits in WAITING_FOR_GRAMMAR and GridGuide blocks on the future only at its FIRST decode step, i.e. after prefill it had to wait for anyway. The child returns pickled frozen dataclasses (LALRTables, ScannerDFA, recorded set) or the pickled exception; Unsupported/GrammarInvalid/LALRConflictError re-raise at join, mapped to the identical per-schema outcome bucket as v0.2.5. Absorbed sub-idea: fork LALR and scanner inside the compile — compile_tables reads only productions, build_scanner only terminals (dfa.py:320) — so TTFM = front-end + max(t_lalr, t_scanner) instead of the sum, with double-raise resolution fixed to the sequential pipeline's order so the recorded error class matches. SingleFlight dedupes concurrent cold builds; memo/table caches must be thread/process-safe and cancellation must not poison the negative cache; VLLM_ENABLE_V1_MULTIPROCESSING=0 in-engine probes constrain executor choice.

**Expected gain:** User-visible TTFM at p90 (217ms) and most of p95 (672ms) disappears under typical multi-hundred-ms prefill; p99 shrinks by one prefill length; co-batched requests stop inheriting a tail schema's compile stall (blast-radius isolation). Measured compile-start TTFM unchanged — report both honestly.

**Correctness risk:** low — no algorithm changes, artifacts bit-identical (differential-test fingerprints parent vs child); hazards are error-timing classification at join and concurrency discipline around SingleFlight/caches.

**Composes with:** Versioned on-disk compile-artifact store (async path checks disk first); Rust scanner build behind a single-call arena FFI (thread instead of subprocess); Persistent T2/ContextJournal warm-set restore (warmup runs in the same background window); Kernel-resident lazy scanner (lazy determinization in grid_core v8)
**Prior art:** vLLM V1 StructuredOutputManager compiles XGrammar/llguidance grammars asynchronously in an executor with requests gated at first constrained decode; TensorRT-LLM guided decoding; standard build-DAG parallelism for the fork.

## 14. Literal-class set terminals (keys and enums)  `[M]`

**Target:** schema_compile emission -> scanner accept/live bookkeeping + LALR columns + kernel terminal count (TBM)

Stop minting one terminal per property key and per enum/const value. Partition an object's keys by equivalence class (value rule name from rule_for(v), required-bit, and GLOBAL membership signature — _key_term at compiler.py:178 is global per compilation, so a key name appearing in two objects with different value schemas must keep a private terminal or maximal-munch declaration-order tie-break silently mis-routes) and all enum literals by ownership signature (set of enum sites containing them); each class becomes ONE named terminal whose pattern is rx.literals_body(members), one unit in the member chain, one alternative in the enum rule. The parser never distinguished values within a class (all alternatives reduce to the same nonterminal), so the language is unchanged — the combined NFA is the same union with coarser accept TAGS, which is exactly where the cost lives: accepts_all/live sets shrink from O(#values) to O(#classes), and LALR terminal columns drop. Merged terminals take the min decl_index of their members so priority ties vs STRING/E/LIT terminals are preserved (Terminal.priority = (is_literal, decl_index), spec.py:55-60); evaluate the (1<<R)*(...)>40_000 cap predicate (compiler.py:1027) on the UNMERGED counts so the set of schemas receiving 'required-not-enforced' records stays bit-identical.

**Expected gain:** 2-10x on grammar+LALR size for machine-generated object-heavy tail schemas (k8s/GitHub-Actions style: hundreds of props sharing a handful of value shapes) and 5-50x on enum-heavy scanner bookkeeping; brings many tail schemas back under the kernel's 512-terminal cap, attacking the shared TTFM+TBM tail. No effect on heterogeneous schemas.

**Correctness risk:** medium — the signature-collision hazard (silent mis-routing via priority tie-break = false rejects) plus re-verification of _pattern_minus_keys/extras key-name interactions; enum merging must happen after the static sibling-constraint filter so kept-sets are unchanged. Recorded sets untouched (no enforcement change) given the cap-predicate discipline.

**Composes with:** Runtime required-key bitset (linear object grammar, kill 2^R); Optional-run factoring of the 2^R member chain; Per-terminal DFA library with lazy product combination; Kernel-resident lazy scanner (lazy determinization in grid_core v8); Tokenizer slicer (per-vocab slice masks + sub-tries in grid_core)
**Prior art:** Classic lexer keyword-class technique (one terminal + membership table); llguidance compiles lexeme sets and enums as single regex alternations; Aho-Corasick literal tries prove the merged automaton stays linear.

## 15. Optional-run factoring of the 2^R member chain (low-risk fallback)  `[S]`

**Target:** schema_compile object machine -> spec_load text size + LALR item-set count

Restructure member_chain (compiler.py:1048-1068) without changing the language: today every subset state S inlines all U pair sources twice (bare + comma-continuation). Instead (a) emit each pair once as a named rule pair_k: K_k ':' v_k; (b) emit ONE shared rule opt_member covering all bit-0 units (optional + generic pairs); (c) state S's rule becomes the required-pair transitions (pair_k ',' m{S|bk}, bare pair_k iff S|bk==FULL, self-loops for already-seen required keys preserving today's duplicate-key acceptance) plus opt_member ',' m{S} and bare opt_member iff S==FULL. Same subset automaton with transitions grouped by target, so LALR(1) adequacy is unchanged (alternatives still keyed by disjoint key terminals, same ','-vs-'}' lookahead pattern); alternatives per state drop from ~2U to ~R+3 and emitted text stops repeating every pair source 2^(R+1) times. Keep the 40k cap predicate computed on the OLD unfactored size so the 'required-not-enforced' record set is bit-identical to v0.2.5.

**Expected gain:** Productions go from 2^R*2U to 2^R*(R+3)+U: at the cap boundary (R=10, U=38) ~6x fewer productions and ~10x less emitted text for that subtree, with LR closure work shrinking proportionally; the safe S-effort win on p95-p99 object-heavy schemas if (or until) candidate 3 lands.

**Correctness risk:** low — same language by construction; the one gate hazard (cap-predicate drift) is handled by evaluating it on unfactored counts.

**Composes with:** Literal-class set terminals (keys and enums); Direct grammar-object emission + front-end fusion; Structural hash-consing + worklist dedupe in schema_compile; Versioned on-disk compile-artifact store
**Conflicts with:** Runtime required-key bitset (linear object grammar, kill 2^R) — supersedes this if it lands; keep this as the low-risk fallback
**Prior art:** Standard grammar left-factoring; RELAX NG interleave implementations (Clark's derivatives) handle permutation languages without per-subset alternative expansion.

## 16. Hierarchical in-process memo tiers (substructure/signature caches)  `[S]`

**Target:** schema_compile + spec_load + scanner + lalr (cross-schema, warm process)

Four content-addressed memo layers above and below whole-schema granularity, all process-global dicts over frozen objects. L0: compile_schema memo keyed by blake2b(json.dumps(schema, sort_keys=True) + strict) — exactly the function's input, no content rewriting (annotation-stripping variants rejected: $ref can path into annotation subtrees). L1: pattern text -> parsed regex AST/NFA fragment (the 9 FORMAT_PATTERNS and STRING_RX/NUMBER_RX/INT_RX recur in nearly every schema and are re-parsed per schema today). L2: ScannerDFA keyed by the ordered (pattern, is_literal) signature — provably the only inputs build_scanner reads (dfa.py:320-336; priority = (is_literal, decl_index) = position in terminal_order), so names stay out of the key; guard future drift with a rebuild-and-compare assertion on a sampled corpus. L3: LALRTables keyed by a shape digest (terminal_order names, per-terminal literal/ignored flags, start, id-encoded prods, identifier_terminals — everything compile_tables:50-69 reads), returning dataclasses.replace(cached, fingerprint=f"{g.fingerprint}:{proj.role_shape_hash}") so downstream T1/T2/kernel keys stay schema-specific; spec.load memo by source hash covers the interim text path.

**Expected gain:** 2-5x aggregate on the p90-p99 band for repeated-structure workloads (XGrammar-2 measured 42-51% substructure reuse when whole-grammar reuse is 0%, and +2.2x from cross-grammar caching); near-free warm hits in serving. No help on a first-ever pathological schema — p100 belongs to candidates 1/2/4.

**Correctness risk:** low-medium — pure memoization of frozen objects; the specific hazard is key under-specification (terminal ORDER and flags must be in L2; L3 must include everything table construction reads or tables silently alias across schemas, and the fingerprint MUST be replaced per schema). Debug mode recompiles on hit and asserts equality.

**Composes with:** Versioned on-disk compile-artifact store (same keys, persisted); Per-terminal DFA library with lazy product combination (L1/L2 are its exact-match tiers); Direct grammar-object emission + front-end fusion (stable fingerprints are the keys); LR(0) + DeRemer-Pennello LALR lookaheads
**Prior art:** XGrammar-2 (arXiv:2601.04426) hierarchical FSM hashing; outlines' one-FSM-per-regex process cache; hash-consing/interning; RFC 8785 canonical JSON hashing; ccache/sccache content addressing.

## 17. Structural hash-consing + worklist dedupe in schema_compile  `[S]`

**Target:** schema_compile (generation + dedupe) -> also fewer productions into spec_load/lalr pre-dedupe

Two coupled fixes in the compiler. (a) rule_for memoizes by id(schema) today (compiler.py:226ff), so repeated INLINE subtrees — endemic in machine-generated tail schemas — regenerate whole rule families that _dedupe_rules only merges post-hoc: add a structural memo keyed by blake2b(json.dumps(node, sort_keys=True)) -> rule name for closed subtrees (nodes on the compile stack keep the id() path so recursion still terminates via pre-registration), and memoize _counted_seq chains globally by (item_rule, m, n, brackets). (b) Replace the _dedupe_rules full-rescan fixpoint (pass count grows with alias-chain depth, up to ~256 for cont/tail chains, each pass touching all rules) with signature-based partition refinement over the rule graph's SCC condensation in reverse topological order — one bottom-up pass plus local refinement for cyclic SCCs, reproducing today's exact merge fixpoint with the same canonical representative (first in rule_order) so emitted text and fingerprints stay byte-identical.

**Expected gain:** 2-10x on the schema_compile phase for repetitive tail schemas and removes a potential O(depth x rules) dedupe blowup; gain contingent on attribution showing schema_compile/dedupe as a nontrivial tail share — cheap enough (S) to do regardless.

**Correctness risk:** low — structural equality implies identical language (x-grid-* extension keys and annotations participate in the hash; $ref resolution is root-global so equal text resolves equally); _record side effects are set-idempotent; byte-identical output is directly assertable.

**Composes with:** Optional-run factoring of the 2^R member chain; Direct grammar-object emission + front-end fusion; Versioned on-disk compile-artifact store; Literal-class set terminals (keys and enums); Hierarchical in-process memo tiers (substructure/signature caches)
**Prior art:** Ershov hash-consing / BDD unique tables; Downey-Sethi-Tarjan congruence closure; Hopcroft partition refinement for the cyclic case.

## 18. Class-compressed flat-arena DFA end-to-end  `[M]`

**Target:** scanner emit + runtime mask serving (the shared TTFM+TBM tail)

build_scanner already runs the subset construction per byte-CLASS but then expands rows back to 256-wide tuples (dfa.py:422-431: row = [DEAD]*256, per-class fill) — which trie/walk.py immediately re-converts via np.array(dfa.trans, dtype=np.int32) and ships as trans.tobytes() to grid_core (walk.py:158). Instead build ONE flat np.int32 arena [n_states x n_classes] directly during construction plus a 256-byte class_of map; ScannerDFA.next becomes trans[state*n_classes + class_of[byte]]. Update the four Python consumers (guide.py, lexer/run.py, lalr/reserve.py, trie/walk.py) and the grid_core blob layout to walk class-compressed tables (kernel changes legal in 0.3.x); store accepts_all/live natively as u64 terminal-bitmask word arrays — the format _term_words already converts to for the kernel (walk.py:115-116). This is also the natural return format for the Rust build (candidate 5).

**Expected gain:** 10-20% of TTFM on large-DFA schemas (tuple materialization + numpy reconversion eliminated); bigger payoff on TBM: tables shrink ~10-25x (observed ~10-30 byte classes), dropping the cold-miss trie-walk working set toward L2 on exactly the schemas where the TBM tail lives.

**Correctness risk:** medium — mechanical but wide (indexing change across four consumers plus the kernel blob format); hazard class is off-by-class/DEAD-handling bugs, mitigated by tests/lexer/test_scan_last_accept.py and the Python-vs-kernel walk parity harness.

**Composes with:** Rust scanner build behind a single-call arena FFI (this arena IS its return format); Bit-parallel bitset subset construction + interval mintermization; Per-terminal DFA library with lazy product combination; Kernel-resident lazy scanner (lazy determinization in grid_core v8) — lazy rows are class-compressed from birth; Versioned on-disk compile-artifact store
**Prior art:** flex/re2c equivalence-class tables (yy_ec indirection); RE2 dense DFAs use byte-class-compressed transition tables for the same locality reason.

## 19. Tokenizer slicer (per-vocab slice masks + sub-tries in grid_core)  `[L]`

**Target:** runtime kernel — cold-miss TBM tail, plus the first mask inside TTFM

Once per PROCESS per tokenizer (amortized across every request, unlike all per-schema work): define slice regexes over the 128k vocab (llguidance's JSON set: [^"\\\x00-\x1f]{1,10}, {1,30}, unbounded, plus the remainder), assign each token to the first containing slice, and build per-slice sub-tries plus one precomputed full bitmask per slice. At mask time in RustWalker, before the cold trie walk over the string-content subtree, test each slice for CONTAINMENT in the currently viable lexeme-continuation language — with grid's regex subset this is structural, not automata-theoretic: slice byte-class ⊆ terminal byte-class AND slice max length ≤ remaining window bound, evaluated against live[] at the current DFA state (or the counter interval under candidate 2). On success, OR the precomputed slice mask into the result and skip the entire subtree; walk only the remainder. Must be a proof-carrying under-approximation (skip only when structurally provable), and slices must exclude bytes that can close/escape the lexeme ('"', '\\', control bytes) so no forced-emission event can occur inside a sliced token, preserving maximal munch.

**Expected gain:** String-content tokens are the bulk of a 128k vocab: expect 5-20x on cold-miss trie walks (TBM p99.9 8ms -> sub-ms), concentrated exactly in the shared TTFM+TBM tail schemas; direct TTFM effect limited to the first mask's cold walk. llguidance reference: ~50us masks with the parser touched in 0.1-1% of token checks.

**Correctness risk:** medium — the hazard class is false ACCEPT (the worst one) from unsound containment (e.g. slice {1,30} vs a (0,20) maxLength window, or vs a pattern-complement body); bounded by the structural-proof-or-walk rule and the existing walk-vs-Python differential harness.

**Composes with:** Counting-set scanner states for {m,n} windows (containment vs counter intervals); Kernel-resident lazy scanner (lazy determinization in grid_core v8); Per-terminal DFA library with lazy product combination; Persistent T2/ContextJournal warm-set restore (slicer fills what warmup misses)
**Prior art:** llguidance's slicer (llg-go-brrr post; shipped in llguidance, used by vLLM's guidance backend): tokens assigned to regex-defined slices, per-slice tries, precomputed masks OR'd in when the slice is contained in the currently allowed lexemes.

**STATUS (Wave B): scoped v1 LANDED** as S2-slicer (`GRID_PERF_SLICER`, default off pending the H100 stamp): ONE vocab-wide unbounded JSON-string-safe slice + rest-trie, containment proved by closure BFS in the walk (kernel + Python spec), all-or-nothing per configuration so output stays byte-identical (blob + entry_id included). Measured: string-interior cold walks 6.83ms -> 0.269ms (25.3x); maskbench sample p95 7.7ms -> 0.9ms, pooled -66%, zero outcome deltas. The length-capped slices ({1,10}/{1,30} vs maxLength windows, needing a group-order canonicalization decision) and lazy-DFA slicing stay follow-on — see ROADMAP.md S2 postscript.

## 20. Persistent T2/ContextJournal warm-set restore  `[M]`

**Target:** runtime TBM tail (first-mask and cold-walk latency of tail schemas on redeployment)

The TBM tail (cold-miss trie walks) is a measured strict subset of the TTFM-tail schemas, and the pre-walk machinery already exists: per-dialect ContextJournal (W4) + admission warmup (W5) in vllm_processor.py, currently in-process only and off by default. Persist journal contents — configuration keys/contexts only, never masks by default — to the artifact store keyed by (dialect hash, adapter/trie fingerprint, kernel version); on template build, restore the journal and let W5 precompute the recorded configurations off-batch before first decode. Optionally persist T2's schema-independent giant entries under the FULL fingerprint key only, since masks are deterministic functions of (configuration, trie) — an incomplete key that replays contexts against a different trie is waste, but persisted masks under a wrong key would be served-wrong-mask, the forbidden class.

**Expected gain:** For redeployments serving the same schema population, the cold-walk tail moves off the token path entirely: first-request TBM approaches warm-cache behavior on exactly the shared TTFM+TBM tail schemas. No help on never-before-seen schemas.

**Correctness risk:** low — warmup only precomputes masks that would be computed identically on demand, so outcomes cannot change, only timing; the one hazard is persistence-key completeness (contexts by default, masks only under the full adapter/vocab/kernel fingerprint).

**Composes with:** Versioned on-disk compile-artifact store (same store, different namespace); Prefill-overlap async compile (subprocess pool + phase fork) — warmup runs in the same background window; Tokenizer slicer (per-vocab slice masks + sub-tries in grid_core)
**Prior art:** pg_prewarm buffer-pool restore; Azul ReadyNow / OpenJDK AppCDS profile+archive warm starts; JIT trace-cache persistence.
