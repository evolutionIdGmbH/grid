# Guard-Rail Design Evolution: From v0.0.5 to v0.0.7 — an Implementable, Near-Linear Constrained-Decoding Engine

*(Lineage: v0.0.5 is our own early, partial design snapshot, authored August
2023. Planned, iterative
engineering carried it forward — v0.0.6 already outperformed guidance (its
July-2023 release) on the scaling benchmarks of that era — and continued to
v0.0.7: the implementation in this repository and the design the final patent
application follows, improved further since the patent. This document records
that design evolution — what changed from the snapshot, what was kept, and why.
The work drew inspiration from published systems (notably the Outlines paper);
the design and implementation are our own.)*

**Goal (requirement R):** per-token guard-rail cost O(1) with respect to current output length *n* (amortized; worst case bounded by grammar/nesting constants, never by *n*); total cost O(n) over a generation; per-stream memory ≤ O(n); holds under batched serving.

**Provenance:** distilled from the planned 6-lens design review of the v0.0.5 snapshot, a 6-topic research sweep (all facts source-checked), 3 independent architecture proposals (precompute-maximal / lazy-maximal / hybrid cache-centric), a 2-member review panel, and 2 independent verification passes that stress-tested the complexity claims and every cited number. Verifier corrections are folded in below and marked ⚠ where they change a guarantee.

---

## 1. What must change (and why)

| # | Original design element | Why it blocks R or correctness | Replacement |
|---|---|---|---|
| 1 | **Sequence-keyed lookup table** over all token sequences ST* (the v0.0.5 spec's core construction) | \|T\|^m entries: 32k vocab at m=3 is already ~3.3×10¹³ (~400 TB); beyond any feasible storage | Key on **parser configurations** (lexer-DFA state × LR state/stack-top), a grammar-sized key space (10⁴–10⁶ keys) |
| 2 | **Complete-string PDA acceptance** as the tag semantics (the v0.0.5 spec) | Every intermediate prefix is "rejected" → pruned token space empty at step 1; masking needs Prefix(L), not L | The **incrementally advanced LR parser itself is the viable-prefix oracle** (correct-prefix property, Aho & Johnson 1974); or Earley+Leo. No separate recognizer needed |
| 3 | **State-set keys without the stack** (the v0.0.5 spec) | Legal next tokens depend on (state, stack); state-sets alone are unsound for nested SQL | **Per-stream persistent stack** — XGrammar-style tree-of-stacks with suffix sharing and O(1) pointer rollback (also powers speculative-decoding rollback) |
| 4 | **No token↔terminal bridge yet** (the subword misalignment the v0.0.5 spec illustrates: 'sel'+'ect' = SELECT — mechanism left as later work) | Breaks completeness; schema identifiers/literals are never single tokens | **Byte-level token-trie walk** from the current lexer state (llguidance: ~13 cycles/node, ~50 µs full mask @128k vocab) or lexer-state × remainder mask tables (SynCode). Exact construction with correctness proof: Koo et al., arXiv:2407.08103 |
| 5 | **EOS as generic stopping criterion** (the v0.0.5 spec) | Viable-but-incomplete output can be emitted ("SELECT name FROM" + EOS) | **EOS legal ⇔ current configuration accepts** (ACCEPT action reachable via reduce closure). ⚠ Exact only on canonical LR(1); LALR/compressed tables have spurious reduces — gate EOS by simulating the reduce chain to ACCEPT, not by a raw row read |
| 6 | **Whole-sequence rejection sampling** and **single-token backtracking** as the v0.0.5 spec's fallback variants | ~169 expected full regenerations for a 100-token statement at 95% per-token validity; 1-token backtrack can't escape deep dead ends | **Per-step hard masking is the only primary mode.** Keep the v0.0.5 spec's multi-token pruning as **jump-forward decoding** on forced sequences. Optional distribution-faithful add-on: GAD/ASAp or CRANE-style two-phase (see §5) |
| 7 | **"Preferably lowering their probability"** (the v0.0.5 spec's soft variant) | Soft down-weighting leaves nonzero mass on illegal tokens → guarantee void | **Hard mask: logits := −∞** for illegal tokens (bitmask kernel, e.g. `apply_token_bitmask_inplace`) |
| 8 | **Grammar check anchored at the request** (the v0.0.5 spec translates the request into the initial sequence) | No SQL grammar accepts prompt-prefixed strings | **Explicit constrained-span anchoring**: constraint activates at a declared delimiter/position; wrapper grammar for surrounding prose if mixed output is wanted |
| 9 | **Full re-precompute on any grammar change** (one-way S1→S5 pipeline) | Schema evolves weekly; tenants × roles × tokenizers multiply tables | **Lazy/JIT grammar specialization + compile cache + write-back mask cache** (see §3). XGrammar-2 shows 4,960 ms → 5.4 ms preprocessing with JIT + cross-grammar caching (arXiv:2601.04426) |

## 2. What survives from the original design (keep these)

- **K1 — decode-time logit masking as primary enforcement** (the v0.0.5 spec's primary mode): the field-proven winner (Outlines, SynCode, XGrammar, llguidance, GBNF, OpenAI Structured Outputs).
- **K2 — the adaptive verdict cache, reframed**: precomputed-where-cached, compute-on-miss via the automaton, **write-back keyed by configuration** (not sequence). This carries v0.0.5's table+acceptor idea to its workable form, and it is genuinely differentiating: XGrammar precomputes per grammar at compile time; a write-back cache exploits cross-request locality **across a per-role/per-tenant grammar family** — our actual deployment shape.
- **K3 — persistent accepted/rejected lists as an audit trail**: immutable, versioned mask-cache entry IDs referenced by per-token audit records, replayable against archived grammar artifacts. No mainstream system (XGrammar, llguidance, SynCode) offers this; first-class compliance feature for the RBAC setting.
- **K4 — grammars derived from API/schema descriptions**: automate as a policy-to-grammar pipeline (L2/L3 generated from the RBAC policy store + `information_schema`, no hand-edited grammars).
- **K5 — the v0.0.5 spec's multi-token pruning → jump-forward decoding** (forced-token spans emitted without model calls; XGrammar `find_jump_forward_string`).
- **K6 — constrained top-k sampling (mask 9)**: keep sampling within the mask; never silently collapse to greedy.

## 3. Target architecture (converged in the design review)

Three layers of grammar, one engine.

**Grammar layers**
- **L1 — dialect core**: the SQL dialect grammar. Production SQL grammars (PostgreSQL `gram.y`, MySQL `sql_yacc.yy`, SQLite Lemon) are **LALR(1)** — deterministic-PDA coverage suffices; no GLR/ALL(*) superlinear machinery needed.
- **L2 — role projection**: RBAC as a **production subset** of L1 (verb subsets, clause bans), compiled on miss and cached by `roleShapeHash` (~tens of distinct shapes org-wide; sub-second LALR compiles; hash-consed). ⚠ **Mandatory reducedness pass**: production subsetting creates unproductive/unreachable nonterminals; run useless-symbol elimination per composed grammar and CI-assert reducedness, else "non-empty chart/action set ⇒ viable prefix" fails and dead ends return.
- **L3 — schema lexicon**: table/column/identifier allow-lists as **lazily-built lexer tries** (Brzozowski-derivative DFAs materialized on demand; zero vocabulary enumeration on cold specialize; target ~1–5 ms per new schema).

**Engine (per stream)**
1. **Incremental LALR(1)/LR(1) parser** with persistent stack (tree-of-stacks, O(1) rollback) = the viable-prefix oracle. Amortized O(1) per shifted terminal.
2. **Mask producer**: allowed-terminal set from the parser row (⚠ simulate reduce closure on compressed tables) → byte-level trie walk from the current lexer state over the tokenizer vocabulary → 0/1 bitmask (16 KB @128k vocab).
3. **Two-tier mask cache (K2)**: offline-seeded entries for context-independent verdicts (XGrammar context-expansion: <1% of tokens are context-dependent for JSON-class grammars; **validate this for nested SQL — gating milestone**) + runtime write-back keyed on (grammar fingerprint, lexer state, allowed-terminal signature). Hit ≈ 1–3 µs; miss = exact trie walk ≈ 20–50 µs @128k (verified: llguidance toktrie, 8-byte nodes, ~13 cycles/node). ⚠ Cache-key soundness must be proven (two configurations sharing a key must yield byte-identical masks — key must refine the Myhill–Nerode classes of the lexer × parser product) and property-tested against the exact walk.
4. ⚠ **Identifier-position composition rule**: never union precomputed generic-IDENT masks at identifier positions — a generic-IDENT verdict admits tokens spelling **forbidden** identifiers, and the parser will NOT reject them later (they are grammatical as identifiers) → silent RBAC violation. At identifier positions the mask must come from the **L3 allow-list trie intersection**, always.
5. **EOS gating**: EOS ⇔ ACCEPT reachable (reduce-closure simulation). **Termination reserve**: cumulative min-completion cost maintained **on the stack** (per-frame shortest-completion costs summed incrementally) — ⚠ a per-state table under-reserves; the reserve depends on the whole stack. When remaining budget = reserve, jump-forward the shortest legal completion; report residual truncation rate as a metric.
6. **Audit record (K3)** per token: (configuration rolling hash, mask-entry version id, chosen token, blocked-count). ⚠ The configuration hash must be an **incrementally maintained rolling hash carried in stack entries** (O(1)/push) — hashing the stack from scratch is Θ(depth)/token, a hidden n-dependence. Speculative rollback must also restore lexer remainder + audit hash chain.
7. **Serving integration**: vLLM structured-output backend interface; mask computed on CPU **overlapped with the GPU forward pass** (XGrammar pattern → near-zero TPOT overhead); batch heterogeneity via per-request matcher state; `BatchGrammarMatcher`-style batching. ⚠ Deadline fallback (coarser over-approximate mask on a p99 spike) reopens the soundness hole — if used, pair it with **defined recovery**: parser-level rejection of the over-admitted token + O(1) rollback and re-mask, never silent continuation.

**Dead-end freedom (provable):** with (a) exact masks, (b) reduced grammars, (c) byte-fallback vocabulary (all 256 byte tokens present — 243 excluding bytes impossible in UTF-8; arXiv:2511.05578), every viable prefix has ≥1 legal token and generation cannot wedge. ⚠ This theorem quantifies over viable prefixes only — it is **incompatible with sound-but-incomplete d-lookahead masks** (SynCode ships d∈{1,2}: Theorem 2 completeness requires d > max token length). One over-admitted token exits Prefix(L) and the theorem no longer applies. **Choose exact byte-level masks** (llguidance/XGrammar style), not lookahead approximation, or define recovery explicitly.

## 4. Formal guarantees — exact statements (post-verification)

1. **Soundness**: every emitted token keeps the detokenized output in Prefix(L(G_role,schema)). Preconditions: exact masks; hard −∞ masking; identifier-trie composition rule (§3.4).
2. **Completeness**: no token is blocked whose byte string can extend the current viable prefix toward a member of L. Preconditions: byte-fallback vocab; reduced grammar; exact trie walk.
3. **Termination**: output ∈ L on every stop. Preconditions: EOS-iff-ACCEPT; cumulative stack reserve (§3.5); finite length budget.
4. **Complexity (the R claim, stated honestly)**: **amortized O(1) per token, total O(n)** — per-step worst case is bounded by **nesting depth** (a single terminal can trigger a reduce cascade proportional to stack depth; SQL prefix operators are inherently right-recursive, so a left-recursion lint helps lists but cannot eliminate cascades). Space O(depth) per stream. Fallback engine: Earley+Leo is O(n) total on all LR-regular grammars (Leo 1991); unambiguous non-LR-regular → O(n²); ambiguous → O(n³).
5. **Distribution**: per-step masking ≠ P(x | x ∈ L) (Grammar-Aligned Decoding, arXiv:2405.21047). Not repaired by default — measured in the benchmark (execution accuracy), with optional CRANE-style two-phase (unconstrained reasoning → constrained emission; up to +10 pts on reasoning benchmarks) and GAD/ASAp as a faithful-sampling research arm. Constraint-induced quality loss is real for small models ("Let Me Speak Freely", arXiv:2408.02442: Llama-3-8B GSM8K ~75 → 48.9 under strict JSON; Claude-3-Haiku 86.5 → 23.4).
6. ⚠ **RBAC scope (honest boundary)**: grammar masking enforces **verb-level and table-level** policy. **Per-table column restrictions are NOT left-to-right enforceable by any CFG mask** — in SQL the SELECT list precedes FROM, so alias→table binding is unknown at column-mask time (and alias binding is context-sensitive). Column-level policy = post-parse semantic check on the completed statement (cheap: one AST walk) or view/rewrite layer in the DB. Market the mask guarantee accordingly.

## 5. Per-token cost budget (against a 5–10 ms decode step, 8B-class model; re-measure ITL on target hardware)

| Item | Cost | Notes |
|---|---|---|
| Parser advance (amortized) | ~0.1–1 µs | reduce cascades bounded by depth |
| Allowed-terminal extraction | ~0.3–1 µs | ⚠ + reduce-closure sim on compressed tables |
| Mask: cache hit | ~1–3 µs | expected ≥90% hit after warmup (validate) |
| Mask: cache miss (exact trie walk) | ~20–50 µs @128k | llguidance-verified 13 cycles/node; worst-case ~0.9M nodes ≈ 4 ms — deadline strategy per §3.7 |
| Bitmask GPU application | ~µs, fused kernel | overlapped with forward pass |
| Audit record | ~0.1–0.3 µs | rolling hash only |
| **Total (typical)** | **< 10 µs hit / < 60 µs miss** | <1% of decode step; overlapped → near-zero TPOT |

**Memory**: shared token trie ~8 MB (8-byte nodes ×~0.9M); LALR tables for full dialect ~5–20k states (canonical LR ~5–10× more — prefer LALR + closure-sim for EOS); mask cache with adaptive storage (accept-list/reject-list/bitset per entry — XGrammar shrank 160 MB → 0.46 MB for JSON); per-stream stack O(depth); per-role compiled variants hash-consed, ~tens org-wide.

## 6. Benchmark plan

**Baselines**: XGrammar v0.2.x **and XGrammar-2** (vLLM default backend), llguidance (vLLM/llama.cpp), Outlines, SynCode (MIT, closest published relative of the original design), llama.cpp GBNF; unconstrained control arm always included.

**Datasets**: Spider 1.0 + BIRD (text-to-SQL; report **execution accuracy (EX)** and executability separately — note: the widely quoted 67.4→81.4 Spider figure is **IterGen** (arXiv:2410.07295, Llama-3.2-3B, execution *success*), not SynCode); JSONSchemaBench (arXiv:2501.10868) for cross-engine comparability; an **internal RBAC suite**: per-role grammars where the violation rate under masking must be exactly 0 for verb/table policy (column-level measured at the post-parse layer per §4.6).

**Metrics**:
1. **The R experiment (headline)**: mask-generation latency (p50/p99) **vs token position** at n = 128 … 16k, with nesting-depth sweep (flat vs deeply nested subqueries). Acceptance: position-slope CI contains 0 at n=16k; total guard-rail cost linear fit R² > 0.99; p99 within per-step budget.
2. TTFT overhead: grammar compile + per-request specialization (target: cold role+schema < 50 ms, warm < 5 ms).
3. TPOT/ITL overhead vs unconstrained at batch 1/8/32/64 (CPU-GPU overlap on).
4. Correctness: syntax validity % (must be 100), EX delta vs unconstrained, truncation rate, dead-end incidents (must be 0).
5. Cache telemetry: hit rate over time, write-back growth, cross-role sharing factor.
6. **Ablations**: cache-off, write-back-off, audit-off, jump-forward-off, CRANE on/off, exact-vs-lookahead masks.

**Fairness**: same model, prompts, sampling params, hardware across engines; per-engine grammar expressed in its native format; measure engine-side preprocessing separately from decode-loop overhead.

## 7. Differentiators vs existing tools (what makes this worth building)

1. **Write-back configuration-keyed mask cache across grammar *families*** (per-role/per-tenant/per-schema) — XGrammar caches per compiled grammar; we exploit locality across the family, which is exactly the enterprise RBAC shape.
2. **Auditability**: replayable per-token permit/block trail bound to versioned grammar artifacts — none of the baselines has it; strong compliance story.
3. **Policy-to-grammar automation** (RBAC store + information_schema → L2/L3), with reducedness verification in CI.
4. **Honest, provable guarantee statements** (§4), including the column-RBAC boundary — competitors don't state theirs.

## 8. Open risks / gating milestones

1. **Nesting-depth sweep** for context-dependent token growth in recursive SQL (XGrammar's <1% figure is JSON-derived; unvalidated for deep subqueries). Fallback budgeted: widen cache keys / runtime-check more tokens.
2. **Leo-item subtleties** if the Earley+Leo fallback engine is used: allowed-terminal extraction under transitive items is known-subtle (Marpa notes); differential-test against brute-force Earley on randomized LR-regular grammars.
3. **Lexer-ambiguity hypothesis bound** (keyword-vs-identifier, maximal munch, case-insensitivity, remainder extendability): assert and property-test a grammar-bounded simultaneous-hypothesis count.
4. **Batch tail behavior**: one slow mask in a batch stalls the step; deadline strategy of §3.7 with defined recovery, measured under load.
5. Re-measure baseline ITL on target hardware (published 8B/H100 figures cluster ~5–10 ms; the budget's headroom claims anchor on it).

## Key sources

Knuth 1965 (LR, viable prefixes regular); Aho & Johnson 1974 (correct-prefix property); Leo 1991, TCS 82:165–176 (linear Earley on LR-regular); Ginsburg & Greibach 1966 (DCFL closure); SynCode arXiv:2403.01632 (TMLR '24; soundness/completeness theorems, mask store 1.06–1.87 GB, build 113–603 s, overhead 1.22–1.76×); XGrammar arXiv:2411.15100 (<40 µs/token JSON, 0.46 MB adaptive cache, tree-of-stacks rollback, vLLM default); XGrammar-2 arXiv:2601.04426 (Earley-based, JIT compile 4,960→5.4 ms, mask 45.5→126.5 µs all-opt); llguidance github.com/guidance-ai/llguidance (toktrie: 8-byte nodes, ~13 cycles/node, ~50 µs @128k); Outlines arXiv:2307.09702; PICARD arXiv:2109.05093; IterGen arXiv:2410.07295; Koo et al. arXiv:2407.08103 (provably correct tokenizer bridge); GAD arXiv:2405.21047; CRANE (constrained reasoning, up to +10 pts); Let Me Speak Freely arXiv:2408.02442; UTF-8 byte-fallback arXiv:2511.05578; JSONSchemaBench arXiv:2501.10868.
