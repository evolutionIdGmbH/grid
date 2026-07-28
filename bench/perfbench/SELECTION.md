# 0.3.x implementation slate: 7 of 20

Selected by a three-judge panel (impact / correctness-risk / composability)
aggregated constraint-first: every measured hang family covered, conflicts
resolved head-to-head, effort mixed. Pool: CANDIDATES.md. Attribution:
profile_phases.py run of Jul 26.

## #4: Per-terminal DFA library with lazy product combination  `[scanner]`

The scanner-family anchor: never determinize the union NFA — per-terminal memoized DFAs plus a lazy, token-length-bounded product kill the dominant 79%-of-tail-mass phase and the 10 scanner caps. Chosen over the conflicting #1 because it is testable by behavioral differential (no concurrency-nondeterministic state numbering), its component library is exactly what #8 persists and #16 would memoize, and it is the only lazy representation that makes #5's rayon parallelism and cross-schema amortization real. Judges: 9/4/9.

## #11: NFA-derived live sets (delete the global live fixpoint)  `[scanner]`

S-effort, zero-conflict, provably-equivalent (two-line subset-construction theorem), and the stated exactness prerequisite for the lazy scanner: #4 cannot serve exact live[]/accepts_all without it. Standalone 1.2-2x by deleting the confirmed wrong-order fixpoint at dfa.py:440-455; verified by full-bench equality before deletion. Highest combined judge confidence in the pool (4/10/9.5). Lands first.

## #2: Counting-set scanner states for {m,n} windows  `[scanner]`

The asymptotic kill of the window dimension that dominates the measured scanner tail (the 0.45s (0,64) build, the 2.3s/145s product cases): O(n) window positions become O(1) counting states with interval invariants. Zero conflicts, slots into #4 as a component type and into #5's kernel op stream, and is the sanctioned path to eventually lifting the <=64 window budget. Phase 1 freezes budget predicates so recorded sets stay byte-identical. Judges: 8/4/9.5.

## #5: Rust scanner build behind a single-call arena FFI  `[scanner]`

The constant-factor carrier for the whole scanner program: 50-200x on construction, rayon per-terminal parallelism that only #4 makes possible, and the best risk profile of any L item — data-only boundary, Python builder kept as oracle, gated on bit-identical arenas over all 11.3k schemas. Its arenas cache directly in #8. Highest raw judge sum in the pool (8/7/8.5). Chosen over the conflicting #7, which is explicitly its interim and whose target loop largely disappears under #4 anyway.

## #17: Structural hash-consing + worklist dedupe in schema_compile (scope-extended to normalize.py _norm)  `[frontend]`

Covers the mandatory schema_compile hang family (~17% of grounded mass, 5 of 16 caps) at S effort with a byte-identical output gate. MANDATORY SCOPE NOTE, verified in-repo: the exponential lives in normalize.py:597-611 ($ref-with-siblings -> _inline_refs deep copy -> merge2 -> recursive _norm of the merged subtree), upstream of the candidate's written rule_for/_dedupe_rules scope. The same structural-memo mechanism must be applied to _norm/_inline_refs (pure functions of node content for a fixed root) or the family's caps survive and criterion (b) fails. Judges: 4/8/8.

## #6: LR(0) + DeRemer-Pennello LALR lookaheads  `[lalr]`

Covers the mandatory LALR family (3.9% of mass but one 75s hang: helm-testsuite, whose canonical-LR(1) frozenset splitting is exactly what DP replaces) at M effort with zero conflicts. LALR(1) is construction-independent, so an isomorphism differential over the corpus is a strong gate; 40 years of Bison prior art plus PLY as a pure-Python reference. Also the substrate #3/#12 would have needed, without their gate hazards. Judges: 3/7/8.5.

## #8: Versioned on-disk compile-artifact store  `[orthogonal]`

The orthogonal/serving pick: repeat-deployment TTFM p99 collapses to unpickle time, and it is the persistence hub of the slate — stores #4's per-terminal component library, #5's arenas, and #6/#17 outputs under keys the code already computes. Conflict-free given #4 was chosen over #1 (whose partial lazy DFA is not persistable). Honesty requirement: cold-run numbers reported separately, since it removes zero first-compile cost by construction. Judges: 2/7/8.5.

## Aggregator rationale

Aggregation: judge 1 = measured-mass removal, judge 2 = correctness/testability, judge 3 = composition. Raw sums rank {5, 11, 4, 2, 7, 17, 1, 6} on top, but the slate was chosen by constraint-first selection, not summation: (1) all three measured hang families must be covered — scanner (79% of grounded mass, 10/16 caps), schema_compile frontend $ref re-normalization (~17%, 5 caps), LALR (3.9%, 1 cap) — which forces #17 and #6 in despite judge 1's low family-size scores, because without them their caps survive and criterion (b) of the decision rule fails; (2) conflicts eliminated head-to-head: #4 over #1 (better testability, composes with #8/#16, enables #5's rayon; #1 additionally conflicts with #8), #5 over #7 (#7 is #5's declared interim and its target loop — eager union subset construction — mostly disappears under #4), #6 over #12 (worst aggregate score, error-timing gate hazard), no 2^R pick since attribution failed to confirm 2^R dominance (LALR 3.9%, completed schema_compile 0.2%) so #3/#15 lose their premise. Effort mix is 3L/2M/2S — heavier than ideal on L, but the three Ls (4, 2, 5) are one designed scanner program with internal dependencies (11 -> 4 -> 2 -> 5), not independent bets, and the two S items (11, 17) land first as standalone wins. One load-bearing verification performed: judge 1's claim that the frontend exponential lives in normalize.py _norm, upstream of #17's written scope, is CONFIRMED (normalize.py:597-611: _inline_refs deep-copy -> merge2 -> recursive _norm(merged) — multiplicative re-normalization of inlined subtrees). #17 is therefore selected WITH an explicit scope extension to structurally memoize _norm/_inline_refs; without it the frontend family is only nominally covered. Decision-rule fit: (a) every pick carries a byte-identical or corpus-differential gate and none changes recorded outcomes (2 and 17 have explicit predicate-freeze / cap-predicate disciplines); (b) 10 scanner caps collapse via 4+2 with 5 as insurance, 5 frontend caps via 17-extended, 1 LALR cap via 6; (c) all three families attacked asymptotically makes >=10x tail-1% cut plausible; (d) p50 safe — 11/17/6 are provably-equivalent rewrites, 4 memoizes the median's repeated components, 8 is off the cold path. Recommended follow-on wave (not in the seven): #13 async compile, #16 memo tiers, #19 tokenizer slicer, #20 warm-set restore, #14 literal-class terminals.

## Notable rejections (overrule paths included)

- **#1 Kernel-resident lazy scanner (grid_core v8)**: Lost the head-to-head with #4 (mutually exclusive lazy representations). Judge 1 rates it marginally higher (10 vs 9), but it is the hardest scanner option to test (concurrency-deterministic state numbering, reserve/audit ports that silently force full builds — J2: 3) and it fights the portfolio: conflicts with #8 (partial lazy DFA not persistable) and drags T1/T2-coherent numbering into the kernel. #4 delivers the same asymptotic collapse with a cross-schema amortization story #1 lacks. Overrule path: if kernel-resident masks-from-NFA become a hard v8 requirement, swap #1 in for #4 and drop #8 to a later wave.

- **#7 Bit-parallel bitset subset construction + interval mintermization**: Highest testability score in the scanner group (byte-identical gate, J2: 9) but conflicts with #5 and is explicitly its interim; with #4 also in the slate, the eager combined-subset-construction loop it accelerates largely ceases to exist. Keeping it would spend an M slot on an oracle fast path. Overrule path: if #5's Rust port slips a milestone, #7 is the drop-in M-effort hedge.

- **#3 Runtime required-key bitset (kill 2^R)**: Its premise — 2^R dominance in LALR/schema_compile — is the one thing the attribution run failed to confirm (LALR 3.9%, completed schema_compile 0.2%). Self-declared high risk in the forbidden hazard class (stale register -> false ACCEPT on '}'-crossing walks), conflicts with #15, and touches the load-bearing configuration-keyed mask machinery. Wrong risk for an unconfirmed payoff.

- **#12 Lazy LALR row materialization**: Worst aggregate score of the 20 (8.5). Late LALRConflictError lands in a different v0.2.5 outcome bucket — a direct gate-(a) hazard — and its main payoff is absorbed by #6. L effort on a 3.9% family.

- **#13 Prefill-overlap async compile**: Strong orthogonal serving win (J3: 8.5, vLLM V1 precedent) but removes ~0 of the measured attribution and compile-start TTFM — the decision rule's metric — is unchanged by its own admission. First candidate for the follow-on wave; #5's GIL release later upgrades it from subprocess to thread.

- **#16 Hierarchical in-process memo tiers**: Its L1/L2 tiers are substantially built into #4 (process-wide per-terminal memoization keyed by pattern source), and #8 persists the same keys. Marginal value in this slate shrinks to L0/L3; S-effort follow-on, not a slot in the seven. No help at p100 cold by its own admission.

- **#9 AST-based scanner cost model with bounded-path routing**: Stage A is routing glue whose value assumes bounded paths are opt-in per terminal; with #4 replacing build_scanner wholesale there is no routing decision left to make. Stage B changes recorded degradation sets and cannot ship inside the gate. Revisit if #4 is descoped to selective routing.

- **#15 Optional-run factoring of the 2^R member chain**: Designed as the fallback leg of the 2^R slot, and the attribution removed the premise for that slot entirely (see #3). Safe S-effort, but it would occupy a slot to shrink phases measuring 0.2%/0.05% plus part of 3.9%. Cheap enough to add opportunistically later.

- **#14 Literal-class set terminals (keys and enums)**: Real inner-loop reduction on machine-generated tail schemas and a genuine TBM/512-terminal-cap win, but its hazard is silent false-reject via signature-collision priority tie-breaks, and #4's component memoization already amortizes the repeated-literal cost it targets. Second wave, after the #4 baseline exists to differential-test against.

- **#19 Tokenizer slicer (per-vocab slice masks)**: The strongest TBM-tail candidate and a designed pairing with #2's counter intervals — but it is L effort against a milliseconds-scale surface while the epoch's gate criteria are seconds-to-120s compiles, and its hazard class is false ACCEPT in the served mask. Prime candidate for the wave after the compile tail is fixed.
