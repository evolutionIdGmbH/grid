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
