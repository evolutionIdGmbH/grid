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
| FACTORED_SCANNER | 5 (to <=1.4s) | 1016s -> 36s (27.9x) | 0.94 / 1.01 | PASS (b,c,d) |
| HASHCONS | 5 (to <=0.1s) | (its wins are all caps) | 0.97 / 0.99 | PASS (b,d) |
| COUNTING_WINDOWS | 0 | 1016s -> 454s (2.2x) | 0.99 / 1.01 | subsumed by FACTORED standalone |
| NFA_LIVE | 0 | 1016s -> 980s (1.0x) | 0.89 / 0.96 | PASS (d); prerequisite value only |
| LALR_DP | 0 (helm still hangs) | ~1.1-1.27x LALR phase | 0.94 / 1.01 | PASS (d); family half-covered |
| RUST_SCANNER | 6 (but to 5-99s, not <1s) | 1016s -> 66s (15.4x) | **2.53 / 4.32** | **FAILS (d)**: +18-22ms FFI floor per build |
| ARTIFACT_STORE | n/a | no standalone win | n/a | standalone caches schema_src only (0.2% of tail mass) |

## Per-cap outcome matrix

Union of caps fixed to sub-second: **10/16** (FACTORED 5 + HASHCONS 5,
perfectly complementary families). RUST drags o47656/o47657 under the 120s
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

The 16 baseline caps: **5 compiled** (o79409 1.4s, o83133 5.8s, o83132 88.9s,
strmprivacy BatchJob 11.9s / DataConnector 13.5s), **5 terminate
deterministically** (frontend family: 4 declared-Unsupported in
schema_compile at ~0s; wp_105 reaches LALR in 0.07s then LALRConflictError
on identical-RHS twin rules - a fixable dedupe gap, tracked), **6 still
time out** (substring-union scanner family x5 + helm LALR).

Tail (>5s baseline, both complete, n=24): **1016s -> 48s (21.3x)**.
Fast schemas (<1s baseline, n=18): ON/OFF median 1.12, p90 1.48.

Interaction findings vs the per-candidate legs (review-wave inputs):
1. o83132/o83133 regress under COMBINED vs FACTORED-alone (0.0s -> 88.9s /
   5.8s): hashcons flag-on changes emitted grammar text, which appears to
   defeat the factored fast path on these schemas. Root-cause in review.
2. Fast-schema median overhead 1.12 vs ~0.94-0.99 for each candidate alone:
   composition overhead (digesting? dispatch? machine noise at n=18) is
   nominally over the 10% p50 bound - needs attribution before v0.3.0.
3. wp_105 twin-rule reduce-reduce (r279_v/r2048_v identical RHS) - dedupe
   should merge; likely turns wp_105 into a full compile.
