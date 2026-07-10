# G8 serving benchmark

Host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner), kernel v7 | heterogeneous schemas (4 distinct grammars) | batches 1, 8, 32

| arm | batch | TTFT p50 (ms) | TPOT mean (ms) | TPOT p99 (ms) | step p99 (ms) | tok/s | overhead vs unconstrained |
|---|--:|--:|--:|--:|--:|--:|--:|
| grid | 1 | — | 6.13 | 6.14 | 6.14 | 162 | +0.15% |
| unconstrained | 1 | — | 6.12 | 6.13 | 6.13 | 163 | — |
| grid | 8 | — | 6.29 | 6.40 | 6.40 | 1229 | +0.73% |
| unconstrained | 8 | — | 6.24 | 6.25 | 6.25 | 1130 | — |
| grid | 32 | — | 6.35 | 6.36 | 6.36 | 4600 | +1.51% |
| unconstrained | 32 | — | 6.25 | 6.27 | 6.27 | 4536 | — |

GRID TTFT split @batch 1: cold specialize **27.3 ms**, warm **1.51 ms**.

**Adversarial cold-miss arm** (fresh never-warmed schema injected into batch-32; both metrics reported, gating metric: **v2**, budget 30 ms):

- **metric v1 — legacy two-point lockstep wall** (assumes every request advances every step; conflates a deferred request's tail into the batch wall): co-batched TPOT degradation **+29.92%**, max step **8.8 ms**.
- **metric v2 — per-request, no lockstep assumption** (raw engine step loop; TPOT = (t_last−t_first)/(T−1) per request over the 31 warm co-batched requests; artifact-robust estimators — median-over-legs degradation, min-over-legs max step (the exogenous once-per-leg vLLM-multiprocess freeze is reported upstream; LESSONS 6.8): co-batched TPOT degradation **+33.81%**, max engine-step wall **15.3 ms** (raw per-leg maxima: ['15', '15', '16', '16', '16'] ms).
- **fresh request (reported, not gated)**: TTFT **0.7 ms**, completion **663.0 ms**, effective TPOT **6.97 ms** (1.00x warm; soft bound <= 3x warm: OK).

The §6 skip-a-round/overlap contract is gated on the metric-v2 values above.

**Concurrent cold start**: 1 build / 8 waiters, same-error-on-FAILED True (E17 single-flight).

## Gate G8

| criterion | pass | value |
|---|---|---|
| TPOT overhead < 2% vs unconstrained @batch 32 | PASS | +1.51% |
| TTFT cold specialize < 50 ms | PASS | 27.3 ms |
| TTFT warm < 5 ms | PASS | 1.51 ms |
| adversarial cold-miss: co-batched TPOT degradation < 5% [gating metric v2: per-request TPOT over warm co-batched requests] | FAIL | +33.81% |
| adversarial cold-miss: max step delay bounded (skip-a-round) [gating metric v2: per-request TPOT over warm co-batched requests] | PASS | 15.3 < 30 ms |
| concurrent cold start: single build, N waiters | PASS | 1 build / 8 waiters |
| concurrent cold start: same error on FAILED | PASS | True |

Gate G8: **FAIL**.

Harness: `bench/vllm_serving_bench.py` (+ `bench/vllm_grid_patch.py`).
