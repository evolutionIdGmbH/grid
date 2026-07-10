# G8 serving benchmark

Host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner) | heterogeneous schemas (4 distinct grammars) | batches 1, 8, 32

| arm | batch | TTFT p50 (ms) | TPOT mean (ms) | TPOT p99 (ms) | step p99 (ms) | tok/s | overhead vs unconstrained |
|---|--:|--:|--:|--:|--:|--:|--:|
| grid | 1 | — | 6.10 | 6.10 | 6.10 | 162 | -0.07% |
| unconstrained | 1 | — | 6.11 | 6.11 | 6.11 | 163 | — |
| grid | 8 | — | 6.25 | 6.26 | 6.26 | 1235 | +0.17% |
| unconstrained | 8 | — | 6.24 | 6.29 | 6.29 | 1261 | — |
| grid | 32 | — | 6.32 | 6.34 | 6.34 | 4606 | +0.73% |
| unconstrained | 32 | — | 6.27 | 6.29 | 6.29 | 4985 | — |

GRID TTFT split @batch 1: cold specialize **27.7 ms**, warm **1.49 ms**.

**Adversarial cold-miss arm** (fresh never-warmed schema injected into batch-32; both metrics reported, gating metric: **v2**, budget 30 ms):

- **metric v1 — legacy two-point lockstep wall** (assumes every request advances every step; conflates a deferred request's tail into the batch wall): co-batched TPOT degradation **+106.85%**, max step **21.6 ms**.
- **metric v2 — per-request, no lockstep assumption** (raw engine step loop; TPOT = (t_last−t_first)/(T−1) per request over the 31 warm co-batched requests; artifact-robust estimators — median-over-legs degradation, min-over-legs max step (the exogenous once-per-leg vLLM-multiprocess freeze is reported upstream; LESSONS 6.8): co-batched TPOT degradation **+114.72%**, max engine-step wall **36.0 ms** (raw per-leg maxima: ['36', '37', '38', '1420', '1687'] ms).
- **fresh request (reported, not gated)**: TTFT **0.8 ms**, completion **924.8 ms**, effective TPOT **9.73 ms** (1.00x warm; soft bound <= 3x warm: OK).

The §6 skip-a-round/overlap contract is gated on the metric-v2 values above.

**Concurrent cold start**: 1 build / 8 waiters, same-error-on-FAILED True (E17 single-flight).

## Gate G8

| criterion | pass | value |
|---|---|---|
| TPOT overhead < 2% vs unconstrained @batch 32 | PASS | +0.73% |
| TTFT cold specialize < 50 ms | PASS | 27.7 ms |
| TTFT warm < 5 ms | PASS | 1.49 ms |
| adversarial cold-miss: co-batched TPOT degradation < 5% [gating metric v2: per-request TPOT over warm co-batched requests] | FAIL | +114.72% |
| adversarial cold-miss: max step delay bounded (skip-a-round) [gating metric v2: per-request TPOT over warm co-batched requests] | FAIL | 36.0 < 30 ms |
| concurrent cold start: single build, N waiters | PASS | 1 build / 8 waiters |
| concurrent cold start: same error on FAILED | PASS | True |

Gate G8: **FAIL**.

Harness: `bench/vllm_serving_bench.py` (+ `bench/vllm_grid_patch.py`).
