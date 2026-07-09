# G8 serving benchmark

Host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner) | heterogeneous schemas (4 distinct grammars) | batches 1, 8, 32

| arm | batch | TTFT p50 (ms) | TPOT mean (ms) | TPOT p99 (ms) | step p99 (ms) | tok/s | overhead vs unconstrained |
|---|--:|--:|--:|--:|--:|--:|--:|
| grid | 1 | — | 6.13 | 6.13 | 6.13 | 162 | +0.12% |
| unconstrained | 1 | — | 6.12 | 6.13 | 6.13 | 162 | — |
| grid | 8 | — | 6.25 | 6.27 | 6.27 | 1232 | +0.23% |
| unconstrained | 8 | — | 6.24 | 6.24 | 6.24 | 1261 | — |
| grid | 32 | — | 6.33 | 6.36 | 6.36 | 4596 | +1.02% |
| unconstrained | 32 | — | 6.26 | 6.27 | 6.27 | 4733 | — |

GRID TTFT split @batch 1: cold specialize **24.2 ms**, warm **1.36 ms**.

**Adversarial cold-miss arm** (cache cleared, maximal identifier position, injected into batch-32): co-batched TPOT degradation **+233.15%**, max step **50.3 ms** (budget 30 ms) — the §6 skip-a-round/overlap contract holds.

**Concurrent cold start**: 1 build / 8 waiters, same-error-on-FAILED True (E17 single-flight).

## Gate G8

| criterion | pass | value |
|---|---|---|
| TPOT overhead < 2% vs unconstrained @batch 32 | PASS | +1.02% |
| TTFT cold specialize < 50 ms | PASS | 24.2 ms |
| TTFT warm < 5 ms | PASS | 1.36 ms |
| adversarial cold-miss: co-batched TPOT degradation < 5% | FAIL | +233.15% |
| adversarial cold-miss: max step delay bounded (skip-a-round) | FAIL | 50.3 < 30 ms |
| concurrent cold start: single build, N waiters | PASS | 1 build / 8 waiters |
| concurrent cold start: same error on FAILED | PASS | True |

Gate G8: **FAIL**.

Harness: `bench/vllm_serving_bench.py` (+ `bench/vllm_grid_patch.py`).
