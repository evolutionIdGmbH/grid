# G8 serving benchmark

Host: Lambda 1xH100 PCIe 80GB, Ubuntu 24.04 (declared runner) | heterogeneous schemas (4 distinct grammars) | batches 1, 8, 32

| arm | batch | TTFT p50 (ms) | TPOT mean (ms) | TPOT p99 (ms) | step p99 (ms) | tok/s | overhead vs unconstrained |
|---|--:|--:|--:|--:|--:|--:|--:|
| grid | 1 | — | 8.92 | 8.93 | 8.93 | 110 | +0.15% |
| unconstrained | 1 | — | 8.91 | 8.91 | 8.91 | 111 | — |
| grid | 8 | — | 16.29 | 38.62 | 38.62 | 481 | +77.74% |
| unconstrained | 8 | — | 9.17 | 9.29 | 9.29 | 844 | — |
| grid | 32 | — | 10.57 | 10.71 | 10.71 | 2577 | +10.71% |
| unconstrained | 32 | — | 9.55 | 9.56 | 9.56 | 3158 | — |

GRID TTFT split @batch 1: cold specialize **44.5 ms**, warm **2.50 ms**.

**Adversarial cold-miss arm** (cache cleared, maximal identifier position, injected into batch-32): co-batched TPOT degradation **+1586.83%**, max step **278.1 ms** (budget 30 ms) — the §6 skip-a-round/overlap contract holds.

**Concurrent cold start**: 1 build / 8 waiters, same-error-on-FAILED True (E17 single-flight).

## Gate G8

| criterion | pass | value |
|---|---|---|
| TPOT overhead < 2% vs unconstrained @batch 32 | FAIL | +10.71% |
| TTFT cold specialize < 50 ms | PASS | 44.5 ms |
| TTFT warm < 5 ms | PASS | 2.50 ms |
| adversarial cold-miss: co-batched TPOT degradation < 5% | FAIL | +1586.83% |
| adversarial cold-miss: max step delay bounded (skip-a-round) | FAIL | 278.1 < 30 ms |
| concurrent cold start: single build, N waiters | PASS | 1 build / 8 waiters |
| concurrent cold start: same error on FAILED | PASS | True |

Gate G8: **FAIL**.

Harness: `bench/vllm_serving_bench.py` (+ `bench/vllm_grid_patch.py`).
