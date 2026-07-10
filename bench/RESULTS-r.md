# G7 R-microharness — mask latency vs position (no model)

Tokenizer: `gpt2` | n=16000 tokens/stream | 20 seeded runs/depth | host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner)

Per seeded run: a synthetic SQL statement with WHERE-chain predicates nested to the
target depth is replayed twice; the warm second pass yields the per-position OLS
slope (requirement R). Epsilon = 0.1 us/1k tokens = 1e-4 us/pos.

| depth | steps | slope (us/pos, mean ± 95% CI) | warm p50 | hit p50 | miss p99 | steady hit rate | cum R² (min) | CD groups (mean/max) | CD pass ids (mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 320000 | -0.000004 ± 0.000003 | 3.8 us | 4.9 us | 6.19 ms | 100.0% | 0.99982 | 143 / 152 | 9277 / 31804 |
| 4 | 320000 | -0.000011 ± 0.000006 | 4.5 us | 5.7 us | 5.82 ms | 100.0% | 0.99918 | 148 / 152 | 12555 / 31803 |
| 8 | 320000 | -0.000004 ± 0.000003 | 4.9 us | 6.0 us | 6.03 ms | 100.0% | 0.99996 | 149 / 152 | 13449 / 31803 |
| 16 | 320000 | -0.000010 ± 0.000003 | 5.7 us | 6.5 us | 6.60 ms | 100.0% | 0.99972 | 150 / 152 | 14135 / 31803 |

Gate criteria (G7, binding on the declared cloud runner): slope 95% CI half-width and upper bound <= 0.0001 us/pos at n=16k; p50 cache-hit < 10 us; p99 miss < the recorded step budget; steady-state hit rate >= 90%; cumulative guard-cost R² > 0.99.

Cross-role T2 hit factor: N/A — the T2 cache tier is deferred to the serving work (grid/mask/cache.py); reported here once T2 lands.

grid_core kernels active: True.
