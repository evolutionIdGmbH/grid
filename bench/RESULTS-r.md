# Flat per-token guard-rail cost — mask latency vs position (no model)

Tokenizer: `gpt2` | n=16000 tokens/stream | 20 seeded runs/depth | host: Lambda 1xH100 SXM5 80GB, Ubuntu 24.04 (declared runner), kernel v7

Per seeded run: a synthetic SQL statement with WHERE-chain predicates nested to the
target depth is replayed twice; the warm second pass yields the per-position OLS
slope (requirement R). Epsilon = 0.1 us/1k tokens = 1e-4 us/pos.

| depth | steps | slope (us/pos, mean ± 95% CI) | warm p50 | hit p50 | miss p99 | steady hit rate | cum R² (min) | CD groups (mean/max) | CD pass ids (mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 320000 | -0.000004 ± 0.000002 | 3.7 us | 4.8 us | 6.56 ms | 100.0% | 0.99985 | 143 / 152 | 9277 / 31804 |
| 4 | 320000 | -0.000007 ± 0.000002 | 4.3 us | 5.6 us | 5.84 ms | 100.0% | 0.99992 | 148 / 152 | 12555 / 31803 |
| 8 | 320000 | -0.000007 ± 0.000002 | 4.7 us | 6.1 us | 6.11 ms | 100.0% | 0.99995 | 149 / 152 | 13449 / 31803 |
| 16 | 320000 | -0.000006 ± 0.000002 | 6.0 us | 6.8 us | 6.31 ms | 100.0% | 0.99994 | 150 / 152 | 14135 / 31803 |

Summary: per-token guard-rail cost is flat with output position at every nesting depth — the OLS slope 95% CI upper bound stays at ≤ 0.0001 us/pos out to n=16k, warm cache-hit p50 is under 10 us, steady-state hit rate is 100%, and the cumulative guard-cost fit holds at R² > 0.99. The reference measurement runs on the declared cloud runner.

Cross-role T2 hit factor: N/A — the T2 cache tier is deferred to the serving work (grid/mask/cache.py); reported here once T2 lands.

grid_core kernels active: True.
