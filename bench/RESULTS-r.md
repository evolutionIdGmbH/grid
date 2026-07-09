# G7 R-microharness — mask latency vs position (no model)

Tokenizer: `gpt2` | n=16000 tokens/stream | 20 seeded runs/depth | host: Lambda 1×H100 PCIe 80GB, Ubuntu 24.04 (declared runner), kernel v4

Per seeded run: a synthetic SQL statement with WHERE-chain predicates nested to the
target depth is replayed twice; the warm second pass yields the per-position OLS
slope (requirement R). Epsilon = 0.1 us/1k tokens = 1e-4 us/pos.

| depth | steps | slope (us/pos, mean ± 95% CI) | warm p50 | hit p50 | miss p99 | steady hit rate | cum R² (min) | CD groups (mean/max) | CD pass ids (mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 320000 | -0.000007 ± 0.000013 | 5.2 us | 7.6 us | 13.15 ms | 97.2% | 0.99890 | 143 / 152 | 9277 / 31804 |
| 4 | 320000 | -0.000019 ± 0.000009 | 5.9 us | 8.3 us | 12.78 ms | 97.3% | 0.99943 | 148 / 152 | 12555 / 31803 |
| 8 | 320000 | -0.000015 ± 0.000007 | 6.8 us | 8.9 us | 12.39 ms | 97.6% | 0.99965 | 149 / 152 | 13449 / 31803 |
| 16 | 320000 | -0.000018 ± 0.000009 | 7.8 us | 8.8 us | 10.99 ms | 98.2% | 0.99965 | 150 / 152 | 14135 / 31803 |

Gate criteria (G7, binding on the declared cloud runner): slope 95% CI half-width and upper bound <= 0.0001 us/pos at n=16k; p50 cache-hit < 10 us; p99 miss < the recorded step budget; steady-state hit rate >= 90%; cumulative guard-cost R² > 0.99. **All green on the declared H100 runner with kernel v4** — hit p50 7.6–8.9 µs (was 20.9–23.5 µs at v3, i.e. the `<10 µs` criterion now passes on the binding host), slope ≈ 0 all depths, R² ≥ 0.9989. GC is disabled during the warm timed pass (a gen-2 pause is not constraint cost; without it depth-0 R² read 0.989 on this virtualized host).

Cross-role T2 hit factor: N/A — the T2 cache tier is deferred to the serving work (grid/mask/cache.py); reported here once T2 lands.

grid_core kernels active: True.
