# Spider dev — execution accuracy (EX), GRID-constrained vs unconstrained

Measured 2026-07-31 at GRID 0.4.0 (grid_core 0.2.0, kernel v8), shipped defaults, plain constrained decoding — no repair loop (the repair-loop record is [RESULTS-spider-repair.md](RESULTS-spider-repair.md), version-pinned).

Model: `Qwen/Qwen2.5-7B-Instruct` (cpu, greedy) | sample: 1034 dev questions (seed 0) | max_tokens 128 | grammar: `grammars/sql_spider.grid` (100% dev-gold coverage) + per-database L3 lexicons | host: Lambda 1xH100 SXM 80GB HBM3, Ubuntu 24.04 (declared runner)

EX = predicted and gold result sets match on the Spider SQLite database (order-sensitive iff gold has ORDER BY). Syntax-valid = sqlite EXPLAIN accepts. GRID generations parse by construction and every identifier is schema-valid via the L3 lexicons; its failures are semantic, not syntactic.

| arm | n | syntax-valid | executes | EX | EX-delta | truncated | tok/query | gen tok/s |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 1034 | 94.6% | 94.6% | **55.2%** | **+2.5%** | 1.0% | 39 | 3.5 |
| unconstrained | 1034 | 91.0% | 91.0% | 52.7% | +0.0% | 0.2% | 33 | 3.5 |

Reading it:

- **Constraining improves the end task.** +2.5 EX points over the same model
  unconstrained, greedy, same prompts — with no repair loop. Masking never
  removes a correct continuation (the gold parse is always in-grammar); what
  it removes is the model's ability to spend probability mass on SQL that
  cannot execute.
- **Both arms generated on CPU** (declared above). EX and syntax-valid are
  device-fair within this run — same device, same greedy decode, per-arm
  identical inputs — but the tok/query and gen tok/s columns are NOT
  throughput evidence; decode-speed claims live in the serving bench
  (RESULTS-serving.md) where generation runs on the GPU.
- Versus the version-pinned v0.0.7 record ([RESULTS-spider.md](RESULTS-spider.md):
  GRID 53.7% EX, syntax-valid 91.3%, unconstrained 52.9%): the GRID arm gained
  +1.5 EX and +3.3 syntax-valid while unconstrained moved -0.2 (noise). The
  two runs differ in engine internals (0.0.7 kernel v7 → 0.4.0 kernel v8 with
  the 0.3.x/0.4.x scanner and budget work) and in generation device
  (cuda → cpu numerics), so the delta is not attributed to a single cause;
  the current-version comparison above is the binding one.

Arms `grid-cache-off`, `grid-audit-off`, `grid-jf-off` are the throughput ablations (write-back cache / audit trail / jump-forward spans); EX is identical by construction — the column that moves is gen tok/s, and it is only meaningful on the GPU runner.

Binding numbers run on the declared cloud runner with the reference model (DESIGN.md SS10); this harness repoints via --model/--device.
