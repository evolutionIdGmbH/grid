# Spider dev — execution accuracy (EX), GRID-constrained vs unconstrained

Model: `Qwen/Qwen2.5-7B-Instruct` (cuda, greedy) | sample: 1034 dev questions (seed 0) | max_tokens 128 | grammar: `grammars/sql_spider.grid` (100% dev-gold coverage) + per-database L3 lexicons | host: Lambda 1x A10 24GB, Ubuntu 24.04 (declared runner; virtualized)

EX = predicted and gold result sets match on the Spider SQLite database (order-sensitive iff gold has ORDER BY). Syntax-valid = sqlite EXPLAIN accepts. GRID generations parse by construction and every identifier is schema-valid via the L3 lexicons; its failures are semantic, not syntactic.

| arm | n | syntax-valid | executes | EX | EX-delta | truncated | tok/query | gen tok/s |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 1034 | 91.3% | 91.3% | 53.7% | +nan% | 0.9% | 35 | 3.1 |
| grid-repair | 1034 | 94.5% | 94.5% | 55.2% | +nan% | 0.5% | 40 | 3.5 |

Arms `grid-cache-off`, `grid-audit-off`, `grid-jf-off` are the G9 ablations (write-back cache / audit trail / jump-forward spans); EX is identical by construction — the column that moves is gen tok/s.

Binding G9 numbers run on the declared cloud runner with the reference model (DESIGN.md SS10); this harness repoints via --model/--device.
