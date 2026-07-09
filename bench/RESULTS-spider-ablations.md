# Spider dev — execution accuracy (EX), GRID-constrained vs unconstrained

Model: `Qwen/Qwen2.5-7B-Instruct` (cuda, greedy) | sample: 20 dev questions (seed 1) | max_tokens 128 | grammar: `grammars/sql_spider.grid` (100% dev-gold coverage) + per-database L3 lexicons | host: Lambda 1x H100 PCIe 80GB, Ubuntu 24.04 (declared runner; virtualized)

EX = predicted and gold result sets match on the Spider SQLite database (order-sensitive iff gold has ORDER BY). Syntax-valid = sqlite EXPLAIN accepts. GRID generations parse by construction and every identifier is schema-valid via the L3 lexicons; its failures are semantic, not syntactic.

| arm | n | syntax-valid | executes | EX | EX-delta | truncated | tok/query | gen tok/s |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 20 | 90.0% | 90.0% | 50.0% | +nan% | 0.0% | 30 | 2.5 |
| grid-cache-off | 20 | 90.0% | 90.0% | 50.0% | +nan% | 0.0% | 30 | 1.7 |
| grid-audit-off | 20 | 90.0% | 90.0% | 50.0% | +nan% | 0.0% | 30 | 2.1 |
| grid-jf-off | 20 | 90.0% | 90.0% | 50.0% | +nan% | 0.0% | 30 | 2.3 |

Arms `grid-cache-off`, `grid-audit-off`, `grid-jf-off` are the G9 ablations (write-back cache / audit trail / jump-forward spans); EX is identical by construction — the column that moves is gen tok/s.

Binding G9 numbers require the pinned runner and the reference model (DESIGN.md SS10); this harness repoints via --model/--device.
