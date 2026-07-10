# Spider dev — execution accuracy (EX), GRID-constrained vs unconstrained

Model: `Qwen/Qwen2.5-0.5B-Instruct` (cuda, greedy) | sample: 100 dev questions (seed 0) | max_tokens 128 | grammar: `grammars/sql_spider.grid` (100% dev-gold coverage) + per-database L3 lexicons | host: local dev (unpinned)

EX = predicted and gold result sets match on the Spider SQLite database (order-sensitive iff gold has ORDER BY). Syntax-valid = sqlite EXPLAIN accepts. GRID generations parse by construction and every identifier is schema-valid via the L3 lexicons; its failures are semantic, not syntactic.

| arm | n | syntax-valid | executes | EX | EX-delta | truncated | tok/query | gen tok/s |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 100 | 57.0% | 57.0% | 29.0% | +13.0% | 4.0% | 41 | 3.2 |
| unconstrained | 100 | 31.0% | 31.0% | 16.0% | +0.0% | 9.0% | 61 | 64.5 |

Arms `grid-cache-off`, `grid-audit-off`, `grid-jf-off` are the throughput ablations (write-back cache / audit trail / jump-forward spans); EX is identical by construction — the column that moves is gen tok/s.

Binding throughput numbers require the pinned runner and the reference model (DESIGN.md SS10); this harness repoints via --model/--device.
