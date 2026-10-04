# Schema churn: cache footprint, hit rate, cold start (GRID 0.4.1, kernel v8)

Produced by `bench/schema_churn.py` (teacher-forced replay of Spider gold queries, consecutive requests usually on different schemas). The T2 cross-schema tier gets no hits: after the soundness fix that scopes every entry of a lexicon-bearing producer by schema fingerprint, schema-independent sharing is gone. T1 has no eviction; footprint keeps growing as new configurations arrive.

## Spider train (140 schemas), T2 on

Split `train_spider`: 7000 gold queries over 140 schemas, shuffled (seed 0); tokenizer `Qwen/Qwen2.5-0.5B-Instruct`; T2 tier on; 195793 mask steps; rejected replays 17 (gold queries outside the grammar). Host: local dev: Apple M3 Max, 36 GB (unpinned).

| measure | value |
|:---|---:|
| grammar + trie build (once) | 0.64 s |
| schema specialization p50 / p90 | 1.8 / 1.9 ms |
| T1 hit rate, all steps | 89.1% |
| T1 hit rate, first query of a schema | 19.0% |
| T1 hit rate, later queries | 90.5% |
| T2 (cross-schema) hit rate | 0.0% |
| T1 hit p50 / p90 / p99 | 1.6 / 16.0 / 72.9 us |
| cold walk p50 / p90 / p99 | 6.8 / 10.1 / 10.6 ms |
| mask time per query, first of schema p50 / p90 | 114 / 231 ms |
| mask time per query, later p50 / p90 | 0.9 / 50.7 ms |
| T1 entries / payload at end | 21,315 / 6,175 MB |
| process RSS growth at end | 14,424 MB |
| audit append / verify per record | 2.91 / 0.48 us |
| audit record size (JSON) | 225 B |

Cache growth over the stream (queries: schemas seen, T1 MB, RSS growth MB): 100: 70, 539, 1,553; 900: 139, 2,616, 6,923; 1700: 140, 3,725, 9,683; 2500: 140, 4,495, 11,593; 3300: 140, 4,977, 12,801; 4100: 140, 5,427, 13,925; 4900: 140, 5,719, 14,424; 5700: 140, 5,936, 14,424; 6500: 140, 6,115, 14,424

## Spider dev (20 schemas), T2 on

Split `dev`: 1034 gold queries over 20 schemas, shuffled (seed 0); tokenizer `Qwen/Qwen2.5-0.5B-Instruct`; T2 tier on; 27793 mask steps; rejected replays 0 (gold queries outside the grammar). Host: local dev: Apple M3 Max, 36 GB (unpinned).

| measure | value |
|:---|---:|
| grammar + trie build (once) | 0.66 s |
| schema specialization p50 / p90 | 1.7 / 1.8 ms |
| T1 hit rate, all steps | 89.7% |
| T1 hit rate, first query of a schema | 25.7% |
| T1 hit rate, later queries | 91.2% |
| T2 (cross-schema) hit rate | 0.0% |
| T1 hit p50 / p90 / p99 | 1.5 / 13.4 / 57.7 us |
| cold walk p50 / p90 / p99 | 6.7 / 10.1 / 11.1 ms |
| mask time per query, first of schema p50 / p90 | 129 / 231 ms |
| mask time per query, later p50 / p90 | 0.6 / 45.4 ms |
| T1 entries / payload at end | 2,860 / 811 MB |
| process RSS growth at end | 2,015 MB |
| audit append / verify per record | 2.44 / 0.49 us |
| audit record size (JSON) | 225 B |

Cache growth over the stream (queries: schemas seen, T1 MB, RSS growth MB): 100: 19, 305, 759; 200: 19, 432, 1,073; 300: 20, 533, 1,327; 400: 20, 602, 1,497; 500: 20, 658, 1,640; 600: 20, 707, 1,762; 700: 20, 745, 1,855; 800: 20, 770, 1,918; 900: 20, 800, 1,989; 1000: 20, 807, 2,006; 1034: 20, 811, 2,015

## Spider dev (20 schemas), T2 off

Split `dev`: 1034 gold queries over 20 schemas, shuffled (seed 0); tokenizer `Qwen/Qwen2.5-0.5B-Instruct`; T2 tier off; 27793 mask steps; rejected replays 0 (gold queries outside the grammar). Host: local dev: Apple M3 Max, 36 GB (unpinned).

| measure | value |
|:---|---:|
| grammar + trie build (once) | 0.65 s |
| schema specialization p50 / p90 | 1.7 / 1.8 ms |
| T1 hit rate, all steps | 89.7% |
| T1 hit rate, first query of a schema | 25.7% |
| T1 hit rate, later queries | 91.2% |
| T2 (cross-schema) hit rate | 0.0% |
| T1 hit p50 / p90 / p99 | 1.5 / 12.3 / 56.7 us |
| cold walk p50 / p90 / p99 | 6.7 / 10.1 / 10.8 ms |
| mask time per query, first of schema p50 / p90 | 130 / 229 ms |
| mask time per query, later p50 / p90 | 0.6 / 45.1 ms |
| T1 entries / payload at end | 2,860 / 811 MB |
| process RSS growth at end | 2,017 MB |
| audit append / verify per record | 2.55 / 0.49 us |
| audit record size (JSON) | 225 B |

Cache growth over the stream (queries: schemas seen, T1 MB, RSS growth MB): 100: 19, 305, 758; 200: 19, 432, 1,073; 300: 20, 533, 1,326; 400: 20, 602, 1,498; 500: 20, 658, 1,642; 600: 20, 707, 1,763; 700: 20, 745, 1,855; 800: 20, 770, 1,919; 900: 20, 800, 1,990; 1000: 20, 807, 2,008; 1034: 20, 811, 2,017

