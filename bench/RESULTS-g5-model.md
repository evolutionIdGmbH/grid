# G5 end-to-end S/C/T at scale — model arm

Host: Lambda 1xH100 PCIe 80GB, Ubuntu 24.04 (declared runner) | grammar: `grammars/sql_subset.grid` + L3 schema lexicons | model: `Qwen/Qwen2.5-0.5B-Instruct` | mode-1 GRID-owned loop, multinomial sampler | 1,000 seeded generations | wall 35.2 min

Real model logits drive the sampler; the mask constrains; GRID owns the loop and writes the audit chain. EOS-only-at-ACCEPT and the mask invariant are asserted inside the loop (`grid/generate/api.py`). This is the model-in-loop complement to the 10k model-free forced-random-walk arm (`RESULTS-g5-walk.md`).

| check | pass | value |
|---|---|---|
| outputs parse under own grammar (binding) | PASS | 1000/1000 |
| audit chain verifies every generation | PASS | 1000/1000 |
| DeadEndError == 0 | PASS | 0 |
| no generation exceeds max_tokens | PASS | 0 over |
| coverage: multi-token identifier events >= 50 | PASS | 1,165 |

Gate G5 (model arm): **PASS**. Reserve stops observed: 706; max paren nesting: 11.

Harness: `bench/g5_scale.py --arm model`.
