# End-to-end soundness/completeness/termination at scale — model arm

Host: Lambda 1xH100 PCIe 80GB, Ubuntu 24.04 (declared runner) | grammar: `grammars/sql_subset.grid` + L3 schema lexicons | model: `Qwen/Qwen2.5-0.5B-Instruct` | mode-1 GRID-owned loop, multinomial sampler | 1,000 seeded generations | wall 35.2 min

Real model logits drive the sampler; the mask constrains; GRID owns the loop and writes the audit chain. EOS-only-at-ACCEPT and the mask invariant are asserted inside the loop (`grid/generate/api.py`). This is the model-in-loop complement to the 10k model-free forced-random-walk arm (`RESULTS-g5-walk.md`).

| check | measured |
|---|---|
| outputs parse under own grammar (binding) | 1000/1000 |
| audit chain verifies every generation | 1000/1000 |
| DeadEndError == 0 | 0 |
| no generation exceeds max_tokens | 0 over |
| coverage: multi-token identifier events >= 50 | 1,165 |

Summary (model arm): all 1,000 model-in-loop generations parse under their own grammar, every audit chain verifies, and there are zero dead-ends. Reserve stops observed: 706; max paren nesting: 11.

Harness: `bench/g5_scale.py --arm model`.
