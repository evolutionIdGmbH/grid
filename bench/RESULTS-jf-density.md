# S1 Stage 0 — forced-run (jump-forward) density (bench/jf_density.py)

Local leg, 2026-07-30: JSONSchemaBench replay (tmp/jsb-src/maskbench/data),
sample 5/split seed 0, tokenizer **gpt2** (the cached local tokenizer; box
reruns must use the serving model's — larger vocabs admit more alternative
tokenizations, so token-level density likely DROPS at 128k+). 99 schemas
compiled, 6 failed/timeout, 106 instances, 19,467 replay steps, **zero
replay desyncs**. Byte leg capped at 200 bytes/schema.

| split | steps | forced | sv@8 (lower) | svb@8 (upper) | byteF% | cov% |
|:---|---:|---:|---:|---:|---:|---:|
| BFCL_simple | 171 | 13.5% | 0.0% | 13.5% | 61.2% | 8% |
| BFCL_parallel | 139 | 10.1% | 2.2% | 10.1% | 51.9% | 7% |
| BFCL_parallel_multiple | 160 | 10.0% | 0.0% | 10.0% | 41.5% | 8% |
| BFCL_sql | 229 | 8.7% | 0.0% | 8.7% | 33.8% | 9% |
| BFCL_multiple | 157 | 8.3% | 0.6% | 8.3% | 46.0% | 6% |
| Github_easy | 409 | 7.8% | 0.0% | 7.8% | 20.7% | 25% |
| WashingtonPost | 3124 | 5.3% | 0.8% | 5.3% | 25.6% | 64% |
| Github_hard | 1824 | 4.7% | 0.2% | 4.7% | 29.0% | 74% |
| (13 more splits) | … | 0–2.5% | ~0% | 0–2.5% | 0–11% | — |
| **TOTAL** | **19,467** | **2.3%** | **0.3%** | **2.3%** | **21.5%** | **20%** |

Full per-split JSON (run histogram included) reproduces via the command in
the module docstring.

## Reading

1. **Decision gate: PASS** — forced steps >= 5% on 8/21 splits (plan rule:
   ">= ~5% on at least one target workload"). Delivery proceeds.
2. **Forced runs are almost all length 1** (isolated forced tokens between
   free spans: `":`, `",`, closing braces merged into multi-byte BPE tokens
   admit several viable tokenizations, killing singletons). Consequence:
   the no-bonus lower bound sv@8 is ~0.3% — the realistic win depends
   ENTIRELY on vLLM's bonus-token mechanics (a fully-accepted draft step
   emits one extra sampled token, so an isolated forced token saves one
   full step). svb@8 then equals forced density: **~2% average, 8–13% on
   rigid function-call-style splits**. Verifying bonus-token behavior is a
   REQUIRED box-probe question (S1 step 3) before any headline claim.
3. **The byte-level ceiling is ~10x the token-level lever** (21.5% forced
   bytes vs 2.3% forced tokens; token chains cover ~20% of forced-byte
   mass, far under the ~half threshold of S1 step 8). This re-confirms the
   DESIGN.md §12 assessment that byte-level JF with re-tokenization is
   where the real mass is — it stays a deliberately deferred v2 item
   (audit-record + `_guide_states` key hazards), NOT part of S1.
4. j_max=8 is a non-binding cap on this corpus (runs of length > 8 are
   rare); raising it buys nothing here.

Spider SQL leg: needs the Spider dataset + generations — not in this repo;
run on the GPU box alongside the serving bake (S1 step 6).
