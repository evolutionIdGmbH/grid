# GRID: Grammar-Railed Decoding

[![CI](https://github.com/evolutionIdGmbH/grid/actions/workflows/ci.yml/badge.svg)](https://github.com/evolutionIdGmbH/grid/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/grid-guardrail)](https://pypi.org/project/grid-guardrail/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21486746-blue)](https://doi.org/10.5281/zenodo.21486746)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

GRID is a structured-outputs engine for LLM serving. Point your stack's
grammar backend at it — vLLM today, a typed-pipeline adapter for DSPy in
[`grid/integrations/`](grid/integrations/dspy_adapter.py), SGLang and
llama.cpp on the [integration list](docs/integrations-plan.md) — and every
JSON-Schema or SQL/CFG-constrained request is enforced by token masks compiled before
sampling: a JSON Schema (or any context-free grammar — SQL was first)
becomes LALR(1) tables with constrained terminals, walked by Rust kernels.
Its founding rule is that **an engine must never fail silently**: every
constraint is either *enforced* by the mask, *recorded* by name in the
result so you know exactly what to re-validate, or *declared* unsupported up
front. Nothing in between.

```bash
pip install "grid-guardrail[kernel]"   # engine + Rust mask kernels (5 platforms)
```

```python
from grid.jsonschema import compile_json_schema

source, recorded = compile_json_schema(schema)   # -> .grid grammar source
# `recorded` names every constraint present but not mask-enforced (default
# mode records; strict=True refuses instead). Nothing is silently ignored.
```

**Deciding in 30 seconds:**

- **Serving schemas you don't control** — users, tenants, model-generated
  tool definitions? Run GRID, and pick it for its worst case, not its
  median: 3 valid-instance rejections and no silent enforcement gaps on the
  11,306-schema corpus below. The full argument — including the two cases
  where you should run llguidance or the stack default instead — is
  [docs/choosing-a-backend.md](docs/choosing-a-backend.md).
- **Need decode-time RBAC, a replayable hash-chained audit trail, or a
  machine-generated list of what was *not* enforced?** That layer is GRID's
  home ground; mask engines don't attempt it:
  [docs/beyond-the-mask.md](docs/beyond-the-mask.md).

## Where GRID stands — full JSONSchemaBench, three engines, one machine

All 11,306 schemas of [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench),
every valid/invalid test instance byte-walked, engines at current versions,
identical runner semantics and tokenizer (Llama-3.1 128k). Full per-schema
statuses and protocol: [`bench/RESULTS-jsonschemabench-v0.4.0.md`](bench/RESULTS-jsonschemabench-v0.4.0.md).

| engine | passing | declared unsupported | valid rejected | invalid accepted | never terminates |
|---|---:|---:|---:|---:|---:|
| **GRID 0.4.0** | 10,159 (89.9%) | 637 | **3** | 507 — **every one named** | **0** |
| llguidance 1.7.6 | 9,487 (83.9%) | 1,797 | 22 | 0 | 0 |
| XGrammar 0.2.3 | 10,212 (90.3%) | 51 | 427 | 627 — **all silent** | 0 |

All counts are schemas; the outcomes table below counts failing *instances*
(GRID's 507 schemas correspond to 876 instances, XGrammar's 627 to 1,493).

Three engines, three philosophies. llguidance refuses what it cannot enforce
— safe, at the cost of 16% of the corpus. XGrammar compiles almost everything
— the highest passing count, bought with the two worst failure modes: 427
schemas where **valid requests are rejected**, and 627 where invalid output
is accepted **with no indication anything was unenforced**. GRID takes the
middle contract: near-XGrammar coverage, near-llguidance safety (3
false-rejects), and when a constraint cannot be mask-enforced, the response
carries its name so downstream code re-checks exactly that.
Keyword-by-keyword status: [`grid/jsonschema/SUPPORT.md`](grid/jsonschema/SUPPORT.md);
the official JSON-Schema-Test-Suite runs in CI under that contract.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/coverage-by-split-dark.svg">
  <img alt="Per-split coverage across GRID, llguidance, XGrammar" src="docs/assets/coverage-by-split-light.svg">
</picture>

### Latency and outcomes (one runner, one accounting)
times in µs , lower is better

| metric | GRID 0.4.0 | llguidance 1.7.6 | XGrammar 0.2.3 |
|:--|--:|--:|--:|
| TBM avg | 486 | **21** | 191 |
| TBM p99 | 7,741 | **294** | 757 |
| TBM p99.9 | 8,638 | **1,063** | 50,665 |
| TTFM avg | 60,905 | **624** | 362,186 |
| TTFM p99 | 1,142,255 | **6,189** | 5,037,974 |
| tokens | 3,491,288 | 2,958,083 | 3,468,252 |
| compile error | 637 | 1,797 | 51 |
| timeout | **0** | **0** | **0** |
| validation error | **5** | 32 | 671 |
| invalidation error | 876 | **0** | 1,493 |

Reading the table:

- The three engines sit at different points of the coverage/upfrontness/
  latency trade-off: compile errors are *declared* non-support (visible,
  safe — which is why that row carries no bold: lowest is not simply best);
  validation errors (valid instance rejected) and invalidation errors
  (invalid instance accepted) are silent correctness gaps — except GRID's,
  whose invalidation errors each carry the names of the unenforced
  constraints involved.
- llguidance's lazy lexer makes it the latency reference point on every
  timing row; nothing else is close. GRID's compile average
  dropped 419ms -> 66ms across the 0.3/0.4 epochs (p50 7.4ms), and — new at
  0.4.0 — **zero schemas fail to terminate**: every one of 11,306
  compiles, declares, or budget-declines deterministically inside the wall.
- GRID's TBM p25-p75 is the grid_core kernel hit path (masks up to 512
  terminals run in-kernel); the p90+ tail is cold-miss trie walks over the
  128k vocabulary. MaskBench runs each schema once — the write-back cache
  that amortizes GRID's misses across requests in serving never warms here;
  the cold walk was cut first 9.3x by the kernel v5.1 verdict-equivalence
  grouping (TBM p90 27.8ms -> 208µs vs the v3-era run), then a further ~2.8x
  by the kernel v7 fused walk->blob->register path that eliminated the
  Python-side per-cold-entry materialization cost and its gen-2 GC pauses
  (208µs -> 75µs on the SQL harness); warm-path mask p50 is 25µs (3.7µs on
  SQL/CFG grammars, GRID's home turf —
  [`bench/RESULTS-engines-sql-v0.4.0.md`](bench/RESULTS-engines-sql-v0.4.0.md)).
- GRID's TTFM is the Python table build per schema. On redeployment the
  artifact store reloads compiled schemas at warm-hit medians of 24-37ms
  for families whose cold compiles are 1.5-7.4s, neutral on cheap schemas,
  declarations identical through the store (H100 runner stamp —
  `bench/perfbench/BAKEOFF.md`).
- GRID's 5 validation errors on 11,306 schemas include 3 genuine
  valid-instance rejections — definition-order properties and spec-default
  additionalProperties (typed extras included) are accepted everywhere else.

GRID notes: grid_core kernels active on 100% of compiled schemas (the rest
exceed the 64-terminal kernel bound and run the pure-Python spec path).

Ignored-but-accepted constraints (counted per schema; the XGrammar-default
convention — these surface as invalidation errors when an invalid instance
hinges on them): oneOf-exclusivity (433), required-not-enforced (350),
scanner-budget: constrained string degraded (240), maxLength-with-pattern
(170), minLength-with-pattern (162), length windows beyond cap (453 across
four cap classes), string-constraint-terminal-too-large (97), minimum (73).

Compile-error reasons (v1 subset boundaries, llguidance-style upfront):
LALRConflictError (519), Unsupported: allOf merge failed (30), RxUnsupported
(9), terminal budget (9), rule budget (8), anyOf/oneOf/$ref sibling-key
families (20), LALRBudgetExceeded (2).

### On the serving box (H100, vLLM 0.26, Qwen2.5-7B)

Mask latency only matters through its effect on decode. Measured end to end
on the declared runner ([`bench/RESULTS-serving-v0.4.0.md`](bench/RESULTS-serving-v0.4.0.md)):

- **TPOT overhead vs unconstrained: −0.00% / +0.21% / +0.71%** at batch
  1/8/32, heterogeneous schemas per batch. Cold schema specialize **14.8 ms
  once**, warm TTFT **1.39 ms**; concurrent cold starts coalesce
  (single-flight: 1 build / 8 waiters). Known limitation, declared in the
  record: a never-seen schema costs its batch ~24% TPOT during its 0.75 s
  specialization window.
- **Constraining improves the end task.** Spider dev, all 1,034 questions,
  greedy, no repair loop: **55.2% execution accuracy constrained vs 52.7%
  unconstrained** ([`bench/RESULTS-spider-v0.4.0.md`](bench/RESULTS-spider-v0.4.0.md)).
  Masking never removes a correct continuation; it removes SQL that cannot
  execute.

## What GRID has that the others are not designed for

- **The recorded-constraints contract.** Per response, the names of every
  constraint that was accepted but not mask-enforced, so downstream code
  re-validates exactly those — or strict mode, which turns them into declared
  errors, llguidance-style. Silent acceptance is not a mode GRID has.
- **Deterministic termination as a feature.** Pathological schemas cannot
  hang the compiler: construction budgets (scanner components, LALR items)
  convert would-be timeouts into declared, recorded declines in bounded time,
  with input-derived fire points that are identical on every machine and run.
- **Conflict-retry compilation.** LALR conflicts from overlapping per-branch
  string values trigger one automatic re-normalization that unifies the
  branches (recorded as widened); schemas that compile normally never execute
  the retry path.
- **A policy and audit layer underneath.** GRID began as an enterprise SQL
  guardrail: grammars are role-projected (forbidden operations unreachable
  *by construction*), every mask decision lands in a hash-chained,
  replayable audit trail, and the provably mask-unenforceable residue goes to
  checker-guided repair. JSON Schema is a front end; any context-free grammar
  in the `.grid` dialect gets the same machinery.
- **Warm redeployments.** A versioned on-disk artifact store (default off)
  reloads grammars, tables, scanners, and cross-schema terminal components,
  keyed by a code epoch that invalidates wholesale on any engine change and
  never executes a module to compute it.
- **Serving-side machinery, measured before advertised.** A jump-forward API
  (forced token runs decoded without forward passes), a tokenizer slicer for
  vocabulary-wide string masks, and an in-kernel lazy scanner (grid_core v8)
  — each behind a flag with a differential parity gate. The slicer's GPU
  stamp is banked (TBM avg 2.3x, outcome parity 315/315) pending a default
  flip; jump-forward failed its probe against vLLM 0.26's spec-token
  accounting and stays off until the port passes. Nothing ships default-on
  without a passing probe, and the probes are in the repo.
- **A measurement discipline you can audit.** `bench/perfbench/` holds the
  two-column TTFM profiler (compile-only and first-mask-included), an outcome
  classifier that refuses to count crashed or partial records as results, the
  per-candidate A/B history of every optimization that shipped (and the ones
  rejected, with reasons), and a roadmap where every deferred claim is named.
  Numbers in this README trace to committed per-schema statuses.

## SQL with policy compiled in

```python
import grid
from grid import generate, samplers
from grid.policy.bundle import PolicyBundle
from grid.policy.schema import SchemaSnapshot

model = grid.models.transformers_model.TransformersModel.from_pretrained("gpt2")
g = generate.sql(
    model,
    open("grammars/sql_subset.grid").read(),
    policy=PolicyBundle.from_store({"analyst": {"verbs": ["select"]}}, "analyst"),
    schema=SchemaSnapshot.from_dict({"users": ["id", "name", "email"]}),
    sampler=samplers.multinomial(temperature=0.7),
)
result = g("List all user names", max_tokens=64, seed=42)
```

Per-role and per-schema grammars make unauthorized verbs, tables, and columns
unreachable at decode time; the audit chain reconstructs, bit for bit, what
the model was permitted to generate at every step. Decode-time masking is
deterministic *capability reduction*; pair it with an independent check where
the SQL executes, both compiled from one policy source. The measured RBAC
and replay records, and why this layer cannot be retrofitted onto a mask
engine, are in [docs/beyond-the-mask.md](docs/beyond-the-mask.md).

## Guarantees

Soundness, completeness, termination, and near-constant per-token cost are
stated with explicit preconditions and paired with empirical tests
(`tests/`, differential against a trial-parse oracle). See
[`DESIGN.md`](DESIGN.md) (architecture), [`GUARDRAIL-REDESIGN.md`](GUARDRAIL-REDESIGN.md)
(design rationale with proofs), [`LESSONS.md`](LESSONS.md) (the measured
history), [`ONBOARDING.md`](ONBOARDING.md) (guided tour).


## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -q                       # verification suite
(cd grid_core && maturin develop --release)      # optional Rust kernels
.venv-bench/bin/python bench/compare_engines.py  # cross-engine SQL harness
```

Benchmark methodology and full reports live in `bench/`: pinned engine
versions, declared runners, full error distributions, no cherry-picking.

## Versions

| epoch | releases | focus |
|---|---|---|
| 0.2.x | 0.2.0-0.2.5 | correctness: JSON coverage 65% -> 89.5%, the honesty contract, official JSON-Schema-Test-Suite gate in CI |
| 0.3.x | 0.3.0 | compile-time tail: measured bake-off of 20 candidates, 7 shipped; p99 4.4s -> 1.2s |
| 0.4.x | 0.4.0 | zero timeouts, kernel v8 (in-kernel lazy product), construction budgets, conflict retry, artifact store |

## Credits

The [Outlines](https://github.com/dottxt-ai/outlines) paper started this line
of work for us in 2023; GRID's design and implementation are its own
throughout. [llguidance](https://github.com/guidance-ai/llguidance) sets the
compile-latency bar and the declare-what-you-cannot-do convention our strict
mode mirrors. [XGrammar](https://github.com/mlc-ai/xgrammar) sets the
coverage bar. [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
is the yardstick for all of it — it found real bugs in GRID that 95 unit
tests missed.

## License

Apache-2.0. Cite via [`CITATION.cff`](CITATION.cff)
([arXiv:2607.11951](https://arxiv.org/abs/2607.11951), artifact DOI
[10.5281/zenodo.21486746](https://doi.org/10.5281/zenodo.21486746)).
