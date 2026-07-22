# GRID - Grammar-Railed Decoding

Constrained decoding for LLMs with **provable guarantees and no silent
errors**: every constraint is either *enforced* by the token mask, or
*named* in the output. Nothing is quietly dropped.

GRID compiles grammars (SQL-first; JSON Schema via `grid.jsonschema`) into
LALR(1) tables with constrained terminals, and masks tokens through a
configuration-keyed viable-prefix walk (Rust kernels on the hot path). On
top of the engine: role/schema policy projection (forbidden operations are
unreachable *by construction*), a hash-chained replayable audit trail, and
checker-guided repair for the provably mask-unenforceable residue.

```bash
pip install grid-guardrail
```

## JSON Schema in one call

```python
from grid.jsonschema import compile_json_schema

source, recorded = compile_json_schema(schema)   # -> .grid grammar source
# `recorded` names every constraint present but not mask-enforced (default
# mode records; strict=True refuses instead). Nothing is silently ignored.
```

Measured on [JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench)
(11,306 real-world schemas, all instance tests, one machine, current
engine versions):

| engine | passing | declared | false-rejects | silent accepts |
|---|---:|---:|---:|---:|
| **GRID 0.2.5** | 10,117 (89.5%) | 668 | **3** | 502, *every one recorded* |
| llguidance 1.7.6 | 9,487 (83.9%) | 1,797 | 22 | 0 |
| XGrammar 0.2.3 | 10,212 (90.3%) | 51 | 427 | 627 |

Three philosophies: llguidance refuses what it can't enforce perfectly;
XGrammar compiles everything and leaks silently. GRID sits at the frontier:
near-best coverage, near-zero false-rejects, and a per-schema record of
anything unenforced. Keyword-by-keyword status:
[`grid/jsonschema/SUPPORT.md`](grid/jsonschema/SUPPORT.md). The official
JSON-Schema-Test-Suite runs in CI under that contract.

Timing note: 0.2.x is the **correctness epoch**; per-token/compile timings
are recorded in the versioned reports but deliberately unoptimized;
performance is the 0.3.x epoch. On GRID's home turf (SQL/CFG grammars) the
warm-path mask is p50 3.7 µs (see `bench/RESULTS.md`).

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

Per-role and per-schema grammars make unauthorized verbs, tables, and
columns unreachable at decode time; the audit chain reconstructs, bit for
bit, what the model was permitted to generate at every step. Decode-time
masking is deterministic *capability reduction*; pair it with an
independent check where the SQL executes, both compiled from one policy
source.

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
The design credits the Outlines paper as inspiration; GRID's design and
implementation are its own throughout.

## License

Apache-2.0. Cite via [`CITATION.cff`](CITATION.cff)
([arXiv:2607.11951](https://arxiv.org/abs/2607.11951)).
