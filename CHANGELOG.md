# Changelog

Versions in the 0.2.x line are **correctness-only** (the coverage epoch,
DESIGN-JSON-COVERAGE.md): error metrics are the headline; timings are
recorded, not optimized (kernel frozen at v7). Speed work is the 0.3.x epoch.

## 0.2.5 — 2026-07-21
- `grid.jsonschema` package: JSON Schema -> grammar compilation promoted out
  of `bench/` with public API `compile_json_schema(schema, strict=False)`.
- Official JSON-Schema-Test-Suite (draft-07 + 2020-12) in CI under the
  honesty contract (valid never rejected; invalid accepted only if recorded).
  Found and fixed: integer-type zero-fraction floats, typeless multipleOf,
  one-sided properties vs additionalProperties under merge, items/prefixItems
  cross-level scoping.
- `grid/jsonschema/SUPPORT.md` keyword matrix; upstream PR kit
  (`bench/upstream/`) verified under jsonschemabench's own runner.
- Full-set (11,306): 10,117 passing (89.5%), 3 false-rejects, all
  unenforced constraints recorded per schema.

## 0.2.4 — 2026-07-20
- patternProperties overlap families -> recorded fallbacks; general object
  negation; record-and-drop for unrewritable narrowing keywords; branch
  string-value unification (kills a token-capture false-reject class);
  required-through-patternProperties satisfiability fix.

## 0.2.2 / 0.2.3 — 2026-07-20
- Full-set validation-error hunt: composite-enum maximal-munch capture,
  untyped type-sniffing, anyOf const harmonization, draft-<=07 $ref-replaces-
  siblings, order-free object machine beyond the required cap, routing-
  terminal degradation exemption, unsatisfiable-schemas -> never-grammar.

## 0.2.1 — 2026-07-19
- Dialect `{m,n}` bounded repetition (parse-time expansion; kernel untouched);
  length windows <= 64 chars enforced via a compact counting form.

## 0.2.0 — 2026-07-19
- Coverage sprint 1: schema normalization (allOf merge, dependencies,
  if/then/else, not), constrained terminals (pattern/format/length/bounds),
  order-free objects, hash-consed rules. Sample passing 206 -> 268.

## 0.0.7 — 2023-11 (baseline)
- SQL-first engine: configuration-keyed viable-prefix masks, byte-level
  token<->terminal bridge, Rust kernels, RBAC/schema projections, audit
  chain, checker-guided repair.
