# JSON Schema support matrix — `grid.jsonschema`

Status of every keyword when compiling JSON Schema to a GRID grammar
(`compile_json_schema(schema, strict=False)`), as of the 0.2.x coverage epoch.

**Legend** — every keyword lands in exactly one bucket per use site:

- **enforced** — compiled into the grammar; the mask makes violations
  unreachable.
- **recorded** — accepted but not (fully) enforced; the constraint name is
  returned in the `recorded` set (per schema). Under `strict=True` these
  raise `Unsupported` instead (declared non-support). This is the honesty
  contract: nothing is ever *silently* unenforced.
- **declared** — compilation refuses (`Unsupported`): the construct cannot be
  expressed soundly, and dropping it would corrupt key routing or masking.
- **unsatisfiable-aware** — statically contradictory schemas compile to the
  never-grammar (all instances correctly rejected), not an error.

Measured on guidance-ai/jsonschemabench (11,306 schemas): 89.6% pass all
instance tests; 3 false-rejects; every unenforced constraint recorded.
Full numbers: `bench/RESULTS-jsonschemabench-v0.2.4-full.md`.

## Core / applicators

| keyword | status | notes |
|---|---|---|
| `type` (incl. lists) | enforced | integer⊂number honored |
| `enum` / `const` | enforced | values statically filtered against sibling constraints; numeric values admit both canonical forms (`2`, `2.0`); composite (object/array) values compile structurally (single-terminal literals would collide with maximal munch) |
| `anyOf` | enforced | branch string-value collisions harmonized (disjoint terminals) so LALR keeps all branches live; residual capture-risky cases may record `branch-string-values-unified` |
| `oneOf` | enforced as anyOf | exclusivity enforced only when branches are provably disjoint (type/discriminator); otherwise `oneOf-exclusivity` recorded |
| `allOf` | enforced via merge | keyword-algebra merge (llguidance-style); unmergeable residue declared |
| `not` | enforced for negatable shapes | type complements, enum/const, required, property-discriminators, min/maxItems, pattern (as complement), anyOf/allOf of those; residue **recorded** (`not-unenforced`) |
| `if`/`then`/`else` | enforced for negatable `if` | rewritten to `anyOf[if∧then, ¬if∧else]`; residue recorded (`if-unenforced`) |
| `$ref` (local, incl. cycles) | enforced | CFG recursion is native; draft-≤07 semantics honored ($ref replaces siblings); 2019-09+ siblings merged |
| `$ref` (external/remote) | declared | |
| `$defs`/`definitions` | enforced | preserved through all rewrites |

## Objects

| keyword | status | notes |
|---|---|---|
| `properties` | enforced | **any key order** (order-free member machine) |
| `required` | enforced (≤10 tracked) | subset machine over required keys; beyond the cap: any order accepted, `required-not-enforced` recorded (never a false-reject) |
| `additionalProperties` (false/schema) | enforced | |
| `patternProperties` | enforced | ECMA-subset patterns; declared-key overlaps merged + pattern-minus-keys subtraction where constructible (else recorded); overlapping patterns take union semantics, recorded |
| `propertyNames` | enforced for pattern/enum/const/length shapes | else declared |
| `minProperties`/`maxProperties` | enforced on generic objects | with declared properties: enforced when statically decidable, else recorded |
| `dependencies`/`dependentRequired`/`dependentSchemas` | enforced via variant expansion | residue recorded |
| `unevaluatedProperties` | recorded | |

## Arrays

| keyword | status | notes |
|---|---|---|
| `items` (schema) | enforced | |
| `items` (list, draft-07) + `additionalItems` | enforced | tuple grammars |
| `prefixItems` + `items` (2020-12) | enforced | |
| `minItems`/`maxItems` | enforced ≤256 | beyond: recorded |
| `uniqueItems` | recorded | not context-free; scheduled for the M3 semantic-refinement layer |
| `contains`/`minContains`/`maxContains` | recorded | M3 |
| `unevaluatedItems` | recorded | |

## Strings

| keyword | status | notes |
|---|---|---|
| `pattern` | enforced (ECMA subset) | unanchored-search semantics; escape/UTF-8-aware over canonical serializations; lookarounds/backrefs/`\b`/`\p` recorded |
| `minLength`/`maxLength` | enforced ≤64 chars | dialect `{m,n}` windows; larger windows recorded (scanner-build cost — 0.3.x) |
| `format` | enforced: date, time, date-time, uuid, ipv4, email, hostname, uri, uri-reference | others recorded by name (`format:<name>`); regexes reference-validated to avoid over-rejection |
| pattern ∧ length / multiple patterns | first dimension enforced, rest recorded | regex-language intersection is out of textual reach |

## Numbers

| keyword | status | notes |
|---|---|---|
| `minimum`/`maximum`/`exclusive*` (integer type) | enforced | exact digit-range terminals |
| bounds on `number` | enforced for integer-valued bounds < 1e15 | over canonical float forms incl. exponent windows; else recorded |
| `multipleOf` | recorded | integer conjunctions merge to lcm; enforcement is M3 |
| draft-04 boolean `exclusiveMinimum/Maximum` | enforced | normalized to numeric form |

## Modes and guarantees

- **Default mode** mirrors the XGrammar-default convention (permissive,
  recorded); **strict mode** mirrors llguidance (declared).
- Exactness is defined over **canonical serializations**
  (`json.dumps(..., ensure_ascii=False)`) — the only forms a masked decode
  can produce.
- Scanner-cost budgets may demote a *value-position* constrained terminal to
  the generic string terminal (always recorded); *key-position* terminals are
  never demoted (they route object pairs).
- Statically unsatisfiable schemas (contradictory bounds/counts, emptied
  enums, required∧forbidden) compile to the never-grammar: every instance is
  correctly rejected.
