"""JSON Schema -> .grid grammar compiler (MaskBench arm; 0.2.x coverage epoch).

Pipeline: jsonschema_normalize.normalize() rewrites allOf/$ref-siblings/
dependencies/if-then-else/not into the core subset (M2), then this compiler
enforces the remaining constraints (M1):

- objects accept properties in ANY key order (an order-free member machine
  tracking the subset of required keys seen — 2^R chain rules, R capped);
  `required` enforced by construction, extras per additionalProperties
  (default: allowed, generic);
- string constraints as constrained terminals: pattern (ECMA subset), format
  (curated table), min/maxLength (unrolled, escape/UTF-8 aware),
  x-grid-not-values (finite complement) — one dimension enforced per
  position, the rest recorded;
- numeric bounds as digit-range terminals (integers exact; numbers over
  canonical float forms for integer-valued bounds);
- min/maxItems (unrolled ≤ cap), tuple arrays (draft-07 items-list +
  additionalItems; 2020-12 prefixItems + items), min/maxProperties on
  generic objects;
- enum/const via exact serialized literals, statically filtered against
  sibling constraints (numeric values admit both canonical forms: 2 and 2.0);
- anyOf/oneOf as alternation; oneOf exclusivity is NOT enforced — recorded,
  except when branches are provably disjoint (types / required-discriminator);
- local $ref/$defs/definitions with cycles (CFG recursion);
- x-grid-forbid-keys (from the dependencies rewrite): forbidden declared
  props dropped, extras key terminal excludes the forbidden names.

Modes: default records unenforced constraints per schema (XGrammar-default
convention; they can surface as invalidation errors); strict=True raises
Unsupported instead (llguidance-style declared non-support).

Still unsupported (raises Unsupported -> "compile error" bucket):
patternProperties, propertyNames (next stage), residual not/if/unevaluated*,
external $ref, false schemas, grammars past the size caps.

Whitespace: %ignore /[ \\t\\n\\r]+/ — the JSON-spec definition.
"""

from __future__ import annotations

import json
import math
from typing import Any

from grid.jsonschema import rx
from grid.jsonschema.normalize import FALSE_SCHEMA, normalize

MAX_PROPERTIES = 256
MAX_NAMED_TERMINALS = 2000
MAX_RULES = 20_000
MAX_ITEMS_UNROLL = 256
MAX_TERMINAL_SRC = 6_000    # scanner-DFA budget per constrained terminal

_ANNOTATIONS = {
    "title", "description", "default", "examples", "$schema", "$id", "$comment",
    "readOnly", "writeOnly", "deprecated", "$defs", "definitions", "id",
    "$vocabulary", "$anchor",
}
# recorded when present but not enforced at this position
_RECORD_ONLY = {
    "uniqueItems", "contains", "minContains", "maxContains", "multipleOf",
    "contentMediaType", "contentEncoding", "unevaluatedItems",
}
_UNSUPPORTED_KEYS = {
    "not", "if", "then", "else",
    "dependencies", "dependentRequired", "dependentSchemas",
    "unevaluatedProperties",
}
_STRING_KEYS = {"pattern", "format", "minLength", "maxLength"}
_NUMBER_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "multipleOf"}

_REGEX_META = set("()[]{}*+?|\\./")

_HEX = "[0-9a-fA-F]"
STRING_RX = (
    r'"([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u' + _HEX + _HEX + _HEX + _HEX + r"))*\""
)
NUMBER_RX = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
INT_RX = r"-?(0|[1-9][0-9]*)(\.0)?"


class Unsupported(Exception):
    """Schema uses a feature outside the supported subset."""


def _regex_literal(text: str) -> str:
    """Exact-match grid-regex for `text` (UTF-8 bytes; metachars escaped)."""
    out = []
    for b in text.encode("utf-8"):
        ch = chr(b)
        if b < 0x20 or b > 0x7E:
            out.append(f"\\x{b:02x}")
        elif ch in _REGEX_META:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _num_literal_rx(v) -> str:
    """Regex admitting every canonical serialization equal to number v."""
    if isinstance(v, bool):
        return _regex_literal(json.dumps(v))
    if isinstance(v, int):
        return _regex_literal(str(v)) + r"(\.0)?"
    if isinstance(v, float) and v.is_integer() and abs(v) < 10**15:
        base = _regex_literal(str(int(v)))
        return base + r"(\.0)?"
    return _regex_literal(json.dumps(v))


def _const_schema(v) -> dict:
    """Structural schema exactly matching the composite constant v."""
    if isinstance(v, dict):
        return {
            "type": "object",
            "properties": {k: _const_schema(x) if isinstance(x, (dict, list))
                           else {"const": x} for k, x in v.items()},
            "required": sorted(v.keys()),
            "additionalProperties": False,
        }
    if isinstance(v, list):
        return {
            "type": "array",
            "prefixItems": [_const_schema(x) if isinstance(x, (dict, list))
                            else {"const": x} for x in v],
            "items": False,
            "minItems": len(v),
        }
    return {"const": v}


class SchemaCompiler:
    def __init__(self, root_schema: Any, strict: bool = False) -> None:
        self.root = root_schema
        self.strict = strict
        self.rules: dict[str, list[str]] = {}
        self.rule_order: list[str] = []
        self.memo: dict[int, str] = {}          # id(schema node) -> rule name
        self._keepalive: list[Any] = []         # nodes behind memo ids
        self.key_terms: dict[str, str] = {}     # property name -> terminal
        self.lit_terms: dict[str, str] = {}     # literal regex -> terminal
        self.rx_terms: dict[str, str] = {}      # constrained regex -> terminal
        self.needs: set[str] = set()            # STRING | NUMBER | INT | generic
        self.ignored: set[str] = set()          # recorded unenforced constraints
        self.degraded: set[str] = set()         # terminals demoted to STRING
        self.degraded_keep: set[str] = set()    # demoted, kept as clones
        self.routing_terms: set[str] = set()    # key-position terminals: NEVER
                                                # degrade (they route pairs; a
                                                # STRING clone breaks pp/pn
                                                # disjointness -> false rejects)
        self.rx_costs: dict[str, int] = {}      # terminal -> scanner-cost proxy
        self._n = 0

    # ------------------------------------------------------------- budget

    def _record(self, feat: str) -> None:
        if self.strict:
            raise Unsupported(f"strict: {feat}")
        self.ignored.add(feat)

    def _check_terms(self) -> None:
        if len(self.key_terms) + len(self.lit_terms) + len(self.rx_terms) \
                > MAX_NAMED_TERMINALS:
            raise Unsupported("terminal budget exceeded (size cap)")

    def _rule(self, hint: str) -> str:
        name = f"r{self._n}_{hint}"
        self._n += 1
        if self._n > MAX_RULES:
            raise Unsupported("rule budget exceeded (size cap)")
        self.rules[name] = []
        self.rule_order.append(name)
        return name

    def _key_term(self, key: str) -> str:
        t = self.key_terms.get(key)
        if t is None:
            t = f"K{len(self.key_terms)}"
            self.key_terms[key] = t
            self._check_terms()
        return t

    def _lit_term(self, value: Any) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            src = _num_literal_rx(value)
        else:
            src = _regex_literal(json.dumps(value, ensure_ascii=False))
        t = self.lit_terms.get(src)
        if t is None:
            t = f"E{len(self.lit_terms)}"
            self.lit_terms[src] = t
            self._check_terms()
        return t

    def _rx_term(self, source: str, cost: int | None = None) -> str:
        t = self.rx_terms.get(source)
        if t is None:
            t = f"S{len(self.rx_terms)}"
            self.rx_terms[source] = t
            self._check_terms()
        self.rx_costs[t] = max(self.rx_costs.get(t, 0),
                               cost if cost is not None else len(source))
        return t

    def _resolve_ref(self, ref: str) -> Any:
        if not ref.startswith("#"):
            raise Unsupported(f"external $ref {ref!r}")
        node: Any = self.root
        for part in ref[1:].split("/"):
            if not part:
                continue
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                node = node[int(part)]
            elif isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise Unsupported(f"unresolvable $ref {ref!r}")
        return node

    # ------------------------------------------------------- schema -> rules

    def rule_for(self, schema: Any) -> str:
        if schema is True or schema == {}:
            return self.generic_value()
        if schema is False or schema == FALSE_SCHEMA:
            # empty language: a grammar no canonical instance can enter —
            # every MaskBench instance is correctly rejected mid-stream
            if "never_v" not in self.rules:
                self.rules["never_v"] = ["NEVER"]
                self.rule_order.append("never_v")
                self.rx_terms.setdefault("\\x00", "NEVER")
            return "never_v"
        if not isinstance(schema, dict):
            raise Unsupported(f"schema node of type {type(schema).__name__}")

        got = self.memo.get(id(schema))
        if got is not None:
            return got

        if "$ref" in schema:
            from grid.jsonschema.normalize import _ASSERTIONS
            extra = (set(schema) - {"$ref"} - _ANNOTATIONS) & _ASSERTIONS
            if extra:
                # normalize() merges mergeable siblings; the residue is real
                raise Unsupported(f"$ref with sibling keys {sorted(extra)}")
            return self.rule_for(self._resolve_ref(schema["$ref"]))

        name = self._rule("v")
        self.memo[id(schema)] = name  # pre-register: recursive schemas terminate
        self._keepalive.append(schema)
        self.rules[name] = self._alternatives(schema)
        return name

    _RECORDABLE_UNSUPPORTED = {
        "not", "if", "then", "else", "dependencies", "dependentRequired",
        "dependentSchemas", "unevaluatedProperties", "unevaluatedItems",
    }

    def _alternatives(self, schema: dict) -> list[str]:
        bad = set(schema) & _UNSUPPORTED_KEYS
        if bad:
            # narrowing keywords normalize() could not rewrite: in default
            # mode they are RECORDED and dropped (XGrammar-default
            # convention; strict mode still declares) — structural keywords
            # (patternProperties/propertyNames) stay hard errors because
            # dropping them breaks key routing
            rec = bad & self._RECORDABLE_UNSUPPORTED
            hard = bad - rec
            if hard:
                raise Unsupported(f"unsupported keys {sorted(hard)}")
            for k in sorted(rec):
                self._record(f"{k}-unenforced")
            schema = {k: v for k, v in schema.items() if k not in rec}
        for k in set(schema) & _RECORD_ONLY:
            if k == "multipleOf":
                continue        # handled (or recorded) in the numeric path
            self._record(k)

        if "allOf" in schema:
            branches = schema["allOf"]
            if len(branches) == 1 and not (set(schema) - {"allOf"} - _ANNOTATIONS):
                return [self.rule_for(branches[0])]
            raise Unsupported("allOf (merge failed)")

        if "enum" in schema or "const" in schema:
            return self._enum_alts(schema)

        if schema.get("x-grid-branch-unified"):
            self._record("branch-string-values-unified")
            schema = {k: v for k, v in schema.items()
                      if k != "x-grid-branch-unified"}
        if "anyOf" in schema or "oneOf" in schema:
            key = "anyOf" if "anyOf" in schema else "oneOf"
            if key == "oneOf" and not self._provably_disjoint(schema[key]):
                self._record("oneOf-exclusivity")
            rest = set(schema) - {key} - _ANNOTATIONS - _RECORD_ONLY
            if rest - {"type"}:
                raise Unsupported(f"{key} with sibling keys {sorted(rest - {'type'})}")
            return [self.rule_for(b) for b in schema[key]]

        types = schema.get("type")
        if isinstance(types, list):
            out = []
            for t in types:
                sub = dict(schema)
                sub["type"] = t
                out.append(self.rule_for(sub))
            return out
        if types is None:
            return self._untyped(schema)

        if types == "object":
            return [self._object(schema)]
        if types == "array":
            return self._array_alts(schema)
        if types == "string":
            return [self._string_term(schema)]
        if types == "number":
            return [self._number_term(schema, integer=False)]
        if types == "integer":
            return [self._number_term(schema, integer=True)]
        if types == "boolean":
            return ['"true"', '"false"']
        if types == "null":
            return ['"null"']
        raise Unsupported(f"type {types!r}")

    # ---------------------------------------------------------- untyped

    def _untyped(self, schema: dict) -> list[str]:
        """Typeless schema: every keyword constrains ONLY instances of its
        own type — other JSON types pass free (e.g. {'properties': {}} on a
        string is vacuously valid)."""
        keys = set(schema) - _ANNOTATIONS - _RECORD_ONLY - {"x-grid-not-values"}
        obj_keys = keys & {"properties", "required", "additionalProperties",
                           "minProperties", "maxProperties",
                           "x-grid-forbid-keys", "patternProperties",
                           "propertyNames"}
        arr_keys = keys & {"items", "prefixItems", "additionalItems",
                           "minItems", "maxItems"}
        s_keys = keys & _STRING_KEYS
        n_keys = keys & (_NUMBER_KEYS - {"multipleOf"})
        if "multipleOf" in schema and schema["multipleOf"] not in (1, 1.0):
            self._record("multipleOf")
        notv = bool(schema.get("x-grid-not-values"))
        if not (obj_keys or arr_keys or s_keys or n_keys):
            if notv:
                return self._not_values_generic(schema["x-grid-not-values"])
            return [self.generic_value()]
        if notv:
            self._record("not-values")
        self.generic_value()
        alts: list[str] = ['"true"', '"false"', '"null"']
        if obj_keys:
            sub = dict(schema)
            sub["type"] = "object"
            alts.append(self._object(sub))
        else:
            alts.append("json_object")
        if arr_keys:
            sub = dict(schema)
            sub["type"] = "array"
            alts.extend(self._array_alts(sub))
        else:
            alts.append("json_array")
        alts.append(self._string_term(schema) if s_keys else "STRING")
        if n_keys:
            alts.append(self._number_term(schema, integer=False))
        else:
            alts.append("NUMBER")
        return alts

    def _not_values_generic(self, values: list) -> list[str]:
        if all(isinstance(v, str) for v in values):
            body = rx.not_literals_body(list(values))
            self.generic_value()
            return ["json_object", "json_array", '"true"', '"false"', '"null"',
                    "NUMBER", self._rx_term(rx.string_terminal_rx(body))]
        self._record("not-values")
        return [self.generic_value()]

    # ------------------------------------------------------------- enums

    def _enum_alts(self, schema: dict) -> list[str]:
        values = [schema["const"]] if "const" in schema else list(schema["enum"])
        if not values:
            raise Unsupported("empty enum")
        rest = {k: v for k, v in schema.items()
                if k not in ("enum", "const") and k not in _ANNOTATIONS}
        if rest:
            from grid.jsonschema.normalize import _valid
            kept = [v for v in values if _valid(v, rest, self.root)]
        else:
            kept = values
        if not kept:
            # genuinely unsatisfiable: every instance must be rejected
            return [self.rule_for(dict(FALSE_SCHEMA))]
        alts = []
        for v in kept:
            if isinstance(v, (dict, list)):
                # composite values CANNOT be single terminals: they'd start
                # with '{'/'[' and hard maximal munch would hold the scanner
                # hostage past the structural literal (observed: an
                # enum-object shadowing every object-open in the grammar)
                alts.append(self.rule_for(_const_schema(v)))
            else:
                alts.append(self._lit_term(v))
        return alts

    # ------------------------------------------------------------ strings

    def _string_term(self, schema: dict) -> str:
        pattern = schema.get("pattern")
        fmt = schema.get("format")
        min_l = schema.get("minLength", 0)
        max_l = schema.get("maxLength")
        notv = schema.get("x-grid-not-values")
        has_len = bool(min_l) or max_l is not None

        body = None
        if pattern is not None:
            try:
                body = rx.pattern_body(pattern)
                for k in ("format", "minLength", "maxLength"):
                    if k in schema:
                        self._record(f"{k}-with-pattern")
                if notv:
                    self._record("not-values")
            except rx.RxUnsupported as e:
                self._record(f"pattern ({e})")
                body = None
        if body is None and fmt is not None:
            fb = rx.format_body(fmt)
            if fb is None:
                self._record(f"format:{fmt}")
            else:
                body = fb
                if has_len:
                    self._record("length-with-format")
                if notv:
                    self._record("not-values")
        if body is None and notv is not None:
            if all(isinstance(v, str) for v in notv):
                body = rx.not_literals_body(list(notv))
                if has_len:
                    self._record("length-with-not-values")
            else:
                self._record("not-values")
        if schema.get("x-grid-extra-patterns"):
            self._record("extra-patterns (pattern conjunction)")
        notp = schema.get("x-grid-not-patterns")
        if notp:
            if body is None and len(notp) == 1 and not has_len:
                try:
                    body = rx.pattern_complement_body(notp[0])
                    if body is None:
                        raise Unsupported(
                            "not-pattern matches every string (false)")
                except rx.RxUnsupported as e:
                    self._record(f"not-pattern ({e})")
            else:
                self._record("not-pattern")
        if body is None and has_len:
            try:
                body = rx.length_body(min_l, max_l)
            except rx.RxUnsupported as e:
                self._record(f"length ({e})")
        if body is not None and len(body) > MAX_TERMINAL_SRC:
            self._record("string-constraint-terminal-too-large")
            body = None
        if body is None:
            self.needs.add("STRING")
            return "STRING"
        cost = None
        if body is not None and "{" in body and (min_l or max_l is not None):
            cost = 40 * (max_l if max_l is not None else min_l)
        return self._rx_term(rx.string_terminal_rx(body), cost=cost)

    # ------------------------------------------------------------ numbers

    def _bounds(self, schema: dict):
        lo, excl_lo = None, False
        if "minimum" in schema:
            lo, excl_lo = schema["minimum"], False
        if "exclusiveMinimum" in schema and isinstance(
                schema["exclusiveMinimum"], (int, float)) and not isinstance(
                schema["exclusiveMinimum"], bool):
            e = schema["exclusiveMinimum"]
            if lo is None or e >= lo:
                lo, excl_lo = e, True
        hi, excl_hi = None, False
        if "maximum" in schema:
            hi, excl_hi = schema["maximum"], False
        if "exclusiveMaximum" in schema and isinstance(
                schema["exclusiveMaximum"], (int, float)) and not isinstance(
                schema["exclusiveMaximum"], bool):
            e = schema["exclusiveMaximum"]
            if hi is None or e <= hi:
                hi, excl_hi = e, True
        return lo, excl_lo, hi, excl_hi

    def _number_term(self, schema: dict, integer: bool) -> str:
        if "multipleOf" in schema and schema["multipleOf"] not in (1, 1.0):
            self._record("multipleOf")
        lo, excl_lo, hi, excl_hi = self._bounds(schema)
        if schema.get("x-grid-not-values"):
            self._record("not-values")
        if lo is None and hi is None:
            self.needs.add("INT" if integer else "NUMBER")
            return "INT" if integer else "NUMBER"
        if integer:
            int_lo = None
            if lo is not None:
                int_lo = math.floor(lo) + 1 if excl_lo else math.ceil(lo)
            int_hi = None
            if hi is not None:
                int_hi = math.ceil(hi) - 1 if excl_hi else math.floor(hi)
            if int_lo is not None and int_hi is not None and int_lo > int_hi:
                return self.rule_for(dict(FALSE_SCHEMA))
            try:
                return self._rx_term(
                    rx.int_range_rx(int_lo, int_hi) + r"(\.0)?")
            except rx.RxUnsupported as e:
                self._record(f"integer-bounds ({e})")
                self.needs.add("INT")
                return "INT"
        src = rx.number_range_rx(lo, hi, excl_lo, excl_hi)
        if src is None:
            for k in _NUMBER_KEYS & set(schema):
                if k != "multipleOf":
                    self._record(k)
            self.needs.add("NUMBER")
            return "NUMBER"
        return self._rx_term(src)

    # ------------------------------------------------------------- arrays

    def _array_alts(self, schema: dict) -> list[str]:
        min_i = schema.get("minItems", 0) or 0
        max_i = schema.get("maxItems")
        # tuple forms: draft-07 items-list (+additionalItems) or 2020-12
        # prefixItems (+items as the tail schema)
        prefix: list | None = None
        tail_schema: Any = True
        items = schema.get("items", None)
        if isinstance(items, list):
            prefix = items
            tail_schema = schema.get("additionalItems", True)
        elif "prefixItems" in schema:
            prefix = schema["prefixItems"]
            tail_schema = items if items is not None else True
        if prefix is not None:
            return self._tuple_array(prefix, tail_schema, min_i, max_i)

        item_rule = self.rule_for(items if items is not None else True)
        if min_i == 0 and max_i is None:
            elems = self._rule("el")
            self.rules[elems] = [item_rule, f'{elems} "," {item_rule}']
            return ['"[" "]"', f'"[" {elems} "]"']
        if (max_i is not None and max_i > MAX_ITEMS_UNROLL) or \
                min_i > MAX_ITEMS_UNROLL:
            for k in ("minItems", "maxItems"):
                if k in schema:
                    self._record(f"{k}-beyond-cap")
            elems = self._rule("el")
            self.rules[elems] = [item_rule, f'{elems} "," {item_rule}']
            return ['"[" "]"', f'"[" {elems} "]"']
        if max_i is not None and min_i > max_i:
            return [self.rule_for(dict(FALSE_SCHEMA))]
        return self._counted_seq(item_rule, min_i, max_i, "[", "]")

    def _tuple_array(self, prefix: list, tail_schema: Any,
                     min_i: int, max_i: int | None) -> list[str]:
        n = len(prefix)
        if n > MAX_ITEMS_UNROLL or min_i > MAX_ITEMS_UNROLL or \
                (max_i is not None and max_i > MAX_ITEMS_UNROLL):
            raise Unsupported("tuple beyond unroll cap")
        prefix_rules = [self.rule_for(s) for s in prefix]
        tail_allowed = tail_schema is not False and tail_schema != FALSE_SCHEMA
        tail_rule = self.rule_for(tail_schema) if tail_allowed else None
        if max_i is not None and min_i > max_i:
            return [self.rule_for(dict(FALSE_SCHEMA))]

        # build right-to-left: pos i may start item i (prefix[i] or tail)
        # count window applies to the total number of items
        eff_max = max_i
        if not tail_allowed and (eff_max is None or eff_max > n):
            eff_max = n
        if eff_max is not None and min_i > eff_max:
            return [self.rule_for(dict(FALSE_SCHEMA))]

        def item_at(i: int) -> str:
            return prefix_rules[i] if i < n else tail_rule  # type: ignore

        # cont(i): continuation after item i-1, i.e. items from index i on
        upper = eff_max if eff_max is not None else max(n, min_i)
        conts: dict[int, str] = {}

        def cont(i: int) -> str:
            # returns a rule name matching ("," item_i ...) continuations,
            # or "" when no continuation is possible
            if eff_max is not None and i >= eff_max:
                return ""
            if i in conts:
                return conts[i]
            name = self._rule(f"tp{i}")
            conts[i] = name
            alts = []
            if i >= min_i:
                alts.append("|EPS|")
            if eff_max is None and i >= n:
                # unbounded tail: left-recursive list of tail items
                more = self._rule("tt")
                self.rules[more] = [f'"," {tail_rule}', f'{more} "," {tail_rule}']
                if i >= min_i:
                    alts.append(more)
                else:
                    # still below min: fixed prefix then unbounded
                    seq = " ".join(f'"," {tail_rule}' for _ in range(i, min_i))
                    alts.append(f"{seq}")
                    alts.append(f"{seq} {more}")
                self.rules[name] = alts
                return name
            if i < upper:
                nxt = item_at(i)
                if nxt is not None:
                    alts.append(f'"," {nxt} {cont(i + 1)}'.strip())
            self.rules[name] = alts
            return name

        out = []
        if min_i == 0:
            out.append('"[" "]"')
        first = item_at(0) if (upper > 0 or eff_max is None) else None
        if first is not None:
            out.append(f'"[" {first} {cont(1)} "]"')
        if not out:
            out.append('"[" "]"')
        return out

    def _counted_seq(self, item_rule: str, m: int, n: int | None,
                     open_b: str, close_b: str) -> list[str]:
        """Bracketed comma list with m..n items (n <= cap or None)."""
        out = []
        if m == 0:
            out.append(f'"{open_b}" "{close_b}"')
        # chain: c_k = continuation when k items already emitted
        conts: dict[int, str] = {}

        def cont(k: int) -> str:
            if n is not None and k >= n:
                return ""                    # no further items possible
            if k in conts:
                return conts[k]
            name = self._rule(f"c{k}")
            conts[k] = name
            alts = []
            if k >= m:
                alts.append("|EPS|")
            if n is None:
                more = self._rule("cm")
                self.rules[more] = [f'"," {item_rule}', f'{more} "," {item_rule}']
                if k >= m:
                    alts.append(more)
                else:
                    seq = " ".join(f'"," {item_rule}' for _ in range(k, m))
                    alts.append(seq)
                    alts.append(f"{seq} {more}")
            else:
                alts.append(f'"," {item_rule} {cont(k + 1)}'.strip())
            self.rules[name] = alts
            return name

        if n is None or n >= 1:
            out.append(f'"{open_b}" {item_rule} {cont(1)} "{close_b}"')
        return out

    # ------------------------------------------------------------- objects

    def _pattern_minus_keys(self, pat: str, keys: list[str]) -> str | None:
        """Serialized body for (pattern-language MINUS declared key names),
        for the supported pattern shapes; None when not constructible."""
        try:
            ast = rx.parse_ecma(pat)
            items = rx._flatten_cat(ast)
            lead = bool(items) and items[0].kind == "caret"
            trail = bool(items) and items[-1].kind == "dollar"
            if not lead and not trail and rx._nullable(ast):
                # matches every string: minus-keys is just NOT_LIT
                return rx.not_literals_body(list(keys))
            if lead and trail:
                atoms = rx._atoms(items[1:-1])
                if atoms and all(a[0] == atoms[0][0] for a in atoms):
                    cls = atoms[0][0]
                    m = sum(1 for a in atoms if a[1] in ("1", "plus"))
                    n = None if any(a[1] in ("star", "plus") for a in atoms) \
                        else len(atoms)
                    return rx.class_window_minus_literals(cls, m, n, list(keys))
        except rx.RxUnsupported:
            return None
        return None

    def _pp_disjoint(self, pats: list[str]) -> bool:
        """Provable pairwise disjointness of key patterns via anchored
        first-char classes."""
        firsts = []
        for p in pats:
            try:
                ast = rx.parse_ecma(p)
            except rx.RxUnsupported:
                return False
            items = rx._flatten_cat(ast)
            if not items or items[0].kind != "caret" or len(items) < 2:
                return False
            head = items[1]
            if head.kind == "ch":
                firsts.append(head.ranges)
            elif head.kind == "plus" and head.kids[0].kind == "ch":
                firsts.append(head.kids[0].ranges)
            else:
                return False
        for i in range(len(firsts)):
            for j in range(i + 1, len(firsts)):
                if rx._subtract(firsts[i], rx._subtract(firsts[i], firsts[j])):
                    return False        # intersection nonempty
        return True

    def _provably_disjoint(self, branches: list) -> bool:
        infos = []
        for b in branches:
            if not isinstance(b, dict):
                return False
            t = b.get("type")
            ts = set([t] if isinstance(t, str) else (t or []))
            if "integer" in ts:
                ts.add("number")
            discs = {}
            req = set(b.get("required", []) or [])
            props = b.get("properties", {}) or {}
            for k in req:
                sub = props.get(k)
                if isinstance(sub, dict):
                    if "const" in sub:
                        discs[k] = {json.dumps(sub["const"])}
                    elif "enum" in sub:
                        discs[k] = {json.dumps(v) for v in sub["enum"]}
            enum_vals = None
            if "enum" in b:
                enum_vals = {json.dumps(v) for v in b["enum"]}
            if "const" in b:
                enum_vals = {json.dumps(b["const"])}
            infos.append((ts, discs, enum_vals))
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                ti, di, ei = infos[i]
                tj, dj, ej = infos[j]
                if ei is not None and ej is not None and not (ei & ej):
                    continue
                if ti and tj and not (ti & tj):
                    continue
                shared = [k for k in di if k in dj and not (di[k] & dj[k])]
                if shared:
                    continue
                return False
        return True

    def _propname_body(self, pn: Any) -> str | None:
        """propertyNames schema -> serialized key-body regex (None = no
        constraint); raises Unsupported outside the supported subset."""
        if pn is True or pn == {}:
            return None
        if pn is False or pn == FALSE_SCHEMA:
            raise Unsupported("propertyNames: false")
        if not isinstance(pn, dict):
            raise Unsupported("propertyNames shape")
        if "$ref" in pn:
            return self._propname_body(self._resolve_ref(pn["$ref"]))
        keys = set(pn) - _ANNOTATIONS - {"type"}
        if pn.get("type") not in (None, "string"):
            raise Unsupported("propertyNames non-string type")
        try:
            if keys == {"const"}:
                return rx.literals_body([pn["const"]])
            if keys == {"enum"}:
                if not all(isinstance(v, str) for v in pn["enum"]):
                    raise Unsupported("propertyNames enum non-string")
                return rx.literals_body(list(pn["enum"]))
            if keys == {"pattern"}:
                return rx.pattern_body(pn["pattern"])
            if keys == {"x-grid-not-values"}:
                vals = pn["x-grid-not-values"]
                if not all(isinstance(v, str) for v in vals):
                    raise Unsupported("propertyNames not-values non-string")
                return rx.not_literals_body(list(vals))
            if keys and keys <= {"minLength", "maxLength"}:
                return rx.length_body(pn.get("minLength", 0), pn.get("maxLength"))
            if not keys:
                return None
        except rx.RxUnsupported as e:
            raise Unsupported(f"propertyNames ({e})")
        raise Unsupported(f"propertyNames keys {sorted(keys)}")

    def _object(self, schema: dict) -> str:
        props: dict[str, Any] = dict(schema.get("properties", {}) or {})
        required = set(schema.get("required", []) or [])
        forbid = set(schema.get("x-grid-forbid-keys", []) or [])
        ap = schema.get("additionalProperties", None)
        min_p = schema.get("minProperties", 0) or 0
        max_p = schema.get("maxProperties")
        pp = dict(schema.get("patternProperties") or {})
        pn = schema.get("propertyNames")

        if forbid & required:
            return self.rule_for(dict(FALSE_SCHEMA))
        for f in forbid:
            props.pop(f, None)
        if pp and pn is not None:
            self._record("propertyNames-with-patternProperties")
            pn = None
        if pp and forbid:
            self._record("forbid-keys-with-patternProperties")
            forbid = set()
        if pn is not None and forbid:
            self._record("propertyNames-with-forbidden-keys")
            forbid = set()

        pn_body = self._propname_body(pn) if pn is not None else None
        if pn is not None:
            from grid.jsonschema.normalize import _valid
            for k in list(props):
                if not _valid(k, pn, self.root):
                    if k in required:
                        raise Unsupported("required key violates propertyNames")
                    props.pop(k)

        if len(props) > MAX_PROPERTIES:
            raise Unsupported(f"{len(props)} properties (size cap)")
        unknown_req = required - set(props)
        if unknown_req and not props:
            self._record("required-without-properties")

        extras = ap is not False and ap != FALSE_SCHEMA
        extra_val = None
        if extras:
            extra_val = self.generic_value() if ap in (None, True) \
                else self.rule_for(ap)

        # patternProperties: pattern-keyed pairs; extras keys must then be
        # the complement (a matching key must take its pattern's pair)
        pp_pairs: list[str] = []
        extras_body: str | None = None      # None -> plain STRING
        pp_key_body: dict[str, str] = {}
        if pp:
            pats = list(pp)
            import re as _re
            from grid.jsonschema.normalize import Unmergeable, merge2, normalize as _n2
            overlap: dict[str, list[str]] = {}
            for k in list(props):
                for pat in pats:
                    try:
                        hit = _re.search(pat, k) is not None
                    except _re.error:
                        raise Unsupported(f"patternProperties bad pattern {pat!r}")
                    if not hit:
                        continue
                    # declared key matching a pattern: both schemas apply
                    try:
                        props[k] = _n2(merge2(props[k], pp[pat], self.root))
                    except Unmergeable:
                        self._record("pp-overlap-merge-unenforced")
                    overlap.setdefault(pat, []).append(k)
            for pat, km in overlap.items():
                # the pattern pair must EXCLUDE the declared names, else a
                # declared key could take the (weaker) pattern path
                body = self._pattern_minus_keys(pat, km)
                if body is None:
                    self._record("pp-overlap-unsubtracted")
                else:
                    pp_key_body[pat] = body
            if len(pats) > 1 and not self._pp_disjoint(pats):
                self._record("patternProperties-overlap")
            for pat in pats:
                try:
                    body = pp_key_body.get(pat) or rx.pattern_body(pat)
                except rx.RxUnsupported as e:
                    raise Unsupported(f"patternProperties pattern {pat!r} ({e})")
                kt = self._rx_term(rx.string_terminal_rx(body))
                self.routing_terms.add(kt)
                vr = self.rule_for(pp[pat])
                pr = self._rule("pp")
                self.rules[pr] = [f'{kt} ":" {vr}']
                pp_pairs.append(pr)
            if extras:
                if len(pats) > 1:
                    self._record("extras-with-multiple-patternProperties")
                elif True:
                    try:
                        comp = rx.pattern_complement_body(pats[0])
                        if comp is None:
                            extras = False   # every key matches the pattern
                            extra_val = None
                        else:
                            extras_body = comp
                    except rx.RxUnsupported:
                        self._record("pp-extras-complement-unavailable")
        elif pn_body is not None:
            extras_body = pn_body

        def extras_key_term() -> str:
            if extras_body is not None:
                t = self._rx_term(rx.string_terminal_rx(extras_body))
                self.routing_terms.add(t)
                return t
            if forbid:
                try:
                    body = rx.not_literals_body(sorted(forbid))
                except rx.RxUnsupported:
                    self._record("forbid-keys-terminal-too-large")
                    self.needs.add("STRING")
                    return "STRING"
                t = self._rx_term(rx.string_terminal_rx(body))
                self.routing_terms.add(t)
                return t
            self.needs.add("STRING")
            return "STRING"

        # generic member pairs: extras pair (if any) + pattern pairs
        generic_pairs: list[str] = list(pp_pairs)
        if extras:
            gp = self._rule("xp")
            self.rules[gp] = [f'{extras_key_term()} ":" {extra_val}']
            generic_pairs.append(gp)

        if not props:
            if not generic_pairs and (required or min_p > 0):
                return self.rule_for(dict(FALSE_SCHEMA))
            if required:
                self._record("required names outside properties")
            if generic_pairs:
                member = self._rule("gp")
                self.rules[member] = list(generic_pairs)
            else:
                obj = self._rule("go")
                self.rules[obj] = ['"{" "}"']
                return obj
            if (min_p == 0 and max_p is None) or \
                    (max_p is not None and max_p > MAX_ITEMS_UNROLL) or \
                    min_p > MAX_ITEMS_UNROLL:
                if not (min_p == 0 and max_p is None):
                    for k in ("minProperties", "maxProperties"):
                        if k in schema:
                            self._record(f"{k}-beyond-cap")
                members = self._rule("gm")
                self.rules[members] = [member, f'{members} "," {member}']
                obj = self._rule("go")
                self.rules[obj] = ['"{" "}"', f'"{{" {members} "}}"']
                return obj
            if max_p is not None and min_p > max_p:
                return self.rule_for(dict(FALSE_SCHEMA))
            alts = self._counted_seq(member, min_p, max_p, "{", "}")
            obj = self._rule("go")
            self.rules[obj] = alts
            return obj

        # required keys outside `properties`: a key matching a
        # patternProperties pattern is NOT "additional" (ap:false does not
        # forbid it) — credit it with the pattern's value schema; otherwise
        # credit via the extras schema; truly impossible only when neither
        if unknown_req:
            import re as _re2
            still = set()
            for k in sorted(unknown_req):
                matched = None
                for pat in (pp or {}):
                    try:
                        if _re2.search(pat, k):
                            matched = pat
                            break
                    except _re2.error:
                        pass
                if matched is not None:
                    props[k] = pp[matched]
                elif extras:
                    props[k] = ap if isinstance(ap, dict) else {}
                else:
                    still.add(k)
            if still:
                return self.rule_for(dict(FALSE_SCHEMA))
        # count constraints with declared properties: enforce only the
        # statically-decidable case
        if min_p or max_p is not None:
            n_req = len(required & set(props))
            n_max = len(props) if not extras else None
            if not extras and set(props) == (required & set(props)):
                count = len(props)
                if count < min_p or (max_p is not None and count > max_p):
                    return self.rule_for(dict(FALSE_SCHEMA))
                # vacuously satisfied — nothing to record
            else:
                if min_p > (n_req if n_req else 0):
                    self._record("minProperties")
                if max_p is not None and (n_max is None or max_p < n_max):
                    self._record("maxProperties")

        # ---- order-free object machine ----
        # Instances present keys in arbitrary order (the bench data is NOT
        # normalized to declaration order). Track only which REQUIRED keys
        # have been seen: 2^R member-chain rules; optional and generic pairs
        # are order- and state-transparent.
        req_list = sorted(required & set(props))
        R = len(req_list)
        if R > 10 or (1 << R) * (len(props) + len(generic_pairs) + 1) > 40_000:
            # subset machine too large. The ordered fallback FALSE-REJECTS
            # out-of-order instances (observed on the full set), which is the
            # forbidden error class — so drop required-tracking instead:
            # any order accepted, missing-required becomes a RECORDED
            # invalidation risk (the allowed class)
            self._record("required-not-enforced (required-set beyond cap)")
            req_list = []
            R = 0
        bit = {k: 1 << i for i, k in enumerate(req_list)}
        FULL = (1 << R) - 1

        units: list[tuple[str, int]] = []       # (pair grammar, state bit)
        for k, v in props.items():
            pair_src = f'{self._key_term(k)} ":" {self.rule_for(v)}'
            units.append((pair_src, bit.get(k, 0)))
        for g in generic_pairs:
            units.append((g, 0))

        mrules: dict[int, str] = {}

        def member_chain(S: int) -> str:
            got = mrules.get(S)
            if got is not None:
                return got
            name = self._rule(f"m{S:x}")
            mrules[S] = name
            alts = []
            for pair_src, b in units:
                S2 = S | b
                if S2 == FULL:
                    alts.append(pair_src)
                alts.append(f'{pair_src} "," {member_chain(S2)}')
            self.rules[name] = alts
            return name

        obj = self._rule("o")
        alts = []
        if R == 0 and min_p == 0:
            alts.append('"{" "}"')
        if units:
            alts.append(f'"{{" {member_chain(0)} "}}"')
        if not alts:
            alts.append('"{" "}"')
        self.rules[obj] = alts
        return obj

    def _ordered_object(self, props: dict, required: set,
                        generic_pairs: list[str], min_p: int) -> str:
        """Declaration-order member machine (fallback beyond the subset cap):
        declared properties in schema order with optional skips, generic
        pairs interleavable. Order violations reject visibly."""
        items = [
            (self._key_term(k), self.rule_for(v), k in required)
            for k, v in props.items()
        ]
        n = len(items)
        has_gen = bool(generic_pairs)

        def pair(i: int) -> str:
            kt, vr, _req = items[i]
            return f'{kt} ":" {vr}'

        tails: list[str] = [""] * (n + 1)
        req_from = [False] * (n + 1)
        for j in range(n - 1, -1, -1):
            req_from[j] = req_from[j + 1] or items[j][2]
        lo = 0 if has_gen else 1
        hi = n if has_gen else n - 1
        for j in range(hi, lo - 1, -1):
            name = self._rule(f"t{j}")
            alts = [] if req_from[j] else ["|EPS|"]
            for k in range(j, n):
                if k > j and any(items[m][2] for m in range(j, k)):
                    break  # skipping a required property is not viable
                alts.append(f'"," {pair(k)} {tails[k + 1]}'.strip())
            for g in generic_pairs:
                alts.append(f'"," {g} {name}')
            self.rules[name] = alts
            tails[j] = name

        heads: list[str] = []
        for k in range(n):
            if k > 0 and any(items[m][2] for m in range(k)):
                break
            heads.append(f"{pair(k)} {tails[k + 1]}".strip())
        for g in generic_pairs:
            heads.append(f"{g} {tails[0]}")

        obj = self._rule("o")
        alts = []
        if not req_from[0] and min_p == 0:
            alts.append('"{" "}"')
        alts += [f'"{{" {h} "}}"' for h in heads]
        self.rules[obj] = alts
        return obj

    def generic_value(self) -> str:
        if "generic" not in self.needs:
            self.needs |= {"generic", "STRING", "NUMBER"}
            self.rules["json_value"] = [
                "json_object", "json_array", "STRING", "NUMBER",
                '"true"', '"false"', '"null"',
            ]
            self.rules["json_pair"] = ['STRING ":" json_value']
            self.rules["json_members"] = ["json_pair", 'json_members "," json_pair']
            self.rules["json_object"] = ['"{" "}"', '"{" json_members "}"']
            self.rules["json_elems"] = ["json_value", 'json_elems "," json_value']
            self.rules["json_array"] = ['"[" "]"', '"[" json_elems "]"']
            for r in ("json_value", "json_pair", "json_members", "json_object",
                      "json_elems", "json_array"):
                self.rule_order.append(r)
        return "json_value"

    # ---------------------------------------------------------- dedupe

    def _dedupe_rules(self, start: str) -> str:
        """Hash-cons the rule set: merge rules with identical alternative
        lists (self-references canonicalized) to a fixpoint. Identical
        per-branch sub-rules otherwise produce LALR reduce-reduce conflicts
        between equal reductions — and needlessly large tables."""
        alias: dict[str, str] = {}

        def resolve(n: str) -> str:
            while n in alias:
                n = alias[n]
            return n

        changed = True
        while changed:
            changed = False
            seen: dict[tuple, str] = {}
            for name in self.rule_order:
                if name in alias or name not in self.rules:
                    continue
                key_parts = []
                for alt in self.rules[name]:
                    toks = tuple("@SELF" if resolve(t) == name else resolve(t)
                                 for t in alt.split())
                    key_parts.append(toks)
                key = tuple(key_parts)
                canon = seen.get(key)
                if canon is None:
                    seen[key] = name
                elif canon != name:
                    alias[name] = canon
                    changed = True

        new_rules: dict[str, list[str]] = {}
        new_order: list[str] = []
        for name in self.rule_order:
            if resolve(name) != name or name not in self.rules:
                continue
            alts = []
            for alt in self.rules[name]:
                if alt in ("", "|EPS|"):
                    if alt not in alts:
                        alts.append(alt)
                    continue
                r = " ".join(resolve(t) for t in alt.split())
                if r not in alts:       # aliasing can create duplicate alts
                    alts.append(r)
            new_rules[name] = alts
            new_order.append(name)
        self.rules = new_rules
        self.rule_order = new_order
        return resolve(start)

    # ---------------------------------------------------------------- emit

    _SCANNER_BIG = 600      # combined-DFA safety: two large position-machines
                            # in one scanner multiply subset states

    def _apply_scanner_budget(self) -> None:
        """The union scanner DFA is a product of live terminal position sets;
        two large constrained terminals in one schema can blow up subset
        construction (observed: email-format x unanchored pattern). Keep the
        largest, degrade the rest to generic STRING — recorded, not silent."""
        # length windows multiply with every co-resident terminal (enum
        # literals and keys share the '"' prefix): enforce them only where
        # the product stays cheap — measured: (0,64) x 100 keys = 2.3s,
        # (0,128) x 196 terminals = 145s
        n_terms = len(self.key_terms) + len(self.lit_terms) + len(self.rx_terms)
        for src, name in list(self.rx_terms.items()):
            if name in self.routing_terms:
                continue
            cost = self.rx_costs.get(name, len(src))
            is_window = "{" in src and cost >= 40 * 32
            if is_window and (cost > 40 * 64 or n_terms > 80) \
                    and name not in self.degraded:
                self._record("scanner-budget: length window degraded")
                self.degraded.add(name)
        costed = [(self.rx_costs.get(name, len(src)), src)
                  for src, name in self.rx_terms.items()
                  if src.startswith('"') and name not in self.degraded
                  and name not in self.routing_terms]
        bigs = [(c, src) for c, src in costed if c > self._SCANNER_BIG]
        if not bigs:
            return
        bigs.sort(reverse=True)
        for _, src in bigs[1:]:
            self._record("scanner-budget: constrained string degraded")
            self.degraded.add(self.rx_terms[src])
        # a very large keeper (length window) multiplies with ANY other
        # nontrivial position machine — degrade those too
        if bigs[0][0] > 2_500:
            for c, src in costed:
                if src != bigs[0][1] and c > 200 and \
                        self.rx_terms[src] not in self.degraded:
                    self._record("scanner-budget: constrained string degraded")
                    self.degraded.add(self.rx_terms[src])

    def compile(self) -> str:
        start = self.rule_for(self.root)
        self._apply_scanner_budget()
        if self.degraded and len(self.degraded) <= 3:
            # few degraded terminals: keep them as separate STRING_RX clones —
            # the small live-set overhead is fine, and distinct names avoid
            # collapsing structurally-different rules into LALR conflicts
            self.degraded_keep = set(self.degraded)
            self.degraded = set()
        if self.degraded:
            # many degraded terminals all sharing STRING_RX — N identical
            # automata would put every one of them in every live set
            # (h_max ~= N) and blow up the scanner closure; alias them to
            # the ONE generic STRING terminal instead
            self.needs.add("STRING")
            self.rx_terms = {src: name for src, name in self.rx_terms.items()
                             if name not in self.degraded}
            for name, alts in self.rules.items():
                self.rules[name] = [
                    " ".join("STRING" if t in self.degraded else t
                             for t in alt.split()) if alt not in ("", "|EPS|")
                    else alt
                    for alt in alts
                ]
            self.degraded = set()
        start = self._dedupe_rules(start)

        # reachability prune: rewrites (enum filtering, degradation, branch
        # drops) can orphan rules/terminals; the spec loader rejects
        # non-reduced grammars
        reach: set[str] = set()
        stack = [start]
        while stack:
            r = stack.pop()
            if r in reach or r not in self.rules:
                continue
            reach.add(r)
            for alt in self.rules[r]:
                for t in alt.split():
                    if t in self.rules and t not in reach:
                        stack.append(t)
        used = {t for r in reach for alt in self.rules[r] for t in alt.split()}

        lines = ["%start start", "%ignore WS", r"WS: /[ \t\n\r]+/"]
        for key, term in self.key_terms.items():
            if term not in used:
                continue
            lines.append(f"{term}: /{_regex_literal(json.dumps(key, ensure_ascii=False))}/")
        for lit, term in self.lit_terms.items():
            if term not in used:
                continue
            lines.append(f"{term}: /{lit}/")
        for src, term in self.rx_terms.items():
            if term not in used:
                continue
            demoted = term in self.degraded or term in self.degraded_keep
            lines.append(f"{term}: /{STRING_RX if demoted else src}/")
        if "STRING" in used:
            lines.append(f"STRING: /{STRING_RX}/")
        if "NUMBER" in used:
            lines.append(f"NUMBER: /{NUMBER_RX}/")
        if "INT" in used:
            lines.append(f"INT: /{INT_RX}/")
        lines.append(f"start: {start}")
        for name in self.rule_order:
            if name not in reach:
                continue
            alts = self.rules.get(name)
            if not alts:
                continue
            rendered = ["" if a == "|EPS|" else a for a in alts]
            lines.append(f"{name}: " + " | ".join(rendered))
        return "\n".join(lines) + "\n"


def compile_schema(schema: Any, strict: bool = False) -> tuple[str, set[str]]:
    """-> (.grid source, recorded-unenforced set). Raises Unsupported."""
    normalized = normalize(schema)
    root = normalized if isinstance(normalized, dict) else schema
    if isinstance(root, dict) and isinstance(schema, dict):
        # rewrites can restructure the root; keep resolution targets reachable
        for dk in ("$defs", "definitions"):
            if dk in schema and dk not in root:
                root = dict(root)
                root[dk] = schema[dk]
    c = SchemaCompiler(root, strict=strict)
    src = c.compile()
    return src, c.ignored
