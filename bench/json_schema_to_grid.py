"""JSON Schema -> .grid grammar compiler (MaskBench arm; v1 subset).

Supported (mirrors the cross-engine conventions from guidance-ai/maskbench):
- types: object, array, string, number, integer, boolean, null; type lists;
- object properties in DEFINITION ORDER (the stable order the MaskBench data is
  normalized to), optional properties skippable, `required` enforced by
  construction, declared-keys-only (XGrammar's default assumption);
- arrays with `items`; enum/const via exact serialized literals (default
  json.dumps separators — byte-identical to the runner's instance serialization);
- anyOf/oneOf as alternation (oneOf exclusivity not enforced — recorded);
- local $ref/$defs/definitions with cycles (CFG recursion).

Ignored-but-accepted (recorded per schema, like XGrammar's default mode):
pattern, format, min/max{Length,imum,Items,Properties}, multipleOf,
uniqueItems, contains, default/title/description/examples annotations.

Unsupported (raises Unsupported -> "compile error" bucket, llguidance-style
honesty): allOf (non-trivial), not, if/then/else, patternProperties,
propertyNames, dependencies, unevaluated*, external $ref, prefixItems,
additionalProperties-as-schema alongside properties, false schemas, and
grammars past the v1 size caps.

Whitespace: %ignore /[ \t\n\r]+/ — the JSON-spec definition (the llama.cpp
MaskBench arm modifies its generator to exactly this).
"""

from __future__ import annotations

import json
from typing import Any

MAX_PROPERTIES = 120
MAX_NAMED_TERMINALS = 300
MAX_RULES = 3000

_ANNOTATIONS = {
    "title", "description", "default", "examples", "$schema", "$id", "$comment",
    "readOnly", "writeOnly", "deprecated", "$defs", "definitions",
}
_IGNORED_CONSTRAINTS = {
    "pattern", "format", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems", "maxItems",
    "uniqueItems", "contains", "minContains", "maxContains", "minProperties",
    "maxProperties", "dependentRequired", "contentMediaType", "contentEncoding",
}
_UNSUPPORTED_KEYS = {
    "not", "if", "then", "else", "patternProperties", "propertyNames",
    "dependencies", "dependentSchemas", "unevaluatedProperties",
    "unevaluatedItems", "prefixItems", "additionalItems",
}

_REGEX_META = set("()[]{}*+?|\\./")


class Unsupported(Exception):
    """Schema uses a feature outside the v1 subset (counted as compile error)."""


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


# JSON string/number terminals (grid regex subset: no bounded repetition — \uXXXX
# hex quads are expanded; escapes and byte classes per grid/lexer/dfa.py).
_HEX = "[0-9a-fA-F]"
STRING_RX = (
    r'"([^"\\\x00-\x1f]|\\(["\\/bfnrt]|u' + _HEX + _HEX + _HEX + _HEX + r"))*\""
)
NUMBER_RX = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
INT_RX = r"-?(0|[1-9][0-9]*)"


class SchemaCompiler:
    def __init__(self, root_schema: dict) -> None:
        self.root = root_schema
        self.rules: dict[str, list[str]] = {}
        self.rule_order: list[str] = []
        self.memo: dict[int, str] = {}          # id(schema node) -> rule name
        self._keepalive: list[Any] = []         # nodes behind memo ids (id-reuse guard)
        self.key_terms: dict[str, str] = {}     # property name -> terminal
        self.lit_terms: dict[str, str] = {}     # serialized enum/const value -> terminal
        self.needs: set[str] = set()            # STRING | NUMBER | INT | generic
        self.ignored: set[str] = set()          # recorded ignored constraints
        self._n = 0

    # ------------------------------------------------------------- utilities

    def _rule(self, hint: str) -> str:
        name = f"r{self._n}_{hint}"
        self._n += 1
        if self._n > MAX_RULES:
            raise Unsupported("rule budget exceeded (v1 size cap)")
        self.rules[name] = []
        self.rule_order.append(name)
        return name

    def _key_term(self, key: str) -> str:
        t = self.key_terms.get(key)
        if t is None:
            t = f"K{len(self.key_terms)}"
            self.key_terms[key] = t
            if len(self.key_terms) + len(self.lit_terms) > MAX_NAMED_TERMINALS:
                raise Unsupported("terminal budget exceeded (v1 size cap)")
        return t

    def _lit_term(self, value: Any) -> str:
        s = json.dumps(value, ensure_ascii=False)  # match the runner's serialization
        t = self.lit_terms.get(s)
        if t is None:
            t = f"E{len(self.lit_terms)}"
            self.lit_terms[s] = t
            if len(self.key_terms) + len(self.lit_terms) > MAX_NAMED_TERMINALS:
                raise Unsupported("terminal budget exceeded (v1 size cap)")
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
        if schema is False:
            raise Unsupported("false schema")
        if not isinstance(schema, dict):
            raise Unsupported(f"schema node of type {type(schema).__name__}")

        got = self.memo.get(id(schema))
        if got is not None:
            return got

        if "$ref" in schema:
            extra = set(schema) - {"$ref"} - _ANNOTATIONS
            if extra - _IGNORED_CONSTRAINTS:
                raise Unsupported(f"$ref with sibling keys {sorted(extra)}")
            self.ignored.update(extra)
            return self.rule_for(self._resolve_ref(schema["$ref"]))

        name = self._rule("v")
        self.memo[id(schema)] = name  # pre-register: recursive schemas terminate
        self._keepalive.append(schema)
        self.rules[name] = self._alternatives(schema, name)
        return name

    def _alternatives(self, schema: dict, self_name: str) -> list[str]:
        bad = set(schema) & _UNSUPPORTED_KEYS
        if bad:
            raise Unsupported(f"unsupported keys {sorted(bad)}")
        for k in set(schema) & _IGNORED_CONSTRAINTS:
            self.ignored.add(k)

        if "allOf" in schema:
            branches = schema["allOf"]
            if len(branches) == 1 and not (set(schema) - {"allOf"} - _ANNOTATIONS):
                return [self.rule_for(branches[0])]
            raise Unsupported("allOf")

        if "enum" in schema or "const" in schema:
            values = schema["enum"] if "enum" in schema else [schema["const"]]
            if not values:
                raise Unsupported("empty enum")
            return [self._lit_term(v) for v in values]

        if "anyOf" in schema or "oneOf" in schema:
            key = "anyOf" if "anyOf" in schema else "oneOf"
            if key == "oneOf":
                self.ignored.add("oneOf-exclusivity")
            rest = set(schema) - {key} - _ANNOTATIONS - _IGNORED_CONSTRAINTS
            if rest - {"type"}:
                raise Unsupported(f"{key} with sibling keys {sorted(rest)}")
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
            if "properties" in schema or "required" in schema or "additionalProperties" in schema:
                types = "object"
            elif "items" in schema:
                types = "array"
            else:
                return [self.generic_value()]

        if types == "object":
            return [self._object(schema)]
        if types == "array":
            items = schema.get("items", True)
            item_rule = self.rule_for(items)
            elems = self._rule("el")
            self.rules[elems] = [item_rule, f'{elems} "," {item_rule}']
            return ['"[" "]"', f'"[" {elems} "]"']
        if types == "string":
            self.needs.add("STRING")
            return ["STRING"]
        if types == "number":
            self.needs.add("NUMBER")
            return ["NUMBER"]
        if types == "integer":
            self.needs.add("INT")
            return ["INT"]
        if types == "boolean":
            return ['"true"', '"false"']
        if types == "null":
            return ['"null"']
        raise Unsupported(f"type {types!r}")

    def _object(self, schema: dict) -> str:
        props: dict[str, Any] = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        ap = schema.get("additionalProperties", None)
        if len(props) > MAX_PROPERTIES:
            raise Unsupported(f"{len(props)} properties (v1 cap {MAX_PROPERTIES})")
        unknown_req = required - set(props)
        if unknown_req and not props:
            # required-only schema: fall back to generic members (can't order)
            self.ignored.add("required-without-properties")

        # JSON Schema default: additionalProperties is ALLOWED unless explicitly
        # false; a schema-valued ap types the extra values.
        extras = ap is not False
        extra_val = None
        if extras:
            extra_val = self.generic_value() if ap in (None, True) else self.rule_for(ap)

        if not props:
            # generic object (typed extras if given)
            self.needs.add("STRING")
            val = extra_val if extras else self.generic_value()
            pair = self._rule("gp")
            self.rules[pair] = [f'STRING ":" {val}']
            members = self._rule("gm")
            self.rules[members] = [pair, f'{members} "," {pair}']
            obj = self._rule("go")
            self.rules[obj] = ['"{" "}"', f'"{{" {members} "}}"']
            return obj
        if unknown_req:
            self.ignored.add("required names outside properties")

        items = [
            (self._key_term(k), self.rule_for(v), k in required)
            for k, v in props.items()
        ]
        n = len(items)

        gpair = ""
        if extras:
            self.needs.add("STRING")
            gpair = self._rule("xp")
            self.rules[gpair] = [f'STRING ":" {extra_val}']

        def pair(i: int) -> str:
            kt, vr, _req = items[i]
            return f'{kt} ":" {vr}'

        # tail_j: what may follow once properties < j have been handled. With
        # extras, tail_j also admits ", <generic pair>" and recurses on itself
        # (extra properties may appear between declared ones); LALR stays
        # conflict-free because the lookahead key terminal (K_i vs STRING)
        # separates the alternatives. Without extras, tails[n] is inline-empty
        # and tails[0] is never referenced (reducedness).
        tails: list[str] = [""] * (n + 1)
        req_from = [False] * (n + 1)
        for j in range(n - 1, -1, -1):
            req_from[j] = req_from[j + 1] or items[j][2]
        lo = 0 if extras else 1
        hi = n if extras else n - 1
        for j in range(hi, lo - 1, -1):
            name = self._rule(f"t{j}")
            alts = [] if req_from[j] else ["|EPS|"]
            for k in range(j, n):
                if k > j and any(items[m][2] for m in range(j, k)):
                    break  # skipping a required property is not viable
                alts.append(f'"," {pair(k)} {tails[k + 1]}'.strip())
            if extras:
                alts.append(f'"," {gpair} {name}')
            self.rules[name] = alts
            tails[j] = name

        heads: list[str] = []
        for k in range(n):
            if k > 0 and any(items[m][2] for m in range(k)):
                break
            heads.append(f"{pair(k)} {tails[k + 1]}".strip())
        if extras:
            heads.append(f"{gpair} {tails[0]}")

        obj = self._rule("o")
        alts = []
        if not req_from[0]:
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

    # ---------------------------------------------------------------- emit

    def compile(self) -> str:
        start = self.rule_for(self.root)
        lines = ["%start start", "%ignore WS", r"WS: /[ \t\n\r]+/"]
        for key, term in self.key_terms.items():
            lines.append(f"{term}: /{_regex_literal(json.dumps(key, ensure_ascii=False))}/")
        for lit, term in self.lit_terms.items():
            lines.append(f"{term}: /{_regex_literal(lit)}/")
        if "STRING" in self.needs:
            lines.append(f"STRING: /{STRING_RX}/")
        if "NUMBER" in self.needs:
            lines.append(f"NUMBER: /{NUMBER_RX}/")
        if "INT" in self.needs:
            lines.append(f"INT: /{INT_RX}/")
        lines.append(f"start: {start}")
        for name in self.rule_order:
            alts = self.rules.get(name)
            if not alts:
                continue
            # eps alternative (only ever first) renders as a leading "|"
            rendered = ["" if a == "|EPS|" else a for a in alts]
            lines.append(f"{name}: " + " | ".join(rendered))
        return "\n".join(lines) + "\n"


def compile_schema(schema: dict) -> tuple[str, set[str]]:
    """-> (.grid source, ignored-feature set). Raises Unsupported."""
    c = SchemaCompiler(schema)
    src = c.compile()
    return src, c.ignored
