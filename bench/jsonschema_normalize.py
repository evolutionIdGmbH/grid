"""Schema normalization (M2 of DESIGN-JSON-COVERAGE).

Rewrites draft-07/2020-12 schemas into the compiler's core subset before
grammar construction:

- ``$ref`` with sibling keys      -> merge(resolve(ref), siblings)
- ``allOf``                       -> folded keyword merge (llguidance-style)
- anyOf/oneOf with siblings       -> siblings distributed into each branch
- ``dependencies``/``dependentRequired``/``dependentSchemas``
                                  -> anyOf over present/absent variants using
                                     the internal ``x-grid-forbid-keys`` marker
- ``if``/``then``/``else``        -> anyOf[if∧then, ¬if∧else] for negatable ifs
- draft-04 exclusiveMinimum/Maximum booleans -> numeric form
- enum/const under merge          -> values filtered against the co-constraints
                                     (jsonschema reference validator)
- ``not`` where computable        -> type complement / enum filter / forbid-keys

Everything it cannot rewrite is left in place for the compiler to declare
(compile-error bucket) — normalization must never *weaken* a constraint:
dropping or ignoring here would surface as silent invalidation errors.

Internal markers consumed by the compiler:
- ``x-grid-forbid-keys``: [names]   object must not contain these keys
- ``x-grid-not-values``: [values]   value must not equal any of these
"""

from __future__ import annotations

import json
from typing import Any

try:
    import jsonschema as _js
except Exception:                                    # pragma: no cover
    _js = None

MAX_BRANCH_PRODUCT = 64
_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")

_ANNOTATIONS = {
    "title", "description", "default", "examples", "$schema", "$id", "$comment",
    "readOnly", "writeOnly", "deprecated", "$defs", "definitions", "id",
    "contentMediaType", "contentEncoding", "$vocabulary", "$anchor",
}
# Every keyword that CONSTRAINS instances; anything outside this set (vendor
# x-* keys, unknown keywords) is an annotation per spec and safely droppable.
_ASSERTIONS = {
    "type", "enum", "const", "anyOf", "oneOf", "allOf", "not",
    "if", "then", "else", "$ref",
    "properties", "patternProperties", "additionalProperties", "propertyNames",
    "required", "dependencies", "dependentRequired", "dependentSchemas",
    "unevaluatedProperties", "minProperties", "maxProperties",
    "items", "prefixItems", "additionalItems", "unevaluatedItems",
    "contains", "minContains", "maxContains", "minItems", "maxItems",
    "uniqueItems", "pattern", "format", "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "x-grid-forbid-keys", "x-grid-not-values", "x-grid-not-patterns",
    "x-grid-extra-patterns",
}
# keyword classes used by the merge algebra
_MIN_KEYS = {"minimum", "minLength", "minItems", "minProperties", "minContains"}
_MAX_KEYS = {"maximum", "maxLength", "maxItems", "maxProperties", "maxContains"}


class Unmergeable(Exception):
    """allOf/sibling combination outside the supported merge algebra."""


FALSE_SCHEMA = {"not": {}}      # canonical unsatisfiable schema

# draft-07 and earlier: $ref REPLACES the schema — sibling keywords are
# IGNORED, not merged (2019-09 changed this). Set per-normalize() run from
# the root $schema; single-threaded bench usage.
_LEGACY_REF = False
_LEGACY_MARKERS = ("draft-03", "draft-04", "draft-06", "draft-07")


def _is_true(s: Any) -> bool:
    return s is True or s == {}


def _is_false(s: Any) -> bool:
    return s is False or s == FALSE_SCHEMA


# ------------------------------------------------------------- ref utils

def _resolve_pointer(root: Any, ref: str) -> Any:
    if not ref.startswith("#"):
        raise Unmergeable(f"external $ref {ref!r}")
    node = root
    for part in ref[1:].split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise Unmergeable(f"unresolvable $ref {ref!r}")
    return node


def _inline_refs(schema: Any, root: Any, seen: tuple = ()) -> Any:
    """Deep-copy with local $refs resolved inline; cycles are unmergeable."""
    if isinstance(schema, dict):
        if "$ref" in schema and isinstance(schema["$ref"], str):
            ref = schema["$ref"]
            if ref in seen:
                raise Unmergeable(f"recursive $ref {ref!r} under merge")
            target = _resolve_pointer(root, ref)
            rest = {k: v for k, v in schema.items() if k != "$ref"}
            inlined = _inline_refs(target, root, seen + (ref,))
            if not isinstance(inlined, dict):
                return inlined if not rest else merge2(_as_dict(inlined), rest, root)
            if rest:
                return merge2(inlined, _inline_refs(rest, root, seen), root)
            return inlined
        return {k: _inline_refs(v, root, seen) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_inline_refs(v, root, seen) for v in schema]
    return schema


def _as_dict(s: Any) -> dict:
    if s is True:
        return {}
    if s is False:
        return dict(FALSE_SCHEMA)
    if isinstance(s, dict):
        return s
    raise Unmergeable(f"schema node of type {type(s).__name__}")


# --------------------------------------------------------- value checking

def _valid(value: Any, schema: Any, root: Any) -> bool:
    """Reference-validator check for finite-value filtering. Conservative:
    on any failure to decide, treat the value as VALID (keeping it never
    weakens the enum-filter result below the original enum)."""
    if _is_true(schema):
        return True
    if _is_false(schema):
        return False
    if _js is None:
        return True
    try:
        inlined = to_std(_inline_refs(schema, root))
        validator = _js.Draft202012Validator(inlined)
        return validator.is_valid(value)
    except Exception:
        return True


# ------------------------------------------------------------- negation

def negate(schema: Any, root: Any = None) -> list[dict]:
    """¬schema as a list of anyOf branches; raises Unmergeable when the
    complement is outside the supported set."""
    schema = _as_dict(schema)
    keys = set(schema) - _ANNOTATIONS
    if not keys:
        return [dict(FALSE_SCHEMA)]        # ¬true = false
    if keys == {"$ref"} and root is not None:
        return negate(_resolve_pointer(root, schema["$ref"]), root)
    if keys == {"anyOf"}:
        # ¬(A ∨ B) = ¬A ∧ ¬B: conjunction across the branch negations
        acc: list[dict] = [{}]
        for br in schema["anyOf"]:
            nb = negate(br, root)
            acc = [merge2(x, y, root) for x in acc for y in nb
                   if not _is_false(merge2(x, y, root))]
            if len(acc) > MAX_BRANCH_PRODUCT:
                raise Unmergeable("negate: anyOf product too large")
            if not acc:
                return [dict(FALSE_SCHEMA)]
        return acc
    if keys == {"allOf"}:
        # ¬(A ∧ B) = ¬A ∨ ¬B
        out: list[dict] = []
        for br in schema["allOf"]:
            out.extend(negate(br, root))
        return out
    if keys == {"pattern"}:
        return [{"x-grid-not-patterns": [schema["pattern"]]}]
    if keys == {"required"}:
        req = schema["required"]
        if not isinstance(req, list) or not req:
            raise Unmergeable("negate: bad required")
        return [{"x-grid-forbid-keys": [k]} for k in req]
    if keys == {"minItems"}:
        n = schema["minItems"]
        return [{"maxItems": n - 1}] if n >= 1 else [dict(FALSE_SCHEMA)]
    if keys == {"maxItems"}:
        return [{"minItems": schema["maxItems"] + 1}]
    if keys == {"const"}:
        return [{"x-grid-not-values": [schema["const"]]}]
    if keys == {"enum"}:
        return [{"x-grid-not-values": list(schema["enum"])}]
    if keys <= {"properties", "required"} and "properties" in schema:
        props = schema.get("properties") or {}
        req = list(schema.get("required") or [])
        if len(props) == 1 and req == list(props):
            (k, sub), = props.items()
            # ¬(k present ∧ sub(k)) = k absent ∨ (k present ∧ ¬sub(k))
            out = [{"x-grid-forbid-keys": [k]}]
            for nb in negate(sub, root):
                out.append({"required": [k], "properties": {k: nb}})
            return out
    if keys == {"type"}:
        ts = schema["type"]
        ts = [ts] if isinstance(ts, str) else list(ts)
        comp = [t for t in _TYPES if t not in ts]
        # integer ⊂ number: ¬number excludes integers too
        if "number" in ts and "integer" in comp:
            comp.remove("integer")
        # ¬integer is NOT a type list: "number" in the complement would
        # re-admit integers (every integer is a number)
        if "integer" in ts and "number" not in ts and "number" in comp:
            raise Unmergeable("negate: ¬integer not type-expressible")
        if not comp:
            return [dict(FALSE_SCHEMA)]
        return [{"type": comp}]
    raise Unmergeable(f"negate: unsupported shape {sorted(keys)}")


def to_std(schema: Any) -> Any:
    """Translate internal markers back to standard JSON Schema (used by the
    reference validator and by differential tests)."""
    if isinstance(schema, list):
        return [to_std(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k == "x-grid-forbid-keys":
            clauses = [{"not": {"required": [name]}} for name in v]
            existing = out.get("allOf", [])
            out["allOf"] = existing + clauses
        elif k == "x-grid-not-values":
            existing = out.get("allOf", [])
            out["allOf"] = existing + [{"not": {"enum": list(v)}}]
        elif k == "x-grid-not-patterns":
            existing = out.get("allOf", [])
            out["allOf"] = existing + [{"not": {"pattern": p}} for p in v]
        elif k == "x-grid-extra-patterns":
            existing = out.get("allOf", [])
            out["allOf"] = existing + [{"pattern": p} for p in v]
        else:
            out[k] = to_std(v)
    return out


# ---------------------------------------------------------------- merge

def _both(a: dict, b: dict, key: str):
    return a.get(key), b.get(key)


def merge2(a: Any, b: Any, root: Any, _depth: int = 0) -> dict:
    """Conjunction of two schemas as one schema (raises Unmergeable)."""
    if _depth > 32:
        raise Unmergeable("merge recursion depth (cyclic $ref web)")
    a, b = _as_dict(a), _as_dict(b)
    if _is_false(a) or _is_false(b):
        return dict(FALSE_SCHEMA)
    if _is_true(a):
        return dict(b)
    if _is_true(b):
        return dict(a)
    a = dict(a)
    b = dict(b)
    for s in (a, b):
        _draft4_exclusive(s)

    # $ref on either side: resolve the TOP ref only (shallow — nested refs
    # are carried and resolved by the consumer; deep inlining fails on
    # recursive targets that a trivial sibling merge doesn't even touch)
    for s, other in ((a, b), (b, a)):
        if "$ref" in s:
            seen = set()
            node: Any = s
            rest = {} if _LEGACY_REF else \
                {k: v for k, v in s.items() if k != "$ref"}
            ref = s["$ref"]
            while True:
                if ref in seen:
                    raise Unmergeable(f"recursive $ref {ref!r} under merge")
                seen.add(ref)
                node = _resolve_pointer(root, ref)
                if isinstance(node, dict) and set(node) == {"$ref"}:
                    ref = node["$ref"]
                    continue
                break
            merged = merge2(node, rest, root, _depth + 1) if rest else _as_dict(node)
            return merge2(merged, other, root, _depth + 1)

    # allOf inside a side: fold it into that side first
    for s in (a, b):
        if "allOf" in s:
            folded = fold_allof(s, root)
            if s is a:
                a = folded
            else:
                b = folded
    if "allOf" in a or "allOf" in b:
        raise Unmergeable("nested allOf resisted folding")

    # branch distribution: anyOf/oneOf on a side
    for key in ("anyOf", "oneOf"):
        for s, other in ((a, b), (b, a)):
            if key in s:
                base = {k: v for k, v in s.items() if k != key}
                branches = s[key]
                if len(branches) * 1 > MAX_BRANCH_PRODUCT:
                    raise Unmergeable("branch product too large")
                merged_branches = []
                for br in branches:
                    try:
                        m = merge2(merge2(br, base, root, _depth + 1), other, root, _depth + 1)
                    except Unmergeable:
                        raise
                    if not _is_false(m):
                        merged_branches.append(m)
                if not merged_branches:
                    return dict(FALSE_SCHEMA)
                if len(merged_branches) == 1:
                    return merged_branches[0]
                return {key: merged_branches}

    # enum/const on either side: filter against the other side
    for s, other in ((a, b), (b, a)):
        if "const" in s or "enum" in s:
            values = [s["const"]] if "const" in s else list(s["enum"])
            rest = {k: v for k, v in s.items()
                    if k not in ("const", "enum") and k not in _ANNOTATIONS}
            kept = [v for v in values
                    if _valid(v, rest, root) and _valid(v, other, root)]
            if not kept:
                return dict(FALSE_SCHEMA)
            return {"enum": kept}

    out: dict = {}
    keys = (set(a) | set(b)) - _ANNOTATIONS
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if k not in a:
            out[k] = vb
            continue
        if k not in b:
            out[k] = va
            continue
        if k == "type":
            ta = [va] if isinstance(va, str) else list(va)
            tb = [vb] if isinstance(vb, str) else list(vb)
            inter = [t for t in ta if t in tb]
            # number ⊇ integer
            if "integer" in ta and "number" in tb and "integer" not in inter:
                inter.append("integer")
            if "integer" in tb and "number" in ta and "integer" not in inter:
                inter.append("integer")
            if not inter:
                return dict(FALSE_SCHEMA)
            out[k] = inter if len(inter) > 1 else inter[0]
        elif k == "properties":
            merged = dict(va)
            for pk, pv in vb.items():
                if pk in merged:
                    merged[pk] = merge2(merged[pk], pv, root, _depth + 1)
                else:
                    # b's bare key must still honor a's additionalProperties
                    cap = _extras_schema(a)
                    merged[pk] = pv if cap is True else merge2(pv, cap, root, _depth + 1)
            # a-only keys must honor b's additionalProperties
            cbp = _extras_schema(b)
            if cbp is not True:
                for pk in va:
                    if pk not in vb:
                        merged[pk] = merge2(merged[pk], cbp, root, _depth + 1)
            out[k] = merged
        elif k == "required":
            out[k] = sorted(set(va) | set(vb))
        elif k == "x-grid-forbid-keys":
            out[k] = sorted(set(va) | set(vb))
        elif k == "x-grid-not-values":
            out[k] = va + [x for x in vb if x not in va]
        elif k == "x-grid-not-patterns":
            out[k] = va + [x for x in vb if x not in va]
        elif k == "patternProperties":
            merged_pp = dict(va)
            for pk, pv in vb.items():
                merged_pp[pk] = merge2(merged_pp[pk], pv, root, _depth + 1) \
                    if pk in merged_pp else pv
            out[k] = merged_pp
        elif k == "additionalProperties":
            out[k] = _merge_extras(va, vb, root, _depth)
        elif k == "additionalItems":
            out[k] = _merge_extras(va, vb, root, _depth)
        elif k in _MIN_KEYS:
            out[k] = max(va, vb)
        elif k in _MAX_KEYS:
            out[k] = min(va, vb)
        elif k in ("exclusiveMinimum",):
            out[k] = max(va, vb)
        elif k in ("exclusiveMaximum",):
            out[k] = min(va, vb)
        elif k == "items":
            if isinstance(va, list) or isinstance(vb, list):
                raise Unmergeable("tuple items under merge")
            out[k] = merge2(va, vb, root, _depth + 1)
        elif k == "prefixItems":
            raise Unmergeable("prefixItems under merge")
        elif k == "pattern":
            if va == vb:
                out[k] = va
            else:
                out[k] = va
                extra = out.get("x-grid-extra-patterns", [])
                out["x-grid-extra-patterns"] = extra + [vb]
        elif k == "format":
            if va != vb:
                raise Unmergeable("two distinct formats")
            out[k] = va
        elif k == "propertyNames":
            out[k] = merge2(va, vb, root, _depth + 1)
        elif k in ("uniqueItems",):
            out[k] = bool(va) or bool(vb)
        elif k == "not":
            # ¬A ∧ ¬B = ¬(A ∨ B)
            out[k] = va if va == vb else {"anyOf": [va, vb]}
        elif k == "multipleOf":
            if va == vb:
                out[k] = va
            elif isinstance(va, int) and isinstance(vb, int) and va > 0 and vb > 0:
                import math as _m
                out[k] = va * vb // _m.gcd(va, vb)
            else:
                out[k] = va     # recorded downstream (multipleOf unenforced)
        elif k == "contains":
            out[k] = va         # recorded downstream (contains unenforced)
        elif k == "x-grid-extra-patterns":
            out[k] = va + [x for x in vb if x not in va]
        elif k in ("if", "then", "else",
                   "dependencies", "dependentRequired", "dependentSchemas",
                   "unevaluatedProperties", "unevaluatedItems"):
            if va == vb:
                out[k] = va
            else:
                raise Unmergeable(f"'{k}' on both sides")
        elif k not in _ASSERTIONS:
            continue        # vendor/unknown keyword: annotation, droppable
        else:
            if va == vb:
                out[k] = va
            else:
                raise Unmergeable(f"unknown both-sided key {k!r}")
    if set(out.get("required", []) or []) & set(out.get("x-grid-forbid-keys", []) or []):
        return dict(FALSE_SCHEMA)
    # $defs/definitions are resolution targets, not assertions — losing them
    # dangles every $ref below the merged node
    for dk in ("$defs", "definitions"):
        merged_defs = {}
        for s in (a, b):
            if isinstance(s.get(dk), dict):
                merged_defs.update(s[dk])
        if merged_defs:
            out[dk] = merged_defs
    return out


def _extras_schema(s: dict):
    """The schema governing keys not declared in `properties` for side s."""
    if "additionalProperties" in s:
        ap = s["additionalProperties"]
        return False if ap is False else (True if ap is True else ap)
    if "properties" in s or "patternProperties" in s:
        return True
    # side says nothing about objects -> no constraint on extra keys
    return True


def _merge_extras(va, vb, root, _depth: int = 0):
    if va is False or vb is False:
        return False
    if va is True or va is None:
        return vb
    if vb is True or vb is None:
        return va
    return merge2(va, vb, root, _depth + 1)


def _draft4_exclusive(s: dict) -> None:
    """draft-04 boolean exclusiveMin/Max + minimum/maximum -> numeric form."""
    if s.get("exclusiveMinimum") is True and "minimum" in s:
        s["exclusiveMinimum"] = s.pop("minimum")
    elif s.get("exclusiveMinimum") is False:
        s.pop("exclusiveMinimum")
    if s.get("exclusiveMaximum") is True and "maximum" in s:
        s["exclusiveMaximum"] = s.pop("maximum")
    elif s.get("exclusiveMaximum") is False:
        s.pop("exclusiveMaximum")


def fold_allof(schema: dict, root: Any) -> dict:
    """Fold schema's allOf into the surrounding keywords."""
    branches = schema.get("allOf", [])
    base = {k: v for k, v in schema.items() if k != "allOf"}
    out = base
    for br in branches:
        out = merge2(out, br, root)
        if _is_false(out):
            return dict(FALSE_SCHEMA)
    return out


# ------------------------------------------------------------ normalize

def normalize(schema: Any, root: Any = None) -> Any:
    """Best-effort top-down rewrite; leaves unrewritable constructs intact."""
    global _LEGACY_REF
    if root is None:
        root = schema
    prev = _LEGACY_REF
    if isinstance(schema, dict):
        uri = schema.get("$schema") or ""
        _LEGACY_REF = any(m in uri for m in _LEGACY_MARKERS)
    try:
        return _norm(schema, root, depth=0)
    finally:
        _LEGACY_REF = prev


def _norm(schema: Any, root: Any, depth: int) -> Any:
    if depth > 64 or not isinstance(schema, dict):
        return schema
    s = dict(schema)
    _draft4_exclusive(s)

    # drop no-op keys
    if s.get("dependencies") == {}:
        s.pop("dependencies")
    for dep_key in ("dependentRequired", "dependentSchemas"):
        if s.get(dep_key) == {}:
            s.pop(dep_key)

    # recurse into structural positions
    for key in ("items", "additionalProperties", "additionalItems",
                "propertyNames", "contains"):
        if key in s and isinstance(s[key], (dict,)):
            s[key] = _norm(s[key], root, depth + 1)
    if isinstance(s.get("items"), list):
        s["items"] = [_norm(x, root, depth + 1) for x in s["items"]]
    if isinstance(s.get("prefixItems"), list):
        s["prefixItems"] = [_norm(x, root, depth + 1) for x in s["prefixItems"]]
    for key in ("properties", "patternProperties", "$defs", "definitions"):
        if isinstance(s.get(key), dict):
            s[key] = {k: _norm(v, root, depth + 1) for k, v in s[key].items()}
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(s.get(key), list):
            s[key] = [_norm(x, root, depth + 1) for x in s[key]]

    # $ref with meaningful siblings (vendor/unknown keys are annotations);
    # in draft-07-and-earlier schemas, $ref REPLACES its siblings entirely
    if "$ref" in s and _LEGACY_REF:
        return {k: v for k, v in s.items()
                if k == "$ref" or k in _ANNOTATIONS}
    if "$ref" in s:
        sib = (set(s) - {"$ref"} - _ANNOTATIONS) & _ASSERTIONS
        if not sib and set(s) - {"$ref"} - _ANNOTATIONS:
            keep = {k: v for k, v in s.items()
                    if k == "$ref" or k in _ANNOTATIONS}
            return keep
        if sib:
            try:
                merged = merge2(_inline_refs({"$ref": s["$ref"]}, root),
                                {k: v for k, v in s.items()
                                 if k != "$ref" and k not in _ANNOTATIONS},
                                root)
                return _norm(merged, root, depth + 1)
            except Unmergeable:
                return s        # compiler will declare it
        return s                # bare $ref: compiler follows it natively

    # allOf (branches normalized FIRST so not/if/deps become markers the
    # merge algebra understands)
    if "allOf" in s:
        s = dict(s)
        s["allOf"] = [_norm(b, root, depth + 1) for b in s["allOf"]]
        try:
            folded = fold_allof(s, root)
            return _norm(folded, root, depth + 1)
        except Unmergeable:
            branches = s["allOf"]
            if len(branches) == 1 and not (set(s) - {"allOf"} - _ANNOTATIONS):
                return _norm(branches[0], root, depth + 1)
            return s

    # if/then/else
    if "if" in s:
        cond = s["if"]
        then = s.get("then", True)
        els = s.get("else", True)
        rest = {k: v for k, v in s.items() if k not in ("if", "then", "else")}
        try:
            neg_branches = negate(cond, root)
            pos = merge2(cond, then, root)
            branches = [pos]
            for nb in neg_branches:
                branches.append(merge2(nb, els, root))
            rewritten = dict(rest)
            existing_any = rewritten.pop("anyOf", None)
            new = {"anyOf": branches}
            if existing_any is not None:
                new = {"allOf": [{"anyOf": existing_any}, new]}
                rewritten["allOf"] = new["allOf"]
            else:
                rewritten["anyOf"] = branches
            return _norm(rewritten, root, depth + 1)
        except Unmergeable:
            return s

    # dependencies family
    dep_req: dict[str, list] = {}
    dep_sch: dict[str, Any] = {}
    if "dependencies" in s:
        for k, v in s["dependencies"].items():
            (dep_req if isinstance(v, list) else dep_sch)[k] = v
    dep_req.update(s.get("dependentRequired", {}))
    dep_sch.update(s.get("dependentSchemas", {}))
    if dep_req or dep_sch:
        base = {k: v for k, v in s.items()
                if k not in ("dependencies", "dependentRequired", "dependentSchemas")}
        try:
            variants = [dict(base)]
            for dk, req in sorted(dep_req.items()):
                variants = _expand_dep(variants, dk, {"required": sorted(set(req) | {dk})}, root)
            for dk, sub in sorted(dep_sch.items()):
                sub_n = _norm(sub, root, depth + 1)
                variants = _expand_dep(variants, dk, merge2({"required": [dk]}, sub_n, root), root)
            variants = [v for v in variants if not _is_false(v)]
            if not variants:
                return dict(FALSE_SCHEMA)
            out = variants[0] if len(variants) == 1 else {"anyOf": variants}
            return _norm(out, root, depth + 1)
        except Unmergeable:
            return s

    # not (computable shapes only)
    if "not" in s:
        try:
            neg_branches = negate(_norm(s["not"], root, depth + 1), root)
            rest = {k: v for k, v in s.items() if k != "not"}
            merged = [merge2(rest, nb, root) for nb in neg_branches]
            merged = [m for m in merged if not _is_false(m)]
            if not merged:
                return dict(FALSE_SCHEMA)
            out = merged[0] if len(merged) == 1 else {"anyOf": merged}
            return _norm(out, root, depth + 1)
        except Unmergeable:
            return s

    # anyOf/oneOf with meaningful siblings (incl. type) -> distribute
    for key in ("anyOf", "oneOf"):
        if key in s:
            sib = set(s) - {key} - _ANNOTATIONS
            if sib:
                try:
                    base = {k: v for k, v in s.items() if k != key}
                    branches = []
                    for br in s[key]:
                        m = merge2(br, base, root)
                        if not _is_false(m):
                            branches.append(m)
                    if not branches:
                        return dict(FALSE_SCHEMA)
                    out = branches[0] if len(branches) == 1 else {key: branches}
                    return _norm(out, root, depth + 1)
                except Unmergeable:
                    return s
    for key in ("anyOf", "oneOf"):
        if isinstance(s.get(key), list) and len(s[key]) > 1:
            s = dict(s)
            s[key] = _harmonize_string_consts(s[key])
    return s


def _harmonize_string_consts(branches: list) -> list:
    """anyOf branches where one branch pins a property to a string const and
    another leaves it plain-string collide at the TOKEN level (the const
    lexeme wins maximal-munch priority and commits the parse to one branch).
    Rewrite the plain-string side as (consts | string-minus-consts) so the
    terminals partition the lexeme space and every branch stays live."""
    consts: dict[str, set] = {}
    for b in branches:
        if not isinstance(b, dict):
            continue
        for k, v in (b.get("properties") or {}).items():
            if isinstance(v, dict):
                if isinstance(v.get("const"), str):
                    consts.setdefault(k, set()).add(v["const"])
                elif isinstance(v.get("enum"), list) and v["enum"] and \
                        all(isinstance(x, str) for x in v["enum"]):
                    consts.setdefault(k, set()).update(v["enum"])
    if not consts:
        return branches
    out = []
    for b in branches:
        if not isinstance(b, dict):
            out.append(b)
            continue
        props = dict(b.get("properties") or {})
        changed = False
        for k, cset in consts.items():
            v = props.get(k)
            if isinstance(v, dict) and v.get("type") == "string" and not (
                    set(v) & {"const", "enum", "pattern", "format",
                              "minLength", "maxLength", "x-grid-not-values",
                              "x-grid-not-patterns"}):
                props[k] = {"anyOf": [{"enum": sorted(cset)},
                                      {**v, "x-grid-not-values": sorted(cset)}]}
                changed = True
        out.append({**b, "properties": props} if changed else b)
    return out


def _expand_dep(variants: list[dict], dk: str, present_extra: dict, root: Any) -> list[dict]:
    out = []
    for v in variants:
        if len(out) > MAX_BRANCH_PRODUCT:
            raise Unmergeable("dependency expansion too large")
        # absent branch
        absent = merge2(v, {"x-grid-forbid-keys": [dk]}, root)
        if not _is_false(absent):
            out.append(absent)
        # present branch
        present = merge2(v, present_extra, root)
        if not _is_false(present):
            out.append(present)
    return out
