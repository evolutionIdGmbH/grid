"""Normalization equivalence tests: for every rewrite, the normalized schema
(markers translated back to standard form) must validate exactly like the
original under the reference validator, over curated + generated instances."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

import jsonschema  # noqa: E402
from jsonschema_normalize import (  # noqa: E402
    FALSE_SCHEMA, Unmergeable, merge2, negate, normalize, to_std,
)


PROBES = [
    {}, {"a": 1}, {"a": "x"}, {"a": 1, "b": "y"}, {"b": "y"}, {"a": 2},
    {"a": 1, "b": 2}, {"c": True}, {"a": [1]}, {"a": {"b": 1}},
    {"extended_address": "x", "street_address": "s"},
    {"extended_address": "x"}, {"street_address": "s"},
    {"post_office_box": "p"}, {"post_office_box": "p", "street_address": "s"},
    {"site": 1}, {"app": 1}, {"site": 1, "app": 2},
    1, 2, 3, 0, -1, "x", "red", "green", True, None, 1.5,
    [], [1], [1, 2], ["a"], [1, "a"], [[1]],
]


def equivalent(original, normalized, probes=PROBES, draft7=False):
    # draft-07 oracle for schemas using pre-2019 keywords (`dependencies`);
    # normalized output uses cross-draft keywords, checked under 2020-12
    v_orig = (jsonschema.Draft7Validator if draft7
              else jsonschema.Draft202012Validator)(original)
    v_norm = jsonschema.Draft202012Validator(to_std(normalized))
    for p in probes:
        o = v_orig.is_valid(p)
        n = v_norm.is_valid(p)
        assert o == n, (f"probe {p!r}: original={o} normalized={n}\n"
                        f"norm={normalized!r}")


def test_allof_object_merge():
    s = {"allOf": [
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
        {"properties": {"b": {"type": "string"}}, "required": ["b"]},
    ]}
    n = normalize(s)
    assert "allOf" not in n
    equivalent(s, n)


def test_allof_ref_plus_props():
    s = {
        "$defs": {"base": {"type": "object",
                           "properties": {"a": {"type": "integer"}},
                           "required": ["a"]}},
        "allOf": [{"$ref": "#/$defs/base"},
                  {"properties": {"b": {"type": "string"}}}],
    }
    n = normalize(s)
    assert "allOf" not in n
    equivalent(s, n)


def test_allof_enum_not():
    s = {"allOf": [{"enum": [1, 2, 3]}, {"not": {"const": 2}}]}
    n = normalize(s)
    assert n.get("enum") == [1, 3]
    equivalent(s, n)


def test_allof_contradictory_types():
    s = {"allOf": [{"type": "string"}, {"type": "integer"}]}
    n = normalize(s)
    equivalent(s, n)      # false schema


def test_ref_with_siblings():
    s = {"$defs": {"t": {"type": "object", "properties": {"a": {"type": "integer"}}}},
         "$ref": "#/$defs/t", "required": ["a"]}
    n = normalize(s)
    assert "$ref" not in n
    equivalent(s, n)


def test_dependent_required():
    s = {"type": "object",
         "properties": {"extended_address": {"type": "string"},
                        "post_office_box": {"type": "string"},
                        "street_address": {"type": "string"}},
         "dependencies": {"extended_address": ["street_address"],
                          "post_office_box": ["street_address"]}}
    n = normalize(s)
    assert "dependencies" not in n
    equivalent(s, n, draft7=True)


def test_dependent_schema():
    s = {"type": "object",
         "properties": {"website": {"type": "string"}, "kind": {"type": "string"}},
         "dependentSchemas": {"website": {"properties": {"kind": {"enum": ["section"]}},
                                          "required": ["kind"]}}}
    n = normalize(s)
    assert "dependentSchemas" not in n
    probes = [{}, {"website": "w"}, {"website": "w", "kind": "section"},
              {"website": "w", "kind": "other"}, {"kind": "other"},
              {"kind": "section"}]
    equivalent(s, n, probes)


def test_if_then_else_minitems():
    s = {"type": "array", "if": {"minItems": 1},
         "then": {"items": {"type": "integer"}},
         "else": {"maxItems": 0}}
    n = normalize(s)
    assert "if" not in n
    equivalent(s, n)


def test_if_required():
    s = {"type": "object",
         "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
         "if": {"required": ["x"]}, "then": {"required": ["y"]}}
    n = normalize(s)
    assert "if" not in n
    probes = [{}, {"x": 1}, {"y": "a"}, {"x": 1, "y": "a"}, {"x": "bad"}]
    equivalent(s, n, probes)


def test_not_type():
    s = {"not": {"type": "object"}}
    n = normalize(s)
    assert "not" not in n
    equivalent(s, n)


def test_not_required_forbid():
    s = {"type": "object", "not": {"required": ["site"]}}
    n = normalize(s)
    assert "not" not in n
    equivalent(s, n)


def test_oneof_with_sibling_ap():
    s = {"type": "object",
         "oneOf": [{"properties": {"a": {"type": "integer"}}, "required": ["a"]},
                   {"properties": {"b": {"type": "string"}}, "required": ["b"]}],
         "additionalProperties": False,
         "properties": {"a": {}, "b": {}}}
    n = normalize(s)
    equivalent(s, n)


def test_draft4_exclusive_booleans():
    s = {"type": "integer", "minimum": 5, "exclusiveMinimum": True}
    n = normalize(s)
    assert n.get("exclusiveMinimum") == 5 and "minimum" not in n
    # draft-04 semantics can't be checked by a 2020-12 validator on the
    # ORIGINAL, so check the normalized meaning directly
    v = jsonschema.Draft202012Validator(to_std(n))
    assert v.is_valid(6) and not v.is_valid(5)


def test_not_property_discriminator_rewrites():
    # ¬(a present ∧ integer) = a absent ∨ (a present ∧ ¬integer)
    s = {"not": {"properties": {"a": {"type": "integer"}}, "required": ["a"]}}
    n = normalize(s)
    equivalent(s, n, probes=[{}, {"a": 1}, {"a": "x"}, {"b": 1}, {"a": 1.5},
                             {"a": None}, 5, "s", [1]])


def test_unrewritable_left_intact():
    s = {"not": {"patternProperties": {"^a": {"type": "integer"}}}}
    n = normalize(s)
    assert n == s


def test_negate_shapes():
    assert negate({"minItems": 1}) == [{"maxItems": 0}]
    assert negate({"maxItems": 2}) == [{"minItems": 3}]
    # required-negation pins type:object — non-objects satisfy `required`
    # vacuously so they are NOT in the complement
    assert negate({"required": ["a", "b"]}) == [
        {"type": "object", "x-grid-forbid-keys": ["a"]},
        {"type": "object", "x-grid-forbid-keys": ["b"]}]
    # a vacuously-true payload has a FALSE complement
    assert negate({"properties": {"a": {}}}) == [FALSE_SCHEMA]
    with pytest.raises(Unmergeable):
        negate({"patternProperties": {"^a": {}}})


def test_merge_bounds_and_lengths():
    m = merge2({"minimum": 1, "maximum": 10, "minLength": 2},
               {"minimum": 5, "maximum": 8, "maxLength": 9}, None)
    assert (m["minimum"], m["maximum"], m["minLength"], m["maxLength"]) == (5, 8, 2, 9)


def test_merge_ap_false_wins():
    m = merge2({"type": "object", "properties": {"a": {"type": "integer"}}},
               {"additionalProperties": False, "properties": {"a": {}}}, None)
    assert m["additionalProperties"] is False


def test_merge_props_respect_other_sides_ap():
    a = {"type": "object", "properties": {"a": {"type": "integer"}},
         "additionalProperties": False}
    b = {"properties": {"b": {"type": "string"}}}
    s = {"allOf": [a, b]}
    n = normalize(s)
    equivalent(s, n, probes=[{}, {"a": 1}, {"b": "x"}, {"a": 1, "b": "x"},
                             {"b": 1}, {"c": 1}])


def test_nested_allof():
    s = {"allOf": [{"allOf": [{"type": "object"},
                              {"properties": {"a": {"type": "integer"}}}]},
                   {"required": ["a"]}]}
    n = normalize(s)
    assert "allOf" not in n
    equivalent(s, n)


def test_normalize_idempotent_on_cases():
    cases = [
        {"allOf": [{"enum": [1, 2, 3]}, {"not": {"const": 2}}]},
        {"type": "object", "not": {"required": ["site"]}},
        {"not": {"type": "object"}},
    ]
    for s in cases:
        n1 = normalize(s)
        n2 = normalize(n1)
        assert n1 == n2, s
