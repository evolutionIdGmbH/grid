"""End-to-end differential: schema -> grammar -> byte-level engine walk must
agree exactly with the reference validator, for every schema that compiles
with an empty recorded set (nothing unenforced => no excuse for disagreement).
"""

import json
import pathlib
import sys
import warnings

import jsonschema
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

from json_schema_to_grid import Unsupported, compile_schema  # noqa: E402

from grid.generate import build_guide  # noqa: E402
from grid.models.tokenizer_adapter import MockTokenizer  # noqa: E402

warnings.filterwarnings("ignore", message=".*L-REC01.*")

TOK = MockTokenizer()
BYTE_ID = {i: TOK.vocabulary[f"<0x{i:02X}>"] for i in range(256)}


def accepts(guide, text: str) -> bool:
    state = guide.initial_state
    for byte in text.encode("utf-8"):
        ids, _ = guide._mask_ids(state)
        tid = BYTE_ID[byte]
        if tid not in set(int(x) for x in ids):
            return False
        state = guide.get_next_state(state, tid)
    ids, _ = guide._mask_ids(state)
    return TOK.eos_token_id in set(int(x) for x in ids)


def check(schema, instances, draft7=False, expect_ignored=frozenset()):
    src, ignored = compile_schema(schema)
    assert ignored == set(expect_ignored), f"ignored={ignored}\n{src}"
    guide = build_guide(src, TOK)
    v = (jsonschema.Draft7Validator if draft7
         else jsonschema.Draft202012Validator)(schema)
    for inst in instances:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        want = v.is_valid(inst)
        got = accepts(guide, s)
        assert got == want, (f"instance {s!r}: engine={got} validator={want}\n"
                             f"schema={json.dumps(schema)[:300]}\n{src[:1500]}")


# ------------------------------------------------------------- strings

def test_string_pattern():
    check({"type": "string", "pattern": "^[a-z_]+$"},
          ["abc", "a_b", "", "ABC", "a1", "aB", "zzz_", "é"])


def test_string_pattern_unanchored():
    check({"type": "string", "pattern": "11"},
          ["11", "011", "110", "x11y", "1", "1x1", ""])


def test_string_lengths():
    check({"type": "string", "minLength": 2, "maxLength": 4},
          ["", "a", "ab", "abc", "abcd", "abcde", "é", "éé", "😀😀", "a\nb",
           'x"y', "a\\b"])


def _check_labeled(schema, labeled, expect_ignored=frozenset()):
    """For format cases: the stock jsonschema validator treats `format` as
    annotation-only, but the bench's ground truth asserts it — so these use
    hand labels (regex-level correctness is covered in test_rx)."""
    src, ignored = compile_schema(schema)
    assert ignored == set(expect_ignored), ignored
    guide = build_guide(src, TOK)
    for inst, want in labeled:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        assert accepts(guide, s) == want, (s, want)


def test_string_format_uuid():
    _check_labeled({"type": "string", "format": "uuid"},
                   [("123e4567-e89b-12d3-a456-426614174000", True),
                    ("not-a-uuid", False), ("", False)])


def test_string_format_date_time():
    _check_labeled({"type": "string", "format": "date-time"},
                   [("2023-01-15T10:30:00Z", True), ("2023-01-15", False),
                    ("x", False)])


def test_string_unknown_format_recorded():
    check({"type": "string", "format": "ipv6"},
          ["anything", ""], expect_ignored={"format:ipv6"})


# ------------------------------------------------------------- numbers

def test_integer_bounds():
    check({"type": "integer", "minimum": 3, "maximum": 27},
          [2, 3, 4, 26, 27, 28, 0, -3, 100, 10])


def test_integer_exclusive_bounds():
    check({"type": "integer", "exclusiveMinimum": 3, "exclusiveMaximum": 27},
          [2, 3, 4, 26, 27, 28])


def test_integer_negative_bounds():
    check({"type": "integer", "minimum": -12, "maximum": -3},
          [-13, -12, -11, -4, -3, -2, 0, 3, -100])


def test_number_bounds():
    check({"type": "number", "minimum": 0},
          [0, 1, -1, 0.5, -0.5, 100, -0.0, 2.25, 1e30, -1e30, 1e-05, -1e-05])


def test_number_window():
    check({"type": "number", "minimum": 1, "maximum": 10},
          [0, 1, 1.5, 9.99, 10, 10.5, 11, -1, 0.999, 5])


def test_number_multiple_of_recorded():
    check({"type": "number", "multipleOf": 3},
          [3, 6], expect_ignored={"multipleOf"})


# ------------------------------------------------------------- enums

def test_enum_type_filtered():
    check({"enum": [1, "a", True, None], "type": "string"},
          ["a", "b", 1, True, None])


def test_numeric_enum_float_twin():
    # 2 and 2.0 are the same JSON number; both serializations must pass
    check({"enum": [2, 5.0]}, [2, 2.0, 5, 5.0, 3])


# ------------------------------------------------------------- arrays

def test_array_counts():
    check({"type": "array", "items": {"type": "integer"}, "minItems": 1,
           "maxItems": 3},
          [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], ["a"], [1, "a"]])


def test_array_min_only():
    check({"type": "array", "items": {"type": "integer"}, "minItems": 2},
          [[], [1], [1, 2], [1, 2, 3, 4, 5]])


def test_tuple_draft7():
    check({"type": "array",
           "items": [{"type": "integer"}, {"type": "string"}],
           "additionalItems": False},
          [[], [1], [1, "a"], [1, "a", 2], ["a"], [1, 2]], draft7=True)


def test_tuple_2020_prefix():
    check({"type": "array",
           "prefixItems": [{"type": "integer"}, {"type": "string"}],
           "items": {"type": "boolean"}},
          [[], [1], [1, "a"], [1, "a", True], [1, "a", True, False],
           [1, "a", 1], [1, 2]])


def test_tuple_with_min():
    check({"type": "array",
           "prefixItems": [{"type": "integer"}], "items": {"type": "string"},
           "minItems": 2},
          [[], [1], [1, "a"], [1, "a", "b"], [1, 2], ["a", "b"]])


# ------------------------------------------------------------- objects

def test_object_basic():
    check({"type": "object",
           "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
           "required": ["a"]},
          [{}, {"a": 1}, {"a": 1, "b": "x"}, {"b": "x"}, {"a": "bad"},
           {"a": 1, "c": [1, 2]}, {"a": 1, "b": 2}])


def test_object_ap_false():
    check({"type": "object", "properties": {"a": {"type": "integer"}},
           "additionalProperties": False},
          [{}, {"a": 1}, {"b": 1}, {"a": 1, "b": 2}])


def test_object_generic_counts():
    check({"type": "object", "minProperties": 1, "maxProperties": 2},
          [{}, {"a": 1}, {"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}])


def test_object_forbid_keys_marker():
    # produced by the dependencies rewrite; validate against the translated form
    schema = {"type": "object",
              "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
              "dependencies": {"a": ["b"]}}
    src, ignored = compile_schema(schema)
    assert ignored == set()
    guide = build_guide(src, TOK)
    v = jsonschema.Draft7Validator(schema)
    for inst in [{}, {"a": 1}, {"b": "x"}, {"a": 1, "b": "x"}, {"c": True},
                 {"a": 1, "c": True}]:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        assert accepts(guide, s) == v.is_valid(inst), s


# ---------------------------------------------------- normalized combos

def test_allof_merged_object():
    check({"allOf": [
        {"type": "object", "properties": {"a": {"type": "integer"}},
         "required": ["a"]},
        {"properties": {"b": {"type": "string"}}, "required": ["b"]}]},
        [{}, {"a": 1}, {"b": "x"}, {"a": 1, "b": "x"}, {"a": 1, "b": 1}])


def test_allof_ref_and_extension():
    check({"$defs": {"base": {"type": "object",
                              "properties": {"id": {"type": "integer"}},
                              "required": ["id"]}},
           "allOf": [{"$ref": "#/$defs/base"},
                     {"properties": {"name": {"type": "string"}}}]},
          [{"id": 1}, {"id": 1, "name": "x"}, {"name": "x"}, {"id": "bad"}])


def test_allof_enum_not():
    check({"allOf": [{"enum": [1, 2, 3]}, {"not": {"const": 2}}]},
          [1, 2, 3, 4])


def test_not_type():
    check({"not": {"type": "object"}},
          [{}, {"a": 1}, [1], "x", 1, True, None])


def test_if_then_else():
    check({"type": "object",
           "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
           "if": {"required": ["x"]}, "then": {"required": ["y"]}},
          [{}, {"x": 1}, {"y": "a"}, {"x": 1, "y": "a"}])


def test_oneof_disjoint_no_record():
    check({"oneOf": [
        {"type": "object", "properties": {"kind": {"const": "a"},
                                          "v": {"type": "integer"}},
         "required": ["kind"]},
        {"type": "object", "properties": {"kind": {"const": "b"},
                                          "v": {"type": "string"}},
         "required": ["kind"]}]},
        [{"kind": "a", "v": 1}, {"kind": "b", "v": "x"}, {"kind": "c"},
         {"kind": "a", "v": "x"}])


def test_oneof_overlapping_recorded():
    src, ignored = compile_schema({"oneOf": [{"type": "integer"},
                                             {"minimum": 5}]})
    assert "oneOf-exclusivity" in ignored


def test_recursive_ref_still_works():
    check({"$defs": {"node": {"type": "object",
                              "properties": {
                                  "v": {"type": "integer"},
                                  "next": {"$ref": "#/$defs/node"}},
                              "required": ["v"],
                              "additionalProperties": False}},
           "$ref": "#/$defs/node"},
          [{"v": 1}, {"v": 1, "next": {"v": 2}},
           {"v": 1, "next": {"v": 2, "next": {"v": 3}}},
           {"next": {"v": 2}}, {"v": "x"}])


def test_strict_mode_raises():
    with pytest.raises(Unsupported):
        compile_schema({"type": "number", "multipleOf": 3}, strict=True)
    src, ignored = compile_schema({"type": "integer", "minimum": 0}, strict=True)
    assert ignored == set()
