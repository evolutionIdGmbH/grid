"""patternProperties / propertyNames / complement machinery."""

import json
import pathlib
import re
import sys
import warnings

import jsonschema
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

import jsonschema_rx as rx  # noqa: E402
from json_schema_to_grid import compile_schema  # noqa: E402

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
    assert ignored == set(expect_ignored), f"ignored={ignored}"
    guide = build_guide(src, TOK)
    v = (jsonschema.Draft7Validator if draft7
         else jsonschema.Draft202012Validator)(schema)
    for inst in instances:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        want = v.is_valid(inst)
        got = accepts(guide, s)
        assert got == want, (f"instance {s!r}: engine={got} validator={want}\n"
                             f"{src[:1200]}")


# ------------------------------------------------------- complement unit

COMPLEMENT_CASES = [
    ("^[a-z_]+$", ["abc", "a_b", "", "ABC", "a1", "Zz", "_"]),
    ("^.+$", ["", "x", "xy", "\n"]),
    ("^.*$", ["", "x", "\n"]),
    ("^x-", ["x-", "x-1", "x", "y-", "", "xx-"]),
    ("^11", ["11", "112", "1", "21", ""]),
    ("^cdeb", ["cdeb", "cdebX", "cde", "cdX", ""]),
    ("a", ["a", "ba", "bab", "b", "", "xyz"]),
    ("^[0-9a-zA-Z_-]{1,25}$", ["a", "a" * 25, "a" * 26, "", "é", "a b"]),
    ("^.{1,2}$", ["", "a", "ab", "abc"]),
    ("^[a-zA-Z_][a-zA-Z0-9_]*$", ["a", "_a1", "1a", "", "a-b", "Z"]),
]


def test_pattern_complement():
    for pat, probes in COMPLEMENT_CASES:
        comp = rx.pattern_complement_body(pat)
        # Python `$` also matches before a trailing \n; ECMA doesn't -> \Z
        oracle = re.compile(pat.replace("$", r"\Z"), re.ASCII)
        for s in probes:
            body = json.dumps(s, ensure_ascii=False)[1:-1] \
                .encode("utf-8").decode("latin-1")
            in_pattern = oracle.search(s) is not None
            if comp is None:
                assert in_pattern, (pat, s)
                continue
            got = re.fullmatch(comp, body) is not None
            assert got == (not in_pattern), (pat, s, got)


def test_pattern_complement_unsupported():
    with pytest.raises(rx.RxUnsupported):
        rx.pattern_complement_body("^a(b|c)d$")


def test_pattern_complement_head_star_end():
    for pat, probes in [
        ("^...*a$", ["", "a", "xa", "xxa", "xxb", "xyza", "ab"]),
        ("^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$",
         ["", "a", "ab", "a-b", "-ab", "ab-", "a--b", "x-", "9"]),
    ]:
        comp = rx.pattern_complement_body(pat)
        oracle = re.compile(pat.replace("$", r"\Z"), re.ASCII)
        for s in probes:
            body = json.dumps(s, ensure_ascii=False)[1:-1] \
                .encode("utf-8").decode("latin-1")
            want = oracle.search(s) is None
            got = re.fullmatch(comp, body) is not None
            assert got == want, (pat, s, got, want)


# ------------------------------------------------------ patternProperties

def test_pp_only_ap_default():
    # keys matching the pattern take the typed value; others are generic
    check({"type": "object",
           "patternProperties": {"^[a-z_]+$": {"type": "integer"}}},
          [{}, {"ab": 1}, {"ab": "x"}, {"AB": "x"}, {"AB": 1},
           {"ab": 1, "AB": "x"}, {"a_b": 2}, {"a1": "free"}])


def test_pp_ap_false():
    check({"type": "object",
           "patternProperties": {"^[a-z]+$": {"type": "integer"}},
           "additionalProperties": False},
          [{}, {"ab": 1}, {"ab": "x"}, {"AB": 1}, {"ab": 1, "cd": 2}])


def test_pp_with_declared_disjoint():
    check({"type": "object",
           "properties": {"ID": {"type": "integer"}},
           "patternProperties": {"^[a-z]+$": {"type": "string"}},
           "additionalProperties": False},
          [{}, {"ID": 1}, {"ab": "x"}, {"ID": 1, "ab": "x"}, {"ab": 1},
           {"ID": "x"}, {"Zz": "x"}])


def test_pp_overlapping_declared_key():
    # declared key matches the pattern: it must satisfy BOTH schemas, and the
    # pattern pair must exclude it (pattern-minus-keys construction)
    schema = {"type": "object",
              "properties": {"abc": {"minimum": 0}},
              "patternProperties": {"^[a-z]+$": {"type": "integer"}},
              "additionalProperties": False}
    src, ignored = compile_schema(schema)
    assert ignored == set()
    guide = build_guide(src, TOK)
    v = jsonschema.Draft202012Validator(schema)
    for inst in [{"abc": 1}, {"abc": -1}, {"abc": "x"}, {"xyz": -5},
                 {"xyz": "x"}, {"abc": 2, "xyz": 3}, {"ABC": 1}]:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        assert accepts(guide, s) == v.is_valid(inst), s


def test_pp_multiple_disjoint():
    check({"type": "object",
           "patternProperties": {"^[a-z]+$": {"type": "integer"},
                                 "^[A-Z]+$": {"type": "string"}},
           "additionalProperties": False},
          [{}, {"ab": 1}, {"AB": "x"}, {"ab": "x"}, {"AB": 1},
           {"ab": 1, "AB": "x"}, {"a1": 1}])


def test_pp_multiple_overlapping_records():
    # overlapping patterns: a multi-matching key takes ONE pair (union
    # over-admission) — recorded, not declared; single-pattern keys stay exact
    schema = {"type": "object",
              "patternProperties": {"^[a-z]+$": {"type": "integer"},
                                    "^[A-C]+$": {"type": "string"},
                                    "^[a-cX]+$": {"type": "string"}},
              "additionalProperties": False}
    src, ignored = compile_schema(schema)
    assert "patternProperties-overlap" in ignored
    guide = build_guide(src, TOK)
    v = jsonschema.Draft202012Validator(schema)
    # probes whose keys match exactly one pattern must agree with the validator
    for inst in [{"zz": 1}, {"zz": "s"}, {"AB": "s"}, {"AB": 1}, {"X": "s"}]:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        assert accepts(guide, s) == v.is_valid(inst), s


def test_pp_prefix_pattern():
    check({"type": "object",
           "patternProperties": {"^x-": {"type": "integer"}}},
          [{}, {"x-a": 1}, {"x-a": "s"}, {"y": "s"}, {"x": "s"},
           {"x-": 1}, {"x-1": 2, "y": []}])


# ------------------------------------------------------- propertyNames

def test_propertynames_pattern():
    check({"type": "object", "propertyNames": {"pattern": "^[a-z]+$"}},
          [{}, {"ab": 1}, {"AB": 1}, {"ab": 1, "cd": "x"}, {"a1": 1}])


def test_propertynames_not_const():
    # Handwritten---existsName1 shape: propertyNames: {not: {const: "0"}}
    check({"type": "object", "propertyNames": {"not": {"const": "0"}}},
          [{}, {"0": 1}, {"1": 1}, {"00": 1}, {"a": "x"}])


def test_propertynames_enum():
    check({"type": "object", "propertyNames": {"enum": ["a", "b"]}},
          [{}, {"a": 1}, {"b": 2}, {"c": 3}, {"a": 1, "b": 2}])


def test_propertynames_with_declared():
    check({"type": "object",
           "properties": {"ok": {"type": "integer"},
                          "toolong": {"type": "string"}},
           "propertyNames": {"maxLength": 3}},
          [{}, {"ok": 1}, {"toolong": "x"}, {"ab": "free"}, {"abcd": 1}])
