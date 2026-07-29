"""LALR-conflict retry: branch string-value unification (wp_105 family).

anyOf branches that pin the same property to overlapping string consts/enums
compile to near-twin value rules (the consts path of _harmonize_string_consts
rewrites only the plain-string branches); when the branches' object shapes
keep the LALR states merged, the twins collide reduce-reduce on the shared
lexeme's terminal (WashingtonPost wp_105: r279_v vs r2048_v on E6).

The fix is retry-shaped BY REQUIREMENT: compile_schema(unify_string_values=
True) is only ever invoked by a caller that already saw compile_tables raise
LALRConflictError, so schemas that compile today never run the new path (a
byte-identical unconditional pass is impossible — the unification widens
per-branch tightness, changing emitted grammars for schemas that compile
fine). The widening is recorded (branch-string-values-unified) in default
mode and declared Unsupported in strict mode.
"""

import json
import pathlib
import sys
import time
import warnings

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

from grid.errors import LALRConflictError  # noqa: E402
from grid.generate import build_guide  # noqa: E402
from grid.grammar import spec  # noqa: E402
from grid.grammar.projection import RoleProjection  # noqa: E402
from grid.jsonschema.compiler import (  # noqa: E402
    Unsupported,
    compile_schema,
)
from grid.jsonschema.normalize import (  # noqa: E402
    _unify_string_values,
    normalize,
)
from grid.lalr.compile import compile_tables  # noqa: E402
from grid.models.tokenizer_adapter import MockTokenizer  # noqa: E402

warnings.filterwarnings("ignore", message=".*L-REC01.*")

DATA = pathlib.Path(__file__).parent / "data" / "hashcons"
OFF = frozenset()
BOTH = frozenset({"norm", "dedupe"})

# Miniature of the wp_105 shape: same property pinned to overlapping string
# values (const "a" vs enum ["a","b"]) across branches whose member machines
# stay LALR-merged until after the value reduce. The distinct second
# required key keeps the branches distinguishable AFTER unification (twin
# member chains — identical required sets — are the documented residual
# family, not fixable by value unification).
SYNTH = {
    "anyOf": [
        {"type": "object",
         "properties": {"t": {"const": "a"}, "x": {"type": "string"}},
         "required": ["t", "x"], "additionalProperties": False},
        {"type": "object",
         "properties": {"t": {"enum": ["a", "b"]}, "y": {"type": "integer"}},
         "required": ["t", "y"], "additionalProperties": False},
    ]
}


def _tables(src: str):
    grammar = spec.load(src)
    return compile_tables(RoleProjection.full(grammar).build())


# ------------------------------------------------------------ retry shape

@pytest.mark.parametrize("hc", [OFF, BOTH], ids=["hc-off", "hc-both"])
def test_synthetic_conflict_then_retry_compiles(hc):
    # normal build: the const/enum twin value rules reduce-reduce collide
    src, recorded = compile_schema(SYNTH, hashcons=hc)
    assert "branch-string-values-unified" not in recorded
    with pytest.raises(LALRConflictError):
        _tables(src)
    # the retry (what callers run after catching LALRConflictError)
    src2, recorded2 = compile_schema(SYNTH, hashcons=hc,
                                     unify_string_values=True)
    assert "branch-string-values-unified" in recorded2
    _tables(src2)  # must not raise


@pytest.mark.parametrize("hc", [OFF, BOTH], ids=["hc-off", "hc-both"])
def test_default_path_byte_identical(hc):
    # unify_string_values=False is the default — flag-off compiles must be
    # byte-identical with and without the parameter spelled out
    src_default, rec_default = compile_schema(SYNTH, hashcons=hc)
    src_off, rec_off = compile_schema(SYNTH, hashcons=hc,
                                      unify_string_values=False)
    assert src_default == src_off and rec_default == rec_off
    # and the unified grammar is genuinely a different (opt-in) artifact
    src_on, _ = compile_schema(SYNTH, hashcons=hc, unify_string_values=True)
    assert src_on != src_default


def test_strict_mode_declares_the_widening():
    # strict callers get llguidance-style declared non-support, not a
    # silently widened grammar
    with pytest.raises(Unsupported, match="branch-string-values-unified"):
        compile_schema(SYNTH, strict=True, unify_string_values=True)


def test_retry_grammar_instance_behavior():
    # the unified grammar stays sound (every valid instance accepted) and
    # the loss is exactly the recorded per-branch tightness
    src, recorded = compile_schema(SYNTH, unify_string_values=True)
    assert "branch-string-values-unified" in recorded
    tok = MockTokenizer()
    byte_id = {i: tok.vocabulary[f"<0x{i:02X}>"] for i in range(256)}
    guide = build_guide(src, tok)

    def accepts(text: str) -> bool:
        state = guide.initial_state
        for byte in text.encode("utf-8"):
            ids, _ = guide._mask_ids(state)
            tid = byte_id[byte]
            if tid not in set(int(x) for x in ids):
                return False
            state = guide.get_next_state(state, tid)
        ids, _ = guide._mask_ids(state)
        return tok.eos_token_id in set(int(x) for x in ids)

    dump = json.dumps
    assert accepts(dump({"t": "a", "x": "s"}, indent=None))   # branch 1
    assert accepts(dump({"t": "b", "y": 3}, indent=None))     # branch 2
    assert not accepts(dump({"t": "c", "y": 3}, indent=None))  # outside union
    assert not accepts(dump({"x": "s"}, indent=None))          # required
    # the recorded widening: branch 1's t="a" pin is no longer enforced
    assert accepts(dump({"t": "b", "x": "s"}, indent=None))


# ------------------------------------------------------- unify mechanics

def test_unified_value_is_one_shared_object():
    # the collapse mechanism is object identity: every branch's value for
    # the overlapping key must be the SAME node, so the compiler's id()
    # rule memo emits ONE rule (twin equal-but-distinct nodes are exactly
    # what conflicted)
    out = normalize(SYNTH, unify_string_values=True)
    b1, b2 = out["anyOf"]
    assert b1["properties"]["t"] is b2["properties"]["t"]
    assert b1["properties"]["t"].get("x-grid-branch-unified") is True


def test_discriminated_union_left_alone():
    # pairwise-disjoint enum sets with no plain-string shape cannot collide
    # at the token level — the discriminator stays tight per branch
    branches = [
        {"properties": {"t": {"const": "a"}, "x": {"type": "integer"}}},
        {"properties": {"t": {"const": "b"}, "y": {"type": "integer"}}},
    ]
    assert _unify_string_values(branches) is branches


def test_identical_enum_sets_left_alone():
    # identical sets dedupe into one rule naturally — nothing to unify
    branches = [
        {"properties": {"t": {"enum": ["a", "b"]}}},
        {"properties": {"t": {"enum": ["a", "b"]}}},
    ]
    assert _unify_string_values(branches) is branches


def test_plain_string_arm_excludes_the_union():
    # a coexisting plain-string shape becomes an arm minus the enum union
    # (terminal partition, same convention as the consts path)
    branches = [
        {"properties": {"t": {"const": "a"}}},
        {"properties": {"t": {"type": "string", "maxLength": 5}}},
    ]
    out = _unify_string_values(branches)
    u1 = out[0]["properties"]["t"]
    assert u1 is out[1]["properties"]["t"]
    assert u1["x-grid-branch-unified"] is True
    assert u1["anyOf"][0] == {"enum": ["a"]}
    assert u1["anyOf"][1] == {"type": "string", "maxLength": 5,
                              "x-grid-not-values": ["a"]}


def test_non_dict_branches_pass_through():
    branches = [
        True,
        {"properties": {"t": {"const": "a"}}},
        {"properties": {"t": {"enum": ["a", "b"]}}},
    ]
    out = _unify_string_values(branches)
    assert out[0] is True
    assert out[1]["properties"]["t"] is out[2]["properties"]["t"]


# ------------------------------------------------------------- wp_105

def test_wp105_conflict_then_retry_compiles():
    # the motivating schema. hashcons=BOTH only: legacy normalize never
    # terminates on wp_105 (see the hang canary), so flag-off there is no
    # conflict to retry from — the schema times out upstream of the tables.
    with open(DATA / "wp_105_Normalized.json") as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]
    t0 = time.monotonic()
    src, recorded = compile_schema(schema, hashcons=BOTH)
    with pytest.raises(LALRConflictError):
        _tables(src)
    src2, recorded2 = compile_schema(schema, hashcons=BOTH,
                                     unify_string_values=True)
    assert "branch-string-values-unified" in recorded2
    # the retry may not weaken anything the normal compile enforced
    assert recorded2 >= recorded
    _tables(src2)  # must not raise
    assert time.monotonic() - t0 < 30
