"""GRID_PERF_DIRECT_EMIT differential tests (P2 direct emission).

Flag-off (compile text -> spec.load) is the oracle: the flag-on object path
(compile_schema_parts -> DialectGrammar.from_parts) must produce the same
grammar identity — terminal_order FIRST (fingerprint hashes sorted terminal
names, so it alone cannot catch a first-use numbering bug that would
silently renumber terminal ids, masks, and kernel T1/T2 keys), then
fingerprint/terminals/productions/start/ignored — the same recorded set,
the same declared outcomes (Unsupported / GrammarInvalid message parity;
they are load-bearing recorded outcomes), and the same L-REC01 warning
counts. GRID_PERF_DIRECT_EMIT_CHECK=1 is exercised end-to-end as the
permanent render+reload oracle. The corpus-scale gate is
bench/perfbench/diff_direct_emit.py; this file is its committed CI subset.
"""

import json
import pathlib
import warnings

import pytest

from grid.errors import GrammarInvalid
from grid.grammar.projection import RoleProjection
from grid.jsonschema import compile_json_schema_grammar
from grid.jsonschema.compiler import Unsupported
from grid.lalr.compile import compile_tables

DATA = pathlib.Path(__file__).parent / "data" / "direct_emit"

IDENTITY_ATTRS = ("terminal_order", "fingerprint", "terminals",
                  "productions", "start", "ignored")

SYNTHETIC = [
    # object with required subset + extras routing (member machine emits
    # right recursion -> L-REC01 fires at grammar load on BOTH paths)
    {"type": "object",
     "properties": {"name": {"type": "string", "pattern": "^[a-z]{2,8}$"},
                    "n": {"type": "integer"}},
     "required": ["name"]},
    # enum/const literal terminals (E-block ordering)
    {"enum": ["red", "green", "blue", "a longer literal", 1, 2.5, True, None]},
    # closed object: no generic-json rules, epsilon-free member chain
    {"type": "object", "properties": {"a": {"type": "string"}},
     "required": ["a"], "additionalProperties": False},
    # tuple array + numeric bounds (S-terminal ordering is load-bearing)
    {"type": "array",
     "prefixItems": [{"type": "integer", "minimum": 0},
                     {"type": "string", "maxLength": 4}],
     "items": {"type": "number"}, "minItems": 1},
    # branch alternation + $ref cycle through a definition
    {"$defs": {"node": {"type": "object",
                        "properties": {"v": {"type": "integer"},
                                       "next": {"$ref": "#/$defs/node"}},
                        "required": ["v"]}},
     "oneOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]},
    # unicode keys + escaped literal content
    {"type": "object",
     "properties": {"héllo\"quote": {"const": "va\\lue"},
                    "π": {"enum": ["α", "β"]}}},
]

CORPUS_FILES = sorted(p.name for p in DATA.glob("*.json"))


def _load_case(name: str) -> dict:
    with open(DATA / name) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]
    return schema


def _arm(schema, monkeypatch, flag: str):
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT", flag)
    # arm comparisons must pin check mode OFF: the render+reload oracle
    # validates twice by design, doubling L-REC01 counts (the CI check leg
    # exports GRID_PERF_DIRECT_EMIT_CHECK=1 suite-wide; the dedicated
    # check-mode test below re-enables it explicitly)
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "0")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            g, recorded = compile_json_schema_grammar(schema)
        except (Unsupported, GrammarInvalid) as e:
            return (type(e).__name__, str(e)), None, None
    lrec = sum(1 for x in w if "L-REC01" in str(x.message))
    return None, (g, sorted(recorded)), lrec


def _assert_arms_equal(schema, monkeypatch):
    err_off, ok_off, lrec_off = _arm(schema, monkeypatch, "0")
    err_on, ok_on, lrec_on = _arm(schema, monkeypatch, "1")
    assert err_off == err_on
    if err_off is not None:
        return None
    (g_off, rec_off), (g_on, rec_on) = ok_off, ok_on
    for attr in IDENTITY_ATTRS:
        assert getattr(g_on, attr) == getattr(g_off, attr), attr
    assert rec_on == rec_off
    assert lrec_on == lrec_off
    return g_off, g_on


@pytest.mark.parametrize("idx", range(len(SYNTHETIC)))
def test_synthetic_identity(idx, monkeypatch):
    _assert_arms_equal(SYNTHETIC[idx], monkeypatch)


@pytest.mark.parametrize("name", CORPUS_FILES)
def test_corpus_identity(name, monkeypatch):
    _assert_arms_equal(_load_case(name), monkeypatch)


def test_lrec01_fires_and_matches(monkeypatch):
    """SYNTHETIC[0]'s member machine is right-recursive: the lint must fire
    on the object path too (shared validate), same count as text."""
    schema = SYNTHETIC[0]
    _, _, lrec_off = _arm(schema, monkeypatch, "0")
    _, _, lrec_on = _arm(schema, monkeypatch, "1")
    assert lrec_off == lrec_on and lrec_off > 0


def test_unproductive_recursion_grammar_invalid_parity(monkeypatch):
    """Unproductive recursive schemas really occur (GrammarInvalid is a
    recorded outcome bucket): message parity, both arms."""
    schema = {"type": "array", "items": {"$ref": "#"}, "minItems": 1}
    err_off, _, _ = _arm(schema, monkeypatch, "0")
    err_on, _, _ = _arm(schema, monkeypatch, "1")
    assert err_off == err_on
    assert err_off[0] == "GrammarInvalid" and "useless" in err_off[1]


def test_external_ref_unsupported_parity(monkeypatch):
    schema = {"$ref": "https://example.com/other.json"}
    err_off, _, _ = _arm(schema, monkeypatch, "0")
    err_on, _, _ = _arm(schema, monkeypatch, "1")
    assert err_off == err_on and err_off[0] == "Unsupported"


@pytest.mark.parametrize("idx", range(len(SYNTHETIC)))
def test_check_mode_oracle_end_to_end(idx, monkeypatch):
    """GRID_PERF_DIRECT_EMIT_CHECK=1: render+reload oracle inside
    from_parts must hold over the subset (no AssertionError)."""
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT", "1")
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g, _ = compile_json_schema_grammar(SYNTHETIC[idx])
        except (Unsupported, GrammarInvalid):
            return
    assert g.state == "FROZEN"


def test_downstream_tables_identity(monkeypatch):
    """role_shape_hash + LALRTables.fingerprint key kernel configurations
    and T1/T2: full_built over the object-path grammar must match the
    text-path full().build() end to end."""
    schema = _load_case(CORPUS_FILES[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setenv("GRID_PERF_DIRECT_EMIT", "0")
        g_off, _ = compile_json_schema_grammar(schema)
        monkeypatch.setenv("GRID_PERF_DIRECT_EMIT", "1")
        g_on, _ = compile_json_schema_grammar(schema)
    p_off = RoleProjection.full(g_off).build()
    p_on = RoleProjection.full_built(g_on)
    assert p_on.role_shape_hash == p_off.role_shape_hash
    t_off, t_on = compile_tables(p_off), compile_tables(p_on)
    assert t_on.fingerprint == t_off.fingerprint
    assert t_on.action == t_off.action and t_on.goto == t_off.goto
