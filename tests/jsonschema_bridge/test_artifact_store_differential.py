"""Cold-vs-warm equivalence gate for GRID_PERF_ARTIFACT_STORE.

Three passes per schema — flag-off baseline (the frozen v0.2.5 path), flag-on
cold (fresh store), flag-on warm (same store) — must produce dataclass-equal
artifacts, identical byte-walk accept/reject decisions, and identical error
outcomes. Warm loads are proven real in the cross-process case, not assumed.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import warnings

import pytest

from grid.errors import GrammarInvalid
from grid.generate import build_guide
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.jsonschema import Unsupported, compile_json_schema
from grid.models.tokenizer_adapter import MockTokenizer
from grid.serving import artifact_store as store

warnings.filterwarnings("ignore", message=".*L-REC01.*")

ROOT = pathlib.Path(__file__).resolve().parents[2]
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


CASES = [
    ({"type": "string", "pattern": "^[a-z_]+$"},
     ["abc", "a_b", "", "ABC", "a1"]),
    ({"type": "integer", "minimum": 3, "maximum": 27},
     [2, 3, 4, 26, 27, 28, 10]),
    ({"type": "object",
      "properties": {"a": {"type": "integer"}, "b": {"type": "string", "maxLength": 3}},
      "required": ["a"], "additionalProperties": False},
     [{"a": 1}, {"a": 1, "b": "xy"}, {"b": "xy"}, {"a": 1, "b": "wxyz"}, {}]),
    ({"type": "array", "items": {"enum": ["r", "g", "b", 7]}, "minItems": 1, "maxItems": 3},
     [["r"], ["r", "g", "b"], [], ["x"], ["r", "g", "b", 7], [7]]),
    ({"type": "string", "format": "uuid"},
     ["123e4567-e89b-12d3-a456-426614174000", "not-a-uuid", ""]),
]


def _compile_all(schema):
    src, recorded = compile_json_schema(schema)
    grammar = spec.load(src)
    proj = RoleProjection.full(grammar).build()
    tables = store.load_or_compile_tables(proj)
    dfa = store.load_or_build_scanner(grammar)
    guide = build_guide(src, TOK)
    return src, recorded, tables, dfa, guide


@pytest.mark.parametrize("schema,instances", CASES,
                         ids=[f"case{i}" for i in range(len(CASES))])
def test_cold_warm_equivalence(schema, instances, tmp_path, monkeypatch):
    # scanner budgets pinned: the store persists DENSE artifacts (lazy facades
    # degrade to a per-process rebuild by design), and the lazy-regime CI legs
    # export budget-0/1 ambients that would otherwise turn every dataclass
    # equality below into a facade identity comparison
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "0")
    base = _compile_all(schema)

    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    cold = _compile_all(schema)
    warm = _compile_all(schema)

    for got in (cold, warm):
        assert got[0] == base[0]                       # .grid source
        assert got[1] == base[1]                       # recorded set
        assert got[2] == base[2]                       # LALRTables
        assert got[3] == base[3]                       # ScannerDFA
        assert got[3].h_max == base[3].h_max           # compare=False field
    for inst in instances:
        s = json.dumps(inst, indent=None, ensure_ascii=False)
        want = accepts(base[4], s)
        assert accepts(cold[4], s) == want, s
        assert accepts(warm[4], s) == want, s


# ------------------------------------------------------------- error outcomes

def test_unsupported_raises_cold_and_warm_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    schema = {"type": "string", "format": "ipv6"}
    with pytest.raises(Unsupported) as cold:
        compile_json_schema(schema, strict=True)
    with pytest.raises(Unsupported) as warm:
        compile_json_schema(schema, strict=True)
    assert str(warm.value) == str(cold.value)
    assert not list(tmp_path.rglob("*.bin"))


def test_grammar_invalid_raises_cold_and_warm_uncached(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    g = spec.load("%start a\nX: /x*/\na: X\n")  # empty-matching terminal
    with pytest.raises(GrammarInvalid, match="empty string") as cold:
        store.load_or_build_scanner(g)
    with pytest.raises(GrammarInvalid, match="empty string") as warm:
        store.load_or_build_scanner(g)
    assert str(warm.value) == str(cold.value)
    assert not list(tmp_path.rglob("*.bin"))


# ------------------------------------------------------------- cross-process

def test_cross_process_warm_hit(tmp_path, monkeypatch):
    """A subprocess populates the store; the parent must serve every artifact
    from disk (builders poisoned) and match a flag-off rebuild — the real
    deployment shape for the atomic-rename write path."""
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}

    # dense-artifact pin, as in test_cold_warm_equivalence (child env inherits)
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "0")
    base = _compile_all(schema)

    child = textwrap.dedent("""
        import json, sys
        from grid.grammar import spec
        from grid.grammar.projection import RoleProjection
        from grid.jsonschema import compile_json_schema
        from grid.serving import artifact_store as store
        schema = json.loads(sys.argv[1])
        src, _ = compile_json_schema(schema)
        g = spec.load(src)
        proj = RoleProjection.full(g).build()
        store.load_or_compile_tables(proj)
        store.load_or_build_scanner(g)
    """)
    env = dict(os.environ, GRID_PERF_ARTIFACT_STORE="1", GRID_CACHE_DIR=str(tmp_path),
               PYTHONPATH=str(ROOT))
    subprocess.run([sys.executable, "-c", child, json.dumps(schema)],
                   check=True, env=env, cwd=str(ROOT))
    assert list(tmp_path.rglob("*.bin"))

    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))

    def _boom(*_a, **_k):
        raise AssertionError("builder called: expected a cross-process warm hit")

    monkeypatch.setattr(store, "build_scanner", _boom)
    monkeypatch.setattr(store, "compile_tables", _boom)
    monkeypatch.setattr("grid.jsonschema.compile_schema", _boom)

    src, recorded = compile_json_schema(schema)
    grammar = spec.load(src)
    proj = RoleProjection.full(grammar).build()
    tables = store.load_or_compile_tables(proj)
    dfa = store.load_or_build_scanner(grammar)
    assert (src, recorded) == (base[0], base[1])
    assert tables == base[2]
    assert dfa == base[3] and dfa.h_max == base[3].h_max
