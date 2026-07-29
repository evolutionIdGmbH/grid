"""grid/perf_flags.py contract tests.

Three gates:

- Grammar oracle: for every flag, the reader must agree with the verbatim
  pre-migration inline expression (copied here as the oracle) on every raw
  value in the grammar, including the nasty ones ("" enables ARTIFACT_STORE
  via != "0" but leaves FACTORED_SCANNER off via == "1"; "true"/"2" are OFF
  for == "1" flags; NFA_LIVE unset defaults ON).
- Call-time read: two calls with an env flip between them see different
  values — kills any future lru_cache / module-level snapshot.
- Leaf import: ``import grid.perf_flags`` pulls no other grid submodule
  (protects the grid.jsonschema flag-off fast path from regaining the
  grid.serving import cost through this module).
"""

import os
import pathlib
import subprocess
import sys

import pytest

from grid import perf_flags

# Raw values crossing every flag's grammar: empty, zeros, ones, tri-state
# words, component lists (with whitespace and unknown names), and garbage.
RAW_VALUES = [
    None,  # unset
    "",
    "0",
    "1",
    "verify",
    "nfa",
    "all",
    "true",
    "2",
    "00",
    "norm",
    "dedupe",
    "norm,dedupe",
    " norm , dedupe ",
    "bogus,norm",
    "abc",
]


def _setenv(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


# ------------------------------------------------------- grammar oracles


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_artifact_store_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_ARTIFACT_STORE", raw)
    # verbatim: grid/serving/artifact_store.py enabled() pre-migration
    oracle = os.environ.get("GRID_PERF_ARTIFACT_STORE", "0") != "0"
    assert perf_flags.artifact_store_enabled() == oracle
    # the grid/jsonschema/__init__.py pre-check is the exact negation
    skip_fast_path = os.environ.get("GRID_PERF_ARTIFACT_STORE", "0") == "0"
    assert (not perf_flags.artifact_store_enabled()) == skip_fast_path


def test_artifact_store_empty_string_enables(monkeypatch):
    # the nasty case the != "0" grammar implies: "" ENABLES the store
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "")
    assert perf_flags.artifact_store_enabled() is True


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_factored_scanner_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_FACTORED_SCANNER", raw)
    # verbatim: grid/lexer/dfa.py build_scanner pre-migration
    oracle = os.environ.get("GRID_PERF_FACTORED_SCANNER", "0") == "1"
    assert perf_flags.factored_scanner_enabled() == oracle
    if raw in ("", "true", "2", "00"):
        assert perf_flags.factored_scanner_enabled() is False


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_nfa_live_mode_predicate_pair(monkeypatch, raw):
    """dfa.py consumes the raw string only through the two predicates
    (mode != "0", mode == "verify"); both must be invariant under the
    normalization to {"0", "verify", "nfa"}."""
    _setenv(monkeypatch, "GRID_PERF_NFA_LIVE", raw)
    raw_mode = os.environ.get("GRID_PERF_NFA_LIVE", "1")  # verbatim dfa.py read
    mode = perf_flags.nfa_live_mode()
    assert mode in ("0", "verify", "nfa")
    assert (mode != "0") == (raw_mode != "0")
    assert (mode == "verify") == (raw_mode == "verify")


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_nfa_live_mode_matches_legacy_live_mode(monkeypatch, raw):
    """Verbatim oracle: factored.py _live_mode() pre-migration body."""
    _setenv(monkeypatch, "GRID_PERF_NFA_LIVE", raw)
    legacy = os.environ.get("GRID_PERF_NFA_LIVE", "1")
    if legacy != "0":
        legacy = "verify" if legacy == "verify" else "nfa"
    assert perf_flags.nfa_live_mode() == legacy


def test_nfa_live_mode_default_is_on(monkeypatch):
    monkeypatch.delenv("GRID_PERF_NFA_LIVE", raising=False)
    assert perf_flags.nfa_live_mode() == "nfa"


@pytest.mark.parametrize("raw", [None, "0", "3", "1000000", "-1", "20000"])
@pytest.mark.parametrize("default", [20_000, 7])
def test_factored_budget_oracle(monkeypatch, raw, default):
    _setenv(monkeypatch, "GRID_PERF_FACTORED_BUDGET", raw)
    # verbatim: grid/lexer/factored.py pre-migration (default injected)
    oracle = int(os.environ.get("GRID_PERF_FACTORED_BUDGET", str(default)))
    assert perf_flags.factored_budget(default) == oracle
    if raw is None:
        assert perf_flags.factored_budget(default) == default


@pytest.mark.parametrize("raw", ["abc", "", " ", "1.5", "0x10"])
def test_factored_budget_garbage_raises(monkeypatch, raw):
    # int() ValueError propagates exactly like the historical inline read
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", raw)
    with pytest.raises(ValueError):
        perf_flags.factored_budget(20_000)


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_lalr_algorithm_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_LALR_DP", raw)
    # verbatim: grid/lalr/compile.py compile_tables pre-migration
    oracle = "dp" if os.environ.get("GRID_PERF_LALR_DP", "0") == "1" else "lr1_merge"
    assert perf_flags.lalr_algorithm() == oracle


_HC = frozenset({"norm", "dedupe"})


def _inline_hashcons(value=None):
    # verbatim: grid/jsonschema/normalize.py _hashcons_components pre-migration
    if value is None:
        value = os.environ.get("GRID_PERF_HASHCONS", "")
    value = value.strip()
    if value in ("", "0"):
        return frozenset()
    if value in ("1", "all"):
        return _HC
    return frozenset(p.strip() for p in value.split(",")) & _HC


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_hashcons_components_env_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_HASHCONS", raw)
    assert perf_flags.hashcons_components() == _inline_hashcons()


@pytest.mark.parametrize("raw", [v for v in RAW_VALUES if v is not None])
def test_hashcons_components_value_oracle(raw):
    # explicit-value path (used by tests and compiler kwarg plumbing)
    assert perf_flags.hashcons_components(raw) == _inline_hashcons(raw)


def test_hashcons_components_constant():
    assert perf_flags.HASHCONS_COMPONENTS == _HC


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_hashcons_debug_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_HASHCONS_DEBUG", raw)
    # verbatim: grid/jsonschema/normalize.py normalize() pre-migration
    oracle = os.environ.get("GRID_PERF_HASHCONS_DEBUG", "0") == "1"
    assert perf_flags.hashcons_debug_enabled() == oracle


# ------------------------------------------------------- call-time reads


def test_every_reader_is_call_time(monkeypatch):
    """Two calls with an env flip between them must observe the flip —
    kills any future lru_cache or module-level snapshot on any reader."""
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "0")
    assert perf_flags.artifact_store_enabled() is False
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    assert perf_flags.artifact_store_enabled() is True

    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "0")
    assert perf_flags.factored_scanner_enabled() is False
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    assert perf_flags.factored_scanner_enabled() is True

    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "0")
    assert perf_flags.nfa_live_mode() == "0"
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "verify")
    assert perf_flags.nfa_live_mode() == "verify"

    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "3")
    assert perf_flags.factored_budget(20_000) == 3
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "4")
    assert perf_flags.factored_budget(20_000) == 4

    monkeypatch.setenv("GRID_PERF_LALR_DP", "1")
    assert perf_flags.lalr_algorithm() == "dp"
    monkeypatch.setenv("GRID_PERF_LALR_DP", "0")
    assert perf_flags.lalr_algorithm() == "lr1_merge"

    monkeypatch.setenv("GRID_PERF_HASHCONS", "norm")
    assert perf_flags.hashcons_components() == frozenset({"norm"})
    monkeypatch.setenv("GRID_PERF_HASHCONS", "all")
    assert perf_flags.hashcons_components() == _HC

    monkeypatch.setenv("GRID_PERF_HASHCONS_DEBUG", "1")
    assert perf_flags.hashcons_debug_enabled() is True
    monkeypatch.setenv("GRID_PERF_HASHCONS_DEBUG", "0")
    assert perf_flags.hashcons_debug_enabled() is False


# ------------------------------------------------------- leaf import


def test_leaf_import_pulls_no_other_grid_module():
    """import grid.perf_flags must leave sys.modules free of every other
    grid submodule — the jsonschema flag-off fast path depends on it."""
    repo_root = pathlib.Path(perf_flags.__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import grid.perf_flags\n"
        "extra = sorted(m for m in sys.modules\n"
        "               if m.startswith('grid') and m not in ('grid', 'grid.perf_flags'))\n"
        "assert not extra, f'perf_flags is not a leaf: {extra}'\n"
    )
    env = dict(os.environ, PYTHONPATH=str(repo_root))
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=env, cwd=str(repo_root)
    )
