"""grid/perf_flags.py contract tests.

Three gates:

- Grammar oracle: for every flag, the reader must agree with the inline
  expression copied here as the oracle (value grammars are the verbatim
  pre-migration ones; unset defaults reflect the E3 flag disposition) on
  every raw value in the grammar, including the nasty ones ("" enables
  ARTIFACT_STORE via != "0" but disables FACTORED_SCANNER via == "1";
  "true"/"2" are OFF for == "1" flags).
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
    # == "1" value grammar unchanged; unset default flipped to ON (E3
    # disposition, sanctioned by the v0.3.0 full-corpus run)
    oracle = os.environ.get("GRID_PERF_FACTORED_SCANNER", "1") == "1"
    assert perf_flags.factored_scanner_enabled() == oracle
    if raw in ("", "true", "2", "00", "0"):
        assert perf_flags.factored_scanner_enabled() is False


def test_factored_scanner_default_is_on(monkeypatch):
    monkeypatch.delenv("GRID_PERF_FACTORED_SCANNER", raising=False)
    assert perf_flags.factored_scanner_enabled() is True


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


@pytest.mark.parametrize("raw", [None, "3", "1000000", "8192", "-1"])
@pytest.mark.parametrize("default", [8192, 7])
def test_component_budget_oracle(monkeypatch, raw, default):
    _setenv(monkeypatch, "GRID_PERF_COMPONENT_BUDGET", raw)
    # int() with injected default, like factored_budget
    oracle = int(os.environ.get("GRID_PERF_COMPONENT_BUDGET", str(default)))
    assert perf_flags.component_budget(default) == oracle
    if raw is None:
        assert perf_flags.component_budget(default) == default


def test_component_budget_zero_disables(monkeypatch):
    # "0" is the kill switch: None = cap disabled = legacy eager component
    # builds (NOT a zero-state cap — that meaning is reserved for the
    # build_factored_scanner component_budget PARAMETER, the test hook)
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "0")
    assert perf_flags.component_budget(8192) is None
    monkeypatch.delenv("GRID_PERF_COMPONENT_BUDGET", raising=False)
    assert perf_flags.component_budget(0) is None  # a 0 default disables too


@pytest.mark.parametrize("raw", ["abc", "", " ", "1.5", "0x10"])
def test_component_budget_garbage_raises(monkeypatch, raw):
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", raw)
    with pytest.raises(ValueError):
        perf_flags.component_budget(8192)


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_lalr_algorithm_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_LALR_DP", raw)
    # == "1" value grammar unchanged; unset default flipped to "dp" (E3)
    oracle = "dp" if os.environ.get("GRID_PERF_LALR_DP", "1") == "1" else "lr1_merge"
    assert perf_flags.lalr_algorithm() == oracle


def test_lalr_algorithm_default_is_dp(monkeypatch):
    monkeypatch.delenv("GRID_PERF_LALR_DP", raising=False)
    assert perf_flags.lalr_algorithm() == "dp"


_HC = frozenset({"norm", "dedupe"})


def _inline_hashcons(value=None):
    # value grammar verbatim from grid/jsonschema/normalize.py
    # _hashcons_components pre-migration; unset default flipped to the
    # rc2-measured component set (E3)
    if value is None:
        value = os.environ.get("GRID_PERF_HASHCONS", "norm,dedupe")
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


def test_hashcons_default_is_measured_set(monkeypatch):
    """Unset must equal the rc2-measured configuration exactly; "0" and ""
    remain the kill switch."""
    monkeypatch.delenv("GRID_PERF_HASHCONS", raising=False)
    assert perf_flags.hashcons_components() == frozenset({"norm", "dedupe"})
    monkeypatch.setenv("GRID_PERF_HASHCONS", "0")
    assert perf_flags.hashcons_components() == frozenset()
    monkeypatch.setenv("GRID_PERF_HASHCONS", "")
    assert perf_flags.hashcons_components() == frozenset()


@pytest.mark.parametrize("raw", RAW_VALUES)
def test_hashcons_debug_oracle(monkeypatch, raw):
    _setenv(monkeypatch, "GRID_PERF_HASHCONS_DEBUG", raw)
    # verbatim: grid/jsonschema/normalize.py normalize() pre-migration
    oracle = os.environ.get("GRID_PERF_HASHCONS_DEBUG", "0") == "1"
    assert perf_flags.hashcons_debug_enabled() == oracle


@pytest.mark.parametrize("flag,reader", [
    ("GRID_PERF_STORE_COMPONENTS", perf_flags.store_components_enabled),
    ("GRID_PERF_STORE_TRIE", perf_flags.store_trie_enabled),
    ("GRID_PERF_STORE_JOURNAL", perf_flags.store_journal_enabled),
])
@pytest.mark.parametrize("raw", RAW_VALUES)
def test_store_namespace_kill_switch_oracles(monkeypatch, flag, raw, reader):
    """S3 namespace sub-flags: default ON under the ARTIFACT_STORE master,
    "0" the only disabling value (the GRID_GENN_KEYS default-on grammar)."""
    _setenv(monkeypatch, flag, raw)
    oracle = os.environ.get(flag, "1") != "0"
    assert reader() == oracle
    if raw is None:
        assert reader() is True


@pytest.mark.parametrize("raw", [None, "1", "64", "1000000"])
@pytest.mark.parametrize("default", [64, 7])
def test_store_journal_flush_every_oracle(monkeypatch, raw, default):
    _setenv(monkeypatch, "GRID_PERF_STORE_JOURNAL_EVERY", raw)
    oracle = int(os.environ.get("GRID_PERF_STORE_JOURNAL_EVERY", str(default)))
    assert perf_flags.store_journal_flush_every(default) == oracle
    if raw is None:
        assert perf_flags.store_journal_flush_every(default) == default


@pytest.mark.parametrize("raw", ["abc", "", " ", "1.5"])
def test_store_journal_flush_every_garbage_raises(monkeypatch, raw):
    monkeypatch.setenv("GRID_PERF_STORE_JOURNAL_EVERY", raw)
    with pytest.raises(ValueError):
        perf_flags.store_journal_flush_every(64)


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

    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "3")
    assert perf_flags.factored_budget(20_000) == 3
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "4")
    assert perf_flags.factored_budget(20_000) == 4

    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "3")
    assert perf_flags.component_budget(8192) == 3
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "0")
    assert perf_flags.component_budget(8192) is None

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

    for flag, reader in [
        ("GRID_PERF_STORE_COMPONENTS", perf_flags.store_components_enabled),
        ("GRID_PERF_STORE_TRIE", perf_flags.store_trie_enabled),
        ("GRID_PERF_STORE_JOURNAL", perf_flags.store_journal_enabled),
    ]:
        monkeypatch.setenv(flag, "0")
        assert reader() is False
        monkeypatch.setenv(flag, "1")
        assert reader() is True

    monkeypatch.setenv("GRID_PERF_STORE_JOURNAL_EVERY", "5")
    assert perf_flags.store_journal_flush_every(64) == 5
    monkeypatch.setenv("GRID_PERF_STORE_JOURNAL_EVERY", "6")
    assert perf_flags.store_journal_flush_every(64) == 6


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
