"""Unit semantics of the on-disk compile-artifact store (grid/serving/artifact_store.py).

Every test runs against a hermetic GRID_CACHE_DIR; warm hits are proven by
poisoning the underlying builder, never inferred from timing.
"""

import os

import pytest

from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import compile_tables
from grid.lexer.dfa import build_scanner
from grid.serving import artifact_store as store


def _boom(*_a, **_k):
    raise AssertionError("builder called: expected a warm store hit")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    return tmp_path


def _bins(root):
    return sorted(p for p in root.rglob("*.bin"))


# ------------------------------------------------------------- roundtrips

def test_scanner_roundtrip(cache, toy_grammar, monkeypatch):
    cold = store.load_or_build_scanner(toy_grammar)
    assert cold == build_scanner(toy_grammar.terminals, toy_grammar.terminal_order)
    assert len(_bins(cache)) == 1
    monkeypatch.setattr(store, "build_scanner", _boom)
    warm = store.load_or_build_scanner(toy_grammar)
    assert warm == cold
    assert warm.h_max == cold.h_max  # h_max is compare=False on the dataclass


def test_lalr_roundtrip(cache, toy_grammar, monkeypatch):
    proj = RoleProjection.full(toy_grammar).build()
    cold = store.load_or_compile_tables(proj)
    assert cold == compile_tables(RoleProjection.full(toy_grammar).build())
    monkeypatch.setattr(store, "compile_tables", _boom)
    warm = store.load_or_compile_tables(proj)
    assert warm == cold


def test_schema_src_roundtrip(cache, monkeypatch):
    from grid.jsonschema import compile_json_schema

    schema = {"type": "string", "format": "ipv6"}  # nonempty recorded set
    cold_src, cold_rec = compile_json_schema(schema)
    assert cold_rec == {"format:ipv6"}
    monkeypatch.setattr("grid.jsonschema.compile_schema", _boom)
    warm_src, warm_rec = compile_json_schema(schema)
    assert (warm_src, warm_rec) == (cold_src, cold_rec)
    assert isinstance(warm_rec, set)


def test_lazy_scanner_never_persisted(cache, toy_grammar, monkeypatch):
    """Store law (P1): product-interner state is never persisted. Forced-lazy
    builds (component budget 1) return the facade and write NO entry — and
    emit no put-failed warning (the pre-P1 behavior pickled the facade and
    warned on its locks)."""
    import warnings as _warnings

    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any store warning fails the test
        dfa = store.load_or_build_scanner(toy_grammar)
    assert getattr(dfa, "lazy", False), "forced-lazy fixture must build lazy"
    assert _bins(cache) == []


# ------------------------------------------------------------- flag off

def test_flag_off_noop(tmp_path, monkeypatch, toy_grammar):
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "0")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    store.put("scanner", "k", (1, 2))
    assert store.get("scanner", "k") is None
    dfa = store.load_or_build_scanner(toy_grammar)
    assert dfa == build_scanner(toy_grammar.terminals, toy_grammar.terminal_order)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------- self-heal

def test_corrupt_entry_selfheals(cache, toy_grammar):
    cold = store.load_or_build_scanner(toy_grammar)
    (path,) = _bins(cache)
    path.write_bytes(path.read_bytes()[:10])
    assert store.get("scanner", path.stem) is None
    assert not path.exists()
    assert store.load_or_build_scanner(toy_grammar) == cold
    assert len(_bins(cache)) == 1


def test_envelope_key_mismatch_rejected(cache):
    store.put("scanner", "aaa", ("payload",))
    (path,) = _bins(cache)
    stolen = path.with_name("bbb.bin")
    stolen.write_bytes(path.read_bytes())
    assert store.get("scanner", "bbb") is None
    assert not stolen.exists()  # unlinked on mismatch
    assert store.get("scanner", "aaa") == ("payload",)


def test_epoch_change_misses(cache, toy_grammar, monkeypatch):
    store.load_or_build_scanner(toy_grammar)
    monkeypatch.setattr(store, "code_epoch", lambda: "f" * 32)
    assert len(_bins(cache)) == 1
    calls = []
    real = build_scanner

    def counting(terminals, order):
        calls.append(1)
        return real(terminals, order)

    monkeypatch.setattr(store, "build_scanner", counting)
    store.load_or_build_scanner(toy_grammar)
    assert calls  # old-epoch entry was not served


# ------------------------------------------------------------- degradation

def test_put_failure_warns_and_compile_succeeds(cache, toy_grammar, monkeypatch):
    monkeypatch.setattr(store, "_put_warned", False)
    os.chmod(cache, 0o500)
    try:
        with pytest.warns(UserWarning, match="artifact store"):
            dfa = store.load_or_build_scanner(toy_grammar)
        assert dfa == build_scanner(toy_grammar.terminals, toy_grammar.terminal_order)
    finally:
        os.chmod(cache, 0o700)


def test_no_tmp_litter(cache, toy_grammar):
    store.load_or_build_scanner(toy_grammar)
    proj = RoleProjection.full(toy_grammar).build()
    store.load_or_compile_tables(proj)
    assert not [p for p in cache.rglob("*") if ".tmp." in p.name]


# ------------------------------------------------------------- key soundness

def test_identifier_terminals_key_separation(cache, sql_grammar, monkeypatch):
    """identifier_terminals is a compile_tables input not covered by
    LALRTables.fingerprint — each set must get its own entry."""
    proj = RoleProjection.full(sql_grammar).build()
    idents = frozenset({"TABLE_NAME", "COLUMN_NAME"})
    plain = store.load_or_compile_tables(proj)
    with_idents = store.load_or_compile_tables(proj, idents)
    assert plain.identifier_terminal_ids == frozenset()
    assert with_idents.identifier_terminal_ids
    assert len([p for p in _bins(cache) if p.parent.name == "lalr"]) == 2
    monkeypatch.setattr(store, "compile_tables", _boom)
    assert store.load_or_compile_tables(proj).identifier_terminal_ids == frozenset()
    assert store.load_or_compile_tables(proj, idents).identifier_terminal_ids \
        == with_idents.identifier_terminal_ids


def test_terminal_order_key_separation(cache):
    """DialectGrammar.fingerprint hashes terminals sorted by name, so two
    sources that differ only in declaration order collide on fingerprint while
    needing different terminal-id numbering — the store key must split them."""
    g1 = spec.load("%start a\nAX: /foo/\nBX: /bar/\na: AX | BX\n")
    g2 = spec.load("%start a\nBX: /bar/\nAX: /foo/\na: AX | BX\n")
    assert g1.fingerprint == g2.fingerprint
    assert g1.terminal_order != g2.terminal_order
    d1 = store.load_or_build_scanner(g1)
    d2 = store.load_or_build_scanner(g2)
    assert d1 == build_scanner(g1.terminals, g1.terminal_order)
    assert d2 == build_scanner(g2.terminals, g2.terminal_order)
    assert d1 != d2  # a shared entry would have served g1's numbering for g2
    t1 = store.load_or_compile_tables(RoleProjection.full(g1).build())
    t2 = store.load_or_compile_tables(RoleProjection.full(g2).build())
    assert t1.terminal_names != t2.terminal_names


def test_schema_insertion_order_key_separation(cache, monkeypatch):
    """compile_schema output depends on dict insertion order (keyword-terminal
    numbering and rule naming follow property order), so dict-equal schemas
    differing only in properties insertion order must get their own entries —
    a sorted-key canon aliases them, and the round-trip faithful check cannot
    split them because dict equality ignores order."""
    from grid.jsonschema import compile_json_schema
    from grid.jsonschema.compiler import compile_schema

    a = {"type": "object",
         "properties": {"alpha": {"type": "integer"}, "beta": {"type": "string"}},
         "required": ["alpha", "beta"], "additionalProperties": False}
    b = {"type": "object",
         "properties": {"beta": {"type": "string"}, "alpha": {"type": "integer"}},
         "required": ["alpha", "beta"], "additionalProperties": False}
    assert a == b  # dict-equal: exactly the pair a sorted-key canon collapses
    own_a, _ = compile_schema(a)  # ground truth, no store involved
    own_b, _ = compile_schema(b)
    assert own_a != own_b  # order-variant sources: a shared entry would lie
    assert compile_json_schema(a)[0] == own_a
    assert compile_json_schema(b)[0] == own_b  # cold: not served a's entry
    assert len([p for p in _bins(cache) if p.parent.name == "schema_src"]) == 2
    monkeypatch.setattr("grid.jsonschema.compile_schema", _boom)
    assert compile_json_schema(a)[0] == own_a  # warm: each hit is its own
    assert compile_json_schema(b)[0] == own_b


def test_epoch_failure_degrades_to_miss(cache, toy_grammar, monkeypatch):
    """code_epoch reads module sources off disk; in a pyc-only deployment it
    raises — get must degrade to a miss instead of breaking the compile."""
    def no_source():
        raise OSError("pyc-only deployment: no module source on disk")

    monkeypatch.setattr(store, "_put_warned", False)
    monkeypatch.setattr(store, "code_epoch", no_source)
    assert store.get("scanner", "k") is None
    with pytest.warns(UserWarning, match="artifact store"):
        dfa = store.load_or_build_scanner(toy_grammar)
    assert dfa == build_scanner(toy_grammar.terminals, toy_grammar.terminal_order)


def test_uncached_projection_raises_like_compile_tables(cache, toy_grammar):
    proj = RoleProjection.full(toy_grammar)  # never .build(): not CACHED
    with pytest.raises(ValueError, match="CACHED"):
        store.load_or_compile_tables(proj)
