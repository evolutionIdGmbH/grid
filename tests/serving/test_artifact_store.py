"""Unit semantics of the on-disk compile-artifact store (grid/serving/artifact_store.py).

Every test runs against a hermetic GRID_CACHE_DIR; warm hits are proven by
poisoning the underlying builder, never inferred from timing.
"""

import os
import pathlib

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
    # exactly one scanner entry (the factored default also writes the
    # per-terminal component namespace alongside — S3)
    assert len([p for p in _bins(cache) if p.parent.name == "scanner"]) == 1
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
    """Store law (P1), scoped per the S3 merge: product-interner state is
    never persisted — forced-lazy builds (component budget 1) return the
    facade and write NO scanner-namespace entry, and emit no put-failed
    warning (the pre-P1 behavior pickled the facade and warned on its
    locks). Component-namespace entries ARE expected: a lazy facade's
    redeploy warmth comes from the cross-schema component store (module
    docstring), and component identity is grammar-independent."""
    import warnings as _warnings

    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")  # pin: legacy leg disables factored
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any store warning fails the test
        dfa = store.load_or_build_scanner(toy_grammar)
    assert getattr(dfa, "lazy", False), "forced-lazy fixture must build lazy"
    by_ns = {p.parent.name for p in _bins(cache)}
    assert "scanner" not in by_ns and "lalr" not in by_ns, by_ns
    assert by_ns <= {"component"}, by_ns


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
    (path,) = [p for p in _bins(cache) if p.parent.name == "scanner"]
    path.write_bytes(path.read_bytes()[:10])
    assert store.get("scanner", path.stem) is None
    assert not path.exists()
    assert store.load_or_build_scanner(toy_grammar) == cold
    assert len([p for p in _bins(cache) if p.parent.name == "scanner"]) == 1


def test_envelope_key_mismatch_rejected(cache):
    store.put("scanner", "aaa", ("payload",))
    (path,) = _bins(cache)
    stolen = path.with_name("bbb.bin")
    stolen.write_bytes(path.read_bytes())
    assert store.get("scanner", "bbb") is None
    assert not stolen.exists()  # unlinked on mismatch
    assert store.get("scanner", "aaa") == ("payload",)


@pytest.mark.parametrize("module", [
    "grid.lexer.factored",   # component + materialized-scanner payload source
    "grid.trie.build",       # trie namespace payload source
    "grid.lexer.subset",     # E2 split: subset construction behind every scanner
    "grid.lexer.nfa",
    "grid.lexer.rx",
])
def test_source_mutation_changes_epoch(module, tmp_path, monkeypatch):
    """Every payload-producing module participates in code_epoch: mutating its
    source must roll the epoch (= a fresh store directory, wholesale miss)."""
    import importlib

    mod = importlib.import_module(module)
    assert module in store._EPOCH_MODULES
    store.code_epoch.cache_clear()
    try:
        before = store.code_epoch()
        mutated = tmp_path / "mutated.py"
        mutated.write_bytes(pathlib.Path(mod.__file__).read_bytes() + b"\n# mutated\n")
        monkeypatch.setattr(mod, "__file__", str(mutated))
        store.code_epoch.cache_clear()
        assert store.code_epoch() != before
    finally:
        # never leak a mutated epoch into other tests (lru_cache outlives
        # monkeypatch teardown)
        monkeypatch.undo()
        store.code_epoch.cache_clear()


def test_code_epoch_executes_no_payload_module():
    """code_epoch LOCATES epoch sources (sys.modules / a PathFinder walk)
    without EXECUTING them: importing grid.trie.build would pull numpy
    (~20ms) into the first store access — a pure import tax on the very
    warm-hit latency the store exists to measure and remove — and even a
    parent package __init__ is off-limits (grid.jsonschema's pulls the
    whole compiler chain into e.g. a grid-source-only serving process).
    Assert NOTHING appears in sys.modules beyond stdlib importlib helpers.
    Fresh process: the pytest process has numpy loaded long before this
    test runs."""
    import subprocess
    import sys

    root = pathlib.Path(store.__file__).resolve().parents[2]
    code = (
        "import sys\n"
        "from grid.serving import artifact_store\n"
        "before = set(sys.modules)\n"
        "artifact_store.code_epoch()\n"
        "new_grid = sorted(m for m in set(sys.modules) - before\n"
        "                  if m == 'grid' or m.startswith('grid.'))\n"
        "assert not new_grid, f'code_epoch executed grid modules: {new_grid}'\n"
        "assert 'numpy' not in sys.modules, 'code_epoch reached numpy'\n"
    )
    env = dict(os.environ, PYTHONPATH=str(root))
    subprocess.run([sys.executable, "-c", code], check=True, env=env,
                   cwd=str(root))


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


# ------------------------------------------------- component namespace (S3)

from grid.errors import GrammarInvalid  # noqa: E402
from grid.grammar.spec import Terminal  # noqa: E402
from grid.lexer import factored  # noqa: E402


@pytest.fixture
def fresh_memo(monkeypatch):
    """Isolate the process-wide component memo: the store consult sits BEHIND
    it, so warm-hit proofs need the memo cold."""
    monkeypatch.setattr(factored, "_COMPONENTS", {})
    return monkeypatch


def _terms(*patterns: str):
    terms = {
        f"T{i}": Terminal(name=f"T{i}", pattern=p, is_literal=False,
                          ignored=False, decl_index=i)
        for i, p in enumerate(patterns)
    }
    return terms, tuple(terms)


# the substring-union shape (BAKEOFF F1) at k=2: breaches small budgets fast
_UNION_PAT = '"([a-e]*ab[a-e]*|[a-e]*cd[a-e]*)"'


def test_component_roundtrip_and_warm_hit(cache, fresh_memo):
    cold = store.load_or_build_component("[a-z]{2,8}", False, 64)
    assert isinstance(cold, factored.TerminalDFA)
    assert [p.parent.name for p in _bins(cache)] == ["component"]
    fresh_memo.setattr(factored, "_build_component", _boom)
    warm = store.load_or_build_component("[a-z]{2,8}", False, 64)
    assert warm == cold  # frozen tuple dataclass: full structural equality


def test_component_breach_marker_skips_eager_attempt(cache, fresh_memo):
    cold = store.load_or_build_component(_UNION_PAT, False, 8)
    assert isinstance(cold, factored.LazyTerminalDFA)
    (path,) = _bins(cache)
    assert store.get("component", path.stem) == store._COMPONENT_BREACH
    # warm: the eager attempt (subset_construct) must never run again
    fresh_memo.setattr(factored, "_build_component", _boom)
    fresh_memo.setattr(factored, "subset_construct", _boom)
    warm = store.load_or_build_component(_UNION_PAT, False, 8)
    assert isinstance(warm, factored.LazyTerminalDFA)
    # value-equal observables: drive both and compare (ids are demand-order
    # on both sides and both are fresh, so numbering agrees too)
    for word in [b'"ab"', b'"cd"', b'"aabcd"', b'"x', b'"e"', b'""']:
        sc, sw = cold, warm
        stc = stw = 0
        for byte in word:
            stc = sc.step(stc, sc.class_of[byte])
            stw = sw.step(stw, sw.class_of[byte])
            assert stc == stw
            if stc == -1:
                break
            assert sc.accepting[stc] == sw.accepting[stw]
            assert sc.co_acc[stc] == sw.co_acc[stw]


def test_component_key_separates_budget_and_kind(cache):
    """Breach is a budget property and literal-vs-regex changes the automaton:
    all three key components must separate entries."""
    k = store.component_key
    assert k("ab", False, 64) != k("ab", True, 64)
    assert k("ab", False, 64) != k("ab", False, 65)
    assert k("ab", False, None) != k("ab", False, 64)
    assert k("ab", False, 64) == k("ab", False, 64)


def test_component_kill_switch(cache, fresh_memo, monkeypatch):
    monkeypatch.setenv("GRID_PERF_STORE_COMPONENTS", "0")
    comp = store.load_or_build_component("[0-9]+", False, 64)
    assert isinstance(comp, factored.TerminalDFA)
    assert _bins(cache) == []  # no component namespace writes


def test_component_bad_regex_raises_before_put_warm_and_cold(cache, fresh_memo):
    for _ in range(2):  # second iteration: store warm for good patterns
        with pytest.raises(GrammarInvalid):
            store.load_or_build_component("a{4,2}", False, 64)
    assert _bins(cache) == []  # failed builds never put


def test_factored_scanner_first_error_ordering_with_partial_warm_store(
        cache, fresh_memo):
    """T0 (good) warms the store; the first GrammarInvalid must still be
    T1's — components build in terminal_order regardless of warmth."""
    terms, order = _terms("[a-z]+", "a{4,2}", "b{9999999}")
    with pytest.raises(GrammarInvalid) as cold:
        factored.build_factored_scanner(terms, order, budget=0, component_budget=64)
    assert len(_bins(cache)) == 1  # T0 stored before T1 raised
    fresh_memo.setattr(factored, "_COMPONENTS", {})
    with pytest.raises(GrammarInvalid) as warm:
        factored.build_factored_scanner(terms, order, budget=0, component_budget=64)
    assert str(warm.value) == str(cold.value)


def test_empty_match_error_parity_with_warm_store(cache, fresh_memo):
    """Empty-matching terminals produce VALID stored components; the
    build-time GrammarInvalid must reproduce identically on a warm store."""
    terms, order = _terms("x", "a*")
    with pytest.raises(GrammarInvalid) as cold:
        factored.build_factored_scanner(terms, order, budget=0, component_budget=64)
    fresh_memo.setattr(factored, "_COMPONENTS", {})
    fresh_memo.setattr(factored, "_build_component", _boom)  # warm-hit proof
    with pytest.raises(GrammarInvalid) as warm:
        factored.build_factored_scanner(terms, order, budget=0, component_budget=64)
    assert str(warm.value) == str(cold.value)


# ------------------------------------------------------ trie namespace (S3)


def test_trie_roundtrip_and_warm_hit(cache, sql_tokenizer, monkeypatch):
    import numpy as np

    from grid.trie import build as tbuild

    cold = store.load_or_build_trie(sql_tokenizer)
    assert [p.parent.name for p in _bins(cache)] == ["trie"]
    # warm: the DFS build must never run (the fingerprint pass still does)
    monkeypatch.setattr(tbuild, "_build_from_entries", _boom)
    warm = store.load_or_build_trie(sql_tokenizer)
    assert np.array_equal(warm.nodes, cold.nodes)
    assert warm.aliases == cold.aliases
    assert warm.n_tokens == cold.n_tokens
    assert warm.tokenizer_fingerprint == cold.tokenizer_fingerprint


def test_trie_kill_switch_and_flag_off(cache, sql_tokenizer, monkeypatch):
    from grid.trie.build import build_trie

    monkeypatch.setenv("GRID_PERF_STORE_TRIE", "0")
    trie = store.load_or_build_trie(sql_tokenizer)
    assert _bins(cache) == []
    import numpy as np

    assert np.array_equal(trie.nodes, build_trie(sql_tokenizer).nodes)


def test_trie_fingerprint_key_separates_tokenizers(cache, sql_tokenizer,
                                                   toy_tokenizer):
    t1 = store.load_or_build_trie(sql_tokenizer)
    t2 = store.load_or_build_trie(toy_tokenizer)
    assert t1.tokenizer_fingerprint != t2.tokenizer_fingerprint
    assert len([p for p in _bins(cache) if p.parent.name == "trie"]) == 2


def test_trie_slicer_variant_key_separation(cache, monkeypatch):
    """S2 interplay: GRID_PERF_SLICER is a BUILD-time input baked into the
    payload (TrieSlices), so the namespace keys per flag state. A slicer-off
    process must never be served a slice-carrying trie (its kill switch
    promises the full walk byte-for-byte), and a slicer-on process must not
    be silently downgraded by an s0 entry; within a variant, hits stay warm."""
    from grid.models.tokenizer_adapter import MockTokenizer
    from grid.trie import build as tbuild

    # mostly JSON-string-safe spellings so the s1 build really carries slices
    tok = MockTokenizer(extra_tokens=(
        "foo", "bar", "da", "ta", " ", "zz", "-1", "12",
        '"', 'a"b', "x\\y", "{", "}", ":", ",",
    ))
    monkeypatch.delenv("GRID_PERF_SLICER", raising=False)
    off = store.load_or_build_trie(tok)
    assert off.slices is None
    monkeypatch.setenv("GRID_PERF_SLICER", "1")
    on = store.load_or_build_trie(tok)  # separate entry, not the s0 hit
    assert on.slices is not None
    assert on.tokenizer_fingerprint == off.tokenizer_fingerprint
    assert len([p for p in _bins(cache) if p.parent.name == "trie"]) == 2
    # both variants warm now: the builder must not run in either flag state
    monkeypatch.setattr(tbuild, "_build_from_entries", _boom)
    assert store.load_or_build_trie(tok).slices is not None
    monkeypatch.delenv("GRID_PERF_SLICER", raising=False)
    assert store.load_or_build_trie(tok).slices is None


def test_trie_warm_mask_parity(cache, sql_source, sql_tokenizer, monkeypatch):
    """Masks are pure functions of (tables, dfa, trie): a store-warm trie must
    yield token-for-token identical instructions to a flag-off build along a
    driven prefix (exercises the kernel walker when grid_core is present)."""
    from grid.generate import build_guide

    store.load_or_build_trie(sql_tokenizer)  # populate the namespace
    warm_guide = build_guide(sql_source, sql_tokenizer)
    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "0")
    off_guide = build_guide(sql_source, sql_tokenizer)

    sw, so = warm_guide.initial_state, off_guide.initial_state
    for _ in range(6):
        iw = warm_guide.get_next_instruction(sw)
        io = off_guide.get_next_instruction(so)
        assert list(iw.tokens) == list(io.tokens)
        if not len(io.tokens):
            break
        t = int(io.tokens[0])
        sw = warm_guide.get_next_state(sw, t)
        so = off_guide.get_next_state(so, t)


def test_scanner_put_skipped_for_lazy_facade(cache, fresh_memo, toy_grammar,
                                             monkeypatch):
    """Over-budget factored products are unpicklable facades: the scanner
    namespace skips them silently (no one-shot degraded warning), while the
    component namespace still persists their per-terminal library."""
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1")
    monkeypatch.setattr(store, "_put_warned", False)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning fails the test
        dfa = store.load_or_build_scanner(toy_grammar)
    assert getattr(dfa, "lazy", False)
    assert store._put_warned is False
    by_ns = {p.parent.name for p in _bins(cache)}
    assert "scanner" not in by_ns
    assert "component" in by_ns
