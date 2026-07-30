"""S2 tokenizer slicer: on/off byte-identity + containment-proof soundness.

The slicer (GRID_PERF_SLICER=1 at build_trie time) may ONLY change walk cost,
never walk output: a slice-carrying trie must produce BYTE-IDENTICAL results
to the plain trie on every configuration — kernel walk()/walk_payload() bytes
(ci i32-le buffer, group tuples, v7 blob — hence identical entry_id), spec
_walk_py results after the consumer's expand+sort, per-step guide masks and
T1 key->entry_id maps, under both GRID_GENN_KEYS regimes. The proof is
all-or-nothing (v1): pass -> rest-trie walk + precomputed ids; fail -> the
full walk byte-for-byte, so a wrong proof is a test failure here, never a
served mask.

Containment refusals proved refused: quote/backslash/control bytes (the class
excludes them — a grammar whose strings CONTINUE past a class byte via DEAD),
maxLength-bounded windows (counting chains hit DEAD or the closure cap),
lexicon-live states (identifier positions), and closure blowups.
"""

import dataclasses
import random

import numpy as np
import pytest

import grid.trie.walk as W
from grid.generate import build_guide
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.guide import COMPLETE
from grid.lalr.compile import compile_tables
from grid.lexer.dfa import build_scanner
from grid.models.tokenizer_adapter import MockTokenizer
from grid.trie.build import JSON_STRING_SAFE, TokenTrie, build_trie

# A JSON-ish grammar whose STRING body class is exactly the slice class
# (grid.jsonschema.compiler.STRING_RX's shape) — the containment proof can
# fire on string-interior states; the object punctuation provides CD-rich
# boundary configurations.
JSON_SOURCE = (
    "%start doc\n"
    "%ignore WS\n"
    "WS: /[ \\t\\n]+/\n"
    'STRING: /"([^"\\\\\\x00-\\x1f]|\\\\(["\\\\\\/bfnrt]'
    "|u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]))*\"/\n"
    "NUM: /-?[0-9]+/\n"
    'doc: "{" pairs "}" | "{" "}"\n'
    'pairs: pair | pairs "," pair\n'
    'pair: STRING ":" value\n'
    "value: STRING | NUM\n"
)

# Mixed safe/unsafe spellings: quotes and backslashes force rest-trie tokens
# whose paths SHARE prefixes with sliced tokens; ':'/','/'{'/'}' spellings
# create CD entries at emission boundaries.
JSON_TOKENS = (
    "foo", "bar", "da", "ta", " ", "  ", "q", "zz", "-1", "12", "3",
    '"', '"da', 'ta"', 'a"b', 'x\\y', '\\n', '":', '",', '" ', '"}',
    '":"', '","', "{", "}", ":", ",", '{"', '"a', 'b":', '1,"', "}\n",
)

# maxLength-analog: the string body is capped at 3 chars — the counting chain
# must refuse the proof (DEAD past the cap on class bytes).
BOUNDED_SOURCE = (
    "%start doc\n"
    "%ignore WS\n"
    "WS: /[ \\t\\n]+/\n"
    'SHORT: /"[^"\\\\\\x00-\\x1f][^"\\\\\\x00-\\x1f]?[^"\\\\\\x00-\\x1f]?"/\n'
    "doc: SHORT | doc SHORT\n"
)


def _parts(source):
    g = spec.load(source)
    tables = compile_tables(RoleProjection.full(g).build())
    dfa = build_scanner(g.terminals, g.terminal_order)
    prio = {tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
            for tid in range(tables.n_terminals)}
    return g, tables, dfa, prio


def _sliced_trie(monkeypatch, tokens=JSON_TOKENS):
    monkeypatch.setenv("GRID_PERF_SLICER", "1")
    trie = build_trie(MockTokenizer(extra_tokens=tokens))
    assert trie.slices is not None
    return trie


def _tid(tables, name):
    return next(t for t in range(tables.n_terminals)
                if tables.terminal_names[t] == name)


# ------------------------------------------------------------ build-time


def test_flag_off_builds_no_slices(monkeypatch):
    monkeypatch.setenv("GRID_PERF_SLICER", "0")
    assert build_trie(MockTokenizer(extra_tokens=JSON_TOKENS)).slices is None
    monkeypatch.delenv("GRID_PERF_SLICER", raising=False)
    assert build_trie(MockTokenizer(extra_tokens=JSON_TOKENS)).slices is None


def test_partition_complete_and_disjoint(monkeypatch):
    trie = _sliced_trie(monkeypatch)
    sl = trie.slices
    tok = MockTokenizer(extra_tokens=JSON_TOKENS)
    special = getattr(tok, "special_token_ids", frozenset())
    all_ids, safe_ids = set(), set()
    for tid in sorted(set(tok.vocabulary.values())):
        if tid in special or not tok.token_bytes(tid):
            continue
        all_ids.add(tid)
        if all(b in JSON_STRING_SAFE for b in tok.token_bytes(tid)):
            safe_ids.add(tid)
    assert set(int(i) for i in sl.ids) == safe_ids  # alias-complete by spelling
    rest_ids = set()
    for w in sl.rest_nodes:
        tid = TokenTrie.unpack(int(w))[1]
        if tid >= 0:
            rest_ids |= set(trie.expand(tid))
    assert safe_ids | rest_ids == all_ids and not (safe_ids & rest_ids)
    # min_ids are the per-spelling smallest aliases, sorted
    assert list(sl.min_ids) == sorted(sl.min_ids)
    assert set(sl.min_ids) <= safe_ids
    expanded = {t for m in sl.min_ids for t in trie.expand(m)}
    assert expanded == safe_ids
    # class round-trips through the bitmap words
    bits = {b for b in range(256) if (sl.class_words[b >> 6] >> (b & 63)) & 1}
    assert bits == set(sl.class_bytes) == set(JSON_STRING_SAFE)


def test_low_coverage_skips_slices(monkeypatch):
    # unsafe spellings swamp the base vocab -> coverage < 50% -> slices
    # skipped at build (the SQL-like-tokenizer disposition, logged)
    monkeypatch.setenv("GRID_PERF_SLICER", "1")
    tok = MockTokenizer(extra_tokens=tuple(f'"{i}"' for i in range(2000)))
    assert build_trie(tok).slices is None


# ------------------------------------------------------------ spec path


def test_spec_walk_identical_after_expand_sort(monkeypatch):
    _g, tables, dfa, prio = _parts(JSON_SOURCE)
    trie_on = _sliced_trie(monkeypatch)
    trie_off = dataclasses.replace(trie_on, slices=None)
    ign = tables.ignored_terminal_ids
    sid = _tid(tables, "STRING")
    engaged = 0
    a_all = frozenset(range(tables.n_terminals - 1)) - ign
    for rem, A in [
        (b'"da', frozenset({sid})),
        (b'"foo bar', frozenset({sid})),
        (b"", a_all),
        (b'"', frozenset({sid})),
        (b"-1", a_all),
        (b" ", a_all),
    ]:
        if dfa.scan_state(rem) < 0:
            continue
        q = dfa.scan_state(rem)
        if W._slice_contained(trie_on.slices, dfa, q, A, ign, None):
            engaged += 1
        r_on = W._walk_py(trie_on, dfa, rem, A, ign, prio)
        r_off = W._walk_py(trie_off, dfa, rem, A, ign, prio)
        exp = lambda r: sorted(t for tid in r.ci_tokens for t in trie_on.expand(tid))  # noqa: E731
        assert exp(r_on) == exp(r_off), rem
        assert r_on.cd_entries == r_off.cd_entries, rem  # order-exact
    assert engaged > 0, "proof never fired (vacuous differential)"


def test_spec_slicing_gated_off_for_lazy_dfa(monkeypatch):
    from grid.lexer.factored import LazyProductDFA, build_factored_scanner

    g, tables, dfa, prio = _parts(JSON_SOURCE)
    lazy = build_factored_scanner(g.terminals, g.terminal_order, budget=0)
    assert isinstance(lazy, LazyProductDFA)
    trie_on = _sliced_trie(monkeypatch)
    trie_off = dataclasses.replace(trie_on, slices=None)
    ign = tables.ignored_terminal_ids
    sid = _tid(tables, "STRING")
    # identical WalkResult (order INCLUDED): the lazy walk must not slice, so
    # it must equal the unsliced eager walk verbatim
    r_lazy = W._walk_py(trie_on, lazy, b'"da', frozenset({sid}), ign, prio)
    r_off = W._walk_py(trie_off, dfa, b'"da', frozenset({sid}), ign, prio)
    assert r_lazy == r_off


# ------------------------------------------------------------ proof refusals


def test_proof_fires_string_interior(monkeypatch):
    _g, tables, dfa, _prio = _parts(JSON_SOURCE)
    trie = _sliced_trie(monkeypatch)
    sid = _tid(tables, "STRING")
    q = dfa.scan_state(b'"da')
    assert W._slice_contained(trie.slices, dfa, q, frozenset({sid}),
                              tables.ignored_terminal_ids, None)


def test_proof_refuses_boundary_state(monkeypatch):
    # start state: class bytes like 0x80 (and most letters when no terminal
    # accepts them as first byte... here 'a' IS a STRING start? no: STRING
    # starts with '"' only) go DEAD -> refuse
    _g, tables, dfa, _prio = _parts(JSON_SOURCE)
    trie = _sliced_trie(monkeypatch)
    a_all = frozenset(range(tables.n_terminals - 1)) - tables.ignored_terminal_ids
    assert not W._slice_contained(trie.slices, dfa, dfa.start, a_all,
                                  tables.ignored_terminal_ids, None)


def test_proof_refuses_bounded_window(monkeypatch):
    # SHORT caps the body at 3 class bytes: the chain dies on the 4th -> the
    # closure sees a DEAD class transition and refuses (fallback engaged)
    _g, tables, dfa, prio = _parts(BOUNDED_SOURCE)
    trie = _sliced_trie(monkeypatch)
    sid = _tid(tables, "SHORT")
    ign = tables.ignored_terminal_ids
    q = dfa.scan_state(b'"a')
    assert q >= 0
    assert not W._slice_contained(trie.slices, dfa, q, frozenset({sid}), ign, None)
    # and the sliced trie still walks byte-for-byte via the fallback
    trie_off = dataclasses.replace(trie, slices=None)
    r_on = W._walk_py(trie, dfa, b'"a', frozenset({sid}), ign, prio)
    r_off = W._walk_py(trie_off, dfa, b'"a', frozenset({sid}), ign, prio)
    assert r_on == r_off  # fallback IS the unsliced walk, order included


def test_proof_refuses_nonviable_live(monkeypatch):
    # A = {NUM} at a string-interior state: live={STRING} disjoint from
    # A|ignored -> refuse (the walk would prune every subtree; slicing must
    # not resurrect them)
    _g, tables, dfa, _prio = _parts(JSON_SOURCE)
    trie = _sliced_trie(monkeypatch)
    nid = _tid(tables, "NUM")
    q = dfa.scan_state(b'"da')
    assert not W._slice_contained(trie.slices, dfa, q, frozenset({nid}),
                                  tables.ignored_terminal_ids, None)


def test_proof_refuses_lexicon_live_states(sql_grammar, sql_tokenizer, monkeypatch):
    # identifier machinery: any reachable state whose live set intersects a
    # lexicon-constrained terminal refuses (prefix_ok would not be vacuous)
    from grid.policy.schema import SchemaSnapshot

    tables = compile_tables(RoleProjection.full(sql_grammar).build(),
                            frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    dfa = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)
    schema = SchemaSnapshot.from_dict({"users": ["id", "name"]})
    lex = schema.lexicons(tables)
    monkeypatch.setenv("GRID_PERF_SLICER", "1")
    trie = build_trie(sql_tokenizer)
    if trie.slices is None:
        pytest.skip("sql mock vocab below slice coverage floor")
    lex_ids = frozenset(lex.allowed)
    q = dfa.scan_state(b"us")
    assert q >= 0 and (dfa.live[q] & lex_ids)
    a_all = frozenset(range(tables.n_terminals - 1)) - tables.ignored_terminal_ids
    assert not W._slice_contained(trie.slices, dfa, q, a_all,
                                  tables.ignored_terminal_ids, lex)


# ------------------------------------------------------------ kernel path

needs_kernel = pytest.mark.skipif(
    W._grid_core is None or not hasattr(W._grid_core, "RustWalker")
    or not hasattr(W._grid_core.RustWalker, "slice_stats"),
    reason="slicer-capable grid_core not installed",
)


@needs_kernel
def test_kernel_walk_and_payload_byte_identical(monkeypatch):
    _g, tables, dfa, prio = _parts(JSON_SOURCE)
    trie_on = _sliced_trie(monkeypatch)
    trie_off = dataclasses.replace(trie_on, slices=None)
    ign = tables.ignored_terminal_ids
    wk_on = W._rust_walker(trie_on, dfa, ign, prio, None)
    wk_off = W._rust_walker(trie_off, dfa, ign, prio, None)
    sid = _tid(tables, "STRING")
    a_all = frozenset(range(tables.n_terminals - 1)) - ign
    checked_groups = 0
    for rem, A in [
        (b'"da', frozenset({sid})),
        (b'"foo bar', frozenset({sid})),
        (b"", a_all),
        (b'"', frozenset({sid})),
        (b'"a', frozenset({sid})),
        (b" ", a_all),
        (b"-1", a_all),
    ]:
        if dfa.scan_state(rem) < 0:
            continue
        aw = W._term_words(A, wk_on.width)
        ci1, g1 = wk_on.walk(bytes(rem), aw)
        ci0, g0 = wk_off.walk(bytes(rem), aw)
        assert ci1 == ci0 and g1 == g0, rem
        assert wk_on.walk_payload(bytes(rem), aw) == \
            wk_off.walk_payload(bytes(rem), aw), rem
        checked_groups += len(g1)
    sliced, fallbacks = wk_on.slice_stats()
    assert sliced > 0, "kernel slicer never engaged (vacuous)"
    assert fallbacks > 0, "no fallback configuration exercised"
    assert wk_off.slice_stats() == (0, 0)
    assert checked_groups > 0, "no CD groups crossed the differential (vacuous)"


@needs_kernel
def test_kernel_matches_spec_when_sliced(monkeypatch):
    """test_rust_parity's contract on a slice-carrying trie: kernel walk vs
    spec _walk_py, both slicing (each proves containment independently)."""
    _g, tables, dfa, prio = _parts(JSON_SOURCE)
    trie = _sliced_trie(monkeypatch)
    ign = tables.ignored_terminal_ids
    wk = W._rust_walker(trie, dfa, ign, prio, None)
    sid = _tid(tables, "STRING")
    for rem, A in [(b'"da', frozenset({sid})), (b'"', frozenset({sid}))]:
        ci_b, _groups = wk.walk(bytes(rem), W._term_words(A, wk.width))
        ci_kernel = np.frombuffer(ci_b, dtype=np.int32).tolist()
        r = W._walk_py(trie, dfa, rem, A, ign, prio)
        ci_spec = sorted(t for tid in r.ci_tokens for t in trie.expand(tid))
        assert ci_kernel == ci_spec, rem


# ------------------------------------------------------- guide differential


def _pair(monkeypatch, source, tokens, genn: str):
    """(slicer-off guide, slicer-on guide) over identical vocab; GENN regime
    pinned; independent producers/kernels/caches like test_v7_differential."""
    monkeypatch.setenv("GRID_GENN_KEYS", genn)
    monkeypatch.setenv("GRID_PERF_SLICER", "0")
    g_off = build_guide(source, MockTokenizer(extra_tokens=tokens))
    monkeypatch.setenv("GRID_PERF_SLICER", "1")
    g_on = build_guide(source, MockTokenizer(extra_tokens=tokens))
    assert g_off.trie.slices is None
    if g_on.trie.slices is None:
        pytest.skip("vocab below slice coverage floor")
    return g_off, g_on


def _drive_pair(g_off, g_on, seed: int, steps: int, ctx: str) -> None:
    rng = random.Random(seed)
    s_off, s_on = g_off.initial_state, g_on.initial_state
    words = (g_off.vocab_size + 31) // 32
    for step in range(steps):
        c = f"{ctx} seed {seed} step {step}"
        ids_off, eid_off = g_off._mask_ids(s_off)
        ids_on, eid_on = g_on._mask_ids(s_on)
        assert eid_on == eid_off, c
        assert list(ids_on) == list(ids_off), c
        row_off = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        row_on = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        g_off.fill_bitmask(s_off, row_off)
        g_on.fill_bitmask(s_on, row_on)
        assert row_on.tolist() == row_off.tolist(), c
        tok = rng.choice(
            sorted(set(int(i) for i in ids_off) - {g_off.eos_token_id})
            or [int(ids_off[0])])
        s_off = g_off.get_next_state(s_off, tok)
        s_on = g_on.get_next_state(s_on, tok)
        if s_off.status == COMPLETE:
            break
    t1_off = {k: e.entry_id for k, e in g_off.producer.cache._t1.items()}
    t1_on = {k: e.entry_id for k, e in g_on.producer.cache._t1.items()}
    assert t1_on == t1_off, f"{ctx}: T1 key/entry-id maps diverge"


@pytest.mark.parametrize("genn", ["1", "0"])
def test_guide_differential_json(monkeypatch, genn):
    g_off, g_on = _pair(monkeypatch, JSON_SOURCE, JSON_TOKENS, genn)
    for seed in (7, 23, 41):
        _drive_pair(g_off, g_on, seed=seed, steps=24, ctx=f"json genn={genn}")


@pytest.mark.parametrize("genn", ["1", "0"])
def test_guide_differential_bounded(monkeypatch, genn):
    # maxLength-analog end to end: fallback path only, still identical
    g_off, g_on = _pair(monkeypatch, BOUNDED_SOURCE, JSON_TOKENS, genn)
    for seed in (5, 19):
        _drive_pair(g_off, g_on, seed=seed, steps=16, ctx=f"bounded genn={genn}")


def test_guide_differential_fuzz(monkeypatch):
    g_off, g_on = _pair(monkeypatch, JSON_SOURCE, JSON_TOKENS, "1")
    for seed in range(30):
        _drive_pair(g_off, g_on, seed=1000 + seed, steps=12, ctx="fuzz")
