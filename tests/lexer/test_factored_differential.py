"""Factored-scanner differential (0.3.x #4, GRID_PERF_FACTORED_SCANNER).

Two gates, both against build_scanner's eager union subset construction:

1. EXACT equality for the under-budget regime: the bounded materializer must
   reproduce the eager ScannerDFA field-for-field, state NUMBERING included
   (the product/subset bijection over the untrimmed components plus identical
   byte-class partition and discovery order make this an equality, not an
   isomorphism).

2. Per-prefix observable equality for the lazy facade (budget=0): after every
   byte of every probe word, (dead?, priority winner, accepts_all, live) must
   match — state ids are instance-local, the sets are not. Plus
   scan/finalize EmissionEvent-stream equality (numbering-free by
   construction), scan_with_last_accept observables, and shortest_lexemes.

Both gates also run with component_budget=0 (P3, GRID_PERF_COMPONENT_BUDGET):
every component a demand-interned LazyTerminalDFA. Gate 1's form there:
build_factored_scanner SKIPS materialization on breach (returns the facade
even under an unbounded product budget), but a FRESH facade materialized in
full must still reproduce the eager artifact exactly — that equality is what
makes the skip a scheduling decision rather than a semantic one. The
substring-union generator scales the o83132/o5195 family shape down to
k=2..6 keywords, small enough for the eager oracle to terminate.
"""

import random
import threading

import pytest

from grid.errors import GrammarInvalid
from grid.grammar import spec
from grid.grammar.spec import Terminal
from grid.jsonschema import compile_json_schema
from grid.lalr.reserve import shortest_lexemes
from grid.lexer.dfa import DEAD, build_scanner
from grid.lexer.factored import (
    LazyProductDFA,
    LazyTerminalDFA,
    build_factored_scanner,
)
from grid.lexer.run import LexerRun, ScanReject, scan


@pytest.fixture(autouse=True, scope="module")
def _expanded_components():
    """The object under test is the EXPANDED factored path's state-for-state
    equality with the eager union build (numbering included). Counting
    components (GRID_PERF_COUNTING) are a different automaton by design —
    control states + counters, count-blind trans reads assert — and their
    equivalence gate is behavioral (tests/lexer/test_counting_windows.py),
    so pin the flag off here whatever the CI leg exports."""
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_PERF_COUNTING", "0")
    yield
    mp.undo()


# ---------------------------------------------------------------- corpora

WINDOW_PATTERNS = [
    "a{3}", "a{2,4}", "a{2,}", "xa{0,2}", "[0-9]{2,3}x", "(ab){2}",
    "a{1,2}b{1,2}", "[a-z]{1,16}x", '"[a-zA-Z0-9]{0,32}"', "[a-f]{4,64}",
]

# m-1/m/n/n+1 boundary probes: deeper than the BFS word-corpus length cap
WINDOW_PROBES: dict[str, list[bytes]] = {
    "a{3}": [b"aa", b"aaa", b"aaaa"],
    "a{2,4}": [b"a", b"aa", b"aaaa", b"aaaaa"],
    "a{2,}": [b"a", b"aa", b"a" * 40],
    "xa{0,2}": [b"x", b"xa", b"xaa", b"xaaa"],
    "[0-9]{2,3}x": [b"1x", b"12x", b"123x", b"1234x"],
    "(ab){2}": [b"ab", b"abab", b"ababab"],
    "a{1,2}b{1,2}": [b"ab", b"aabb", b"aaabb"],
    "[a-z]{1,16}x": [b"a" * 15 + b"x", b"a" * 16 + b"x", b"a" * 17 + b"x", b"z" * 16],
    '"[a-zA-Z0-9]{0,32}"': [b'""', b'"' + b"Q" * 31 + b'"', b'"' + b"Q" * 32 + b'"',
                            b'"' + b"Q" * 33 + b'"'],
    "[a-f]{4,64}": [b"a" * 3, b"a" * 4, b"f" * 63, b"f" * 64, b"f" * 65],
}

# the untrimmed-component hazard cases: empty byte classes make NFA states
# reachable but co-inaccessible, so the union DFA holds ZOMBIE states
# (non-DEAD, live == {}) that a trimmed product would kill one byte early
ZOMBIE_PATTERNS = [
    ("Z", "a|[^\\x00-\\xff]b"),
    ("Z", "z[^\\x00-\\xff]y"),           # empty language: zombie from byte 1
    ("Z", "x([^\\x00-\\xff]y)?q"),
]

JSON_SCHEMAS = [
    {"type": "object",
     "properties": {"name": {"type": "string", "pattern": "^[a-z]{2,8}$"},
                    "n": {"type": "integer"}},
     "required": ["name"]},
    {"enum": ["red", "green", "blue", "a longer literal", 1, 2.5, True, None]},
    {"type": "string", "format": "date-time"},
    {"type": "object",
     "properties": {"a": {"type": "string", "minLength": 2, "maxLength": 6},
                    "b": {"type": "array", "items": {"type": "number"}}}},
]

JSON_PROBES = [
    b'{"name": "abc", "n": 42}', b'{"name": "a"}', b'{"name": ', b'{"n": -3.5e+7}',
    b'"red"', b'"gree', b'"2025-01-31T23:59:59Z"', b'"2025-13-99"', b'null', b'tru',
    b'{"a": "xy", "b": [1, 2.5]}', b'{"a": "x"}', b'[1,', b'"h\xc3\xa9llo"',
    '"héllo wörld"'.encode(), b'"\xf0\x9f\x8d\x8e"', b'"ab\xc3', b'  {  ', b'0.1e',
]


def _rx_terms(*patterns: str) -> tuple[dict[str, Terminal], tuple[str, ...]]:
    terms = {
        f"T{i}": Terminal(name=f"T{i}", pattern=p, is_literal=False,
                          ignored=False, decl_index=i)
        for i, p in enumerate(patterns)
    }
    return terms, tuple(terms)


def _pair(terminals, order):
    eager = build_scanner(terminals, order, factored=False)
    lazy = build_factored_scanner(terminals, order, budget=0)
    assert isinstance(lazy, LazyProductDFA)
    return eager, lazy


def _words(eager, seed: int, max_len: int = 6, cap: int = 350) -> list[bytes]:
    """Probe corpus: BFS-enumerated live paths of the eager DFA (byte order,
    so window boundaries m-1/m/n/n+1 all appear), plus seeded mutations and
    dead-end probes."""
    words: list[bytes] = []
    frontier: list[tuple[int, bytes]] = [(eager.start, b"")]
    for _ in range(max_len):
        nxt: list[tuple[int, bytes]] = []
        for st, path in frontier:
            row = eager.trans[st]
            taken: set[int] = set()
            for byte in range(256):
                ns = row[byte]
                if ns == DEAD or ns in taken:
                    continue
                taken.add(ns)
                w = path + bytes([byte])
                words.append(w)
                nxt.append((ns, w))
                if len(words) >= cap:
                    break
            if len(words) >= cap:
                break
        frontier = nxt
        if len(words) >= cap:
            break
    rng = random.Random(seed)
    for w in list(words[: cap // 2]):
        if not w:
            continue
        m = bytearray(w)
        m[rng.randrange(len(m))] = rng.randrange(256)
        words.append(bytes(m))
        words.append(w + bytes([rng.randrange(256)]))
    words.extend([b"", b"\x00", b"\xff" * 3, "é9".encode()])
    return words


def compare_prefixes(eager, fact, word: bytes) -> None:
    es, fs = eager.start, fact.start
    for i, b in enumerate(word):
        es = eager.trans[es][b]
        fs = fact.trans[fs][b]
        assert (es == DEAD) == (fs == DEAD), (word, i, es, fs)
        if es == DEAD:
            return
        assert eager.accept[es] == fact.accept[fs], (word, i)
        assert eager.accepts_all[es] == fact.accepts_all[fs], (word, i)
        assert eager.live[es] == fact.live[fs], (word, i)


def compare_swla(eager, fact, word: bytes) -> None:
    eq, el, ep = eager.scan_with_last_accept(word)
    fq, fl, fp = fact.scan_with_last_accept(word)
    assert el == fl, word
    assert (eq == DEAD) == (fq == DEAD), word
    assert (ep == -1) == (fp == -1), word
    if eq != DEAD:
        assert eager.accepts_all[eq] == fact.accepts_all[fq], word
        assert eager.live[eq] == fact.live[fq], word
    if ep != -1:
        assert eager.accepts_all[ep] == fact.accepts_all[fp], word


def compare_streams(eager, fact, buf: bytes) -> None:
    """scan + finalize equality: EmissionEvent = (candidates frozenset, length)
    is numbering-independent; ScanReject offsets match because the state
    graphs are isomorphic."""
    try:
        ev_e, rem_e = scan(eager, buf)
        got_e: tuple = (ev_e, rem_e, LexerRun(remainder=rem_e).finalize(eager))
    except ScanReject as exc:
        got_e = ("reject", str(exc))
    try:
        ev_f, rem_f = scan(fact, buf)
        got_f: tuple = (ev_f, rem_f, LexerRun(remainder=rem_f).finalize(fact))
    except ScanReject as exc:
        got_f = ("reject", str(exc))
    assert got_e == got_f, buf


def _full_differential(terminals, order, seed: int) -> None:
    eager, lazy = _pair(terminals, order)
    assert eager.h_max == lazy.h_max
    assert eager.accepts_all[0] == lazy.accepts_all[0]
    assert eager.live[0] == lazy.live[0]
    words = _words(eager, seed)
    for w in words:
        compare_prefixes(eager, lazy, w)
        compare_swla(eager, lazy, w)
        compare_streams(eager, lazy, w)
    assert shortest_lexemes(eager, len(order)) == shortest_lexemes(lazy, len(order))
    # under-budget materialization: EXACT reproduction of the eager artifact
    # (both budgets pinned: CI legs export GRID_PERF_FACTORED_BUDGET=0 and
    # GRID_PERF_COMPONENT_BUDGET=1, and this assertion is about the
    # under-budget regime in every leg)
    mat = build_factored_scanner(terminals, order, budget=10**9, component_budget=10**9)
    if isinstance(mat, LazyProductDFA):  # pragma: no cover
        pytest.fail("materialization aborted under an unbounded budget")
    assert mat == eager
    assert mat.h_max == eager.h_max  # h_max is compare=False on the dataclass

    # component-budget=0 leg (P3): every component demand-interned; same
    # per-prefix observables, emission streams, swla, and reserve BFS words
    lazy_c = build_factored_scanner(terminals, order, budget=0, component_budget=0)
    assert isinstance(lazy_c, LazyProductDFA)
    assert all(isinstance(c, LazyTerminalDFA) for c in lazy_c.comps)
    assert eager.h_max == lazy_c.h_max
    for w in words:
        compare_prefixes(eager, lazy_c, w)
        compare_swla(eager, lazy_c, w)
        compare_streams(eager, lazy_c, w)
    assert shortest_lexemes(eager, len(order)) == shortest_lexemes(lazy_c, len(order))
    # breached components under an unbounded product budget: the builder
    # SKIPS materialization (facade returned), but a fresh facade
    # materialized in full still equals the eager artifact EXACTLY —
    # numbering included (fresh = no demand walks perturbing discovery order)
    skip = build_factored_scanner(terminals, order, budget=10**9, component_budget=0)
    assert isinstance(skip, LazyProductDFA)
    mat_c = skip.materialize(10**9)
    assert mat_c is not None
    assert mat_c == eager
    assert mat_c.h_max == eager.h_max


# ---------------------------------------------------------------- grammars


def test_toy_grammar(toy_grammar):
    _full_differential(toy_grammar.terminals, toy_grammar.terminal_order, seed=1)


def test_sql_grammar(sql_grammar):
    _full_differential(sql_grammar.terminals, sql_grammar.terminal_order, seed=2)


def test_wide_grammar(wide_source):
    g = spec.load(wide_source)
    _full_differential(g.terminals, g.terminal_order, seed=3)


def test_window_terminals():
    for i, pat in enumerate(WINDOW_PATTERNS):
        terms, order = _rx_terms(pat)
        _full_differential(terms, order, seed=10 + i)
        eager, lazy = _pair(terms, order)
        for w in WINDOW_PROBES[pat]:
            compare_prefixes(eager, lazy, w)
            compare_swla(eager, lazy, w)
            compare_streams(eager, lazy, w)


def test_window_product():
    terms, order = _rx_terms("[a-z]{1,16}x", "[a-y]{2,8}", "z{0,4}q")
    _full_differential(terms, order, seed=29)


def _union_terms(k: int) -> tuple[dict[str, Terminal], tuple[str, ...]]:
    """The substring-union family (BAKEOFF.md F1 / o83132 S2) scaled to k
    keywords: unanchored keyword alternatives inside one quoted terminal,
    each branch with the trailing closure that keeps the per-keyword matched
    bit alive — eager subsets scale with 2^k, so the eager oracle terminates
    for k<=6 while the shape stays the pathological one. Siblings: a
    disjoint terminal and an overlapping quoted window (shared '"' prefix)."""
    kws = ["ab", "cd", "ace", "bd", "abc", "cab"][:k]
    alts = "|".join(f"[a-e]*{kw}[a-e]*" for kw in kws)
    return _rx_terms(f'"({alts})"', "[0-9]+", '"[0-9]{1,4}"')


UNION_PROBES = [
    b'"ab"', b'"cd"', b'"ace"', b'"aabbcd"', b'"abcd"', b'"eeabee"',
    b'"e"', b'"', b'"ab', b'"abx', b'"1234"', b'"12"', b'123', b'"abcab"',
    b'"acecab"', b'"' + b"e" * 30 + b'ab"', b'""',
]


def test_substring_union_family():
    for k in range(2, 7):
        terms, order = _union_terms(k)
        _full_differential(terms, order, seed=90 + k)
        eager, lazy = _pair(terms, order)
        lazy_c = build_factored_scanner(terms, order, budget=0, component_budget=0)
        for w in UNION_PROBES:
            compare_prefixes(eager, lazy, w)
            compare_prefixes(eager, lazy_c, w)
            compare_swla(eager, lazy_c, w)
            compare_streams(eager, lazy_c, w)


def test_component_budget_flag_dispatch(monkeypatch):
    """Env wiring: GRID_PERF_COMPONENT_BUDGET breaches -> the builder skips
    materialization and returns the facade even under a huge product budget;
    "0" is the kill switch restoring eager components (dense result)."""
    terms, order = _union_terms(4)
    eager = build_scanner(terms, order, factored=False)
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "2")
    lazy = build_scanner(terms, order)
    assert isinstance(lazy, LazyProductDFA)
    assert any(isinstance(c, LazyTerminalDFA) for c in lazy.comps)
    assert lazy.materialize(10**9) == eager   # fresh facade: exact numbering
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "0")   # kill switch
    dense = build_scanner(terms, order)
    assert dense == eager


def test_mixed_lazy_and_eager_components():
    """A budget between the union component's size and the small siblings':
    only the union goes lazy; the mixed product still matches the oracle."""
    terms, order = _union_terms(5)
    eager = build_scanner(terms, order, factored=False)
    mixed = build_factored_scanner(terms, order, budget=0, component_budget=24)
    assert isinstance(mixed, LazyProductDFA)
    kinds = {name: type(c).__name__ for name, c in zip(order, mixed.comps, strict=True)}
    assert kinds["T0"] == "LazyTerminalDFA"   # the union breaches 24 states
    assert "TerminalDFA" in kinds.values()    # a sibling stays eager
    for w in _words(eager, seed=88) + UNION_PROBES:
        compare_prefixes(eager, mixed, w)
        compare_swla(eager, mixed, w)
    fresh = build_factored_scanner(terms, order, budget=10**9, component_budget=24)
    assert isinstance(fresh, LazyProductDFA)   # breach skips materialization
    assert fresh.materialize(10**9) == eager


def test_lazy_component_threaded_walks():
    """The component memo is shared across producer prefetch threads: races
    must produce duplicate COMPUTES at worst, never duplicate ids or torn
    annotations (the LazyProductDFA._intern idiom, one level down)."""
    terms, order = _union_terms(5)
    eager = build_scanner(terms, order, factored=False)
    lazy_c = build_factored_scanner(terms, order, budget=0, component_budget=0)
    words = _words(eager, seed=97) + UNION_PROBES
    errs: list[BaseException] = []

    def storm(offset: int) -> None:
        try:
            for w in words[offset::4]:
                compare_swla(eager, lazy_c, w)
        except BaseException as e:  # noqa: BLE001 - surface into the main thread
            errs.append(e)

    threads = [threading.Thread(target=storm, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs
    # post-storm coherence: a full single-threaded sweep over every word
    for w in words:
        compare_prefixes(eager, lazy_c, w)
        compare_streams(eager, lazy_c, w)


def test_zombie_states():
    """Empty-class patterns: the union DFA keeps co-inaccessible subsets as
    live states (non-DEAD, live == {}); the untrimmed product must reproduce
    them byte-for-byte, forced emissions included."""
    for i, (name, pat) in enumerate(ZOMBIE_PATTERNS):
        terms = {
            name: Terminal(name=name, pattern=pat, is_literal=False,
                           ignored=False, decl_index=0),
            "W": Terminal(name="W", pattern="[a-z]", is_literal=False,
                          ignored=False, decl_index=1),
        }
        _full_differential(terms, ("Z", "W"), seed=40 + i)
    # the pure-zombie prefix: eager scan_state(b"z...") is a live-empty state
    terms, order = _rx_terms("z[^\\x00-\\xff]y")
    eager, lazy = _pair(terms, order)
    st = eager.scan_state(b"z")
    assert st != DEAD and eager.live[st] == frozenset()
    fst = lazy.scan_state(b"z")
    assert fst != DEAD and lazy.live[fst] == frozenset()


def test_priority_ties():
    # literal beats named at equal length; declaration order breaks named ties
    terms = {
        "RX": Terminal(name="RX", pattern="a[b]", is_literal=False,
                       ignored=False, decl_index=0),
        "LIT": Terminal(name="LIT", pattern="ab", is_literal=True,
                        ignored=False, decl_index=1),
        "RX2": Terminal(name="RX2", pattern="ab|cd", is_literal=False,
                        ignored=False, decl_index=2),
    }
    order = ("RX", "LIT", "RX2")
    eager, lazy = _pair(terms, order)
    for w in [b"ab", b"cd", b"a", b"abx"]:
        compare_prefixes(eager, lazy, w)
    st = lazy.scan_state(b"ab")
    assert lazy.accepts_all[st] == frozenset({0, 1, 2})
    assert lazy.accept[st] == 1  # the literal wins
    assert build_factored_scanner(terms, order, budget=10**9, component_budget=10**9) == eager


def test_empty_match_rejected_same_message():
    for pats in [("a*",), ("a*", "b?"), ("x", "a*")]:
        terms, order = _rx_terms(*pats)
        with pytest.raises(GrammarInvalid) as e_eager:
            build_scanner(terms, order, factored=False)
        with pytest.raises(GrammarInvalid) as e_fact:
            build_factored_scanner(terms, order)
        assert str(e_eager.value) == str(e_fact.value)
        # lazy components carry the same matches_empty (accept in the start
        # closure), so the budget-0 path raises the identical message too
        with pytest.raises(GrammarInvalid) as e_lazy:
            build_factored_scanner(terms, order, component_budget=0)
        assert str(e_eager.value) == str(e_lazy.value)


def test_bad_regex_same_first_error():
    terms, order = _rx_terms("a{4,2}", "b{9999999}")
    with pytest.raises(GrammarInvalid) as e_eager:
        build_scanner(terms, order, factored=False)
    with pytest.raises(GrammarInvalid) as e_fact:
        build_factored_scanner(terms, order)
    assert str(e_eager.value) == str(e_fact.value)


def test_jsonschema_corpus():
    for i, schema in enumerate(JSON_SCHEMAS):
        src, _rec = compile_json_schema(schema)
        g = spec.load(src)
        eager, lazy = _pair(g.terminals, g.terminal_order)
        for w in _words(eager, seed=60 + i, max_len=5, cap=250) + JSON_PROBES:
            compare_prefixes(eager, lazy, w)
            compare_swla(eager, lazy, w)
            compare_streams(eager, lazy, w)
        assert (shortest_lexemes(eager, len(g.terminal_order))
                == shortest_lexemes(lazy, len(g.terminal_order)))
        mat = build_factored_scanner(g.terminals, g.terminal_order,
                                     budget=10**9, component_budget=10**9)
        assert mat == eager
        assert mat.h_max == eager.h_max


# ---------------------------------------------------------------- regimes


def test_flag_dispatch(monkeypatch, sql_grammar):
    eager = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order, factored=False)
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1000000")
    # pinned so the ambient component-lazy CI leg keeps this a DENSE dispatch
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1000000")
    mat = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)
    assert mat == eager
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "3")
    lazy = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)
    assert isinstance(lazy, LazyProductDFA)
    monkeypatch.setenv("GRID_PERF_FACTORED_SCANNER", "0")
    assert build_scanner(sql_grammar.terminals, sql_grammar.terminal_order) == eager


def test_store_warm_differential(tmp_path, monkeypatch):
    """S3 component namespace: a store-warm factored build (fresh memo,
    builder poisoned — every component unpickled or marker-rebuilt) must
    (a) materialize the BIT-IDENTICAL ScannerDFA in the dense regime and
    (b) match the eager oracle's per-prefix observables in the lazy/breach
    regime, over the substring-union family shape."""
    from grid.lexer import factored as fmod

    monkeypatch.setenv("GRID_PERF_ARTIFACT_STORE", "1")
    monkeypatch.setenv("GRID_CACHE_DIR", str(tmp_path))
    terms, order = _union_terms(4)
    eager = build_scanner(terms, order, factored=False)
    words = _words(eager, seed=123) + UNION_PROBES

    real_build = fmod._build_component

    def _boom(*_a, **_k):
        raise AssertionError("builder called: expected a warm component hit")

    # dense regime: cold populate, then warm rebuild from the store alone
    monkeypatch.setattr(fmod, "_COMPONENTS", {})
    cold = build_factored_scanner(terms, order, budget=10**9, component_budget=10**9)
    assert cold == eager
    monkeypatch.setattr(fmod, "_COMPONENTS", {})
    monkeypatch.setattr(fmod, "_build_component", _boom)
    warm = build_factored_scanner(terms, order, budget=10**9, component_budget=10**9)
    assert warm == eager  # bit-identical arrays, numbering included
    assert warm.h_max == eager.h_max

    # breach regime: cold writes the marker for the union terminal, warm
    # builds its LazyTerminalDFA directly (no eager attempt — builder AND
    # subset_construct poisoned); observables must match the oracle
    monkeypatch.setattr(fmod, "_build_component", real_build)
    monkeypatch.setattr(fmod, "_COMPONENTS", {})
    cold_lazy = build_factored_scanner(terms, order, budget=0, component_budget=24)
    assert isinstance(cold_lazy, LazyProductDFA)
    monkeypatch.setattr(fmod, "_COMPONENTS", {})
    monkeypatch.setattr(fmod, "_build_component", _boom)
    monkeypatch.setattr(fmod, "subset_construct", _boom)
    warm_lazy = build_factored_scanner(terms, order, budget=0, component_budget=24)
    assert isinstance(warm_lazy, LazyProductDFA)
    assert any(isinstance(c, LazyTerminalDFA) for c in warm_lazy.comps)
    assert eager.h_max == warm_lazy.h_max
    for w in words:
        compare_prefixes(eager, warm_lazy, w)
        compare_swla(eager, warm_lazy, w)
        compare_streams(eager, warm_lazy, w)
    assert shortest_lexemes(eager, len(order)) == shortest_lexemes(warm_lazy, len(order))


def test_materialize_after_demand_walks(sql_grammar):
    """Budget breach then late materialization on the SAME instance: demand
    walks reorder state discovery, so numbering may differ from eager — the
    per-prefix observables must not."""
    eager, lazy = _pair(sql_grammar.terminals, sql_grammar.terminal_order)
    for w in [b"select * from users;", b"insert into orders", b"where x = 1"]:
        lazy.scan_with_last_accept(w)
    dense = lazy.materialize(10**9)
    assert dense is not None
    for w in _words(eager, seed=77, max_len=5, cap=200):
        compare_prefixes(eager, dense, w)
        compare_swla(eager, dense, w)
