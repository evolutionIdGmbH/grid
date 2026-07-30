"""Counting-set scanner components for {m,n} windows (GRID_PERF_COUNTING, P4).

The hard gate is EXACT product equivalence: a synchronized BFS over
(expanded state, (control state, counts)) pairs asserting per-pair equality
of accept, accepts_all, live and per-byte DEAD-ness over the whole reachable
space — total equivalence, not sampling (feasible because the counting
machine's reachable configuration count equals the expanded DFA's state
count). The expanded flag-off build is the oracle. Boundary lexemes,
walk/guide differentials on compiled schemas, cache-key counter separation,
fallback determinism, phase-1 kernel routing and the factored-seam specifics
(lazy counting facade, memo/selection isolation, geps-aware component
co_acc) ride on top.

Ported from the held COUNTING_WINDOWS suite (worktree wf_12480d7a-e5d-1,
59 tests) onto the ScannerComponent seam; deltas: the flag is
GRID_PERF_COUNTING, counting rides the factored path only, and guide builds
toggle the env flag (production reads it through perf_flags at build time).
"""

import os

import numpy as np
import pytest

from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.grammar.spec import Terminal
from grid.jsonschema import compile_json_schema, rx
from grid.lalr.compile import compile_tables
from grid.lexer.dfa import DEAD, build_scanner
from grid.lexer.factored import build_factored_scanner
from grid.models.tokenizer_adapter import MockTokenizer
from grid.trie.build import build_trie
from grid.trie.walk import make_verdict_kernel, walk


@pytest.fixture(autouse=True, scope="module")
def _factored_default_budgets():
    """Counting components exist only on the factored path, and the
    dense-regime oracles here (ScannerDFA field equality, guard_rows
    inspection, genN counter keys) require materialized products — so pin
    the factored scanner ON (the legacy kill-switch CI leg exports
    GRID_PERF_FACTORED_SCANNER=0, under which ``counting=True`` is ignored
    by design) and the default budgets (the CI lazy legs export
    GRID_PERF_FACTORED_BUDGET=0 / GRID_PERF_COMPONENT_BUDGET=1, which would
    hand every build a facade). The lazy x counting regime is covered HERE
    by the explicit budget=0 tests (facade product equivalence,
    dense-vs-lazy scans, genN gating, reserve completion), not by the env;
    the expanded oracle legs (counting=False) materialize to the eager
    artifact exactly, which is the factored path's own CI-pinned
    invariant."""
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_PERF_FACTORED_SCANNER", "1")
    mp.delenv("GRID_PERF_FACTORED_BUDGET", raising=False)
    mp.delenv("GRID_PERF_COMPONENT_BUDGET", raising=False)
    yield
    mp.undo()


def _term(pattern: str, name: str = "T", decl: int = 0, literal: bool = False) -> Terminal:
    return Terminal(name=name, pattern=pattern, is_literal=literal,
                    ignored=False, decl_index=decl)


def _window_terminal(m: int, n: int | None) -> Terminal:
    return _term(rx.string_terminal_rx(rx.length_body(m, n)))


def _both(terms: dict[str, Terminal], order: tuple[str, ...]):
    return (build_scanner(terms, order, counting=False),
            build_scanner(terms, order, counting=True))


def _assert_product_equal(d0, d1) -> None:
    """Synchronized BFS over (expanded state, (control state, counts))."""
    start = (d0.start, d1.start, d1.zero_counts())
    seen = {start}
    work = [start]
    while work:
        s0, s1, c1 = work.pop()
        assert d0.accept[s0] == d1.accept[s1], (s0, s1, c1)
        assert d0.accepts_all[s0] == d1.accepts_all[s1], (s0, s1, c1)
        assert d0.live[s0] == d1.live[s1], (s0, s1, c1, d0.live[s0] ^ d1.live[s1])
        for b in range(256):
            n0 = d0.trans[s0][b]
            n1, nc1 = d1.step(s1, c1, b)
            assert (n0 == DEAD) == (n1 == DEAD), (s0, s1, c1, b, n0, n1)
            if n0 != DEAD:
                k = (n0, n1, nc1)
                if k not in seen:
                    seen.add(k)
                    work.append(k)
    assert d0.h_max == d1.h_max


def _product_check(terms: dict[str, Terminal], order: tuple[str, ...],
                   expect_counting: bool | None = None) -> None:
    d0, d1 = _both(terms, order)
    if expect_counting is not None:
        assert bool(d1.counters) == expect_counting, d1.counters
    if not d1.counters:
        assert d0 == d1 and d0.h_max == d1.h_max
        return
    _assert_product_equal(d0, d1)


WINDOWS = [(0, 8), (3, 8), (8, 8), (0, 64), (17, 64), (64, 64), (8, None), (0, 128)]


@pytest.mark.parametrize("m,n", WINDOWS)
def test_product_equivalence_string_windows(m, n):
    _product_check({"T": _window_terminal(m, n)}, ("T",), expect_counting=True)


def test_product_equivalence_dialect_window():
    _product_check({"T": _term("[xy]{9,17}z")}, ("T",), expect_counting=True)


def test_product_equivalence_union_shared_first_byte():
    # window + literal key + number: the key/enum literals share the '"' first
    # byte with the window terminal, forcing mixed control sets
    terms = {
        "T": _window_terminal(2, 12),
        "K": _term('"name"', "K", 1, literal=True),
        "N": _term(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?", "N", 2),
    }
    _product_check(terms, ("T", "K", "N"), expect_counting=True)


def test_product_equivalence_two_window_union():
    terms = {
        "T": _window_terminal(0, 10),
        "U": _term("[a-f]{8,16}!", "U", 1),
    }
    _product_check(terms, ("T", "U"), expect_counting=True)


# continuation bytes inside the body class: the iterate and exit hypotheses
# co-occupy control states, so counts must keep running through the overlap
@pytest.mark.parametrize("pattern", [
    "[a-z]{8,12}x",       # single-char continuation from the body class
    "[0-9]{8,}5",         # open window: saturating INC under overlap
    "(ab){8,12}ab",       # continuation equal to one full body word
    "x[a-z]{8,64}x",      # entry byte re-enterable as body/continuation
])
def test_product_equivalence_overlapping_continuation(pattern):
    _product_check({"T": _term(pattern)}, ("T",), expect_counting=True)


def test_product_equivalence_corunning_counters():
    # shared first bytes force both loops (and both counters) into one control set
    terms = {
        "A": _term("[a-z]{8,20}x", "A", 0),
        "B": _term("[a-y]{10,15}!", "B", 1),
    }
    _product_check(terms, ("A", "B"), expect_counting=True)


# tiny windows expand (span < _COUNTING_MIN_SPAN); the flag-on DFA must be the
# flag-off object field-for-field
@pytest.mark.parametrize("m,n", [(0, 0), (0, 1), (1, 1), (2, 4)])
def test_tiny_windows_expand(m, n):
    _product_check({"T": _window_terminal(m, n)}, ("T",), expect_counting=False)


@pytest.mark.parametrize("pattern", [
    "(ab){12}",           # loop is terminal-final: exit eps-reaches accept
    "(a{2,3}){2}",        # nested rep + non-prefix-code outer body
    "(a{8,10})*x",        # rep under a quantifier
    "(a?){8,16}x",        # epsilon-capable body
    "a+[ab]{8,10}x",      # ambiguous entry: fresh L while the loop runs
    "a{8,10}b{8,10}c",    # two loops in one terminal (conservative fallback)
])
def test_fallback_determinism(pattern):
    d0, d1 = _both({"T": _term(pattern)}, ("T",))
    assert not d1.counters
    assert d0 == d1 and d0.h_max == d1.h_max


def test_counter_budget_largest_spans_win():
    """More eligible windows than _MAX_COUNTERS: the 8 largest spans keep
    counters, the rest expand — deterministically, and still product-equal."""
    terms = {}
    order = []
    for i in range(10):
        pat = f"{chr(ord('a') + i)}[0-9]{{{8 + i},{12 + i}}}Z"
        name = f"T{i}"
        terms[name] = _term(pat, name, i)
        order.append(name)
    d1 = build_scanner(terms, tuple(order), counting=True)
    assert len(d1.counters) == 8
    assert min(n for _m, n in d1.counters) == 14, "smallest spans must expand"
    _product_check(terms, tuple(order), expect_counting=True)
    d2 = build_scanner(terms, tuple(order), counting=True)
    assert (d1.trans, d1.counters, d1.guard_rows) == (d2.trans, d2.counters, d2.guard_rows)


# ---------------------------------------------------------- boundary lexemes

_CHAR_FORMS = ["a", "\\n", "\\u0041", "é", "世", "\U0001f600"]
# 1 decoded char each: ascii, short escape, \uXXXX, 2/3/4-byte UTF-8


@pytest.mark.parametrize("m,n", [(3, 8), (8, 8), (17, 64), (8, None)])
@pytest.mark.parametrize("form", _CHAR_FORMS)
def test_boundary_lexemes(m, n, form):
    d0, d1 = _both({"T": _window_terminal(m, n)}, ("T",))
    assert d1.counters
    tops = [m - 1, m, m + 1] + ([n, n + 1] if n is not None else [m + 9])
    for k in tops:
        if k < 0:
            continue
        data = ('"' + form * k + '"').encode("utf-8")
        q0, l0, p0 = d0.scan_with_last_accept(data)
        q1, l1, p1 = d1.scan_with_last_accept(data)
        assert (q0 == DEAD) == (q1 == DEAD), (k, data)
        assert l0 == l1, (k, data)
        assert (p0 == -1) == (p1 == -1), (k, data)
        accepted = q0 != DEAD and l0 == len(data)
        in_window = m <= k and (n is None or k <= n)
        assert accepted == in_window, (k, data, accepted)


# ------------------------------------------------- compiled-schema differential

_TOKENS = (
    '"', '"a', 'ab', 'abcd', 'hello', '"hello"', 'x', '\\', '\\n', '\\u0041',
    '{', '}', '[', ']', ',', ':', ' ', '1', '42', 'true', 'null',
    '"k1"', '"k2"', 'k1', '\xc3\xa9', 'a', 'b', '"ab', 'cd"',
)  # '\xc3\xa9' is a 2-byte UTF-8 é through the latin-1 MockTokenizer


def _pipeline(schema: dict, counting: bool):
    src, _rec = compile_json_schema(schema)
    g = spec.load(src)
    proj = RoleProjection.full(g).build()
    tables = compile_tables(proj)
    dfa = build_scanner(g.terminals, g.terminal_order, counting=counting)
    return g, tables, dfa


WINDOW_SCHEMA = {"type": "string", "minLength": 2, "maxLength": 12}
WINDOW_OBJ_SCHEMA = {
    "type": "object",
    "properties": {
        "k1": {"type": "string", "minLength": 3, "maxLength": 20},
        "k2": {"type": "integer"},
    },
    "required": ["k1", "k2"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("schema", [WINDOW_SCHEMA, WINDOW_OBJ_SCHEMA])
def test_walk_differential_on_schema(schema):
    g0, tables0, dfa0 = _pipeline(schema, counting=False)
    g1, tables1, dfa1 = _pipeline(schema, counting=True)
    assert dfa1.counters, "schema window must keep a counter"
    tok = MockTokenizer(extra_tokens=_TOKENS)
    trie = build_trie(tok)
    priority = {tid: (0 if tid in tables0.literal_terminal_ids else 1, tid)
                for tid in range(tables0.n_terminals)}
    ignored = tables0.ignored_terminal_ids
    from grid.lalr.stack import allowed_terminals, root_node
    A = allowed_terminals(tables0, root_node(tables0))
    remainders = [b"", b'"', b'"a', b'"ab', b'"abcde', b'"\xc3\xa9', b'"\\u00']
    for rem in remainders:
        if dfa0.scan_state(rem) == DEAD:
            continue
        r0 = walk(trie, dfa0, rem, A, ignored, priority)
        r1 = walk(trie, dfa1, rem, A, ignored, priority)
        ci0 = sorted(int(t) for t in r0.ci_tokens)
        ci1 = sorted(int(t) for t in r1.ci_tokens)
        # kernel results arrive alias-expanded; normalize the spec path too
        if r0.groups is not None and r1.groups is None:
            ci1 = sorted(int(t) for tid in ci1 for t in trie.expand(tid))
        elif r1.groups is not None and r0.groups is None:
            ci0 = sorted(int(t) for tid in ci0 for t in trie.expand(tid))
        assert ci0 == ci1, rem
        cd0 = sorted(e.token_id for e in r0.cd_entries)
        cd1 = sorted(e.token_id for e in r1.cd_entries)
        if r0.groups is not None:
            cd0 = sorted(int(t) for _rep, tids, _pl in r0.groups for t in tids)
            cd1 = sorted(int(t) for tid in cd1 for t in trie.expand(tid))
        assert cd0 == cd1, rem


def _build_guide_counting(src: str, tok, counting: bool):
    """build_guide with the scanner built on the requested path. Production
    reads GRID_PERF_COUNTING through perf_flags at build_scanner time, so the
    honest toggle is the env flag itself (the artifact store is default-off;
    when enabled, flag-on entries live under a counting-scoped key and
    counting-carrying scanners are never persisted)."""
    from grid.generate import build_guide

    prev = os.environ.get("GRID_PERF_COUNTING")
    os.environ["GRID_PERF_COUNTING"] = "1" if counting else "0"
    try:
        return build_guide(src, tok)
    finally:
        if prev is None:
            del os.environ["GRID_PERF_COUNTING"]
        else:
            os.environ["GRID_PERF_COUNTING"] = prev


def test_guide_step_masks_equal():
    """End-to-end per-step mask equality while driving a canonical instance."""
    src, _rec = compile_json_schema(WINDOW_OBJ_SCHEMA)
    tok = MockTokenizer(extra_tokens=_TOKENS)
    g_off = _build_guide_counting(src, tok, counting=False)
    g_on = _build_guide_counting(src, tok, counting=True)
    assert g_on.dfa.counters and not g_off.dfa.counters
    text = b'{"k1": "hello", "k2": 42}'
    s_off, s_on = g_off.initial_state, g_on.initial_state
    for t in tok.greedy_tokenize(text):
        ids_off, _ = g_off._mask_ids(s_off)
        ids_on, _ = g_on._mask_ids(s_on)
        assert sorted(int(x) for x in np.asarray(ids_off)) == \
            sorted(int(x) for x in np.asarray(ids_on)), (s_off, s_on)
        assert int(t) in {int(x) for x in np.asarray(ids_off)}, t
        s_off = g_off.get_next_state(s_off, int(t))
        s_on = g_on.get_next_state(s_on, int(t))
    assert s_off.status == s_on.status


# ------------------------------------------------------------ cache-key counts

def test_cache_key_separates_counter_values():
    """Two remainders with equal (p, q, v) but different counter values must
    not share a genN key (T1/T2 aliasing across counts is the forbidden
    false-accept/false-reject class)."""
    schema = {"type": "string", "minLength": 4, "maxLength": 10}
    src, _rec = compile_json_schema(schema)
    tok = MockTokenizer(extra_tokens=_TOKENS)
    guide = _build_guide_counting(src, tok, counting=True)
    dfa = guide.dfa
    assert dfa.counters
    prod = guide.producer
    A = prod.allowed(guide.initial_state.stack)
    # 5 and 7 chars: both counts inside [m, n-1], so the CONTROL states alias
    # (same in-window set) while the counter values differ
    r1, r2 = b'"abcde', b'"abcdefg'
    q1, cq1, l1, p1, _cp1 = dfa.scan_full(r1)
    q2, cq2, l2, p2, _cp2 = dfa.scan_full(r2)
    assert q1 == q2 and p1 == p2 == -1 and l1 == l2 == 0, "premise: states alias"
    assert cq1 != cq2
    k1 = prod.cache_key(r1, A)
    k2 = prod.cache_key(r2, A)
    assert k1[0] == k2[0] == "genN"
    assert k1 != k2, "counter values must separate genN keys"


# --------------------------------------------------------- phase-1 kernel gate

def test_counting_dfa_routes_to_python_walk():
    _g, tables, dfa = _pipeline(WINDOW_SCHEMA, counting=True)
    assert dfa.counters
    assert make_verdict_kernel(tables, dfa, None) is None
    tok = MockTokenizer(extra_tokens=_TOKENS)
    trie = build_trie(tok)
    priority = {tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
                for tid in range(tables.n_terminals)}
    from grid.lalr.stack import allowed_terminals, root_node
    A = allowed_terminals(tables, root_node(tables))
    result = walk(trie, dfa, b'"ab', A, tables.ignored_terminal_ids, priority)
    assert result.groups is None, "counting DFA must use the Python spec walk"


def test_next_asserts_on_counting_dfa():
    _g, _tables, dfa = _pipeline(WINDOW_SCHEMA, counting=True)
    with pytest.raises(AssertionError):
        dfa.next(dfa.start, ord('"'))


# ------------------------------------------------------------- budget freeze

def test_budget_predicates_frozen(monkeypatch):
    """The flag changes only the scanner built FROM already-emitted grammar
    source: emitted text and recorded degradation sets stay byte-identical."""
    big = {"type": "string", "minLength": 0, "maxLength": 4096}  # beyond cap: recorded
    for schema in (WINDOW_SCHEMA, WINDOW_OBJ_SCHEMA, big):
        monkeypatch.delenv("GRID_PERF_COUNTING", raising=False)
        off = compile_json_schema(schema)
        monkeypatch.setenv("GRID_PERF_COUNTING", "1")
        on = compile_json_schema(schema)
        assert off[0] == on[0], "emitted grammar text must be byte-identical"
        assert off[1] == on[1], "recorded sets must be byte-identical"


def test_env_flag_dispatch(monkeypatch):
    terms = {"T": _window_terminal(3, 16)}
    monkeypatch.setenv("GRID_PERF_COUNTING", "1")
    d_on = build_scanner(terms, ("T",))
    assert d_on.counters
    monkeypatch.delenv("GRID_PERF_COUNTING")
    d_off = build_scanner(terms, ("T",))
    assert not d_off.counters and not d_off.guard_rows


# ------------------------------------------- factored-seam specifics (P4 new)

def test_product_equivalence_lazy_counting_facade():
    """Over-budget counting products serve the LazyProductDFA facade; its
    step()/annotations must match the expanded oracle configuration-for-
    configuration exactly like the dense counting ScannerDFA (the per-
    (sid, gclass) cache stores variant PLANS, never a count-dependent
    successor — this is the adversarial test for that hazard)."""
    terms = {
        "T": _window_terminal(2, 12),
        "K": _term('"name"', "K", 1, literal=True),
    }
    d0 = build_scanner(terms, ("T", "K"), counting=False)
    d1 = build_factored_scanner(terms, ("T", "K"), budget=0, counting=True)
    assert getattr(d1, "lazy", False) and d1.counters
    _assert_product_equal(d0, d1)


def test_lazy_counting_facade_scans_match_dense():
    terms = {"T": _window_terminal(3, 9), "N": _term("[0-9]+", "N", 1)}
    dense = build_scanner(terms, ("T", "N"), counting=True)
    lazy = build_factored_scanner(terms, ("T", "N"), budget=0, counting=True)
    assert dense.counters and getattr(lazy, "lazy", False)
    probes = [b'"', b'"ab', b'"abc', b'"abcdefghi', b'"abcdefghij', b'"ab"', b"12",
              b'"\\u0041', b'"\\n' + b"a" * 7 + b'"']
    for rem in probes:
        qd, cqd, ld, pd, cpd = dense.scan_full(rem)
        ql, cql, ll, pl, cpl = lazy.scan_full(rem)
        # state ids are instance-local; compare the observable knowledge
        assert (qd == DEAD) == (ql == DEAD), rem
        assert (ld, cqd if qd != DEAD else None, cpd) == (ll, cql if ql != DEAD else None, cpl), rem
        if qd != DEAD:
            assert dense.accepts_all[qd] == lazy.accepts_all[ql], rem
            assert dense.live[qd] == lazy.live[ql], rem


def test_lazy_counting_gated_off_genn_keys():
    """Lazy + counting keeps the lazy raw ("generic", ...) schema-scoped key:
    demand-order state ids must never enter shared T2 keys."""
    from grid.lalr.stack import root_node
    from grid.mask.producer import MaskProducer

    src, _rec = compile_json_schema(WINDOW_OBJ_SCHEMA)
    g = spec.load(src)
    proj = RoleProjection.full(g).build()
    tables = compile_tables(proj)
    dfa = build_factored_scanner(g.terminals, g.terminal_order, budget=0, counting=True)
    assert getattr(dfa, "lazy", False) and dfa.counters
    tok = MockTokenizer(extra_tokens=_TOKENS)
    trie = build_trie(tok)
    prod = MaskProducer(tables=tables, dfa=dfa, trie=trie,
                        vocab_size=max(tok.vocabulary.values()) + 1,
                        schema_fingerprint="fp-test")
    A = prod.allowed(root_node(tables))
    key = prod.cache_key(b'"ab', A)
    assert key[0] == "generic" and key[-1] == "fp-test"


def test_counting_component_coacc_oracle():
    """Component co_acc must be geps-aware: a forward BFS over (state, counts)
    configurations is the oracle. A geps-blind reach would mark loop states
    zombie and fire forced emissions a byte early (outcome-changing)."""
    from grid.lexer.counting import build_counting_component

    for pat in ["[a-z]{8,12}x", '"([^"\\\\]|\\\\.){3,8}"', "[0-9]{8,}5"]:
        comp = build_counting_component(pat, None)
        assert comp is not None, pat
        # reachable configurations
        start = (0, comp.zero_counts())
        seen = {start}
        work = [start]
        edges: dict[tuple, set[tuple]] = {}
        while work:
            st, cts = work.pop()
            for byte in range(256):
                ns, ncts = comp.step_counts(st, cts, comp.class_of[byte])
                if ns == DEAD:
                    continue
                edges.setdefault((st, cts), set()).add((ns, ncts))
                if (ns, ncts) not in seen:
                    seen.add((ns, ncts))
                    work.append((ns, ncts))
        # backward closure from accepting configurations
        acc = {cfg for cfg in seen if comp.accepting[cfg[0]]}
        changed = True
        co = set(acc)
        while changed:
            changed = False
            for src, dsts in edges.items():
                if src not in co and dsts & co:
                    co.add(src)
                    changed = True
        # per control state: co_acc iff SOME configuration is co-accessible;
        # eligibility makes it count-independent, so ALL configurations of a
        # control state must agree with the bit
        by_state: dict[int, set[bool]] = {}
        for cfg in seen:
            by_state.setdefault(cfg[0], set()).add(cfg in co)
        for st, vals in by_state.items():
            assert vals == {comp.co_acc[st]}, (pat, st, vals, comp.co_acc[st])


def test_selection_does_not_poison_memo():
    """A terminal dropped by the _MAX_COUNTERS selection in one schema must
    still count in another build (the memo is keyed by pattern+budget only;
    selection happens at product assembly)."""
    terms = {}
    order = []
    for i in range(10):
        pat = f"{chr(ord('a') + i)}[0-9]{{{8 + i},{12 + i}}}Z"
        name = f"T{i}"
        terms[name] = _term(pat, name, i)
        order.append(name)
    d_big = build_scanner(terms, tuple(order), counting=True)
    assert len(d_big.counters) == 8   # T0/T1 (spans 12, 13) dropped
    # the dropped smallest-span pattern alone still counts
    solo = {"T0": terms["T0"]}
    d_solo = build_scanner(solo, ("T0",), counting=True)
    assert d_solo.counters == ((8, 12),)
    _product_check(solo, ("T0",), expect_counting=True)
    # and rebuilding the big grammar after the solo build is unchanged
    d_big2 = build_scanner(terms, tuple(order), counting=True)
    assert (d_big.trans, d_big.counters, d_big.guard_rows) == \
        (d_big2.trans, d_big2.counters, d_big2.guard_rows)


def test_flag_off_products_share_flag_on_objects():
    """counting=True with no eligible windows returns the exact flag-off
    product (same component objects via the plain memo, so the built
    ScannerDFA is equal field-for-field)."""
    terms = {"K": _term('"name"', "K", 0, literal=True), "N": _term("[0-9]+", "N", 1)}
    d0, d1 = _both(terms, ("K", "N"))
    assert not d1.counters and not d1.guard_rows
    assert d0 == d1


def test_reserve_completion_on_counting_dfa():
    """ReserveTable's shortest_lexemes over a dense counting DFA: identical
    per-terminal shortest lexemes to the expanded oracle (min-window strings
    must reflect m, not the control-state shortcut)."""
    from grid.lalr.reserve import shortest_lexemes

    for schema in (WINDOW_SCHEMA, {"type": "string", "minLength": 9, "maxLength": 40}):
        _g0, t0, d0 = _pipeline(schema, counting=False)
        _g1, _t1, d1 = _pipeline(schema, counting=True)
        assert d1.counters
        lex0 = shortest_lexemes(d0, t0.n_terminals)
        lex1 = shortest_lexemes(d1, t0.n_terminals)
        assert lex0 == lex1


def test_reserve_completion_on_lazy_counting_facade():
    terms = {"T": _window_terminal(4, 11), "N": _term("[0-9]+", "N", 1)}
    from grid.lalr.reserve import shortest_lexemes

    d0 = build_scanner(terms, ("T", "N"), counting=False)
    lazy = build_factored_scanner(terms, ("T", "N"), budget=0, counting=True)
    assert getattr(lazy, "lazy", False) and lazy.counters
    assert shortest_lexemes(d0, 2) == shortest_lexemes(lazy, 2)
