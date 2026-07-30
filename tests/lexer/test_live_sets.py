"""Live-set gate for the NFA terminal-reach computation (dfa._terminal_reach,
the only live-set path since the legacy DFA-graph fixpoint and its env flag
were deleted — sanctioned by the 11.3k zero-divergence verify pass on
v0.3.0rc1, see CHANGELOG). The oracle here is a forward BFS over the
finished DFA graph: independent of BOTH build paths (eager union and
factored product), covering grammar shapes including the empty-byte-class
over-report hazard."""

import pytest
from test_bounded_repetition import CASES

from grid.grammar import spec
from grid.grammar.spec import Terminal
from grid.jsonschema import compile_json_schema
from grid.lexer.dfa import DEAD, build_scanner


@pytest.fixture(autouse=True, scope="module")
def _expanded_scanner():
    """The dense-graph forward BFS oracle below only reads ``trans`` rows,
    which are advisory on a counting DFA's guarded cells (GRID_PERF_COUNTING)
    — the counting live gate is the configuration-space oracle in
    tests/lexer/test_counting_windows.py. Pin the flag off whatever the CI
    leg exports."""
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_PERF_COUNTING", "0")
    yield
    mp.undo()


def _dfa_of(patterns: list[str]):
    terms = {
        f"T{i}": Terminal(name=f"T{i}", pattern=pat, is_literal=False,
                          ignored=False, decl_index=i)
        for i, pat in enumerate(patterns)
    }
    return build_scanner(terms, tuple(terms))


def _bfs_live(dfa) -> list[frozenset[int]]:
    """Oracle, independent of both build paths: forward BFS over the DFA
    graph, unioning accepts_all over everything reachable."""
    out: list[frozenset[int]] = []
    for s0 in range(len(dfa.trans)):
        seen = {s0}
        stack = [s0]
        while stack:
            s = stack.pop()
            for t in dfa.trans[s]:
                if t != DEAD and t not in seen:
                    seen.add(t)
                    stack.append(t)
        out.append(frozenset().union(*(dfa.accepts_all[s] for s in seen)))
    return out


def _dense(dfa):
    """Lazy-regime CI legs (GRID_PERF_FACTORED_BUDGET=0 and/or
    GRID_PERF_COMPONENT_BUDGET=1) make build_scanner return the facade; the
    dense-graph oracles here need the full artifact, and materializing a
    facade reproduces the eager one exactly, so the oracles lose nothing."""
    if getattr(dfa, "lazy", False):
        dfa = dfa.materialize(10**9)
        assert dfa is not None
    return dfa


def _assert_matches_bfs(dfa) -> None:
    dfa = _dense(dfa)
    oracle = _bfs_live(dfa)
    assert list(dfa.live) == oracle
    assert dfa.h_max == max((len(s) for s in oracle), default=0)


SCHEMAS = {
    "pattern_len": {"type": "string", "pattern": "^[a-z]{1,8}[0-9]{2,5}$",
                    "minLength": 3, "maxLength": 40},
    "wide_enum": {"enum": [f"choice_{i:02d}" for i in range(24)] + [1, 2.5, True, None]},
    "required_obj": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "name": {"type": "string"},
                       "email": {"type": "string"}, "active": {"type": "boolean"}},
        "required": ["id", "name"],
        "additionalProperties": False,
    },
    "formats": {"type": "string", "format": "date-time"},
    "format_obj": {
        "type": "object",
        "properties": {"id": {"type": "string", "format": "uuid"},
                       "host": {"type": "string", "format": "hostname"},
                       "when": {"type": "string", "format": "date-time"}},
        "required": ["id"],
    },
    "array_pattern": {"type": "array",
                      "items": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}},
}


def test_differential_grammar_fixtures(toy_grammar, sql_grammar, wide_source):
    for g in (toy_grammar, sql_grammar, spec.load(wide_source)):
        _assert_matches_bfs(build_scanner(g.terminals, g.terminal_order))


def test_differential_bounded_repetition_patterns():
    for pat, _yes, _no in CASES:
        _assert_matches_bfs(_dfa_of([pat]))
    _assert_matches_bfs(_dfa_of(["[a-z]{1,64}x"]))          # deep window chain


def test_differential_empty_class_edges():
    _assert_matches_bfs(_dfa_of(["a|[^\\x00-\\xff]b"]))
    # empty-class edge inside a state where its terminal is otherwise dead:
    # after 'a', T0 can never accept (the empty class blocks it) — reverse
    # reachability must not resurrect it through the untakeable edge.
    dfa = _dfa_of(["a[^\\x00-\\xff]c", "ab"])
    _assert_matches_bfs(dfa)
    assert dfa.live[dfa.scan_state(b"a")] == frozenset({1})


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_differential_jsonschema_grammars(name):
    src, _recorded = compile_json_schema(SCHEMAS[name])
    g = spec.load(src)
    _assert_matches_bfs(build_scanner(g.terminals, g.terminal_order))


def test_ground_truth_bfs(toy_dfa, sql_dfa):
    for dfa in (toy_dfa, sql_dfa):
        dfa = _dense(dfa)
        assert list(dfa.live) == _bfs_live(dfa)


def test_live_monotone_and_start(toy_grammar, sql_grammar):
    for g in (toy_grammar, sql_grammar):
        dfa = _dense(build_scanner(g.terminals, g.terminal_order))
        assert dfa.live[dfa.start] == frozenset(range(len(g.terminal_order)))
        assert dfa.h_max == len(dfa.live[dfa.start])
        for src_state, row in enumerate(dfa.trans):
            for dst in set(row):
                if dst != DEAD:
                    assert dfa.live[dst] <= dfa.live[src_state]


def test_eager_and_factored_paths_equal(monkeypatch, sql_grammar):
    """The two surviving build paths must agree exactly (live/h_max included);
    the factored path's co_acc bits are transitively pinned through this and
    the byte-identical differential in test_factored_differential.py.
    Budgets pinned: this assertion is about the under-budget regime in every
    CI leg (the lazy legs export budget-0/1 ambients)."""
    monkeypatch.setenv("GRID_PERF_FACTORED_BUDGET", "1000000")
    monkeypatch.setenv("GRID_PERF_COMPONENT_BUDGET", "1000000")
    dfa_eager = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order,
                              factored=False)
    dfa_fact = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order,
                             factored=True)
    assert dfa_eager == dfa_fact
    assert dfa_eager.h_max == dfa_fact.h_max   # h_max is compare=False
