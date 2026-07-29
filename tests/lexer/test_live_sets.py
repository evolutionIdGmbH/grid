"""Differential gate for GRID_PERF_NFA_LIVE (NFA-derived live sets): the new
path must equal the legacy DFA-graph fixpoint AND a third, independent
forward-BFS oracle over the finished DFA, across grammar shapes including the
empty-byte-class over-report hazard."""

import pytest
from test_bounded_repetition import CASES

from grid.grammar import spec
from grid.grammar.spec import Terminal
from grid.jsonschema import compile_json_schema
from grid.lexer.dfa import DEAD, _live_fixpoint, build_scanner


def _dfa_of(patterns: list[str]):
    terms = {
        f"T{i}": Terminal(name=f"T{i}", pattern=pat, is_literal=False,
                          ignored=False, decl_index=i)
        for i, pat in enumerate(patterns)
    }
    return build_scanner(terms, tuple(terms))


def _assert_matches_fixpoint(dfa) -> None:
    legacy = _live_fixpoint([list(r) for r in dfa.trans], list(dfa.accepts_all))
    assert list(dfa.live) == legacy
    assert dfa.h_max == max((len(s) for s in legacy), default=0)


def _bfs_live(dfa) -> list[frozenset[int]]:
    """Third oracle, independent of both build paths: forward BFS over the DFA
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


def test_differential_grammar_fixtures(toy_grammar, sql_grammar, wide_source, monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    for g in (toy_grammar, sql_grammar, spec.load(wide_source)):
        _assert_matches_fixpoint(build_scanner(g.terminals, g.terminal_order))


def test_differential_bounded_repetition_patterns(monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    for pat, _yes, _no in CASES:
        _assert_matches_fixpoint(_dfa_of([pat]))
    _assert_matches_fixpoint(_dfa_of(["[a-z]{1,64}x"]))     # deep window chain


def test_differential_empty_class_edges(monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    _assert_matches_fixpoint(_dfa_of(["a|[^\\x00-\\xff]b"]))
    # empty-class edge inside a state where its terminal is otherwise dead:
    # after 'a', T0 can never accept (the empty class blocks it) — reverse
    # reachability must not resurrect it through the untakeable edge.
    dfa = _dfa_of(["a[^\\x00-\\xff]c", "ab"])
    _assert_matches_fixpoint(dfa)
    assert dfa.live[dfa.scan_state(b"a")] == frozenset({1})


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_differential_jsonschema_grammars(name, monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    src, _recorded = compile_json_schema(SCHEMAS[name])
    g = spec.load(src)
    _assert_matches_fixpoint(build_scanner(g.terminals, g.terminal_order))


def test_ground_truth_bfs(toy_dfa, sql_dfa):
    for dfa in (toy_dfa, sql_dfa):
        assert list(dfa.live) == _bfs_live(dfa)


def test_live_monotone_and_start(toy_grammar, sql_grammar, monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    for g in (toy_grammar, sql_grammar):
        dfa = build_scanner(g.terminals, g.terminal_order)
        assert dfa.live[dfa.start] == frozenset(range(len(g.terminal_order)))
        assert dfa.h_max == len(dfa.live[dfa.start])
        for src_state, row in enumerate(dfa.trans):
            for dst in set(row):
                if dst != DEAD:
                    assert dfa.live[dst] <= dfa.live[src_state]


def test_flag_paths_equal(sql_grammar, monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "0")
    dfa0 = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "1")
    dfa1 = build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)
    assert dfa0 == dfa1
    assert dfa0.h_max == dfa1.h_max     # h_max is compare=False: check it separately


def test_verify_mode(sql_grammar, wide_source, monkeypatch):
    monkeypatch.setenv("GRID_PERF_NFA_LIVE", "verify")
    wide = spec.load(wide_source)
    for g in (sql_grammar, wide):
        dfa = build_scanner(g.terminals, g.terminal_order)
        assert dfa.h_max >= 1
