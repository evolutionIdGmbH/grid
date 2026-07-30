"""GRID_PERF_LALR_DP differential: DP (LR(0) + DeRemer-Pennello) vs canonical
LR(1)-merge. LALR(1) is uniquely defined and the DP path reproduces the merged
state numbering, so the gate is plain field-by-field LALRTables equality; on
non-LALR grammars both paths must raise LALRConflictError with the same
normalized conflict set (cur/act pair order inside a legacy conflict tuple is
set-iteration dependent, so pairs compare unordered)."""

import dataclasses

import pytest

from grid.errors import LALRConflictError
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.jsonschema import compile_json_schema
from grid.lalr.compile import compile_tables


def _both(g, identifier_terminals=frozenset()):
    a = compile_tables(RoleProjection.full(g).build(), identifier_terminals, algorithm="lr1_merge")
    b = compile_tables(RoleProjection.full(g).build(), identifier_terminals, algorithm="dp")
    return a, b


def _assert_tables_equal(a, b):
    for f in dataclasses.fields(a):
        assert getattr(a, f.name) == getattr(b, f.name), f"LALRTables.{f.name} differs"


def _norm_conflicts(err: LALRConflictError) -> set:
    return {(st, term, frozenset((c, a))) for (st, term, c, a) in err.report}


# Small grammars aimed at the classic DP implementation bugs: nullable
# handling in reads/includes and non-trivial includes-SCCs.
ADVERSARIAL = {
    # reads chain: goto states with consecutive nullable-nonterminal
    # transitions (terminals distinct — sharing one makes the grammar ambiguous)
    "nullable_reads_chain": (
        "%start s\n"
        "A: /a/\n"
        "B: /b/\n"
        "C1: /x/\n"
        "C2: /y/\n"
        "C3: /z/\n"
        "s: A n1 n2 n3 B\n"
        "n1: C1 |\n"
        "n2: C2 |\n"
        "n3: C3 |\n"
    ),
    # includes through a fully-nullable production tail
    "nullable_includes_tail": (
        "%start s\n"
        "A: /a/\n"
        "B: /b/\n"
        "C1: /x/\n"
        "C2: /y/\n"
        "C3: /z/\n"
        "s: A n1 B | B t\n"
        "t: n1 n2 n3\n"
        "n1: C1 |\n"
        "n2: C2 |\n"
        "n3: C3 |\n"
    ),
    # mutual right recursion: a non-trivial includes-SCC (the y-transition
    # state and the x-transition state after "b" include each other), with an
    # eps production whose nullable-trailing includes edge feeds $end into it
    "includes_cycle": (
        "%start s\n"
        "s: x tail\n"
        'x: "a" y | "c"\n'
        'y: "b" x | "d"\n'
        'tail: "!" |\n'
    ),
    # LALR(1)-but-not-SLR(1): lookaheads must come from context, not FOLLOW
    "lalr_not_slr": (
        "%start s\n"
        "ID: /i/\n"
        's: l "=" r | r\n'
        'l: "*" r | ID\n'
        "r: l\n"
    ),
    "left_right_lists": (
        "%start s\n"
        "A: /a/\n"
        "B: /b/\n"
        "s: left | right\n"
        "left: left A | A\n"
        "right: B right | B\n"
    ),
}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_grammars_equal(name):
    g = spec.load(ADVERSARIAL[name])
    _assert_tables_equal(*_both(g))


def test_toy_equal(toy_grammar):
    _assert_tables_equal(*_both(toy_grammar))


def test_sql_equal(sql_grammar):
    _assert_tables_equal(*_both(sql_grammar, frozenset({"TABLE_NAME", "COLUMN_NAME"})))


def test_wide_equal(wide_source):
    _assert_tables_equal(*_both(spec.load(wide_source)))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_json_schema_member_chain_equal():
    # the 2^R member-chain shape DP targets, in-CI at R=6 (2 optional keys)
    schema = {
        "type": "object",
        "properties": {f"k{i}": {"type": "string"} for i in range(8)},
        "required": [f"k{i}" for i in range(6)],
        "additionalProperties": False,
    }
    src, _recorded = compile_json_schema(schema)
    _assert_tables_equal(*_both(spec.load(src)))


CONFLICTING = {
    # dangling-else-style ambiguity (shift/reduce), from test_tables.py
    "ambiguous_seq": "%start s\nA: /a/\ns: e\ne: A | e e\n",
    # LR(1)-but-not-LALR(1): reduce/reduce conflicts appear only after the
    # core merge — exactly what DP lookahead unions must reproduce
    "lr1_not_lalr": (
        "%start s\n"
        's: "a" x "d" | "b" y "d" | "a" y "e" | "b" x "e"\n'
        'x: "c"\n'
        'y: "c"\n'
    ),
}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", sorted(CONFLICTING))
def test_conflict_parity(name):
    g = spec.load(CONFLICTING[name])
    with pytest.raises(LALRConflictError) as legacy:
        compile_tables(RoleProjection.full(g).build(), algorithm="lr1_merge")
    with pytest.raises(LALRConflictError) as dp:
        compile_tables(RoleProjection.full(g).build(), algorithm="dp")
    assert _norm_conflicts(legacy.value) == _norm_conflicts(dp.value)


def test_env_flag_selects_dp(monkeypatch, toy_grammar):
    legacy = compile_tables(RoleProjection.full(toy_grammar).build(), algorithm="lr1_merge")
    monkeypatch.setenv("GRID_PERF_LALR_DP", "1")
    flagged = compile_tables(RoleProjection.full(toy_grammar).build())
    _assert_tables_equal(legacy, flagged)
    # explicit algorithm overrides the env var
    monkeypatch.setenv("GRID_PERF_LALR_DP", "0")
    _assert_tables_equal(legacy, compile_tables(RoleProjection.full(toy_grammar).build(), algorithm="dp"))


def test_unknown_algorithm_rejected(toy_grammar):
    with pytest.raises(ValueError, match="unknown LALR algorithm"):
        compile_tables(RoleProjection.full(toy_grammar).build(), algorithm="slr")


def test_budget_scoping_of_the_differential(monkeypatch):
    """P5 scoping rule: table equality holds for under-budget grammars; an
    over-budget grammar raises the declared LALRBudgetExceeded under BOTH
    algorithms (fire counts may differ: LR(1) materializes >= LR(0) items,
    so the oracle can fire where dp completes — dp defines shipped
    outcomes). tests/lalr/test_budget.py holds the budget's own gates."""
    from grid.errors import LALRBudgetExceeded

    g = spec.load(ADVERSARIAL["left_right_lists"])
    monkeypatch.setenv("GRID_LALR_BUDGET", "0")
    _assert_tables_equal(*_both(g))
    monkeypatch.setenv("GRID_LALR_BUDGET", "4")
    for algorithm in ("lr1_merge", "dp"):
        with pytest.raises(LALRBudgetExceeded):
            compile_tables(RoleProjection.full(g).build(), algorithm=algorithm)
