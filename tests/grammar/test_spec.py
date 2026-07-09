import pytest

from grid.errors import EmptyLanguageError, GrammarInvalid, IllegalTransition
from grid.grammar import spec
from grid.grammar.projection import RoleProjection, _prod_key
from grid.grammar.reduction import reduce_productions, useless_symbols


def test_load_toy(toy_grammar):
    assert toy_grammar.state == "FROZEN"
    assert toy_grammar.start == "expr"
    assert "WS" in toy_grammar.ignored
    assert toy_grammar.fingerprint
    # canonical numbering: named terminals first, then literals, stable
    assert toy_grammar.terminal_order[0] == "WS"
    assert all(n.startswith("LIT_") for n in toy_grammar.terminal_order[3:])


def test_fingerprint_deterministic(toy_source):
    a = spec.load(toy_source)
    b = spec.load(toy_source)
    assert a.fingerprint == b.fingerprint
    c = spec.load(toy_source + "\n# comment only")
    assert c.fingerprint == a.fingerprint  # comments don't change canonical form


def test_unknown_rule_is_invalid():
    src = "%start a\nWS: / /\na: b\n"
    g = spec.DialectGrammar(source=src).parse()
    with pytest.raises(GrammarInvalid):
        g.validate()
    assert g.state == "INVALID"
    with pytest.raises(IllegalTransition):
        g.freeze()  # INVALID is terminal


def test_unreduced_grammar_rejected():
    src = "%start a\nX: /x/\na: X\ndead: X dead\n"
    with pytest.raises(GrammarInvalid, match="useless"):
        spec.load(src)


def test_ignored_terminal_in_rule_rejected():
    src = "%start a\n%ignore WS\nWS: / /\na: WS\n"
    with pytest.raises(GrammarInvalid, match="ignored"):
        spec.load(src)


def test_right_recursion_lint_warns():
    src = "%start a\nX: /x/\na: X | X a\n"
    with pytest.warns(UserWarning, match="L-REC01"):
        spec.load(src)


def test_reduction_removes_unproductive_and_unreachable(toy_grammar):
    prods = list(toy_grammar.productions)
    # keep only 'expr: term' and term/factor productions minus factor alternatives
    kept = [p for p in prods if not (p.lhs == "expr" and len(p.rhs) == 3)]
    reduced = reduce_productions(kept, "expr")
    assert useless_symbols(reduced, "expr") == set()


def test_projection_lifecycle(toy_grammar):
    proj = RoleProjection.full(toy_grammar).build()
    assert proj.state == "CACHED"
    assert proj.role_shape_hash


def test_projection_empty_language(toy_grammar):
    # drop every 'factor' production -> nothing is productive
    keep = frozenset(
        _prod_key(p) for p in toy_grammar.productions if p.lhs != "factor"
    )
    with pytest.raises(EmptyLanguageError):
        RoleProjection(base=toy_grammar, keep=keep).build()


def test_random_projections_reduced_or_rejected(sql_grammar):
    """G1 property: every composed projection is REDUCED+VERIFIED or INVALID."""
    import random

    keys = [_prod_key(p) for p in sql_grammar.productions]
    rng = random.Random(42)
    for _ in range(50):
        keep = frozenset(k for k in keys if rng.random() < 0.8)
        proj = RoleProjection(base=sql_grammar, keep=keep)
        try:
            proj.build()
        except (EmptyLanguageError, GrammarInvalid):
            assert proj.state == "INVALID"
            continue
        assert proj.state == "CACHED"
        assert useless_symbols(proj.productions, sql_grammar.start) == set()
