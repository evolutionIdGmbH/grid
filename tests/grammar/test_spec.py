import pytest

from grid.errors import EmptyLanguageError, GrammarInvalid, IllegalTransition
from grid.grammar import spec
from grid.grammar.parts import GrammarParts, render_text
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


# -- from_parts (P2 direct emission) ----------------------------------------

# manifest shaped exactly like compiler emission: WS first, named terminals,
# quoted literals in rule bodies (incl. an escaped one), an epsilon alt
PARTS = GrammarParts(
    terminal_defs=(("WS", r"[ \t\n\r]+"), ("K0", '"a"'), ("S0", "[0-9]+")),
    start_target="r0_v",
    rules=(
        ("r0_v", (('"{"', "r1_m", '"}"'),)),
        ("r1_m", (("K0", '":"', "S0"), ('"\\""',), ())),
    ),
)


def test_from_parts_identity_with_text_path():
    """from_parts(parts) == load(render_text(parts)) on the full identity."""
    g_text = spec.load(render_text(PARTS))
    g_obj = spec.DialectGrammar.from_parts(PARTS)
    assert g_obj.state == "FROZEN"
    for attr in ("start", "ignored", "terminals", "productions",
                 "terminal_order", "fingerprint"):
        assert getattr(g_obj, attr) == getattr(g_text, attr), attr


def test_from_parts_decl_and_literal_first_use():
    g = spec.DialectGrammar.from_parts(PARTS)
    # named terminals take manifest order; literals append in first-use
    # order over rules ({ before } before : before "), NOT sorted
    assert [t.name for t in sorted(g.terminals.values(),
                                   key=lambda t: t.decl_index)] == [
        "WS", "K0", "S0", "LIT__7B", "LIT__7D", "LIT__3A", "LIT__22"]
    assert g.terminal_order == (
        "WS", "K0", "S0", "LIT__7B", "LIT__7D", "LIT__3A", "LIT__22")
    # escaped literal unescapes exactly like _parse_rhs
    assert g.terminals["LIT__22"].pattern == '"'
    assert g.terminals["LIT__22"].is_literal
    assert g.ignored == frozenset({"WS"}) and g.terminals["WS"].ignored


def test_from_parts_epsilon_alt():
    g = spec.DialectGrammar.from_parts(PARTS)
    assert spec.Production("r1_m", ()) in g.productions
    # production order: synthetic start rule first, then manifest order
    assert g.productions[0] == spec.Production("start", ("r0_v",))


def test_from_parts_unproductive_invalid_parity():
    """GrammarInvalid outcomes are load-bearing: the object path must reject
    an unproductive manifest with the text path's exact message."""
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"), ("X", "x")),
        start_target="r0",
        rules=(("r0", (("X", "r1"),)), ("r1", (('"["', "r1", '"]"'),))),
    )
    with pytest.raises(GrammarInvalid) as obj_err:
        spec.DialectGrammar.from_parts(parts)
    with pytest.raises(GrammarInvalid) as text_err:
        spec.load(render_text(parts))
    assert str(obj_err.value) == str(text_err.value)
    assert "useless" in str(obj_err.value)


def test_from_parts_unknown_rule_invalid_parity():
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"),),
        start_target="r0",
        rules=(("r0", (("missing_rule",),)),),
    )
    with pytest.raises(GrammarInvalid, match="unknown rule") as obj_err:
        spec.DialectGrammar.from_parts(parts)
    with pytest.raises(GrammarInvalid) as text_err:
        spec.load(render_text(parts))
    assert str(obj_err.value) == str(text_err.value)


def test_from_parts_duplicate_terminal_rejected():
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"), ("X", "x"), ("X", "y")),
        start_target="r0",
        rules=(("r0", (("X",),)),),
    )
    with pytest.raises(GrammarInvalid, match="duplicate terminal"):
        spec.DialectGrammar.from_parts(parts)


def test_from_parts_lrec01_warning_parity(monkeypatch):
    # count parity is only meaningful without the render+reload oracle
    # (check mode validates twice by design; the CI check leg sets it
    # suite-wide)
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "0")
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"), ("X", "x")),
        start_target="r0",
        rules=(("r0", (("X",), ("X", "r0"))),),
    )
    with pytest.warns(UserWarning, match="L-REC01") as obj_w:
        spec.DialectGrammar.from_parts(parts)
    with pytest.warns(UserWarning, match="L-REC01") as text_w:
        spec.load(render_text(parts))
    assert [str(w.message) for w in obj_w] == [str(w.message) for w in text_w]


def test_from_parts_check_mode_passes(monkeypatch):
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "1")
    g = spec.DialectGrammar.from_parts(PARTS)
    assert g.state == "FROZEN"
    assert g.fingerprint == spec.load(render_text(PARTS)).fingerprint


def test_from_parts_check_mode_catches_divergence(monkeypatch):
    """A token with embedded whitespace renders as TWO tokens: the object
    path must not silently diverge from what the text path would load."""
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "1")
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"), ("X", "x"), ("Y", "y")),
        start_target="r0",
        rules=(("r0", (("X Y",),)),),   # object: one bogus symbol "X Y"
    )
    with pytest.raises(AssertionError, match="direct-emit check"):
        spec.DialectGrammar.from_parts(parts)


def test_from_parts_check_mode_invalid_message_parity(monkeypatch):
    """Check mode on an unproductive manifest: both paths raise the same
    GrammarInvalid, and the object-path error propagates."""
    monkeypatch.setenv("GRID_PERF_DIRECT_EMIT_CHECK", "1")
    parts = GrammarParts(
        terminal_defs=(("WS", r"[ \t\n\r]+"), ("X", "x")),
        start_target="r0",
        rules=(("r0", (("X", "r1"),)), ("r1", (("r1", "X"),))),
    )
    with pytest.raises(GrammarInvalid, match="useless"):
        spec.DialectGrammar.from_parts(parts)



# -- full_built (P2 trusted full-projection fast path) -----------------------


def test_full_built_matches_full_build(toy_grammar):
    a = RoleProjection.full(toy_grammar).build()
    b = RoleProjection.full_built(toy_grammar)
    assert b.state == "CACHED"
    assert b.role_shape_hash == a.role_shape_hash
    assert b.productions == a.productions


def test_full_built_requires_frozen():
    g = spec.DialectGrammar(source="%start a\nX: /x/\na: X\n").parse()
    with pytest.raises(GrammarInvalid):
        RoleProjection.full_built(g)


def test_full_built_tables_fingerprint(toy_grammar):
    from grid.lalr.compile import compile_tables

    ta = compile_tables(RoleProjection.full(toy_grammar).build())
    tb = compile_tables(RoleProjection.full_built(toy_grammar))
    assert ta.fingerprint == tb.fingerprint
    assert ta.action == tb.action and ta.goto == tb.goto
