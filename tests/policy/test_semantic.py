"""G6(d): SemanticChecker flags what masks cannot enforce (column-table binding)."""

from grid.policy.schema import SchemaSnapshot
from grid.policy.semantic import SemanticChecker


def _checker(sql_tables, sql_dfa):
    schema = SchemaSnapshot.from_dict(
        {"users": ["id", "name", "email"], "orders": ["id", "user_id", "total"]}
    )
    return SemanticChecker(sql_tables, sql_dfa, schema)


def test_valid_statement_passes(sql_tables, sql_dfa):
    assert _checker(sql_tables, sql_dfa).check("select name from users;") == []
    assert _checker(sql_tables, sql_dfa).check("select total from orders where user_id = 1;") == []


def test_cross_table_column_flagged(sql_tables, sql_dfa):
    """'name' is a users column: grammatical on orders (mask uses the union by
    proof), caught here."""
    v = _checker(sql_tables, sql_dfa).check("select name from orders;")
    assert [x.kind for x in v] == ["column_not_in_referenced_tables"]
    assert v[0].lexeme == "name"


def test_unknown_table_flagged(sql_tables, sql_dfa):
    v = _checker(sql_tables, sql_dfa).check("delete from salaries;")
    assert [x.kind for x in v] == ["unknown_table"]


def test_unparseable_flagged(sql_tables, sql_dfa):
    v = _checker(sql_tables, sql_dfa).check("select from where;")
    assert [x.kind for x in v] == ["parse_error"]


def test_fixture_suite_100_percent(sql_tables, sql_dfa):
    """G6(d) pass criterion: checker flags 100% of violation fixtures."""
    fixtures = [
        ("select email from orders;", True),
        ("select user_id from users;", True),
        ("update orders set name = 'x' where id = 1;", True),
        ("select id from users;", False),
        ("select user_id from orders where total > 5;", False),
    ]
    checker = _checker(sql_tables, sql_dfa)
    for sql, should_flag in fixtures:
        v = checker.check(sql)
        assert bool(v) == should_flag, f"{sql!r}: expected flag={should_flag}, got {v}"


def _spider_checker():
    import pathlib

    from grid.grammar import spec
    from grid.grammar.projection import RoleProjection
    from grid.lalr.compile import compile_tables
    from grid.lexer.dfa import build_scanner
    from grid.policy.schema import SchemaSnapshot
    from grid.policy.semantic import SemanticChecker

    src = (pathlib.Path(__file__).parent.parent.parent / "grammars" / "sql_spider.grid").read_text()
    g = spec.load(src)
    tables = compile_tables(RoleProjection.full(g).build(), frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    dfa = build_scanner(g.terminals, g.terminal_order)
    schema = SchemaSnapshot.from_dict({
        "orchestra": ["orchestra_id", "conductor_id", "year_of_founded"],
        "performance": ["performance_id", "orchestra_id", "type"],
    })
    return SemanticChecker(tables, dfa, schema)


def test_alias_bound_to_wrong_table_flagged():
    """The observed Spider failure: year_of_founded belongs to orchestra (t1),
    referenced through t2 (performance) — grammatical, lexicon-valid, unbound."""
    chk = _spider_checker()
    bad = ("select t2.year_of_founded from orchestra as t1 "
           "join performance as t2 on t1.orchestra_id = t2.orchestra_id")
    kinds = [v.kind for v in chk.check(bad)]
    assert "column_not_in_aliased_table" in kinds


def test_alias_bound_correctly_is_clean():
    chk = _spider_checker()
    good = ("select t1.year_of_founded from orchestra as t1 "
            "join performance as t2 on t1.orchestra_id = t2.orchestra_id")
    assert chk.check(good) == []


def test_unknown_alias_flagged():
    chk = _spider_checker()
    kinds = [v.kind for v in chk.check(
        "select t3.type from performance as t2")]
    assert "unknown_alias" in kinds


def test_table_qualified_wrong_column_flagged():
    chk = _spider_checker()
    kinds = [v.kind for v in chk.check(
        "select performance.year_of_founded from performance")]
    assert "column_not_in_table" in kinds


def test_bare_column_union_rule_still_applies():
    chk = _spider_checker()
    kinds = [v.kind for v in chk.check("select year_of_founded from performance")]
    assert "column_not_in_referenced_tables" in kinds
