import pathlib

import pytest

from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import compile_tables
from grid.lexer.dfa import build_scanner
from grid.models.mock import MockModel
from grid.models.tokenizer_adapter import MockTokenizer

ROOT = pathlib.Path(__file__).parent.parent

TOY_TOKENS = (
    "foo", "bar", "baz", "12", "34", "1", "2", "+", "-", "*", "(", ")",
    " ", "  ", " + ", "+ ", " (", ") ", "12*", "(ba", "o+", "3)", "fo", "sel",
)

SQL_TOKENS = (
    "select", "sel", "ect", "insert", "update", "delete", "from", "where", "and", "or",
    "limit", "into", "values", "set", " ", "*", ",", ";", "=", "<", ">", "(", ")",
    "users", "orders", "user_id", "name", "email", "total", "id", "salaries",
    " from ", " where ", "select ", "'x'", "'", "1", "42", "0",
    "s;", "rs;", "us", "ers", " users", ",name", "us_x", "sala", "ries",
)


@pytest.fixture(scope="session")
def toy_source() -> str:
    return (ROOT / "grammars" / "toy_expr.grid").read_text()


@pytest.fixture(scope="session")
def sql_source() -> str:
    return (ROOT / "grammars" / "sql_subset.grid").read_text()


@pytest.fixture(scope="session")
def toy_grammar(toy_source):
    return spec.load(toy_source)


@pytest.fixture(scope="session")
def sql_grammar(sql_source):
    return spec.load(sql_source)


@pytest.fixture(scope="session")
def toy_tables(toy_grammar):
    return compile_tables(RoleProjection.full(toy_grammar).build())


@pytest.fixture(scope="session")
def toy_dfa(toy_grammar):
    return build_scanner(toy_grammar.terminals, toy_grammar.terminal_order)


@pytest.fixture(scope="session")
def sql_tables(sql_grammar):
    return compile_tables(
        RoleProjection.full(sql_grammar).build(),
        frozenset({"TABLE_NAME", "COLUMN_NAME"}),
    )


@pytest.fixture(scope="session")
def sql_dfa(sql_grammar):
    return build_scanner(sql_grammar.terminals, sql_grammar.terminal_order)


@pytest.fixture(scope="session")
def toy_tokenizer():
    return MockTokenizer(extra_tokens=TOY_TOKENS)


@pytest.fixture(scope="session")
def sql_tokenizer():
    return MockTokenizer(extra_tokens=SQL_TOKENS)


@pytest.fixture()
def toy_model(toy_tokenizer):
    return MockModel(toy_tokenizer, seed=7)


@pytest.fixture()
def sql_model(sql_tokenizer):
    return MockModel(sql_tokenizer, seed=3)


@pytest.fixture(scope="session")
def wide_source(sql_source) -> str:
    """A >64-terminal grammar (SQL subset + 60 extra keyword literals reachable
    from the start): exercises the widened [u64; W>1] kernel mask paths."""
    kws = [f'"w{i:02d}"' for i in range(60)]
    return (
        sql_source
        + "wide_stmt: wide_kw | wide_stmt wide_kw\n"
        + "wide_kw: " + " | ".join(kws) + "\n"
    ).replace("stmt: query \";\"", "stmt: query \";\" | wide_stmt \";\"")


@pytest.fixture(scope="session")
def wide_tokenizer():
    return MockTokenizer(extra_tokens=SQL_TOKENS + tuple(
        f"w{i:02d}" for i in range(0, 60, 3)
    ) + ("w0", "w1", " w", "1 w"))
