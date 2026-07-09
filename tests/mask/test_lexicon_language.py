"""The identifier composition rule's precondition: L3 allow-list words must lie
in their terminal's language. A word the DFA cannot scan (or that the terminal
does not accept) would let every prefix pass prefix_ok while no token can ever
complete the lexeme — an empty mask at a viable state ("bug by theorem").
Found by the Spider EX harness: a real schema column named
``Official_ratings_(millions)`` against COLUMN_NAME ``[a-z_][a-z0-9_]*``.
"""

import pytest

from grid.errors import GrammarInvalid
from grid.generate import build_guide
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import compile_tables
from grid.trie.walk import Lexicons


def _lexicons(sql_source, words: set[bytes]):
    grammar = spec.load(sql_source)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    c_id = tables.terminal_names.index("COLUMN_NAME")
    t_id = tables.terminal_names.index("TABLE_NAME")
    return Lexicons({t_id: {b"users"}, c_id: words})


def test_lexicon_word_outside_terminal_language_raises(sql_source, sql_tokenizer):
    lex = _lexicons(sql_source, {b"id", b"official_ratings_(millions)"})
    with pytest.raises(GrammarInvalid, match="official_ratings_"):
        build_guide(sql_source, sql_tokenizer, lexicons=lex, schema_fingerprint="t")


def test_lexicon_words_inside_language_build_fine(sql_source, sql_tokenizer):
    lex = _lexicons(sql_source, {b"id", b"official_ratings_millions"})
    guide = build_guide(sql_source, sql_tokenizer, lexicons=lex, schema_fingerprint="t")
    assert guide.producer.lexicons is lex
