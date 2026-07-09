import pytest

from grid.errors import GrammarInvalid
from grid.grammar import spec
from grid.lexer.dfa import build_scanner
from grid.lexer.run import LexerRun, ScanReject


def _names(grammar, dfa, event):
    return sorted(grammar.terminal_order[c] for c in event.candidates)


def test_maximal_munch_and_remainder(toy_grammar, toy_dfa):
    run = LexerRun()
    run, events = run.advance(toy_dfa, b"foo+12")
    assert [_names(toy_grammar, toy_dfa, e) for e in events] == [["IDENT"], ["LIT__2B"]]
    assert run.remainder == b"12"  # NUMBER could extend: not emitted yet
    run, events = run.advance(toy_dfa, b"3*")
    assert [_names(toy_grammar, toy_dfa, e) for e in events] == [["NUMBER"]]
    assert run.remainder == b"*"  # '*' cannot extend, but nothing follows to force it
    run, events = run.advance(toy_dfa, b"(")
    assert [_names(toy_grammar, toy_dfa, e) for e in events] == [["LIT__2A"]]
    assert run.remainder == b"("


def test_partial_lexeme_not_emitted(toy_dfa):
    run, events = LexerRun().advance(toy_dfa, b"fo")
    assert events == () and run.remainder == b"fo"


def test_illegal_byte_rejects(toy_dfa):
    with pytest.raises(ScanReject):
        LexerRun().advance(toy_dfa, b"@")


def test_finalize_segments_greedily(toy_grammar, toy_dfa):
    run, _ = LexerRun().advance(toy_dfa, b"12")
    events = run.finalize(toy_dfa)
    assert events is not None and len(events) == 1
    assert _names(toy_grammar, toy_dfa, events[0]) == ["NUMBER"]
    assert LexerRun(remainder=b"").finalize(toy_dfa) == ()


def test_keyword_vs_identifier_candidates(sql_grammar, sql_dfa):
    """'select' accepts as both the keyword literal and TABLE/COLUMN_NAME —
    the candidate set carries all three; context picks (E7)."""
    run, events = LexerRun().advance(sql_dfa, b"select ")
    assert len(events) == 1
    names = _names(sql_grammar, sql_dfa, events[0])
    assert "LIT_SELECT" in names and "TABLE_NAME" in names and "COLUMN_NAME" in names


def test_empty_matching_terminal_rejected():
    src = "%start a\nX: /x*/\na: X\n"
    g = spec.load(src)
    with pytest.raises(GrammarInvalid, match="empty string"):
        build_scanner(g.terminals, g.terminal_order)


def test_hypotheses_bounded(sql_dfa):
    run, _ = LexerRun().advance(sql_dfa, b"sel")
    assert 1 <= len(run.hypotheses(sql_dfa)) <= sql_dfa.h_max
