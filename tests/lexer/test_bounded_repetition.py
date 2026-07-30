"""Dialect {m,n} bounded repetition: parse-time expansion, scanner
equivalence with hand-unrolled forms, literal-brace fallback, caps."""

import pytest

from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal
from grid.lexer.dfa import _parse_regex, build_scanner


def _accepts(pattern: str, text: str, counting: bool = False) -> bool:
    term = Terminal(name="T", pattern=pattern, is_literal=False,
                    ignored=False, decl_index=0)
    dfa = build_scanner({"T": term}, ["T"], counting=counting)
    state = dfa.scan_state(text.encode("latin-1"))
    return state >= 0 and dfa.accept[state] >= 0


CASES = [
    ("a{3}", ["aaa"], ["", "a", "aa", "aaaa"]),
    ("a{2,4}", ["aa", "aaa", "aaaa"], ["a", "aaaaa", ""]),
    ("a{2,}", ["aa", "aaa", "a" * 10], ["a", ""]),
    # bare nullable terminals are illegal (scanner-loop guard), so {0,n}
    # windows are exercised with an anchor char, as real string terminals are
    ("xa{0,2}", ["x", "xa", "xaa"], ["xaaa", "a"]),
    ("[0-9]{2,3}x", ["12x", "123x"], ["1x", "1234x", "12"]),
    ("(ab){2}", ["abab"], ["ab", "ababab"]),
    ("a{1,2}b{1,2}", ["ab", "aab", "abb", "aabb"], ["a", "b", "aaab"]),
    # counting-eligible spans (>= _COUNTING_MIN_SPAN, non-final loop)
    ("[0-9]{2,12}x", ["12x", "123456789012x"], ["1x", "1234567890123x", "12"]),
    ("a{8,}b", ["a" * 8 + "b", "a" * 20 + "b"], ["a" * 7 + "b", "a" * 8]),
]


@pytest.mark.parametrize("counting", [False, True])
def test_bounded_repetition_semantics(counting):
    for pat, yes, no in CASES:
        for s in yes:
            assert _accepts(pat, s, counting), (pat, s)
        for s in no:
            assert not _accepts(pat, s, counting), (pat, s)


@pytest.mark.parametrize("counting", [False, True])
def test_equivalent_to_unrolled(counting):
    pairs = [("a{2,4}", "aaa?a?"), ("a{2,}", "aaa*"), ("a{3}", "aaa"),
             ("[xy]{1,3}", "[xy][xy]?[xy]?"),
             ("[xy]{8,10}z", "[xy]{8}[xy]?[xy]?z")]
    probes = ["", "a", "aa", "aaa", "aaaa", "aaaaa", "x", "xy", "xyx", "xyxy",
              "xyxyxyxy" + "z", "xyxyxyxyx" + "z", "xyxyxyxyxyx" + "z"]
    for braced, unrolled in pairs:
        for s in probes:
            assert _accepts(braced, s, counting) == _accepts(unrolled, s), (braced, s)


def test_literal_brace_fallback():
    # '{' not followed by a valid quantifier stays literal
    assert _accepts("a{b", "a{b")
    assert _accepts("a{,2}", "a{,2}")
    assert _accepts("{2}", "{2}") is False or True  # quantifier without atom:
    # '{2}' at pattern start quantifies nothing — parse_atom takes '{' literal,
    # then '2', then '}' — so it matches the literal text
    assert _accepts("{2}", "{2}")


def test_bad_ranges_raise():
    with pytest.raises(GrammarInvalid):
        _parse_regex("a{3,2}")
    with pytest.raises(GrammarInvalid):
        _parse_regex("a{9999999}")


def test_large_window_builds():
    # the coverage epoch relies on multi-thousand windows building sanely
    # (anchored: bare {0,n} would be nullable and correctly rejected)
    term = Terminal(name="T", pattern="x[a-z]{0,2048}", is_literal=False,
                    ignored=False, decl_index=0)
    dfa = build_scanner({"T": term}, ["T"])
    assert dfa is not None
