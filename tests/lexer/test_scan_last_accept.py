"""W1 ScannerDFA.scan_with_last_accept: (q, l, p) must equal the brute-force
per-prefix re-scan on every input — random byte strings, grammar-derived
fixtures, and the sql_spider terminal corpus — and the '1e'/'1E' pair must
yield IDENTICAL (q, l, p) (documenting why the genN key must carry the
post-accept suffix v separately; see tests/mask/test_genn_keys.py)."""

import pathlib
import random

import pytest

from grid.grammar import spec
from grid.lexer.dfa import DEAD, build_scanner

ROOT = pathlib.Path(__file__).parent.parent.parent

EXPONENT_GRAMMAR = """%start s
%ignore WS
WS: /[ \\t\\n]+/
NUMBER: /[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?/
NAME: /[a-z_][a-z0-9_]*/
s: item | s item
item: NUMBER | NAME | "+"
"""


def _dfa_of(source: str):
    g = spec.load(source)
    return build_scanner(g.terminals, g.terminal_order)


@pytest.fixture(scope="module")
def spider_dfa():
    return _dfa_of((ROOT / "grammars" / "sql_spider.grid").read_text())


@pytest.fixture(scope="module")
def exp_dfa():
    return _dfa_of(EXPONENT_GRAMMAR)


def _brute(dfa, r: bytes) -> tuple[int, int, int]:
    """Reference: re-scan every prefix from scratch."""
    q = dfa.scan_state(r)
    length, p = 0, -1
    for k in range(1, len(r) + 1):
        st = dfa.scan_state(r[:k])
        if st == DEAD:
            break
        if dfa.accept[st] != -1:
            length, p = k, st
    return q, length, p


def _check(dfa, r: bytes) -> None:
    got = dfa.scan_with_last_accept(r)
    want = _brute(dfa, r)
    assert got == want, f"{r!r}: fast {got} != brute {want}"
    q, length, p = got
    # structural invariants of the contract
    assert 0 <= length <= len(r)
    if p == -1:
        assert length == 0
    else:
        assert length >= 1 and dfa.accept[p] != -1
        assert dfa.scan_state(r[:length]) == p


# corpus: sql_spider terminal fixtures (keywords, numbers, strings, idents,
# operators, whitespace) plus boundary-crossing composites; every prefix of
# each fixture is also checked.
SPIDER_FIXTURES = [
    b"select", b"distinct", b"from", b"where", b"group", b"having", b"order",
    b"union", b"intersect", b"except", b"join", b"like", b"between", b"not",
    b"count", b"sum", b"avg", b"min", b"max", b"asc", b"desc", b"limit",
    b"1", b"42", b"-3", b"2000.5", b"3.14", b"42.", b"-0.5", b"1e", b"1E",
    b"'", b"'abc", b"'senior band", b"'abc'", b"'zz qq'",
    b"name", b"z_col_0_0", b"t1", b"t9", b"dept_i", b"salary_band",
    b" ", b"\t\n ", b"=", b"<", b">", b"<=", b"<>", b"*", b"(", b")", b".",
    b"73301 ", b"select *", b"name like", b"1.2.3", b"''", b"'a'b",
]


def test_spider_fixtures_and_prefixes(spider_dfa):
    for r in SPIDER_FIXTURES:
        for k in range(len(r) + 1):
            _check(spider_dfa, r[:k])


def test_sql_subset_fixtures(sql_dfa):
    for r in SPIDER_FIXTURES:
        _check(sql_dfa, r)


def test_random_bytes_differential(spider_dfa, exp_dfa, sql_dfa):
    rng = random.Random(20260709)
    alphabet = b"abcz_019.'e E+-*<>()\t\n\x00\xff"
    for dfa in (spider_dfa, exp_dfa, sql_dfa):
        for _ in range(400):
            n = rng.randrange(0, 12)
            r = bytes(rng.choice(alphabet) for _ in range(n))
            _check(dfa, r)
        for _ in range(100):  # fully random bytes (mostly instant-dead scans)
            r = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 8)))
            _check(dfa, r)


def test_empty_remainder(spider_dfa):
    # the empty prefix never accepts (empty-matching terminals rejected at build)
    assert spider_dfa.scan_with_last_accept(b"") == (spider_dfa.start, 0, -1)


def test_1e_1E_share_qlp(exp_dfa):
    """The refuted v-less key's counterexample seed: b'1e' and b'1E' have
    IDENTICAL (q, l, p) — only the suffix bytes v = r[l:] separate them, which
    is why the genN key must embed v verbatim when p >= 0."""
    s1 = exp_dfa.scan_with_last_accept(b"1e")
    s2 = exp_dfa.scan_with_last_accept(b"1E")
    assert s1 == s2
    q, length, p = s1
    assert length == 1 and p != -1 and q not in (DEAD, p)
    assert exp_dfa.accept[q] == -1  # dangling exponent: not accepting
