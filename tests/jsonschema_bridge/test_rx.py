"""jsonschema_rx property tests.

The grid regex dialect emitted by jsonschema_rx is a strict subset of Python
`re` syntax (byte-level classes as \\xHH, unrolled quantifiers, no anchors),
so Python `re.fullmatch` over the latin-1-decoded serialized bytes is a valid
oracle. Every emitted regex is additionally parsed with grid's own regex
parser to pin dialect compatibility.
"""

import json
import pathlib
import random
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

import jsonschema_rx as rx  # noqa: E402
from grid.lexer.dfa import _parse_regex  # noqa: E402


def ser_body(s: str) -> str:
    """Canonical serialized string body, one latin-1 char per byte."""
    return json.dumps(s, ensure_ascii=False)[1:-1].encode("utf-8").decode("latin-1")


def full(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, text) is not None


def grid_parses(pattern: str) -> None:
    _parse_regex(pattern)   # raises GrammarInvalid on dialect violations


# ------------------------------------------------------------- int ranges

def test_int_range_property():
    rng = random.Random(0)
    cases = []
    for _ in range(200):
        lo = rng.choice([None, rng.randint(-10**6, 10**6)])
        hi = rng.choice([None, rng.randint(-10**6, 10**6)])
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        cases.append((lo, hi))
    cases += [(0, 0), (None, 0), (0, None), (-1, 1), (10, 100), (99, 100),
              (-100, -10), (None, -1), (7, 7), (1, 9), (0, 9), (10, 19)]
    for lo, hi in cases:
        pat = rx.int_range_rx(lo, hi)
        grid_parses(pat)
        probes = {0, 1, -1, 7, 9, 10, 11, 99, 100, 101, 999999, -999999}
        for b in (lo, hi):
            if b is not None:
                probes |= {b - 1, b, b + 1}
        for _ in range(30):
            probes.add(random.Random(hash((lo, hi))).randint(-10**7, 10**7))
        for x in probes:
            want = (lo is None or x >= lo) and (hi is None or x <= hi)
            got = full(pat, str(x))
            assert got == want, f"[{lo},{hi}] x={x}: got {got} want {want}\n{pat}"


# ------------------------------------------------------------- numbers

def test_number_range_property():
    rng = random.Random(1)
    cases = [(0, None, False, False), (0, None, True, False),
             (None, 0, False, False), (None, 0, False, True),
             (1, 10, False, False), (1, 10, True, True),
             (-5, 5, False, False), (0, 1, False, False),
             (-1, 0, False, False), (0, 100, False, True),
             (None, None, False, False), (-3, None, False, False)]
    for _ in range(60):
        lo = rng.choice([None, rng.randint(-1000, 1000)])
        hi = rng.choice([None, rng.randint(-1000, 1000)])
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        cases.append((lo, hi, rng.random() < 0.3, rng.random() < 0.3))
    for lo, hi, xl, xh in cases:
        pat = rx.number_range_rx(lo, hi, xl, xh)
        if pat is None:
            continue
        grid_parses(pat)
        probes: list = [0, 1, -1, 0.5, -0.5, 1.5, -1.5, 0.25, 9.75, 100,
                        -0.0, 1e-05, -1e-05, 1e+30, -1e+30, 3.0, -3.0, 0.1]
        for b in (lo, hi):
            if b is not None:
                probes += [b, b - 1, b + 1, b + 0.5, b - 0.5]
        for x in probes:
            lo_ok = lo is None or (x > lo if xl else x >= lo)
            hi_ok = hi is None or (x < hi if xh else x <= hi)
            want = lo_ok and hi_ok
            s = json.dumps(x)
            got = full(pat, s)
            assert got == want, (f"[{lo},{hi}] excl=({xl},{xh}) x={s}: "
                                 f"got {got} want {want}\n{pat}")


# ------------------------------------------------------------- lengths

def test_length_window():
    samples = ["", "a", "ab", "abc", "abcd", 'a"b', "a\\b", "a\nb", "\x00\x01",
               "é", "éé", "日本語", "😀", "😀x", "ab😀d", " ", "\t\t\t"]
    for m, n in [(0, None), (1, None), (2, 4), (0, 0), (3, 3), (0, 2), (2, None)]:
        body = rx.length_body(m, n)
        grid_parses(body)
        for s in samples:
            want = len(s) >= m and (n is None or len(s) <= n)
            got = full(body, ser_body(s))
            assert got == want, f"len window ({m},{n}) s={s!r}: got {got} want {want}"


# ------------------------------------------------------------- patterns

CROSS_PATTERNS = [
    r"^[a-z_]+$", r"^.+$", r"^.*$", r"a", r"^x-", r"^11", r"^...*a$",
    r"^[a-zA-Z0-9._-]+$", r"^[a-zA-Z_][a-zA-Z0-9_]*$", r"^cdeb",
    r"^[0-9a-zA-Z_-]{1,25}$", r"^.{1,2}$", r"^(foo|bar)+$", r"^a{2}b{1,3}$",
    r"^\d{2,4}-x$", r"[^a-z]", r"^[^b]*$", r"foo|bar", r"^a(b|c)?d*$",
    r"^\w+$", r"^\W$", r"^[\d]{3}$", r"^-?\d+(\.\d+)?$", r"x",
]
CROSS_STRINGS = ["", "a", "ab", "abc", "aaab", "x-", "x-1", "11", "112", "cdeb",
                 "cdebX", "foo", "bar", "foobar", "foox", "12-x", "1234-x",
                 "abd", "acdd", "ad", "b", "B", "_", "-", ".", "a.b-c_d",
                 "A9", "999", "12.5", "-3", "hello world", 'q"q', "q\\q",
                 "a\nb", "é", "éa", "日本", "😀", "aa😀", "zzz", "1", "M"]


def test_pattern_cross_python():
    for pat in CROSS_PATTERNS:
        body = rx.pattern_body(pat)
        grid_parses(body)
        oracle = re.compile(pat, re.ASCII)
        for s in CROSS_STRINGS:
            want = oracle.search(s) is not None
            got = full(body, ser_body(s))
            assert got == want, f"pattern {pat!r} s={s!r}: got {got} want {want}"


def test_pattern_unicode_class_negation():
    # negated ASCII class includes all non-ASCII codepoints (UTF-8 multibyte)
    body = rx.pattern_body(r"^[^a-z]+$")
    grid_parses(body)
    for s, want in [("é", True), ("日本", True), ("😀", True), ("AB", True),
                    ("a", False), ("Za", False), ("", False), ('"', True),
                    ("\\", True), ("\n", True)]:
        assert full(body, ser_body(s)) == want, (s, want)


def test_pattern_ecma_s_includes_nbsp():
    body = rx.pattern_body(r"a\sb")
    for s, want in [("a b", True), ("a\tb", True), ("a b", True),
                    ("a b", True), ("axb", False)]:
        assert full(body, ser_body(s)) == want, (s, want)


def test_pattern_unsupported():
    for pat in [r"(?=x)", r"(?!x)", r"(?<=x)y", r"\bword\b", r"(a)\1",
                r"\p{L}", r"a{1000}"]:
        with pytest.raises(rx.RxUnsupported):
            rx.pattern_body(pat)


# ------------------------------------------------------------- not-literals

def test_not_literals():
    cases = [
        (["a"], ["", "a", "ab", "b", "aa", "ba"]),
        (["a", "ab"], ["", "a", "ab", "abc", "b", "aa", "ba", "abab"]),
        (["site", "app"], ["", "site", "app", "sit", "sites", "apps", "ap",
                           "s", "x", "papp", "appsite"]),
        (["日"], ["", "日", "日本", "本", "x"]),
        ([""], ["", "a", "ab"]),
        (["k0", "k1", "key"], ["k0", "k1", "key", "k", "k2", "ke", "keys", "0"]),
    ]
    for lits, probes in cases:
        body = rx.not_literals_body(lits)
        grid_parses(body)
        for p in probes:
            want = p not in lits
            got = full(body, ser_body(p))
            assert got == want, f"not_literals({lits}) probe={p!r}: got {got} want {want}"


# ------------------------------------------------------------- formats

FORMAT_CASES = {
    "date": [("2023-01-15", True), ("2023-12-31", True), ("2023-13-01", False),
             ("2023-00-10", False), ("2023-01-32", False), ("23-01-01", False)],
    "date-time": [("2023-01-15T10:30:00Z", True), ("2023-01-15t10:30:00z", True),
                  ("2023-01-15T10:30:00.123+05:30", True),
                  ("2023-01-15T24:00:00Z", False), ("2023-01-15T10:30:00", False),
                  ("2023-01-15 10:30:60Z", True)],
    "time": [("10:30:00Z", True), ("23:59:60+23:59", True), ("24:00:00Z", False),
             ("10:30Z", False)],
    "uuid": [("123e4567-e89b-12d3-a456-426614174000", True),
             ("123E4567-E89B-12D3-A456-426614174000", True),
             ("123e4567-e89b-12d3-a456-42661417400", False),
             ("123e4567e89b12d3a456426614174000", False)],
    "ipv4": [("0.0.0.0", True), ("255.255.255.255", True), ("192.168.1.1", True),
             ("256.1.1.1", False), ("1.2.3", False), ("01.2.3.4", False)],
    "email": [("a@b.co", True), ("first.last+tag@example.org", True),
              ("no-at-sign", False), ("a@-bad.com", False)],
    "hostname": [("example.com", True), ("a-b.c-d.e", True), ("-bad.com", False),
                 ("ok", True)],
    "uri": [("https://example.com/x?y=1", True), ("mailto:a@b.co", True),
            ("not a uri", False), ("//missing-scheme", False)],
}


def test_formats():
    for name, cases in FORMAT_CASES.items():
        body = rx.format_body(name)
        assert body is not None
        grid_parses(body)
        for s, want in cases:
            got = full(body, ser_body(s))
            assert got == want, f"format {name} s={s!r}: got {got} want {want}"
    assert rx.format_body("ipv6") is None       # declared, not guessed


# ------------------------------------------------ serialized specials

def test_pattern_matches_serialized_escapes():
    # decoded quote/backslash/control chars must match through their escapes;
    # ECMA `.` excludes line terminators, so those go through [\s\S]
    body = rx.pattern_body(r'^.+$')
    for s in ['"', "\\", "\b", "\f", "\t", "\x01", '"x"', "a\\b"]:
        assert full(body, ser_body(s)), s
    assert not full(body, ser_body("\n"))
    body_all = rx.pattern_body(r'^[\s\S]+$')
    for s in ["\n", "\r", "\n\r", "a\nb"]:
        assert full(body_all, ser_body(s)), s
    body2 = rx.pattern_body(r'^["\\]+$')
    for s, want in [('"', True), ("\\", True), ('"\\"', True), ("a", False)]:
        assert full(body2, ser_body(s)) == want, s
