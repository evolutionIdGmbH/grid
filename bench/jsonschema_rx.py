"""JSON-Schema string/number constraints -> grid-dialect terminal regexes.

Value level vs serialized level: JSON Schema constraints (pattern, length,
format, bounds) speak about the DECODED string value; grid terminals match the
SERIALIZED bytes the MaskBench runner feeds the engine. The runner serializes
every instance with ``json.dumps(x, indent=None, ensure_ascii=False)``, so only
canonical serializations ever reach the mask:

- ``"`` -> ``\\"``, ``\\`` -> ``\\\\``,
- \\b \\f \\n \\r \\t short escapes; other C0 controls -> ``\\u00xx`` (lowercase),
- everything else (incl. DEL and all non-ASCII) -> raw UTF-8 bytes,
- numbers via int/float repr (plain ints, ``D.D`` floats with no trailing
  zeros, exponent floats only for |x| >= 1e16 or 0 < |x| < 1e-4).

This module therefore compiles value-level constraints into byte-level grid
regexes that are exact over canonical serializations (non-canonical forms never
reach the mask, in either the valid or the invalid direction).

Grid regex dialect (grid/lexer/dfa.py): byte-level; ``* + ?`` (no ``{m,n}`` —
we unroll), ``|``, ``( )``, classes ``[..]`` with ranges, ``\\xHH``; no
anchors (terminals are whole-match). We never emit ``.``, negated classes, or
raw ``/ ] ^ -`` inside classes.

ECMA-262 pattern subset: literals, escapes (\\d \\D \\w \\W \\s \\S \\xHH
\\uHHHH \\n \\t \\r \\f \\v \\0, identity escapes), classes, ``.``, groups
``()``/``(?:)``, quantifiers ``* + ? {m} {m,} {m,n}`` (lazy suffix stripped),
anchors ``^``/``$`` at branch edges. Unsupported (raises RxUnsupported):
lookarounds, backrefs, \\b/\\B, \\p{..}, \\u{..}, \\c, named groups.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_UNROLL = 512            # per-quantifier repetition cap
MAX_EMIT = 60_000           # emitted regex source cap (chars)

_MAX_CP = 0x10FFFF
_SURR = (0xD800, 0xDFFF)


class RxUnsupported(Exception):
    """Constraint outside the supported subset."""


# ---------------------------------------------------------------- value AST
#
# ranges: tuple of (lo, hi) codepoint pairs, sorted, disjoint, non-adjacent.

def _norm(ranges) -> tuple:
    rs = sorted((lo, hi) for lo, hi in ranges if lo <= hi)
    out: list[list[int]] = []
    for lo, hi in rs:
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return tuple((lo, hi) for lo, hi in out)


def _complement(ranges) -> tuple:
    """Complement within [0, 0x10FFFF] minus surrogates."""
    out = []
    prev = 0
    for lo, hi in _norm(ranges):
        if prev < lo:
            out.append((prev, lo - 1))
        prev = hi + 1
    if prev <= _MAX_CP:
        out.append((prev, _MAX_CP))
    final = []
    for lo, hi in out:
        if hi < _SURR[0] or lo > _SURR[1]:
            final.append((lo, hi))
        else:
            if lo < _SURR[0]:
                final.append((lo, _SURR[0] - 1))
            if hi > _SURR[1]:
                final.append((_SURR[1] + 1, hi))
    return _norm(final)


def _subtract(a: tuple, b: tuple) -> tuple:
    if not b:
        return _norm(a)
    out = []
    for lo, hi in a:
        segs = [(lo, hi)]
        for blo, bhi in b:
            nxt = []
            for slo, shi in segs:
                if bhi < slo or blo > shi:
                    nxt.append((slo, shi))
                    continue
                if slo < blo:
                    nxt.append((slo, blo - 1))
                if shi > bhi:
                    nxt.append((bhi + 1, shi))
            segs = nxt
        out.extend(segs)
    return _norm(out)


ANY_CHAR = _complement(())          # every scalar codepoint
_D = _norm([(0x30, 0x39)])
_W = _norm([(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)])
_S = _norm([(0x09, 0x0D), (0x20, 0x20), (0xA0, 0xA0), (0x1680, 0x1680),
            (0x2000, 0x200A), (0x2028, 0x2029), (0x202F, 0x202F),
            (0x205F, 0x205F), (0x3000, 0x3000), (0xFEFF, 0xFEFF)])
_DOT = _complement([(0x0A, 0x0A), (0x0D, 0x0D), (0x2028, 0x2029)])


@dataclass(frozen=True)
class N:
    kind: str                # ch|cat|alt|star|plus|opt|eps|caret|dollar
    ranges: tuple = ()
    kids: tuple = ()


EPS = N("eps")
ANY = N("ch", ranges=ANY_CHAR)
ANYSTAR = N("star", kids=(ANY,))


def lit(s: str) -> N:
    return N("cat", kids=tuple(N("ch", ranges=((ord(c), ord(c)),)) for c in s))


# ------------------------------------------------------------ ECMA parser

_CLASS_ESC = {"d": _D, "w": _W, "s": _S,
              "D": _complement(_D), "W": _complement(_W), "S": _complement(_S)}
_CTRL_ESC = {"n": 0x0A, "t": 0x09, "r": 0x0D, "f": 0x0C, "v": 0x0B, "0": 0x00}


def parse_ecma(pattern: str) -> N:
    pos = 0

    def peek(k: int = 0):
        return pattern[pos + k] if pos + k < len(pattern) else None

    def take() -> str:
        nonlocal pos
        ch = pattern[pos]
        pos += 1
        return ch

    def restore(p: int) -> None:
        nonlocal pos
        pos = p

    def esc_char(in_class: bool):
        """-> codepoint (int) or ranges (tuple) for a class-set escape."""
        if peek() is None:
            raise RxUnsupported("dangling backslash")
        e = take()
        if e in _CLASS_ESC:
            return _CLASS_ESC[e]
        if e in _CTRL_ESC:
            return _CTRL_ESC[e]
        if e == "b":
            if in_class:
                return 0x08
            raise RxUnsupported("\\b word boundary")
        if e == "B":
            raise RxUnsupported("\\B")
        if e == "x":
            h = take() + take()
            return int(h, 16)
        if e == "u":
            if peek() == "{":
                raise RxUnsupported("\\u{...}")
            h = take() + take() + take() + take()
            cp = int(h, 16)
            if _SURR[0] <= cp <= _SURR[1]:
                raise RxUnsupported("surrogate escape")
            return cp
        if e in ("p", "P"):
            raise RxUnsupported("\\p{...}")
        if e in "123456789":
            raise RxUnsupported("backreference")
        if e == "c":
            raise RxUnsupported("\\c control escape")
        return ord(e)          # identity escape

    def parse_class() -> N:
        negate = False
        if peek() == "^":
            take()
            negate = True
        items: list[tuple] = []
        first = True
        while True:
            c = peek()
            if c is None:
                raise RxUnsupported("unclosed class")
            if c == "]" and not first:
                take()
                break
            first = False
            if c == "\\":
                take()
                got = esc_char(in_class=True)
                if isinstance(got, tuple):
                    items.append(got)
                    continue
                lo = got
            else:
                lo = ord(take())
            if peek() == "-" and peek(1) not in (None, "]"):
                take()
                if peek() == "\\":
                    take()
                    hi = esc_char(in_class=True)
                    if isinstance(hi, tuple):
                        raise RxUnsupported("class-set escape in range")
                else:
                    hi = ord(take())
                if hi < lo:
                    raise RxUnsupported("reversed class range")
                items.append(((lo, hi),))
            else:
                items.append(((lo, lo),))
        ranges = _norm([r for rs in items for r in rs])
        if negate:
            ranges = _complement(ranges)
        if not ranges:
            raise RxUnsupported("empty class")
        return N("ch", ranges=ranges)

    def parse_int() -> int | None:
        s = ""
        while peek() is not None and peek().isdigit():
            s += take()
        return int(s) if s else None

    def apply_quant(node: N) -> N:
        while True:
            c = peek()
            if c in ("*", "+", "?"):
                take()
                node = N({"*": "star", "+": "plus", "?": "opt"}[c], kids=(node,))
            elif c == "{":
                save = pos
                take()
                m = parse_int()
                if m is None:
                    restore(save)   # ECMA: literal '{'
                    break
                n: int | None = m
                if peek() == ",":
                    take()
                    n = parse_int()     # None -> open
                if peek() != "}":
                    restore(save)       # literal '{...' without close
                    break
                take()
                if m > MAX_UNROLL or (n is not None and n > MAX_UNROLL):
                    raise RxUnsupported(f"quantifier too large ({m},{n})")
                if n is not None and n < m:
                    raise RxUnsupported("bad {m,n} order")
                node = unroll(node, m, n)
            else:
                break
            if peek() == "?":     # lazy — same language
                take()
        return node

    def parse_atom() -> N:
        c = take()
        if c == "(":
            if peek() == "?":
                take()
                k = peek()
                if k == ":":
                    take()
                else:
                    raise RxUnsupported(f"(?{k}...) group")
            node = parse_alt()
            if peek() != ")":
                raise RxUnsupported("unclosed group")
            take()
            return node
        if c == "[":
            return parse_class()
        if c == ".":
            return N("ch", ranges=_DOT)
        if c == "^":
            return N("caret")
        if c == "$":
            return N("dollar")
        if c == "\\":
            got = esc_char(in_class=False)
            if isinstance(got, tuple):
                return N("ch", ranges=got)
            return N("ch", ranges=((got, got),))
        return N("ch", ranges=((ord(c), ord(c)),))

    def parse_cat() -> N:
        items: list[N] = []
        while peek() not in (None, "|", ")"):
            items.append(apply_quant(parse_atom()))
        if not items:
            return EPS
        return items[0] if len(items) == 1 else N("cat", kids=tuple(items))

    def parse_alt() -> N:
        branches = [parse_cat()]
        while peek() == "|":
            take()
            branches.append(parse_cat())
        return branches[0] if len(branches) == 1 else N("alt", kids=tuple(branches))

    node = parse_alt()
    if pos != len(pattern):
        raise RxUnsupported(f"trailing input at {pattern[pos:]!r}")
    return node


def unroll(node: N, m: int, n: int | None) -> N:
    """{m,n} -> m copies + (n-m) optionals (or a star tail when n is None)."""
    kids: list[N] = [node] * m
    if n is None:
        kids.append(N("star", kids=(node,)))
    else:
        kids.extend([N("opt", kids=(node,))] * (n - m))
    if not kids:
        return EPS
    return kids[0] if len(kids) == 1 else N("cat", kids=tuple(kids))


def anchor(node: N) -> N:
    """JSON-Schema `pattern` is an unanchored search: pad unanchored branch
    edges with ANY*. ^/$ allowed only at branch edges."""
    branches = node.kids if node.kind == "alt" else (node,)
    out = []
    for b in branches:
        items = list(b.kids) if b.kind == "cat" else ([] if b.kind == "eps" else [b])
        lead = bool(items) and items[0].kind == "caret"
        trail = bool(items) and items[-1].kind == "dollar"
        if lead:
            items = items[1:]
        if trail:
            items = items[:-1]
        for it in items:
            if _contains_anchor(it):
                raise RxUnsupported("inner ^/$ anchor")
        if not lead:
            items = [ANYSTAR] + items
        if not trail:
            items = items + [ANYSTAR]
        out.append(N("cat", kids=tuple(items)) if len(items) != 1 else items[0])
    return out[0] if len(out) == 1 else N("alt", kids=tuple(out))


def _contains_anchor(node: N) -> bool:
    if node.kind in ("caret", "dollar"):
        return True
    return any(_contains_anchor(k) for k in node.kids)


# ---------------------------------------------- serialized-form emission

_SHORT_ESC = {0x08: "b", 0x0C: "f", 0x0A: "n", 0x0D: "r", 0x09: "t"}
_RAW_LO, _RAW_HI = 0x20, 0x7E       # raw-safe ASCII window (minus " and \)


def _grid_byte(b: int) -> str:
    if (0x30 <= b <= 0x39) or (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A):
        return chr(b)
    return f"\\x{b:02x}"


def _grid_class(ranges: list[tuple[int, int]]) -> str:
    """Byte ranges -> grid class (positive members, hex-escaped)."""
    ranges = [(lo, hi) for lo, hi in ranges if lo <= hi]
    if not ranges:
        raise RxUnsupported("empty byte class")
    if len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
        return _grid_byte(ranges[0][0])
    parts = []
    for lo, hi in ranges:
        if lo == hi:
            parts.append(f"\\x{lo:02x}")
        else:
            parts.append(f"\\x{lo:02x}-\\x{hi:02x}")
    return "[" + "".join(parts) + "]"


def _enc(cp: int) -> tuple[int, ...]:
    return tuple(chr(cp).encode("utf-8"))


def _seq_range(a: tuple[int, ...], b: tuple[int, ...]) -> list[list[tuple[int, int]]]:
    """Byte sequences a..b (equal length, same UTF-8 length class)."""
    if len(a) == 1:
        return [[(a[0], b[0])]]
    if a[0] == b[0]:
        return [[(a[0], a[0])] + t for t in _seq_range(a[1:], b[1:])]
    out: list[list[tuple[int, int]]] = []
    cont_min = (0x80,) * (len(a) - 1)
    cont_max = (0xBF,) * (len(a) - 1)
    out += [[(a[0], a[0])] + t for t in _seq_range(a[1:], cont_max)]
    if a[0] + 1 <= b[0] - 1:
        out.append([(a[0] + 1, b[0] - 1)] + [(0x80, 0xBF)] * (len(a) - 1))
    out += [[(b[0], b[0])] + t for t in _seq_range(cont_min, b[1:])]
    return out


def _utf8_seqs(lo: int, hi: int) -> list[list[tuple[int, int]]]:
    """Codepoint range (>= 0x80, surrogate-free) -> exact byte-range seqs."""
    out: list[list[tuple[int, int]]] = []
    for blo, bhi in ((0x80, 0x7FF), (0x800, 0xFFFF), (0x10000, 0x10FFFF)):
        s, e = max(lo, blo), min(hi, bhi)
        if s > e:
            continue
        out.extend(_seq_range(_enc(s), _enc(e)))
    return out


def emit_char(ranges: tuple) -> str:
    """One decoded char (codepoint ranges) -> grid regex over its canonical
    serialization(s)."""
    ascii_raw: list[tuple[int, int]] = []
    branches: list[str] = []
    for lo, hi in ranges:
        for c in range(max(lo, 0), min(hi, 0x1F) + 1):      # C0 controls
            if c in _SHORT_ESC:
                branches.append("\\\\" + _SHORT_ESC[c])
            else:
                branches.append(f"\\\\u00{c:02x}")
        s, e = max(lo, _RAW_LO), min(hi, _RAW_HI)
        if s <= e:
            sub = [(s, e)]
            for special in (0x22, 0x5C):
                sub = [seg for r in sub for seg in _split_out(r, special)]
            ascii_raw.extend(sub)
        if lo <= 0x22 <= hi:
            branches.append('\\\\"')
        if lo <= 0x5C <= hi:
            branches.append("\\\\\\\\")
        if lo <= 0x7F <= hi:
            ascii_raw.append((0x7F, 0x7F))
        s, e = max(lo, 0x80), min(hi, _MAX_CP)
        if s <= e:
            for seq in _utf8_seqs(s, e):
                branches.append("".join(_grid_class([br]) for br in seq))
    core = []
    if ascii_raw:
        core.append(_grid_class(_merge_byte_ranges(ascii_raw)))
    core.extend(branches)
    if not core:
        raise RxUnsupported("char class serializes to nothing")
    return core[0] if len(core) == 1 else "(" + "|".join(core) + ")"


def _split_out(r: tuple[int, int], x: int) -> list[tuple[int, int]]:
    lo, hi = r
    if x < lo or x > hi:
        return [r]
    out = []
    if lo <= x - 1:
        out.append((lo, x - 1))
    if x + 1 <= hi:
        out.append((x + 1, hi))
    return out


def _merge_byte_ranges(rs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rs = sorted(rs)
    out: list[list[int]] = []
    for lo, hi in rs:
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


# Unbounded ANY runs emit in compact byte form (same looseness convention as
# the engine's generic STRING_RX): identical verdicts over canonical
# serializations, ~10x smaller than the char-precise alternation — the
# char-precise form is only needed where CHARS ARE COUNTED (length windows,
# fixed positions).
_BYTE_ANY = (r'([^"\\\x00-\x1f]|\\(["\\/bfnrt]'
             r"|u[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]))")


def emit(node: N) -> str:
    """Value AST -> grid regex over the serialized JSON string BODY."""
    s = _emit(node)
    if len(s) > MAX_EMIT:
        raise RxUnsupported(f"emitted regex too large ({len(s)})")
    return s


def _emit(node: N) -> str:
    k = node.kind
    if k == "eps":
        return ""
    if k == "ch":
        return emit_char(node.ranges)
    if k == "cat":
        return "".join(_emit(c) for c in node.kids)
    if k == "alt":
        return "(" + "|".join(_emit(c) for c in node.kids) + ")"
    if k in ("star", "plus", "opt"):
        kid = node.kids[0]
        if k in ("star", "plus") and kid.kind == "ch" and kid.ranges == ANY_CHAR:
            return _BYTE_ANY + ("*" if k == "star" else "+")
        inner = _emit(kid)
        if not inner:
            return ""
        if not _is_atom(inner):
            inner = "(" + inner + ")"
        return inner + {"star": "*", "plus": "+", "opt": "?"}[k]
    if k in ("caret", "dollar"):
        raise RxUnsupported("unprocessed anchor (call anchor() first)")
    raise AssertionError(k)


def _is_atom(s: str) -> bool:
    """Is `s` already a single quantifiable regex atom?"""
    if len(s) == 1:
        return True
    if len(s) == 2 and s[0] == "\\":
        return True
    if s.startswith("\\x") and len(s) == 4:
        return True
    if s.startswith("[") and s.endswith("]") and "[" not in s[1:-1]:
        return True
    if s.startswith("(") and s.endswith(")"):
        depth = 0
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return False
            i += 1
        return depth == 0
    return False


def string_terminal_rx(body: str) -> str:
    """Wrap an emitted BODY regex into a full JSON-string terminal regex."""
    return '"' + body + '"'


# --------------------------------------------------------------- lengths

def length_body(min_len: int, max_len: int | None) -> str:
    """CHAR-count window over decoded chars (escape/UTF-8 aware)."""
    if min_len > MAX_UNROLL or (max_len is not None and max_len > MAX_UNROLL):
        raise RxUnsupported(f"length window ({min_len},{max_len}) beyond cap")
    return emit(unroll(ANY, min_len, max_len))


# ----------------------------------------------------- NOT-literals (keys)

def not_literals_body(literals: list[str]) -> str:
    """Complement of a finite decoded-string set (over string BODIES).
    Used for forbidden keys / extras-exclusion."""
    trie: dict = {}
    for s in literals:
        node = trie
        for ch in s:
            node = node.setdefault(ch, {})
        node[None] = True   # member marker

    branches: list[N] = []

    def walk(node: dict, path: list[str]) -> None:
        conts = [c for c in node if c is not None]
        cont_ranges = _norm([(ord(c), ord(c)) for c in conts])
        diverge = _subtract(ANY_CHAR, cont_ranges)
        prefix = [N("ch", ranges=((ord(c), ord(c)),)) for c in path]
        if diverge:
            branches.append(N("cat", kids=tuple(prefix + [N("ch", ranges=diverge), ANYSTAR])))
        if None not in node and path:
            branches.append(N("cat", kids=tuple(prefix)) if len(prefix) > 1 else prefix[0])
        for c in conts:
            walk(node[c], path + [c])

    walk(trie, [])
    parts = [_emit(b) for b in branches]
    if "" not in literals:
        parts.insert(0, "")             # admit the empty string
    body = "(" + "|".join(parts) + ")"
    if len(body) > MAX_EMIT:
        raise RxUnsupported("not-literals regex too large")
    return body


# ------------------------------------------------------------ int ranges
#
# Canonical JSON integers: -?(0|[1-9][0-9]*); no leading zeros, no "-0".

def _same_len_ge(a: str) -> list[str]:
    """digit strings, same length as `a`, numerically >= a (leading digit
    keeps the no-leading-zero rule via position-0 floor of the caller)."""
    n = len(a)
    out = [a]
    for i in range(n):
        d = int(a[i])
        if d + 1 > 9:
            continue
        cls = "9" if d + 1 == 9 else f"[{d + 1}-9]"
        out.append(a[:i] + cls + "[0-9]" * (n - 1 - i))
    return out


def _same_len_le(b: str) -> list[str]:
    n = len(b)
    out = [b]
    for i in range(n):
        d = int(b[i])
        floor = 1 if i == 0 and n > 1 else 0
        if d - 1 < floor:
            continue
        cls = str(floor) if d - 1 == floor else f"[{floor}-{d - 1}]"
        out.append(b[:i] + cls + "[0-9]" * (n - 1 - i))
    return out


def _suffix_ge(a: str) -> list[str]:
    """digit strings of len(a) (leading zeros OK) >= a"""
    return _same_len_ge(a)


def _suffix_le(b: str) -> list[str]:
    n = len(b)
    out = [b]
    for i in range(n):
        d = int(b[i])
        if d - 1 < 0:
            continue
        cls = "0" if d - 1 == 0 else f"[0-{d - 1}]"
        out.append(b[:i] + cls + "[0-9]" * (n - 1 - i))
    return out


def _same_len_range(a: str, b: str) -> list[str]:
    assert len(a) == len(b) and int(a) <= int(b)
    if a == b:
        return [a]
    i = 0
    while a[i] == b[i]:
        i += 1
    pre = a[:i]
    da, db = int(a[i]), int(b[i])
    n = len(a)
    if i + 1 == n:
        cls = f"[{da}-{db}]" if db > da else str(da)
        return [pre + cls]
    out = []
    out.extend(pre + a[i] + t for t in _suffix_ge(a[i + 1:]))
    out.extend(pre + b[i] + t for t in _suffix_le(b[i + 1:]))
    if da + 1 <= db - 1:
        mid = str(da + 1) if da + 1 == db - 1 else f"[{da + 1}-{db - 1}]"
        out.append(pre + mid + "[0-9]" * (n - i - 1))
    return out


def _uint_ge_open(a: str) -> list[str]:
    """digit strings >= a (no upper bound)"""
    if int(a) == 0:
        return ["0", "[1-9][0-9]*"]
    out = ["[1-9]" + "[0-9]" * len(a) + "[0-9]*"]   # strictly longer
    out.extend(_same_len_ge(a))
    return out


def _uint_le(b: str) -> list[str]:
    if int(b) == 0:
        return ["0"]
    out = ["0"]
    for ln in range(1, len(b)):
        out.append("[1-9]" + "[0-9]" * (ln - 1))
    out.extend(_same_len_le(b))
    return out


def _uint_range(a: str, b: str) -> list[str]:
    ia, ib = int(a), int(b)
    assert 0 <= ia <= ib
    if ia == 0:
        return _uint_le(b)
    if len(a) == len(b):
        return _same_len_range(a, b)
    out = _same_len_ge(a)
    for ln in range(len(a) + 1, len(b)):
        out.append("[1-9]" + "[0-9]" * (ln - 1))
    out.extend(_same_len_le(b))
    return out


def int_range_rx(lo: int | None, hi: int | None) -> str:
    """Grid regex for canonical JSON integers in [lo, hi] (None = open)."""
    branches: list[str] = []
    # positive side (x >= 0)
    if hi is None or hi >= 0:
        plo = 0 if lo is None else max(lo, 0)
        if hi is None:
            branches.extend(_uint_ge_open(str(plo)))
        elif plo <= hi:
            branches.extend(_uint_range(str(plo), str(hi)))
    # negative side (x <= -1): |x| in [nlo, nhi]
    if lo is None or lo <= -1:
        nlo = 1 if hi is None or hi >= -1 else -hi
        if lo is None:
            mags = _uint_ge_open(str(nlo))
        else:
            nhi = -lo
            mags = _uint_range(str(nlo), str(nhi)) if nlo <= nhi else []
        branches.extend("-" + m for m in mags if m != "0")
    if not branches:
        raise RxUnsupported(f"empty int range [{lo},{hi}]")
    return "(" + "|".join(branches) + ")"


# ------------------------------------------------------------- numbers
#
# Canonical float forms (json.dumps == repr): "K.F" with F trailing-zero-free
# (float ints -> "K.0"), "-0.0"; exponent forms only for |x| >= 1e16 or
# 0 < |x| < 1e-4: "M(.F)?e[+-]E" with M in 1..9. Ints stay plain.

_NZ_FRAC = r"\.([0-9]*[1-9])"       # nonzero canonical fraction
_ZERO_FRAC = r"(\.0)?"              # canonical zero fraction is exactly ".0"
_E_NEG = r"(\.[0-9]*[1-9])?[eE]-[0-9]+"      # |x| < 0.1 canonical e- forms


def number_range_rx(lo, hi, excl_lo: bool = False, excl_hi: bool = False) -> str | None:
    """Grid regex for canonical JSON numbers within bounds; None when the case
    is outside the supported set (bounds must be integer-valued or None, with
    |bound| < 1e15 so canonical e+ forms are decidable by side)."""
    def intv(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return None

    ilo = intv(lo)
    ihi = intv(hi)
    if (lo is not None and ilo is None) or (hi is not None and ihi is None):
        return None
    if (ilo is not None and abs(ilo) >= 10**15) or (ihi is not None and abs(ihi) >= 10**15):
        return None

    branches: list[str] = []

    # 1. integer lattice (plain ints and K.0 float-ints)
    int_lo = None if ilo is None else (ilo + 1 if excl_lo else ilo)
    int_hi = None if ihi is None else (ihi - 1 if excl_hi else ihi)
    if int_lo is None or int_hi is None or int_lo <= int_hi:
        branches.append(int_range_rx(int_lo, int_hi) + _ZERO_FRAC)

    # 2. nonzero-fraction values: x in (k, k+1), admitted iff [k, k+1] within
    # bounds (open interval => exclusivity at the integer endpoints is free)
    #   positive buckets k >= 0: "k.F"
    pk_lo = 0 if ilo is None else max(ilo, 0)
    pk_hi = None if ihi is None else ihi - 1
    if pk_hi is None or pk_lo <= pk_hi:
        branches.append(int_range_rx(pk_lo, pk_hi) + _NZ_FRAC)
    #   negative buckets: x in (-(m+1), -m), serialized "-m.F" (m >= 0)
    nm_lo = 0 if ihi is None else max(-ihi, 0)          # -m <= hi
    nm_hi = None if ilo is None else -ilo - 1           # -(m+1) >= lo
    if nm_hi is None or nm_lo <= nm_hi:
        if nm_lo == 0:
            branches.append(r"-0" + _NZ_FRAC)
            nm_lo = 1
        if nm_hi is None or nm_lo <= nm_hi:
            mags = _uint_ge_open(str(nm_lo)) if nm_hi is None else \
                (_uint_range(str(nm_lo), str(nm_hi)) if nm_lo <= nm_hi else [])
            mags = [m for m in mags if m != "0"]
            if mags:
                branches.append("-(" + "|".join(mags) + ")" + _NZ_FRAC)

    # 3. -0.0 == 0
    zero_in = ((ilo is None or ilo < 0 or (ilo == 0 and not excl_lo)) and
               (ihi is None or ihi > 0 or (ihi == 0 and not excl_hi)))
    if zero_in:
        branches.append(r"-0\.0")

    # 4. tiny e- forms: 0 < |x| < 1e-4
    pos_tiny = ((ilo is None or ilo <= 0) and (ihi is None or ihi >= 1))
    neg_tiny = ((ihi is None or ihi >= 0) and (ilo is None or ilo <= -1))
    if pos_tiny:
        branches.append(r"[1-9]" + _E_NEG)
    if neg_tiny:
        branches.append(r"-[1-9]" + _E_NEG)

    # 5. huge e+ forms: |x| >= 1e16 — admitted only on an open side
    if ihi is None:
        branches.append(r"[1-9](\.[0-9]*[1-9])?[eE]\+[0-9]+")
    if ilo is None:
        branches.append(r"-[1-9](\.[0-9]*[1-9])?[eE]\+[0-9]+")

    if not branches:
        return None
    return "(" + "|".join(branches) + ")"


# ------------------------------------------------------------- formats

# Value-level ECMA patterns (anchored); compiled through the same pipeline.
FORMAT_PATTERNS: dict[str, str] = {
    "date": r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$",
    "time": (
        r"^([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?"
        r"([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$"
    ),
    "date-time": (
        r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
        r"[Tt ]([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?"
        r"([Zz]|[+-]([01][0-9]|2[0-3]):[0-5][0-9])$"
    ),
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "ipv4": (
        r"^(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
        r"(\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])){3}$"
    ),
    "email": (
        r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
    ),
    "hostname": (
        r"^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(\.[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$"
    ),
    "uri": r"^[A-Za-z][A-Za-z0-9+.-]*:[!-~]*$",
    "uri-reference": r"^[!-~]*$",
}


def format_body(name: str) -> str | None:
    pat = FORMAT_PATTERNS.get(name)
    if pat is None:
        return None
    return emit(anchor(parse_ecma(pat)))


def pattern_body(pattern: str) -> str:
    """JSON-Schema `pattern` (unanchored ECMA search) -> serialized body."""
    return emit(anchor(parse_ecma(pattern)))


def literals_body(strings: list[str]) -> str:
    """Alternation of exact decoded strings, over serialized bodies."""
    return emit(N("alt", kids=tuple(lit(s) for s in strings)))


# ------------------------------------------------ pattern complement
#
# Complement of a JSON-Schema `pattern` over all strings — needed for the
# additionalProperties key terminal when patternProperties is present
# (keys matching NO pattern). Supported shape family (covers the schemas
# observed in jsonschemabench):
#   - unanchored single char class:      c          -> [¬c]*
#   - anchored prefix of classes:        ^C1C2..    -> shorter | divergence
#   - fixed class sequence:              ^C1..Ck$   -> len≠k | divergence
#   - head + class window:               ^H1..Hj [c]{m,n}$  (n may be open;
#     covers [c]+, [c]*, .{1,2}, [h][t]*) -> shorter | head divergence |
#     class divergence | too long
# Returns None when the complement is EMPTY (pattern matches everything):
# callers then know "non-matching keys" are impossible.

def _flatten_cat(node: N) -> list[N]:
    if node.kind == "cat":
        out = []
        for k in node.kids:
            out.extend(_flatten_cat(k))
        return out
    if node.kind == "eps":
        return []
    return [node]


def _atoms(items: list[N]) -> list[tuple[tuple, str]]:
    """-> [(ranges, quant)] with quant in {'1','opt','star','plus'};
    raises RxUnsupported for anything else."""
    out = []
    for it in items:
        if it.kind == "ch":
            out.append((it.ranges, "1"))
        elif it.kind in ("opt", "star", "plus") and it.kids[0].kind == "ch":
            out.append((it.kids[0].ranges, it.kind))
        else:
            raise RxUnsupported(f"complement: {it.kind} atom")
    return out


def _too_short_ast(k: int) -> N | None:
    """All strings with length < k (any chars)."""
    if k <= 0:
        return None
    return unroll(ANY, 0, k - 1)


def class_window_minus_literals(cls: tuple, m: int, n: int | None,
                                literals: list[str]) -> str:
    """Strings over class `cls` with length in [m, n] MINUS a finite literal
    set (used when patternProperties overlaps declared keys: the pattern pair
    must exclude the declared names). Literals outside the class language are
    harmless to exclude."""
    if n is not None and (n > MAX_UNROLL or m > MAX_UNROLL):
        raise RxUnsupported("class window beyond cap")
    cnode = N("ch", ranges=cls)

    def window(d: int) -> N | None:
        """suffix of cls-chars so total length lands in [m, n]"""
        lo = max(m - d, 0)
        hi = None if n is None else n - d
        if hi is not None and hi < 0:
            return None
        return unroll(cnode, lo, hi)

    trie: dict = {}
    for s in literals:
        node = trie
        for ch in s:
            node = node.setdefault(ch, {})
        node[None] = True

    branches: list[N] = []

    def walk(node: dict, path: list[str]) -> None:
        d = len(path)
        conts = [c for c in node if c is not None]
        cont_ranges = _norm([(ord(c), ord(c)) for c in conts])
        diverge_in_cls = _subtract(cls, cont_ranges)
        prefix = [N("ch", ranges=((ord(c), ord(c)),)) for c in path]
        if diverge_in_cls:
            tail = window(d + 1)
            if tail is not None:
                branches.append(N("cat", kids=tuple(
                    prefix + [N("ch", ranges=diverge_in_cls), tail])))
        if None not in node and path and m <= d and (n is None or d <= n):
            branches.append(N("cat", kids=tuple(prefix)) if d > 1 else prefix[0])
        for c in conts:
            if _subtract(_norm([(ord(c), ord(c))]), cls) == ():
                walk(node[c], path + [c])
            # a literal char outside cls: its whole subtree is outside the
            # class language — nothing to subtract there

    walk(trie, [])
    parts = [_emit(b) for b in branches]
    if m == 0 and "" not in literals:
        parts.insert(0, "")
    if not parts:
        raise RxUnsupported("class-minus-literals: empty result")
    body = "(" + "|".join(parts) + ")"
    if len(body) > MAX_EMIT:
        raise RxUnsupported("class-minus-literals too large")
    return body


def _head_star_end(items: list[N]) -> str | None:
    """Complement for ^H1..Hj [t]* e$ and ^H1..Hj ([t]* e)?$: matches fail by
    (a) too short, (b) head divergence, (c) a non-t char before the last
    position, (d) a last char outside e."""
    if not items:
        return None
    optional_tail = False
    parts = list(items)
    last = parts[-1]
    if last.kind == "opt" and last.kids[0].kind == "cat":
        inner = _flatten_cat(last.kids[0])
        parts = parts[:-1] + inner
        optional_tail = True
    if len(parts) < 2:
        return None
    star, end = parts[-2], parts[-1]
    if not (star.kind == "star" and star.kids[0].kind == "ch"
            and end.kind == "ch"):
        return None
    head = parts[:-2]
    if any(h.kind != "ch" for h in head):
        return None
    hrs = [h.ranges for h in head]
    t_cls = star.kids[0].ranges
    e_cls = end.ranges
    j = len(hrs)
    min_len = j if optional_tail else j + 1
    branches: list[N] = []
    ts = _too_short_ast(min_len)
    if ts is not None:
        branches.append(ts)
    if not optional_tail and j == 0:
        pass
    for i in range(j):
        bad = _subtract(ANY_CHAR, hrs[i])
        if bad:
            kids = [N("ch", ranges=r) for r in hrs[:i]]
            kids += [N("ch", ranges=bad), ANYSTAR]
            branches.append(N("cat", kids=tuple(kids)))
    hk = [N("ch", ranges=r) for r in hrs]
    bad_t = _subtract(ANY_CHAR, t_cls)
    if bad_t:
        # a non-t char strictly before the final char
        branches.append(N("cat", kids=tuple(
            hk + [N("star", kids=(N("ch", ranges=t_cls),)),
                  N("ch", ranges=bad_t), ANY, ANYSTAR])))
    bad_e = _subtract(ANY_CHAR, e_cls)
    if bad_e:
        # length >= j+1 with a final char outside e
        branches.append(N("cat", kids=tuple(
            hk + [ANYSTAR, N("ch", ranges=bad_e)])))
    if not branches:
        return None
    return emit(N("alt", kids=tuple(branches)))


def _nullable(node: N) -> bool:
    k = node.kind
    if k in ("eps", "star", "opt", "caret", "dollar"):
        return True
    if k == "ch":
        return False
    if k == "cat":
        return all(_nullable(c) for c in node.kids)
    if k == "alt":
        return any(_nullable(c) for c in node.kids)
    if k == "plus":
        return _nullable(node.kids[0])
    return False


def pattern_complement_body(pattern: str) -> str | None:
    ast = parse_ecma(pattern)
    if ast.kind == "alt":
        raise RxUnsupported("complement: alternation")
    items = _flatten_cat(ast)
    lead = bool(items) and items[0].kind == "caret"
    trail = bool(items) and items[-1].kind == "dollar"
    # an unanchored nullable pattern finds an empty match in every string:
    # it matches everything, so its complement is empty
    if not lead and not trail and _nullable(ast):
        return None
    if lead:
        items = items[1:]
    if trail:
        items = items[:-1]
    for it in items:
        if _contains_anchor(it):
            raise RxUnsupported("complement: inner anchor")

    # shape: ^H1..Hj ([t]* e)? $  or  ^H1..Hj [t]* e $   (hostname-label /
    # '^...*a$' style: fixed head, class star, ONE closing class)
    if lead and trail:
        got = _head_star_end(items)
        if got is not None:
            return got
    atoms = _atoms(items)

    branches: list[N] = []

    def diverge_branch(prefix_ranges: list[tuple], i: int, bad: tuple) -> N:
        kids = [N("ch", ranges=r) for r in prefix_ranges[:i]]
        kids.append(N("ch", ranges=bad))
        kids.append(ANYSTAR)
        return N("cat", kids=tuple(kids))

    if not lead:
        # unanchored search: only a single one-char atom is supported
        if len(atoms) == 1 and atoms[0][1] == "1":
            comp = _subtract(ANY_CHAR, atoms[0][0])
            if not comp:
                return None         # pattern matches any nonempty... and ""? a
                                    # 1-char search never matches "" — but comp
                                    # empty means every char matches; complement
                                    # is just the empty string
            return emit(N("alt", kids=(EPS, N("star", kids=(N("ch", ranges=comp),)))))
        raise RxUnsupported("complement: unanchored multi-atom")

    fixed = [a for a in atoms if a[1] == "1"]
    if not trail:
        # ^prefix...: all atoms must be fixed classes
        if len(fixed) != len(atoms):
            raise RxUnsupported("complement: quantifier in open prefix")
        k = len(atoms)
        if k == 0:
            return None             # ^ alone matches everything
        ts = _too_short_ast(k)
        if ts is not None:
            branches.append(ts)
        rs = [a[0] for a in atoms]
        for i in range(k):
            bad = _subtract(ANY_CHAR, rs[i])
            if bad:
                branches.append(diverge_branch(rs, i, bad))
        if not branches:
            return None
        return emit(N("alt", kids=tuple(branches)))

    # fully anchored
    if len(fixed) == len(atoms):
        k = len(atoms)
        if k == 0:
            # ^$: complement = all nonempty strings
            return emit(N("cat", kids=(ANY, ANYSTAR)))
        rs = [a[0] for a in atoms]
        ts = _too_short_ast(k)
        if ts is not None:
            branches.append(ts)
        # too long: k matching-or-not chars... any string of length > k
        branches.append(N("cat", kids=tuple([ANY] * (k + 1) + [ANYSTAR])))
        for i in range(k):
            bad = _subtract(ANY_CHAR, rs[i])
            if bad:
                branches.append(diverge_branch(rs, i, bad))
        return emit(N("alt", kids=tuple(branches)))

    # head of fixed classes + ONE quantified class run, nothing after
    qpos = [i for i, a in enumerate(atoms) if a[1] != "1"]
    head = atoms[:qpos[0]]
    tail_atoms = atoms[qpos[0]:]
    cls = tail_atoms[0][0]
    if any(a[0] != cls for a in tail_atoms):
        raise RxUnsupported("complement: mixed classes in window")
    m = sum(1 for a in tail_atoms if a[1] in ("1", "plus"))
    n: int | None = None
    if all(a[1] in ("1", "opt") for a in tail_atoms):
        n = len(tail_atoms)
    elif any(a[1] in ("star", "plus") for a in tail_atoms):
        n = None
    hrs = [a[0] for a in head]
    j = len(head)
    min_len = j + m
    ts = _too_short_ast(min_len)
    if ts is not None:
        branches.append(ts)
    for i in range(j):
        bad = _subtract(ANY_CHAR, hrs[i])
        if bad:
            branches.append(diverge_branch(hrs, i, bad))
    bad_cls = _subtract(ANY_CHAR, cls)
    if bad_cls:
        # head matched, then class chars, then a divergence char
        kids = [N("ch", ranges=r) for r in hrs]
        kids.append(N("star", kids=(N("ch", ranges=cls),)))
        kids.append(N("ch", ranges=bad_cls))
        kids.append(ANYSTAR)
        branches.append(N("cat", kids=tuple(kids)))
    if n is not None:
        # too many class chars (all matching)
        kids = [N("ch", ranges=r) for r in hrs]
        kids.extend([N("ch", ranges=cls)] * (n + 1))
        kids.append(N("star", kids=(N("ch", ranges=cls),)))
        branches.append(N("cat", kids=tuple(kids)))
    if not branches:
        return None
    return emit(N("alt", kids=tuple(branches)))
