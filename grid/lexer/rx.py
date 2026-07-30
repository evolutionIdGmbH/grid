"""Grid-regex subset -> parse tree (the lexer pipeline's first stage).

Moved verbatim from grid/lexer/dfa.py, which remains the import facade for
the lexer pipeline (tests and the jsonschema bridge's dialect docs reference
dfa._parse_regex). All patterns operate on BYTES: literals encode latin-1;
multi-byte UTF-8 enters through byte classes (e.g. [\\x80-\\xff]).
"""

from __future__ import annotations

from dataclasses import dataclass

from grid.errors import GrammarInvalid

_ESCAPES = {"n": ord("\n"), "t": ord("\t"), "r": ord("\r"), "0": 0}
_MAX_REPEAT = 8192      # {m,n} bound cap: expansion is linear in n


def _expand_repeat(node: _Node, m: int, n: int | None) -> _Node:
    """{m,n} -> m copies + (n-m) optionals ({m,} -> m copies + a star tail).
    Expansion happens at parse time; the NFA builder is unchanged and shared
    subtrees are safe (construction walks per visit)."""
    kids: list[_Node] = [node] * m
    if n is None:
        kids.append(_Node("star", kids=(node,)))
    else:
        kids.extend([_Node("opt", kids=(node,))] * (n - m))
    if not kids:
        return _Node("eps")
    return kids[0] if len(kids) == 1 else _Node("cat", kids=tuple(kids))


@dataclass
class _Node:
    kind: str                      # char|class|any|cat|alt|star|plus|opt|eps|rep
    chars: frozenset[int] = frozenset()
    kids: tuple[_Node, ...] = ()
    bounds: tuple[int, int | None] | None = None  # rep only: (m, n); n=None open


def _parse_regex(pattern: str, keep_reps: bool = False) -> _Node:
    pos = 0

    def peek() -> str | None:
        return pattern[pos] if pos < len(pattern) else None

    def take() -> str:
        nonlocal pos
        ch = pattern[pos]
        pos += 1
        return ch

    def parse_alt() -> _Node:
        branches = [parse_cat()]
        while peek() == "|":
            take()
            branches.append(parse_cat())
        return branches[0] if len(branches) == 1 else _Node("alt", kids=tuple(branches))

    def parse_cat() -> _Node:
        items: list[_Node] = []
        while peek() not in (None, "|", ")"):
            items.append(parse_post())
        if not items:
            return _Node("eps")
        return items[0] if len(items) == 1 else _Node("cat", kids=tuple(items))

    def parse_post() -> _Node:
        node = parse_atom()
        while True:
            c = peek()
            if c in ("*", "+", "?"):
                op = take()
                node = _Node({"*": "star", "+": "plus", "?": "opt"}[op], kids=(node,))
            elif c == "{":
                rep = try_parse_repeat()
                if rep is None:
                    break               # literal '{' consumed by parse_atom later
                m, n = rep
                if keep_reps:
                    # counting-set candidate ({m,n} kept as a counted-loop
                    # node; grid/lexer/counting.py expands the ineligible
                    # ones via the same _expand_repeat)
                    node = _Node("rep", kids=(node,), bounds=(m, n))
                else:
                    node = _expand_repeat(node, m, n)
            else:
                break
        return node

    def try_parse_repeat():
        """Parse {m} / {m,} / {m,n} after an atom; None (no input consumed)
        when the braces are not a valid quantifier — the '{' then reads as a
        literal, matching the ECMA convention."""
        nonlocal pos
        save = pos
        take()                          # '{'
        digits = ""
        while peek() is not None and peek().isdigit():
            digits += take()
        if not digits:
            pos = save
            return None
        m = int(digits)
        n: int | None = m
        if peek() == ",":
            take()
            digits = ""
            while peek() is not None and peek().isdigit():
                digits += take()
            n = int(digits) if digits else None
        if peek() != "}":
            pos = save
            return None
        take()                          # '}'
        if n is not None and n < m:
            raise GrammarInvalid(f"bad repetition {{{m},{n}}} in regex {pattern!r}")
        if m > _MAX_REPEAT or (n is not None and n > _MAX_REPEAT):
            raise GrammarInvalid(
                f"repetition bound over {_MAX_REPEAT} in regex {pattern!r}")
        return m, n

    def parse_atom() -> _Node:
        ch = take()
        if ch == "(":
            node = parse_alt()
            if peek() != ")":
                raise GrammarInvalid(f"unclosed group in regex {pattern!r}")
            take()
            return node
        if ch == "[":
            return parse_class()
        if ch == ".":
            return _Node("class", chars=frozenset(range(256)) - {ord("\n")})
        if ch == "\\":
            esc = take()
            if esc in _ESCAPES:
                return _Node("char", chars=frozenset({_ESCAPES[esc]}))
            if esc == "x":
                hexs = take() + take()
                return _Node("char", chars=frozenset({int(hexs, 16)}))
            return _Node("char", chars=frozenset({ord(esc)}))
        return _Node("char", chars=frozenset({ord(ch)}))

    def parse_class() -> _Node:
        negate = False
        if peek() == "^":
            take()
            negate = True
        chars: set[int] = set()
        first = True
        while peek() != "]" or first:
            if peek() is None:
                raise GrammarInvalid(f"unclosed class in regex {pattern!r}")
            first = False
            ch = take()
            if ch == "\\":
                esc = take()
                if esc == "x":
                    code = int(take() + take(), 16)
                else:
                    code = _ESCAPES.get(esc, ord(esc))
            else:
                code = ord(ch)
            if peek() == "-" and pos + 1 < len(pattern) and pattern[pos + 1] != "]":
                take()
                hi_ch = take()
                if hi_ch == "\\":
                    esc = take()
                    hi = int(take() + take(), 16) if esc == "x" else _ESCAPES.get(esc, ord(esc))
                else:
                    hi = ord(hi_ch)
                chars.update(range(code, hi + 1))
            else:
                chars.add(code)
        take()  # ']'
        if negate:
            chars = set(range(256)) - chars
        return _Node("class", chars=frozenset(chars))

    node = parse_alt()
    if pos != len(pattern):
        raise GrammarInvalid(f"trailing regex input at {pattern[pos:]!r}")
    return node


def _literal_node(pattern: str) -> _Node:
    """Literal terminal text -> concatenation of single-byte char nodes."""
    if len(pattern) > 1:
        return _Node(
            "cat",
            kids=tuple(_Node("char", chars=frozenset({c})) for c in pattern.encode("latin-1")),
        )
    return _Node("char", chars=frozenset({ord(pattern)}))
