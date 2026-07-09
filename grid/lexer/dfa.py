"""Terminal regexes -> combined scanner DFA (DESIGN.md SS3 lexer/, E7).

Pipeline: grid-regex subset -> NFA (Thompson) -> combined NFA (one tagged accept
per terminal) -> subset-construction DFA over bytes.

Scanner DFA state knowledge:
- ``accept[state]``: the winning terminal if the scan stopped here (maximal munch
  resolves length; at equal length, literal terminals beat named ones, then
  declaration order — Terminal.priority).
- ``live[state]``: the set of terminals still reachable from this state — the
  lexer hypothesis set (E7). INV-LEX1's H_max is the max |live| over states,
  computed here at build time (eager L1/L2 product; L3 identifier categories add
  at most +1 hypothesis by construction).

All automata operate on BYTES: patterns are encoded latin-1; multi-byte UTF-8
in identifiers enters through byte classes (e.g. [\\x80-\\xff]).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal

# ---------------------------------------------------------------- regex parser

_ESCAPES = {"n": ord("\n"), "t": ord("\t"), "r": ord("\r"), "0": 0}


@dataclass
class _Node:
    kind: str                      # char|class|any|cat|alt|star|plus|opt|eps
    chars: frozenset[int] = frozenset()
    kids: tuple[_Node, ...] = ()


def _parse_regex(pattern: str) -> _Node:
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
        while peek() in ("*", "+", "?"):
            op = take()
            node = _Node({"*": "star", "+": "plus", "?": "opt"}[op], kids=(node,))
        return node

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


# ---------------------------------------------------------------- NFA

@dataclass
class _NFA:
    """Byte-labelled NFA with epsilon edges; single (start, accept) pair."""

    start: int
    accept: int
    eps: dict[int, list[int]]
    edges: dict[int, list[tuple[frozenset[int], int]]]


class _NFABuilder:
    def __init__(self) -> None:
        self.n = 0
        self.eps: dict[int, list[int]] = {}
        self.edges: dict[int, list[tuple[frozenset[int], int]]] = {}

    def new(self) -> int:
        self.n += 1
        return self.n - 1

    def add_eps(self, a: int, b: int) -> None:
        self.eps.setdefault(a, []).append(b)

    def add_edge(self, a: int, chars: frozenset[int], b: int) -> None:
        self.edges.setdefault(a, []).append((chars, b))

    def build(self, node: _Node) -> tuple[int, int]:
        s, a = self.new(), self.new()
        if node.kind in ("char", "class"):
            self.add_edge(s, node.chars, a)
        elif node.kind == "eps":
            self.add_eps(s, a)
        elif node.kind == "cat":
            prev = s
            for kid in node.kids:
                ks, ka = self.build(kid)
                self.add_eps(prev, ks)
                prev = ka
            self.add_eps(prev, a)
        elif node.kind == "alt":
            for kid in node.kids:
                ks, ka = self.build(kid)
                self.add_eps(s, ks)
                self.add_eps(ka, a)
        elif node.kind == "star":
            ks, ka = self.build(node.kids[0])
            self.add_eps(s, ks)
            self.add_eps(s, a)
            self.add_eps(ka, ks)
            self.add_eps(ka, a)
        elif node.kind == "plus":
            ks, ka = self.build(node.kids[0])
            self.add_eps(s, ks)
            self.add_eps(ka, ks)
            self.add_eps(ka, a)
        elif node.kind == "opt":
            ks, ka = self.build(node.kids[0])
            self.add_eps(s, ks)
            self.add_eps(s, a)
            self.add_eps(ka, a)
        else:  # pragma: no cover
            raise AssertionError(node.kind)
        return s, a


# ---------------------------------------------------------------- scanner DFA

DEAD = -1


@dataclass(frozen=True)
class ScannerDFA:
    """Combined byte DFA over all terminals of one grammar (immutable, shared).

    - ``accept[s]``: priority-winning terminal accepting exactly at ``s`` (or -1).
    - ``accepts_all[s]``: every terminal accepting exactly at ``s`` (EOS finalization
      and keyword-vs-identifier hypotheses need the full set, not just the winner).
    - ``live[s]``: terminals whose accept is reachable from ``s`` in >= 0 bytes —
      the E7 hypothesis set; ``h_max = max |live|`` (INV-LEX1).
    """

    start: int
    trans: tuple[tuple[int, ...], ...]      # [state][byte] -> state or DEAD
    accept: tuple[int, ...]                 # [state] -> terminal id or -1 (priority winner)
    accepts_all: tuple[frozenset[int], ...]
    live: tuple[frozenset[int], ...]
    h_max: int = field(compare=False, default=0)

    def next(self, state: int, byte: int) -> int:
        return self.trans[state][byte]

    def scan_state(self, remainder: bytes) -> int:
        """DFA state after scanning ``remainder`` from start (DEAD if impossible)."""
        st = self.start
        for b in remainder:
            st = self.trans[st][b]
            if st == DEAD:
                return DEAD
        return st


def build_scanner(terminals: dict[str, Terminal], terminal_order: tuple[str, ...]) -> ScannerDFA:
    """Combined NFA over all terminals -> subset-construction byte DFA."""
    b = _NFABuilder()
    root = b.new()
    accept_terminal: dict[int, int] = {}  # NFA accept state -> terminal id
    for tid, name in enumerate(terminal_order):
        t = terminals[name]
        if t.is_literal:
            node: _Node = _Node(
                "cat",
                kids=tuple(_Node("char", chars=frozenset({c})) for c in t.pattern.encode("latin-1")),
            ) if len(t.pattern) > 1 else _Node("char", chars=frozenset({ord(t.pattern)}))
        else:
            node = _parse_regex(t.pattern)
        s, a = b.build(node)
        b.add_eps(root, s)
        accept_terminal[a] = tid

    # eps-closure distributes over union: precompute the per-state closure once
    # (graph reachability over eps edges), then closure(S) = union of eps_star[s].
    eps_star: dict[int, frozenset[int]] = {}

    def _star(s0: int) -> frozenset[int]:
        got = eps_star.get(s0)
        if got is None:
            stack, seen = [s0], {s0}
            while stack:
                st = stack.pop()
                for nxt in b.eps.get(st, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            got = eps_star[s0] = frozenset(seen)
        return got

    def eps_closure(states) -> frozenset[int]:
        out: frozenset[int] = frozenset()
        for s in states:
            out |= _star(s)
        return out

    prio = {tid: terminals[name].priority for tid, name in enumerate(terminal_order)}

    # Alphabet compression: partition the 256 byte values into equivalence
    # classes over the distinct edge charsets (JSON/SQL grammars have ~10-30
    # classes), run the subset construction per CLASS, and expand to 256-wide
    # rows at the end. Same DFA modulo state numbering; the dominant TTFM cost
    # (the per-byte inner loop) drops by the compression factor.
    distinct_charsets = {chars for edges in b.edges.values() for chars, _dst in edges}
    blocks: list[set[int]] = [set(range(256))]
    for chars in distinct_charsets:
        nxt_blocks: list[set[int]] = []
        for blk in blocks:
            inside = blk & chars
            outside = blk - chars
            if inside:
                nxt_blocks.append(inside)
            if outside:
                nxt_blocks.append(outside)
        blocks = nxt_blocks
    blocks.sort(key=min)  # deterministic class order (stable across processes)
    class_of = [0] * 256
    class_rep: list[int] = []
    for ci_, blk in enumerate(blocks):
        class_rep.append(min(blk))
        for c in blk:
            class_of[c] = ci_
    n_classes = len(blocks)
    # per NFA state: class -> destination set (edges evaluated once, per class)
    edge_by_class: dict[int, list[list[int] | None]] = {}
    for st, edges in b.edges.items():
        per = edge_by_class[st] = [None] * n_classes
        for chars, dst in edges:
            seen_cls: set[int] = set()
            for c in chars:
                cl = class_of[c]
                if cl in seen_cls:
                    continue
                seen_cls.add(cl)
                lst = per[cl]
                if lst is None:
                    per[cl] = [dst]
                else:
                    lst.append(dst)

    start_set = eps_closure(frozenset({root}))
    ids: dict[frozenset[int], int] = {start_set: 0}
    order = [start_set]
    trans: list[list[int]] = []
    accepts_all: list[frozenset[int]] = []
    i = 0
    while i < len(order):
        cur = order[i]
        i += 1
        by_class: dict[int, set[int]] = {}
        for st in cur:
            per = edge_by_class.get(st)
            if per is None:
                continue
            for cl, dsts in enumerate(per):
                if dsts is not None:
                    by_class.setdefault(cl, set()).update(dsts)
        row = [DEAD] * 256
        for cl, dsts in sorted(by_class.items()):
            nxt = eps_closure(frozenset(dsts))
            if nxt not in ids:
                ids[nxt] = len(order)
                order.append(nxt)
            dst_id = ids[nxt]
            for c in blocks[cl]:
                row[c] = dst_id
        trans.append(row)
        accepts_all.append(frozenset(accept_terminal[st] for st in cur if st in accept_terminal))

    if accepts_all[0]:
        bad = ", ".join(terminal_order[t] for t in accepts_all[0])
        raise GrammarInvalid(f"terminals match the empty string (scanner would loop): {bad}")

    accepts = [min(acc, key=lambda t: prio[t]) if acc else -1 for acc in accepts_all]

    # live[s] = union of accepts_all over states reachable from s (incl. s itself),
    # computed by one reverse-topological fixpoint over the DFA graph.
    n = len(order)
    succ: list[frozenset[int]] = [frozenset(t for t in row if t != DEAD) for row in trans]
    live_sets: list[set[int]] = [set(acc) for acc in accepts_all]
    changed = True
    while changed:
        changed = False
        for s in range(n):
            before = len(live_sets[s])
            for nx in succ[s]:
                live_sets[s] |= live_sets[nx]
            if len(live_sets[s]) != before:
                changed = True

    lives = [frozenset(s) for s in live_sets]
    h_max = max((len(s) for s in lives), default=0)
    return ScannerDFA(
        start=0,
        trans=tuple(tuple(r) for r in trans),
        accept=tuple(accepts),
        accepts_all=tuple(accepts_all),
        live=tuple(lives),
        h_max=h_max,
    )
