"""Thompson byte-NFAs + terminal-accept reachability (lexer pipeline stage 2).

Moved verbatim from grid/lexer/dfa.py, which remains the import facade.
_terminal_reach is THE live-set computation for both scanner paths: the
eager union DFA ORs it per subset state, the factored path runs it per
component NFA (the legacy DFA-graph fixpoint and its env flag were deleted
after the 11.3k zero-divergence verify pass on v0.3.0rc1, see CHANGELOG).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid.lexer.rx import _Node


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


def _terminal_reach(b: _NFABuilder, accept_terminal: dict[int, int]) -> list[int]:
    """term_reach[q] = bitmask of terminals whose accept state is reachable from
    NFA state q via eps/byte edges (>= 0 bytes). One reverse DFS per accept
    state; per-terminal sub-NFAs are disjoint (fresh states per build call), so
    total work is O(NFA edges). Empty byte classes (e.g. [^\\x00-\\xff]) produce
    edges the DFA can never take — following them in reverse would over-report
    live terminals, so they are skipped."""
    rev: list[list[int]] = [[] for _ in range(b.n)]
    for src, dsts in b.eps.items():
        for dst in dsts:
            rev[dst].append(src)
    for src, edges in b.edges.items():
        for chars, dst in edges:
            if not chars:
                continue
            rev[dst].append(src)
    reach: list[int] = [0] * b.n
    for acc, tid in accept_terminal.items():
        bit = 1 << tid
        stack = [acc]
        while stack:
            q = stack.pop()
            if reach[q] & bit:
                continue
            reach[q] |= bit
            stack.extend(rev[q])
    return reach
