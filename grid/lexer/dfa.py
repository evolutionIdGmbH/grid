"""Terminal regexes -> combined scanner DFA (DESIGN.md SS3 lexer/, E7).

Pipeline: grid-regex subset (rx.py) -> NFA (Thompson, nfa.py) -> combined NFA
(one tagged accept per terminal) -> subset-construction DFA over bytes
(shared core in subset.py). This module holds the ScannerDFA artifact and the
build_scanner entry point (+ the factored-path dispatch), and re-exports the
pipeline's historically-public names (DEAD, _parse_regex, _literal_node,
_NFABuilder, _terminal_reach) — importers treat dfa.py as the lexer's stable
facade; keep it that way.

Scanner DFA state knowledge:
- ``accept[state]``: the winning terminal if the scan stopped here (maximal munch
  resolves length; at equal length, literal terminals beat named ones, then
  declaration order — Terminal.priority).
- ``live[state]``: the set of terminals still reachable from this state — the
  lexer hypothesis set (E7). INV-LEX1's H_max is the max |live| over states,
  computed at build time from NFA terminal-reachability accumulated per subset
  state (_terminal_reach; the legacy DFA-graph fixpoint and its env flag were
  deleted after the 11.3k zero-divergence verify pass on v0.3.0rc1, see
  CHANGELOG — tests/lexer/test_live_sets.py keeps an independent forward-BFS
  oracle; L3 identifier categories add at most +1 hypothesis by
  construction).

All automata operate on BYTES: patterns are encoded latin-1; multi-byte UTF-8
in identifiers enters through byte classes (e.g. [\\x80-\\xff]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from grid import perf_flags
from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal
from grid.lexer.nfa import _NFA, _NFABuilder, _terminal_reach  # noqa: F401  (re-export)
from grid.lexer.rx import _literal_node, _Node, _parse_regex
from grid.lexer.subset import (
    DEAD,
    byte_classes,
    edges_by_class,
    eps_closure_fn,
    subset_construct,
)

if TYPE_CHECKING:
    # guard is mandatory: factored.py imports ScannerDFA from this module at
    # runtime, so an unguarded import would cycle
    from grid.lexer.factored import LazyProductDFA


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

    def scan_with_last_accept(self, remainder: bytes) -> tuple[int, int, int]:
        """One-pass scan returning ``(q, l, p)``:

        - ``q``: state after the FULL remainder (DEAD if the scan dies; every
          longer prefix of a dead scan is dead too, so ``l``/``p`` are final),
        - ``l``: length of the LONGEST accepting prefix (0 if none — the empty
          prefix never accepts: empty-matching terminals are rejected at build),
        - ``p``: state after ``remainder[:l]`` (-1 when no prefix accepts).

        Foundation for the genN cache-key normal form (mask/producer.cache_key):
        under the lexicon-visibility guard, remainders with equal ``(p, q)`` and
        equal post-accept suffix ``v = remainder[l:]`` are walk-indistinguishable.
        Pure; differentially bound to per-prefix re-scanning in
        tests/lexer/test_scan_last_accept.py."""
        st = self.start
        length, p = 0, -1
        for i, b in enumerate(remainder):
            st = self.trans[st][b]
            if st == DEAD:
                return DEAD, length, p
            if self.accept[st] != -1:
                length, p = i + 1, st
        return st, length, p


def build_scanner(
    terminals: dict[str, Terminal],
    terminal_order: tuple[str, ...],
    *,
    factored: bool | None = None,
) -> ScannerDFA | LazyProductDFA:
    """Combined NFA over all terminals -> subset-construction byte DFA.

    ``factored`` (None = read GRID_PERF_FACTORED_SCANNER, default ON since
    the v0.3.0 full-corpus run) selects the per-terminal-DFA product path
    (grid/lexer/factored.py), which may return a ScannerDFA-protocol lazy
    facade instead of an eager ScannerDFA when the product exceeds its state
    budget. GRID_PERF_FACTORED_SCANNER=0 restores the eager union builder
    below — kept as the factored path's exactness oracle (LazyProductDFA.
    materialize reproduces it exactly, numbering included). Live sets on
    both paths come from the one NFA terminal-reach computation
    (_terminal_reach; per component on the factored path — see
    factored.py)."""
    if factored is None:
        factored = perf_flags.factored_scanner_enabled()
    if factored:
        from grid.lexer.factored import build_factored_scanner

        return build_factored_scanner(terminals, terminal_order)
    b = _NFABuilder()
    root = b.new()
    accept_terminal: dict[int, int] = {}  # NFA accept state -> terminal id
    for tid, name in enumerate(terminal_order):
        t = terminals[name]
        if t.is_literal:
            node: _Node = _literal_node(t.pattern)
        else:
            node = _parse_regex(t.pattern)
        s, a = b.build(node)
        b.add_eps(root, s)
        accept_terminal[a] = tid

    term_reach = _terminal_reach(b, accept_terminal)
    prio = {tid: terminals[name].priority for tid, name in enumerate(terminal_order)}

    # shared subset-construction core (grid/lexer/subset.py): eps-closure
    # memoization, byte-class alphabet compression, per-class edge index,
    # FIFO subset loop over class-compressed rows — the same helpers behind
    # factored._build_component.
    eps_closure = eps_closure_fn(b.eps)
    blocks, class_of = byte_classes(b.edges)
    n_classes = len(blocks)
    order, class_rows = subset_construct(
        eps_closure(frozenset({root})),
        edges_by_class(b.edges, class_of, n_classes),
        eps_closure,
        n_classes,
    )

    # post-passes over the discovery order (each depends only on the subset
    # order[i], so the lists are positionally identical to the legacy in-loop
    # annotation):
    # - 256-wide rows: class_of[c] == cl exactly when c in blocks[cl], and
    #   transition-free classes hold DEAD in the class row, so the expansion
    #   reproduces the legacy sparse per-byte writes row-for-row;
    # - live(S) = OR of term_reach over S: the subset state after a word is
    #   exactly the NFA states reachable via it (Rabin-Scott), so terminal-
    #   accept reachability distributes over the union;
    # - accepts_all[i] = the tagged accept states inside order[i].
    trans = [[crow[cl] for cl in class_of] for crow in class_rows]
    accepts_all = [
        frozenset(accept_terminal[st] for st in cur if st in accept_terminal)
        for cur in order
    ]
    live_masks: list[int] = []
    for cur in order:
        mask = 0
        for st in cur:
            mask |= term_reach[st]
        live_masks.append(mask)

    if accepts_all[0]:
        bad = ", ".join(terminal_order[t] for t in accepts_all[0])
        raise GrammarInvalid(f"terminals match the empty string (scanner would loop): {bad}")

    accepts = [min(acc, key=lambda t: prio[t]) if acc else -1 for acc in accepts_all]

    # equal masks share one frozenset (deep window chains repeat one live set)
    by_mask: dict[int, frozenset[int]] = {}
    lives: list[frozenset[int]] = []
    for m in live_masks:
        got = by_mask.get(m)
        if got is None:
            tids: list[int] = []
            r = m
            while r:
                low = r & -r
                tids.append(low.bit_length() - 1)
                r ^= low
            got = by_mask[m] = frozenset(tids)
        lives.append(got)
    h_max = max((len(s) for s in lives), default=0)
    return ScannerDFA(
        start=0,
        trans=tuple(tuple(r) for r in trans),
        accept=tuple(accepts),
        accepts_all=tuple(accepts_all),
        live=tuple(lives),
        h_max=h_max,
    )
