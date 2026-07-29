"""Factored scanner: per-terminal DFAs + lazy product combination (0.3.x #4).

GRID_PERF_FACTORED_SCANNER=1 replaces build_scanner's eager union subset
construction. One small byte DFA is built per terminal — memoized process-wide
by (pattern source, is_literal), the identity the schema compiler already
dedupes terminals on — and the scanner state becomes a sparse tuple of
(terminal id, component state) pairs, materialized on demand.

Components are NOT trimmed: a component state that can no longer reach its
accept stays in the tuple until its subset dies. The sub-NFAs of the combined
automaton share no states, so the combined subset after any word is exactly
the disjoint union of the per-terminal subsets — the product below is the
eager union DFA state-for-state, and the bounded materializer reproduces
build_scanner's output EXACTLY, state numbering included (identical byte-class
partition: the joint refinement of the per-component partitions; identical
FIFO/min-byte-class discovery order). Per-state knowledge is local:
``accepts_all`` = the accepting components, ``live`` = the components whose
state can still reach accept (a per-component co-accessibility bit — equal to
the union-DFA live set by the same disjointness), so the global live fixpoint
never runs on this path. The co-accessibility bit itself honors
GRID_PERF_NFA_LIVE: on (default) it comes from dfa._terminal_reach over the
component NFA — the same NFA-derived live computation build_scanner uses —
while =0 falls back to a reverse BFS over the component DFA graph and
=verify cross-checks the two (see _build_component).

Over GRID_PERF_FACTORED_BUDGET product states the LazyProductDFA facade is
returned instead: it serves the ScannerDFA protocol with transitions
materialized on demand — token-length-bounded along trie paths, so blowup
states are never built. Lazy DFAs are gated off the Rust kernel
(trie/walk.py), off genN cache keys (mask/producer.py: demand-order state ids
are instance-local, and T2 is shared across template instances), and off the
full-enumeration reserve BFS (lalr/reserve.py dispatches to
``shortest_lexemes_factored``).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal
from grid.lexer.dfa import (
    DEAD,
    ScannerDFA,
    _literal_node,
    _NFABuilder,
    _parse_regex,
    _terminal_reach,
)

_DEFAULT_BUDGET = 20_000
_COMPONENT_CAP = 4096   # wholesale-reset guard: per-schema key literals in a long-lived server


@dataclass(frozen=True)
class TerminalDFA:
    """One terminal's byte DFA (untrimmed subset construction, start state 0).

    - ``trans[s][class_of[byte]]``: successor state or DEAD.
    - ``accepting[s]``: the terminal accepts exactly at ``s``.
    - ``co_acc[s]``: an accepting state is reachable from ``s`` in >= 0 bytes
      (False = zombie: the union DFA keeps such subsets alive, so the product
      must too — dropping them early would fire forced emissions one byte
      sooner than build_scanner's automaton).
    """

    trans: tuple[tuple[int, ...], ...]
    class_of: tuple[int, ...]
    accepting: tuple[bool, ...]
    co_acc: tuple[bool, ...]
    matches_empty: bool


_COMPONENTS: dict[tuple[str, bool, str], TerminalDFA] = {}


def _live_mode() -> str:
    """GRID_PERF_NFA_LIVE -> component co-accessibility computation:
    '0' = DFA-graph reverse BFS (the legacy live-set idiom, per component),
    'verify' = both + cross-check, anything else = NFA terminal-reach
    (dfa._terminal_reach, the GRID_PERF_NFA_LIVE default path). All three are
    provably equal (per-component Rabin-Scott, empty byte classes skipped),
    so cached components are interchangeable across modes."""
    raw = os.environ.get("GRID_PERF_NFA_LIVE", "1")
    if raw == "0":
        return "0"
    return "verify" if raw == "verify" else "nfa"


def _graph_co_acc(trans: list[list[int]], accepting: list[bool]) -> list[bool]:
    """Legacy (GRID_PERF_NFA_LIVE=0) co-accessibility: reverse reachability
    from the accepting states over the component DFA graph."""
    n = len(trans)
    preds: list[list[int]] = [[] for _ in range(n)]
    for s, row in enumerate(trans):
        for d in row:
            if d != DEAD:
                preds[d].append(s)
    co_acc = [False] * n
    stack = [s for s in range(n) if accepting[s]]
    for s in stack:
        co_acc[s] = True
    while stack:
        s = stack.pop()
        for p in preds[s]:
            if not co_acc[p]:
                co_acc[p] = True
                stack.append(p)
    return co_acc


def _build_component(pattern: str, is_literal: bool, live_mode: str) -> TerminalDFA:
    node = _literal_node(pattern) if is_literal else _parse_regex(pattern)
    b = _NFABuilder()
    s0, acc = b.build(node)

    eps_star: dict[int, frozenset[int]] = {}

    def _star(st0: int) -> frozenset[int]:
        got = eps_star.get(st0)
        if got is None:
            stack, seen = [st0], {st0}
            while stack:
                st = stack.pop()
                for nxt in b.eps.get(st, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            got = eps_star[st0] = frozenset(seen)
        return got

    def eps_closure(states) -> frozenset[int]:
        out: frozenset[int] = frozenset()
        for s in states:
            out |= _star(s)
        return out

    # per-component byte classes: same partition-refinement idiom as
    # build_scanner, over this terminal's edge charsets only (classes stay tiny)
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
    blocks.sort(key=min)
    class_of = [0] * 256
    for ci_, blk in enumerate(blocks):
        for c in blk:
            class_of[c] = ci_
    n_classes = len(blocks)
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

    start_set = eps_closure(frozenset({s0}))
    ids: dict[frozenset[int], int] = {start_set: 0}
    order = [start_set]
    trans: list[list[int]] = []
    accepting: list[bool] = []
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
        row = [DEAD] * n_classes
        for cl, dsts in sorted(by_class.items()):
            nxt = eps_closure(frozenset(dsts))
            if nxt not in ids:
                ids[nxt] = len(order)
                order.append(nxt)
            row[cl] = ids[nxt]
        trans.append(row)
        accepting.append(acc in cur)

    # co-accessibility: the same NFA-derived live computation as
    # dfa.build_scanner when GRID_PERF_NFA_LIVE is on — reach[q] != 0 iff NFA
    # state q can reach the accept via eps/non-empty byte edges, and a subset
    # state is co-accessible iff any member is (Rabin-Scott, per component).
    # GRID_PERF_NFA_LIVE=0 keeps the DFA-graph reverse BFS; 'verify' runs both.
    co_acc: list[bool]
    if live_mode != "0":
        reach = _terminal_reach(b, {acc: 0})
        co_acc = [any(reach[q] for q in subset) for subset in order]
        if live_mode == "verify":
            legacy = _graph_co_acc(trans, accepting)
            if co_acc != legacy:
                bad = [s for s in range(len(order)) if co_acc[s] != legacy[s]]
                raise AssertionError(
                    f"GRID_PERF_NFA_LIVE=verify: factored component co_acc for "
                    f"pattern {pattern!r} diverges from the DFA-graph BFS at "
                    f"states {bad}")
    else:
        co_acc = _graph_co_acc(trans, accepting)

    return TerminalDFA(
        trans=tuple(tuple(r) for r in trans),
        class_of=tuple(class_of),
        accepting=tuple(accepting),
        co_acc=tuple(co_acc),
        matches_empty=accepting[0],
    )


def _component(pattern: str, is_literal: bool, live_mode: str) -> TerminalDFA:
    # live_mode in the key so 'verify' actually verifies on a cold entry and
    # a mode flip mid-process (tests) never serves the other path's build; the
    # values themselves are equal across modes, so hits stay interchangeable
    key = (pattern, is_literal, live_mode)
    got = _COMPONENTS.get(key)
    if got is None:
        if len(_COMPONENTS) >= _COMPONENT_CAP:
            _COMPONENTS.clear()
        # setdefault: a concurrent duplicate build is benign, duplicate values are equal
        got = _COMPONENTS.setdefault(key, _build_component(pattern, is_literal, live_mode))
    return got


class _LazyRow:
    """One state's transition row: ``row[byte]`` materializes the successor."""

    __slots__ = ("_p", "_sid")

    def __init__(self, p: LazyProductDFA, sid: int) -> None:
        self._p = p
        self._sid = sid

    def __getitem__(self, byte: int) -> int:
        p = self._p
        return p._class_step(self._sid, p._gclass_of[byte])


class _LazyTrans:
    __slots__ = ("_p",)

    def __init__(self, p: LazyProductDFA) -> None:
        self._p = p

    def __getitem__(self, sid: int) -> _LazyRow:
        return _LazyRow(self._p, sid)


class LazyProductDFA:
    """ScannerDFA-protocol facade over the per-terminal component product.

    State = canonical sparse tuple ((tid, comp_state), ...) ascending by tid,
    interned to dense ids in creation order; ``accept``/``accepts_all``/
    ``live`` are list-backed and valid for every id already handed out.
    State creation is locked (producer prefetch pools walk on threads);
    duplicate-compute races are benign, duplicate ids are not.
    """

    lazy = True
    start = 0

    def __init__(self, comps: list[TerminalDFA], prio: dict[int, tuple[int, int]]) -> None:
        self.comps = comps
        self._prio = prio
        # global byte classes: the joint refinement of the component partitions,
        # ordered by first occurrence = ascending min byte (the eager builder's
        # block order — this makes the materialized numbering match exactly)
        keys: dict[tuple[int, ...], int] = {}
        gclass_of = [0] * 256
        reps: list[int] = []
        for byte in range(256):
            k = tuple(c.class_of[byte] for c in comps)
            g = keys.get(k)
            if g is None:
                g = keys[k] = len(reps)
                reps.append(byte)
            gclass_of[byte] = g
        self._gclass_of = tuple(gclass_of)
        self._n_g = len(reps)
        self._cmap = [tuple(c.class_of[rb] for rb in reps) for c in comps]

        start_state = tuple((t, 0) for t in range(len(comps)))
        self._states: list[tuple[tuple[int, int], ...]] = [start_state]
        self._crows: list[list[int | None]] = [[None] * self._n_g]
        self.accept: list[int] = []
        self.accepts_all: list[frozenset[int]] = []
        self.live: list[frozenset[int]] = []
        self._annotate(start_state)
        self._ids: dict[tuple[tuple[int, int], ...], int] = {start_state: 0}
        # live is monotone non-increasing along product transitions, so the
        # start state carries the maximum (INV-LEX1's H_max) — no global pass
        self.h_max = len(self.live[0])
        self._lock = threading.Lock()
        self.trans = _LazyTrans(self)

    def _annotate(self, st: tuple[tuple[int, int], ...]) -> None:
        comps = self.comps
        acc = frozenset(t for t, cs in st if comps[t].accepting[cs])
        self.accepts_all.append(acc)
        self.accept.append(min(acc, key=lambda t: self._prio[t]) if acc else -1)
        self.live.append(frozenset(t for t, cs in st if comps[t].co_acc[cs]))

    def _class_step(self, sid: int, g: int) -> int:
        got = self._crows[sid][g]
        if got is None:
            comps = self.comps
            cmap = self._cmap
            nxt = [
                (t, nc) for t, cs in self._states[sid]
                if (nc := comps[t].trans[cs][cmap[t][g]]) != DEAD
            ]
            got = self._intern(tuple(nxt)) if nxt else DEAD
            self._crows[sid][g] = got
        return got

    def _intern(self, st: tuple[tuple[int, int], ...]) -> int:
        got = self._ids.get(st)
        if got is not None:
            return got
        with self._lock:
            got = self._ids.get(st)
            if got is None:
                got = len(self._states)
                self._states.append(st)
                self._crows.append([None] * self._n_g)
                self._annotate(st)
                self._ids[st] = got   # published last: lists are indexable first
        return got

    # -- ScannerDFA protocol (grid/lexer/dfa.py semantics, verbatim) ---------

    def next(self, state: int, byte: int) -> int:
        return self._class_step(state, self._gclass_of[byte])

    def scan_state(self, remainder: bytes) -> int:
        st = 0
        step, gof = self._class_step, self._gclass_of
        for b in remainder:
            st = step(st, gof[b])
            if st == DEAD:
                return DEAD
        return st

    def scan_with_last_accept(self, remainder: bytes) -> tuple[int, int, int]:
        st = 0
        step, gof, accept = self._class_step, self._gclass_of, self.accept
        length, p = 0, -1
        for i, b in enumerate(remainder):
            st = step(st, gof[b])
            if st == DEAD:
                return DEAD, length, p
            if accept[st] != -1:
                length, p = i + 1, st
        return st, length, p

    # -- bounded materialization ---------------------------------------------

    def materialize(self, budget: int) -> ScannerDFA | None:
        """BFS in the eager builder's discovery order (FIFO states, byte
        classes ascending by min byte): within budget the result equals
        build_scanner's ScannerDFA EXACTLY — numbering included — by the
        product/subset bijection. None on breach (already-materialized states
        stay; the facade then serves them demand-order)."""
        i = 0
        n_g = self._n_g
        step = self._class_step
        while i < len(self._states):
            if len(self._states) > budget:
                return None
            for g in range(n_g):
                step(i, g)
            i += 1
        gof = self._gclass_of
        trans = tuple(tuple(row[g] for g in gof) for row in self._crows)
        lives = tuple(self.live)
        return ScannerDFA(
            start=0,
            trans=trans,
            accept=tuple(self.accept),
            accepts_all=tuple(self.accepts_all),
            live=lives,
            h_max=max((len(s) for s in lives), default=0),
        )


def build_factored_scanner(
    terminals: dict[str, Terminal],
    terminal_order: tuple[str, ...],
    budget: int | None = None,
) -> ScannerDFA | LazyProductDFA:
    """The GRID_PERF_FACTORED_SCANNER=1 path behind dfa.build_scanner."""
    # components in terminal_order: the first GrammarInvalid from a bad regex
    # is raised for the same terminal as the eager path
    live_mode = _live_mode()
    comps = [
        _component(terminals[name].pattern, terminals[name].is_literal, live_mode)
        for name in terminal_order
    ]
    empty = frozenset(tid for tid, c in enumerate(comps) if c.matches_empty)
    if empty:
        # same frozenset value + join as build_scanner: identical message text
        bad = ", ".join(terminal_order[t] for t in empty)
        raise GrammarInvalid(f"terminals match the empty string (scanner would loop): {bad}")
    prio = {tid: terminals[name].priority for tid, name in enumerate(terminal_order)}
    if budget is None:
        budget = int(os.environ.get("GRID_PERF_FACTORED_BUDGET", str(_DEFAULT_BUDGET)))
    product = LazyProductDFA(comps, prio)
    dense = product.materialize(budget)
    return dense if dense is not None else product


def shortest_lexemes_factored(dfa: LazyProductDFA) -> dict[int, bytes]:
    """Per-component BFS: the lexicographically-least shortest accepted word
    per terminal — exactly what reserve.shortest_lexemes' union-DFA BFS
    returns (acceptance is a per-component state property; smallest-byte
    level order is lexicographic order) — without touching the product."""
    out: dict[int, bytes] = {}
    for tid, comp in enumerate(dfa.comps):
        if not comp.co_acc[0]:
            continue   # empty language: the union BFS never sees this terminal
        word = _shortest_word(comp)
        if word is not None:
            out[tid] = word
    return out


def _shortest_word(comp: TerminalDFA) -> bytes | None:
    frontier: list[tuple[int, bytes]] = [(0, b"")]
    seen = {0}
    while frontier:
        for st, path in frontier:
            if comp.accepting[st]:
                return path
        nxt: list[tuple[int, bytes]] = []
        for st, path in frontier:
            row = comp.trans[st]
            cof = comp.class_of
            for byte in range(256):
                ns = row[cof[byte]]
                if ns != DEAD and ns not in seen:
                    seen.add(ns)
                    nxt.append((ns, path + bytes([byte])))
        frontier = nxt
    return None
