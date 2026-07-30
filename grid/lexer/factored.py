"""Factored scanner: per-terminal DFAs + lazy product combination (0.3.x #4).

This path is the default behind dfa.build_scanner since the v0.3.0
full-corpus run; GRID_PERF_FACTORED_SCANNER=0 restores the eager union
subset construction (retained as the exactness oracle — see materialize).
One small byte DFA is built per terminal — memoized process-wide by
(pattern source, is_literal), the identity the schema compiler already
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
the union-DFA live set by the same disjointness), so no global live pass
ever runs on this path. The co-accessibility bit comes from
nfa._terminal_reach over the component NFA — the same NFA-derived live
computation build_scanner uses (the legacy DFA-graph alternatives and their
env flag were deleted after the 11.3k zero-divergence verify pass on
v0.3.0rc1, see CHANGELOG).

Over GRID_PERF_FACTORED_BUDGET product states the LazyProductDFA facade is
returned instead: it serves the ScannerDFA protocol with transitions
materialized on demand — token-length-bounded along trie paths, so blowup
states are never built. Lazy DFAs are gated off the Rust kernel
(trie/walk.py), off genN cache keys (mask/producer.py: demand-order state
ids are instance-local, and T2 is shared across template instances), and
off the full-enumeration reserve BFS (lalr/reserve.py dispatches to
``shortest_lexemes_factored``).

The product budget bounds only the PRODUCT; GRID_PERF_COMPONENT_BUDGET
(default 16384, "0" = cap disabled) bounds each COMPONENT's eager subset
construction the same way. The substring-union terminal family (13-17
unanchored keywords inside one JSON-string terminal — BAKEOFF.md F1) keeps a
persistent per-keyword matched bit, so its eager component discovers ~2^k
subsets (o83132: 268,803 states / ~87s / 2.2GB; o5195: >200k states with the
frontier still open) — while a WALK materializes at most one new subset per
scanned byte. On breach the component comes back as a LazyTerminalDFA
(subset states interned on demand, exact same subsets/annotations as the
eager build) and the scanner skips straight to the lazy product: by the
product/component projection the union DFA has at least as many states as
any component, so a breached component means the product would overrun any
smaller product budget anyway — materializing would just re-pay the
pathological subset cost the cap avoided. The 16384 default is fixed by the
manifest-set sweep (853 schemas / 21,223 unique terminal patterns): the
largest terminating NON-family component is 7,210 states, but the family's
own terminating members reach 15,865 inside products that compile DENSE
today (strmprivacy Stream — eager digest == factored digest — and
DataContract), and 16384 is the smallest power of two that keeps every
dense-today build byte-identical while every 2^k member still breaches
(in 1.8-11.9s, vs 87s..timeout to complete eagerly). A hypothetical
out-of-corpus component in (16384, 20000] whose product fits the product
budget would have compiled dense before this cap and now serves lazily —
masks identical, serving gated off the kernel;
GRID_PERF_COMPONENT_BUDGET=0 restores it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from grid import perf_flags
from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal
from grid.lexer.dfa import ScannerDFA
from grid.lexer.nfa import _NFABuilder, _terminal_reach
from grid.lexer.rx import _literal_node, _parse_regex
from grid.lexer.subset import (
    DEAD,
    SubsetBudgetExceeded,
    byte_classes,
    edges_by_class,
    eps_closure_fn,
    subset_construct,
)

_DEFAULT_BUDGET = 20_000
# sweep-fixed (module docstring): above every terminating component that
# compiles dense today (max 15,865), below every 2^k family member
_DEFAULT_COMPONENT_BUDGET = 16384
_COMPONENT_CAP = 4096   # wholesale-reset guard: per-schema key literals in a long-lived server


class ScannerComponent(Protocol):
    """Structural seam for product components (type-only): the four
    attributes + step() the lazy product (_annotate/_class_step), the bounded
    materializer, and the reserve BFS (shortest_lexemes_factored/
    _shortest_word) actually consume. TerminalDFA is the eager regex/literal
    case, LazyTerminalDFA the over-component-budget case; the held
    COUNTING_WINDOWS component type plugs in as a sibling without touching
    the product. ``accepting``/``co_acc`` are indexable for every state id
    ``step`` has returned (LazyTerminalDFA grows them on intern)."""

    class_of: tuple[int, ...]
    accepting: Sequence[bool]
    co_acc: Sequence[bool]
    matches_empty: bool

    def step(self, state: int, cls: int) -> int:
        """Successor state id (or DEAD) on byte-class ``cls``."""
        ...  # pragma: no cover - Protocol body


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

    def step(self, state: int, cls: int) -> int:
        return self.trans[state][cls]


class LazyTerminalDFA:
    """One terminal's byte DFA with subset states interned on demand — the
    over-GRID_PERF_COMPONENT_BUDGET regime (module docstring: the
    substring-union family discovers ~2^k subsets eagerly, at most one new
    subset per walked byte on demand).

    Same automaton as the eager TerminalDFA state-for-state, numbered in
    demand order instead of BFS order: ``step`` performs the identical
    per-class NFA move + eps-closure as subset.subset_construct, and
    ``accepting[s]`` (accept in subset) / ``co_acc[s]`` (NFA terminal-reach
    ORed over the subset — nfa._terminal_reach, the same live computation as
    the eager path) are pure functions of the subset value, so every
    product-recorded set (accepts_all / accept / live / h_max via
    LazyProductDFA._annotate) is EXACTLY what the eager component yields; a
    fresh instance driven in the materializer's FIFO/ascending-class order
    even reproduces the eager NUMBERING (tests pin ScannerDFA equality).

    Interning is locked (producer prefetch pools walk on threads; the
    LazyProductDFA._intern idiom): duplicate step computations are benign —
    equal subsets intern to one id — duplicate ids are not; annotation lists
    are appended before the id is published, so any id ``step`` returns is
    already indexable. Memo growth is bounded by distinct subsets demanded
    (<= 1 per scanned byte per component, trie-shared across walks)."""

    matches_empty: bool

    def __init__(
        self,
        start: frozenset[int],
        acc: int,
        eps_closure: Callable[[frozenset[int]], frozenset[int]],
        edge_by_class: dict[int, list[list[int] | None]],
        n_classes: int,
        class_of: list[int],
        reach: list[int],
    ) -> None:
        self._acc = acc
        self._eps_closure = eps_closure   # keeps the builder's eps_star memo
        self._edge_by_class = edge_by_class
        self._n_classes = n_classes
        self.class_of = tuple(class_of)
        self._reach = reach
        self._states: list[frozenset[int]] = [start]
        self._rows: list[list[int | None]] = [[None] * n_classes]
        self.accepting: list[bool] = [acc in start]
        self.co_acc: list[bool] = [any(reach[q] for q in start)]
        self.matches_empty = self.accepting[0]
        self._ids: dict[frozenset[int], int] = {start: 0}
        self._lock = threading.Lock()

    def step(self, state: int, cls: int) -> int:
        got = self._rows[state][cls]
        if got is None:
            # the subset_construct per-class move: union the members' class
            # destinations, eps-close, intern (DEAD when no member moves —
            # exactly the classes the eager row leaves DEAD)
            dsts: set[int] = set()
            for st in self._states[state]:
                per = self._edge_by_class.get(st)
                if per is not None:
                    lst = per[cls]
                    if lst is not None:
                        dsts.update(lst)
            got = self._intern(self._eps_closure(frozenset(dsts))) if dsts else DEAD
            self._rows[state][cls] = got   # racing writes store the same id
        return got

    def _intern(self, subset: frozenset[int]) -> int:
        got = self._ids.get(subset)
        if got is not None:
            return got
        with self._lock:
            got = self._ids.get(subset)
            if got is None:
                got = len(self._states)
                self._states.append(subset)
                self._rows.append([None] * self._n_classes)
                self.accepting.append(self._acc in subset)
                self.co_acc.append(any(self._reach[q] for q in subset))
                self._ids[subset] = got   # published last: lists are indexable first
        return got


_COMPONENTS: dict[tuple[str, bool, int | None], TerminalDFA | LazyTerminalDFA] = {}


def _nfa_artifacts(pattern: str, is_literal: bool):
    """The subset-free front half of a component build: NFA + byte classes +
    per-class edge index + eps-closure memo. Shared by the eager build and the
    direct lazy build (store breach-marker hits skip the doomed subset
    attempt), so both regimes construct from identical artifacts."""
    node = _literal_node(pattern) if is_literal else _parse_regex(pattern)
    b = _NFABuilder()
    s0, acc = b.build(node)

    # shared subset-construction core (grid/lexer/subset.py — the same helpers
    # behind build_scanner); byte classes cover this terminal's edge charsets
    # only, so they stay tiny.
    eps_closure = eps_closure_fn(b.eps)
    blocks, class_of = byte_classes(b.edges)
    n_classes = len(blocks)
    edge_by_class = edges_by_class(b.edges, class_of, n_classes)
    start = eps_closure(frozenset({s0}))
    return b, acc, start, eps_closure, class_of, n_classes, edge_by_class


def _lazy_component(pattern: str, is_literal: bool) -> LazyTerminalDFA:
    """Demand-interned component built DIRECTLY (no eager subset attempt) —
    the store's breach-marker warm path. Value-equal to the LazyTerminalDFA
    the breach branch below returns: same NFA artifacts, and every subset/
    annotation is a pure function of the demanded words (the eps memo starts
    cold instead of warm from the failed attempt — a speed detail only)."""
    b, acc, start, eps_closure, class_of, n_classes, edge_by_class = \
        _nfa_artifacts(pattern, is_literal)
    return LazyTerminalDFA(
        start, acc, eps_closure, edge_by_class, n_classes, class_of,
        _terminal_reach(b, {acc: 0}),
    )


def _build_component(
    pattern: str, is_literal: bool, budget: int | None,
) -> TerminalDFA | LazyTerminalDFA:
    # acceptance is a post-pass over the discovery order: order[i] is the
    # i-th subset, so the list is positionally identical to the legacy
    # in-loop append
    b, acc, start, eps_closure, class_of, n_classes, edge_by_class = \
        _nfa_artifacts(pattern, is_literal)
    try:
        order, trans = subset_construct(
            start, edge_by_class, eps_closure, n_classes, max_states=budget,
        )
    except SubsetBudgetExceeded:
        # component budget breach (substring-union family): discard the
        # partial subset arrays, keep the subset-free NFA artifacts — byte
        # classes, per-class edge index, the eps_star memo inside
        # eps_closure, and NFA accept-reachability — and intern subset
        # states on demand instead
        return LazyTerminalDFA(
            start, acc, eps_closure, edge_by_class, n_classes, class_of,
            _terminal_reach(b, {acc: 0}),
        )
    accepting = [acc in cur for cur in order]

    # co-accessibility: the same NFA-derived live computation as
    # dfa.build_scanner — reach[q] != 0 iff NFA state q can reach the accept
    # via eps/non-empty byte edges, and a subset state is co-accessible iff
    # any member is (Rabin-Scott, per component). Provably equal to a reverse
    # BFS over the component DFA graph (11.3k zero-divergence verify pass,
    # v0.3.0rc1); test_live_sets.py pins it with an independent forward-BFS
    # oracle.
    reach = _terminal_reach(b, {acc: 0})
    co_acc = [any(reach[q] for q in subset) for subset in order]

    return TerminalDFA(
        trans=tuple(tuple(r) for r in trans),
        class_of=tuple(class_of),
        accepting=tuple(accepting),
        co_acc=tuple(co_acc),
        matches_empty=accepting[0],
    )


def _component(
    pattern: str, is_literal: bool, budget: int | None = None,
) -> TerminalDFA | LazyTerminalDFA:
    """Memoized component build. ``budget`` is the component state budget:
    None resolves GRID_PERF_COMPONENT_BUDGET at call time (the reader maps
    the "0" kill switch to no cap); explicit 0 caps at zero states, i.e.
    every non-degenerate component comes back lazy (the all-lazy test hook,
    mirroring build_factored_scanner's product budget=0 convention). The memo
    keys on the RESOLVED budget so test-forced budgets never poison default
    builds."""
    if budget is None:
        budget = perf_flags.component_budget(_DEFAULT_COMPONENT_BUDGET)
    key = (pattern, is_literal, budget)
    got = _COMPONENTS.get(key)
    if got is None:
        if len(_COMPONENTS) >= _COMPONENT_CAP:
            _COMPONENTS.clear()
        # memo miss -> the on-disk component namespace (S3), then the build.
        # perf_flags is checked BEFORE importing the store so the flag-off
        # fast path never pays the grid.serving import chain (the E1
        # contract grid/jsonschema relies on); the store loader is a pure
        # passthrough to _build_component under either flag.
        if perf_flags.artifact_store_enabled() and perf_flags.store_components_enabled():
            from grid.serving.artifact_store import load_or_build_component

            built = load_or_build_component(pattern, is_literal, budget)
        else:
            built = _build_component(pattern, is_literal, budget)
        # setdefault: a concurrent duplicate build is benign, duplicate values
        # are equal (for a lazy duplicate the loser instance is discarded
        # before any caller holds it)
        got = _COMPONENTS.setdefault(key, built)
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

    def __init__(self, comps: Sequence[ScannerComponent], prio: dict[int, tuple[int, int]]) -> None:
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
                if (nc := comps[t].step(cs, cmap[t][g])) != DEAD
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
    component_budget: int | None = None,
) -> ScannerDFA | LazyProductDFA:
    """The default (GRID_PERF_FACTORED_SCANNER, "0" = legacy eager) path
    behind dfa.build_scanner. ``budget``/``component_budget`` default (None)
    to the GRID_PERF_FACTORED_BUDGET / GRID_PERF_COMPONENT_BUDGET call-time
    reads (see _component for the component budget's 0-vs-kill-switch
    convention)."""
    # components in terminal_order: the first GrammarInvalid from a bad regex
    # is raised for the same terminal as the eager path
    comps = [
        _component(terminals[name].pattern, terminals[name].is_literal, component_budget)
        for name in terminal_order
    ]
    empty = frozenset(tid for tid, c in enumerate(comps) if c.matches_empty)
    if empty:
        # same frozenset value + join as build_scanner: identical message text
        bad = ", ".join(terminal_order[t] for t in empty)
        raise GrammarInvalid(f"terminals match the empty string (scanner would loop): {bad}")
    prio = {tid: terminals[name].priority for tid, name in enumerate(terminal_order)}
    if budget is None:
        budget = perf_flags.factored_budget(_DEFAULT_BUDGET)
    product = LazyProductDFA(comps, prio)
    if any(isinstance(c, LazyTerminalDFA) for c in comps):
        # a component breached its budget, so the union DFA has at least that
        # many states (the product projects onto every component's reachable
        # set) — materializing would re-pay the pathological subset cost the
        # component cap just avoided, so serve the lazy product directly.
        # (Only if the true component size were inside (component budget,
        # product budget] could this skip flip a would-be-dense scanner to
        # lazy — the corpus sweep places no terminating component there; see
        # module docstring.)
        return product
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


def _shortest_word(comp: ScannerComponent) -> bytes | None:
    """Smallest-byte level-order BFS from the component start.

    Visits each distinct state at most once (``seen``), so on a
    LazyTerminalDFA the work is O(states within shortest-word depth) — for
    the substring-union family that is keyword depth, a few hundred states —
    and NEVER the eager 2^k construction (which explores every state
    regardless of depth; a visited-states cap could only trade this bounded
    exactness for that strictly larger cost, so there is none)."""
    frontier: list[tuple[int, bytes]] = [(0, b"")]
    seen = {0}
    while frontier:
        for st, path in frontier:
            if comp.accepting[st]:
                return path
        nxt: list[tuple[int, bytes]] = []
        for st, path in frontier:
            cof = comp.class_of
            for byte in range(256):
                ns = comp.step(st, cof[byte])
                if ns != DEAD and ns not in seen:
                    seen.add(ns)
                    nxt.append((ns, path + bytes([byte])))
        frontier = nxt
    return None
