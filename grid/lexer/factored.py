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
states are never built. Lazy DFAs are gated off genN cache keys
(mask/producer.py: demand-order state ids are instance-local, and T2 is
shared across template instances) and off the full-enumeration reserve BFS
(lalr/reserve.py dispatches to ``shortest_lexemes_factored``). Trie walks
serve through the grid_core v8 in-kernel lazy product (GRID_PERF_KERNEL_LAZY,
trie/walk.py; component payloads via ``kernel_lazy_payload`` below, this
facade remaining the executable specification and the fallback), while
RustVerdicts stays gated off — CD re-checks and guide advance still consume
this facade directly. Counting products (GRID_PERF_COUNTING; counters != ())
stay off the kernel entirely — Python spec walk until kernel counter frames
(P4 phase 2) land.

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

import itertools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from grid import perf_flags
from grid.errors import GrammarInvalid
from grid.grammar.spec import Terminal
from grid.lexer.counting import _MAX_COUNTERS, CountingTerminalDFA, build_counting_component
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

# counting components (GRID_PERF_COUNTING) memoize separately so the plain
# memo above stays byte-identical flag-off; None = "no eligible loops / fell
# back" is a cached outcome (the caller then uses the plain component)
_COUNTING_COMPONENTS: dict[tuple[str, int | None], CountingTerminalDFA | None] = {}


def _build_component(
    pattern: str, is_literal: bool, budget: int | None,
) -> TerminalDFA | LazyTerminalDFA:
    node = _literal_node(pattern) if is_literal else _parse_regex(pattern)
    b = _NFABuilder()
    s0, acc = b.build(node)

    # shared subset-construction core (grid/lexer/subset.py — the same helpers
    # behind build_scanner); byte classes cover this terminal's edge charsets
    # only, so they stay tiny. Acceptance is a post-pass over the discovery
    # order: order[i] is the i-th subset, so the list is positionally
    # identical to the legacy in-loop append.
    eps_closure = eps_closure_fn(b.eps)
    blocks, class_of = byte_classes(b.edges)
    n_classes = len(blocks)
    edge_by_class = edges_by_class(b.edges, class_of, n_classes)
    start = eps_closure(frozenset({s0}))
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
        # setdefault: a concurrent duplicate build is benign, duplicate values
        # are equal (for a lazy duplicate the loser instance is discarded
        # before any caller holds it)
        got = _COMPONENTS.setdefault(key, _build_component(pattern, is_literal, budget))
    return got


def _counting_component(
    pattern: str, budget: int | None = None,
) -> CountingTerminalDFA | None:
    """Memoized counting-component build (non-literal patterns only; the
    caller falls back to _component on None). The memo caches the None
    outcome too — eligibility lowering re-runs are pure waste — and is keyed
    by (pattern, resolved budget) ONLY: the _MAX_COUNTERS selection happens
    at product-assembly time and never leaks into this memo (a dropped
    terminal in one schema must not poison another schema's build; the
    _COMPONENT_CAP wholesale reset is likewise selection-agnostic)."""
    if budget is None:
        budget = perf_flags.component_budget(_DEFAULT_COMPONENT_BUDGET)
    key = (pattern, budget)
    if key in _COUNTING_COMPONENTS:   # None is a valid cached outcome
        return _COUNTING_COMPONENTS[key]
    if len(_COUNTING_COMPONENTS) >= _COMPONENT_CAP:
        _COUNTING_COMPONENTS.clear()
    return _COUNTING_COMPONENTS.setdefault(key, build_counting_component(pattern, budget))


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

    Counting members (GRID_PERF_COUNTING; CountingTerminalDFA components)
    extend the product with a global counter table: ``counters`` is the
    concatenation of member counter tables in tid order and each member's
    local cids are offset into it. Transitions on counting products go
    through ``step(sid, counts, byte)``: the per-(sid, gclass) cache stores a
    count-independent transition PLAN (per-member variant tables with cids
    remapped, drop-resets precomputed) — never a single successor id, which
    for a guarded transition would be the forbidden wrong-mask class — and
    the plan is evaluated under the live counts per step. ``_class_step``
    (the count-blind protocol surface) asserts the plan is guard-free before
    serving it. materialize() lowers the plans to the held eager format:
    ScannerDFA.guard_rows keyed (state, byte) with variants (conds, dst,
    ops), so every ported runtime dispatcher works unchanged.
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

        # global counter table: member tables concatenated in tid order,
        # member-local cid k -> global cid _coffs[tid] + k. () = plain product
        # (every code path below then matches the pre-counting behavior).
        self._coffs: dict[int, int] = {}
        counters: list[tuple[int, int]] = []
        for t, c in enumerate(comps):
            cc = getattr(c, "counters", ())
            if cc:
                self._coffs[t] = len(counters)
                counters.extend(cc)
        self.counters: tuple[tuple[int, int], ...] = tuple(counters)

        start_state = tuple((t, 0) for t in range(len(comps)))
        self._states: list[tuple[tuple[int, int], ...]] = [start_state]
        self._crows: list[list[int | None]] = [[None] * self._n_g]
        self._prows: list[list[tuple | None]] = [[None] * self._n_g] if counters else []
        self.accept: list[int] = []
        self.accepts_all: list[frozenset[int]] = []
        self.live: list[frozenset[int]] = []
        self._annotate(start_state)
        self._ids: dict[tuple[tuple[int, int], ...], int] = {start_state: 0}
        # live is monotone non-increasing along product transitions (variant
        # successors included), so the start state carries the maximum
        # (INV-LEX1's H_max) — no global pass
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
        if self.counters:
            # counting product: transitions resolve through the count-aware
            # plan; the count-blind protocol surface (trans facade) is only
            # valid where the plan is guard- and op-free
            plan = self._plan(sid, g)
            assert plan[0] == 0, \
                "counting product: count-dependent transition needs step()"
            return plan[1]
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

    def _plan(self, sid: int, g: int) -> tuple:
        """Count-independent transition plan for (sid, gclass) — cached, and
        the ONLY thing cached for counting products (caching a successor id
        for a guarded transition is the forbidden wrong-mask class).

        Shapes: ``(0, dst_sid)`` when no member step is count-dependent or
        op-carrying (dst interned eagerly — safe, it is count-independent);
        ``(1, entries, static_ops)`` otherwise, with entries per surviving
        member either ``(0, tid, dst_comp_state)`` (fixed) or ``(1, tid,
        variants, drop_resets)`` (variants = the member's guard_rows row with
        local cids remapped global, covering the whole count space with DEAD
        boxes explicit; drop_resets = reset ops for the member's counters
        occupied at its current state, applied when it dies under the live
        counts). static_ops carries the count-INDEPENDENT resets: members
        whose fixed step drops them while their loop region was occupied."""
        got = self._prows[sid][g]
        if got is None:
            comps, cmap, coffs = self.comps, self._cmap, self._coffs
            entries: list[tuple] = []
            static_ops: list[tuple[int, int]] = []
            guarded = False
            for t, cs in self._states[sid]:
                cls = cmap[t][g]
                comp = comps[t]
                off = coffs.get(t)
                if off is None:
                    nc = comp.step(cs, cls)
                    if nc != DEAD:
                        entries.append((0, t, nc))
                    continue
                var = comp.guard_rows.get((cs, cls))
                if var is None:
                    nc = comp.trans[cs][cls]
                    if nc != DEAD:
                        entries.append((0, t, nc))
                    else:
                        occ = comp.occupied[cs]
                        if occ:
                            # member drops on this class regardless of counts:
                            # its counters reset to canonical 0
                            static_ops.extend((cid + off, 2) for cid in sorted(occ))
                    continue
                guarded = True
                remapped = tuple(
                    (
                        tuple((cid + off, lo, hi) for cid, lo, hi in conds),
                        dst,
                        tuple((cid + off, op) for cid, op in ops),
                    )
                    for conds, dst, ops in var
                )
                drop_resets = tuple((cid + off, 2) for cid in sorted(comp.occupied[cs]))
                entries.append((1, t, remapped, drop_resets))
            if not guarded and not static_ops:
                nxt = tuple((t, nc) for _k, t, nc in entries)
                got = (0, self._intern(nxt) if nxt else DEAD)
            else:
                got = (1, tuple(entries), tuple(static_ops))
            self._prows[sid][g] = got   # racing writes store equal plans
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
                if self.counters:
                    self._prows.append([None] * self._n_g)
                self._annotate(st)
                self._ids[st] = got   # published last: lists are indexable first
        return got

    # -- ScannerDFA protocol (grid/lexer/dfa.py semantics, verbatim) ---------

    def next(self, state: int, byte: int) -> int:
        assert not self.counters, "counting DFA: hand-stepping must go through step()"
        return self._class_step(state, self._gclass_of[byte])

    def zero_counts(self) -> tuple[int, ...]:
        return (0,) * len(self.counters)

    def step(self, sid: int, counts: tuple[int, ...], byte: int) -> tuple[int, tuple[int, ...]]:
        """One counter-aware byte step: (state', counts'). Counter-free
        products reduce to the plain class step (counts pass through)."""
        g = self._gclass_of[byte]
        if not self.counters:
            return self._class_step(sid, g), counts
        plan = self._plan(sid, g)
        if plan[0] == 0:
            return plan[1], counts
        _tag, entries, ops_all = plan
        nxt: list[tuple[int, int]] = []
        extra_ops: list[tuple[int, int]] = []
        for e in entries:
            if e[0] == 0:
                nxt.append((e[1], e[2]))
                continue
            _k, t, variants, drop_resets = e
            for conds, dst, ops in variants:
                ok = True
                for cid, lo, hi in conds:
                    c = counts[cid]
                    if c < lo or c > hi:
                        ok = False
                        break
                if not ok:
                    continue
                if dst != DEAD:
                    nxt.append((t, dst))
                    if ops:
                        extra_ops.extend(ops)
                elif drop_resets:
                    extra_ops.extend(drop_resets)
                break   # variant boxes are disjoint: first match is the match
        if not nxt:
            return DEAD, counts
        if ops_all or extra_ops:
            lst = list(counts)
            for cid, op in itertools.chain(ops_all, extra_ops):
                if op == 1:
                    m, n = self.counters[cid]
                    cap = n if n >= 0 else m
                    v = lst[cid] + 1
                    lst[cid] = cap if v > cap else v
                else:
                    lst[cid] = 0
            counts = tuple(lst)
        return self._intern(tuple(nxt)), counts

    def scan_state(self, remainder: bytes) -> int:
        if self.counters:
            st, counts = 0, self.zero_counts()
            for b in remainder:
                st, counts = self.step(st, counts, b)
                if st == DEAD:
                    return DEAD
            return st
        st = 0
        step, gof = self._class_step, self._gclass_of
        for b in remainder:
            st = step(st, gof[b])
            if st == DEAD:
                return DEAD
        return st

    def scan_with_last_accept(self, remainder: bytes) -> tuple[int, int, int]:
        if self.counters:
            q, _cq, length, p, _cp = self.scan_full(remainder)
            return q, length, p
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

    def scan_full(self, remainder: bytes) -> tuple[int, tuple[int, ...], int, int, tuple[int, ...]]:
        """Counter-carrying scan_with_last_accept (dfa.ScannerDFA.scan_full
        semantics verbatim): ``(q, counts_q, l, p, counts_p)``; ``counts_p``
        is () when no prefix accepts."""
        st, counts = 0, self.zero_counts()
        length, p = 0, -1
        counts_p: tuple[int, ...] = ()
        accept = self.accept
        for i, b in enumerate(remainder):
            st, counts = self.step(st, counts, b)
            if st == DEAD:
                return DEAD, counts, length, p, counts_p
            if accept[st] != -1:
                length, p, counts_p = i + 1, st, counts
        return st, counts, length, p, counts_p

    # -- bounded materialization ---------------------------------------------

    def materialize(self, budget: int) -> ScannerDFA | None:
        """BFS in the eager builder's discovery order (FIFO states, byte
        classes ascending by min byte): within budget the result equals
        build_scanner's ScannerDFA EXACTLY — numbering included — by the
        product/subset bijection. None on breach (already-materialized states
        stay; the facade then serves them demand-order). Counting products
        (GRID_PERF_COUNTING) lower their transition plans to the held eager
        format instead — counters + guard_rows keyed (state, byte) — there is
        no expanded-numbering oracle for those (equivalence to the expanded
        DFA is behavioral, tests/lexer/test_counting_windows.py)."""
        if self.counters:
            return self._materialize_counting(budget)
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

    def _materialize_counting(self, budget: int) -> ScannerDFA | None:
        """Counting-product materialization: same FIFO/ascending-class BFS,
        with count-dependent cells lowered to product-level variant tables.
        Per cell the guarded members' variant boxes are crossed (ascending
        tid; per-member variant order preserved) — each combo is a box over
        the global counter space whose successor tuple drops the members dead
        in it; combos where every member drops are skipped, so uncovered
        count regions fall through to DEAD exactly like the held step().
        Successor states intern in combo-enumeration order (deterministic:
        genN keys embed materialized state ids). The dense trans value on a
        guarded cell is the largest-successor variant (the held dense-row
        convention — advisory only; ScannerDFA.next asserts)."""
        i = 0
        n_g = self._n_g
        guard_cells: dict[tuple[int, int], tuple] = {}   # (sid, gclass) -> variants
        while i < len(self._states):
            if len(self._states) > budget:
                return None
            for g in range(n_g):
                plan = self._plan(i, g)
                if plan[0] == 0:
                    self._crows[i][g] = plan[1]
                    continue
                variants, dense = self._lower_plan(plan)
                self._crows[i][g] = dense
                if variants:
                    guard_cells[(i, g)] = variants
            i += 1
        gof = self._gclass_of
        trans = tuple(tuple(row[g] for g in gof) for row in self._crows)
        guard_rows: dict[tuple[int, int], tuple] = {}
        for (sid, g), vt in guard_cells.items():
            for byte in range(256):
                if gof[byte] == g:
                    guard_rows[(sid, byte)] = vt
        lives = tuple(self.live)
        return ScannerDFA(
            start=0,
            trans=trans,
            accept=tuple(self.accept),
            accepts_all=tuple(self.accepts_all),
            live=lives,
            h_max=max((len(s) for s in lives), default=0),
            counters=self.counters,
            guard_rows=guard_rows,
        )

    def _lower_plan(self, plan: tuple) -> tuple[tuple, int]:
        """One count-dependent plan -> (product variants, dense advisory)."""
        _tag, entries, static_ops = plan
        var_lists = [e[2] for e in entries if e[0] == 1]
        out: list[tuple[tuple, int, tuple]] = []
        best: tuple[int, int] | None = None   # (n_members, dst_sid)
        for combo in itertools.product(*var_lists):
            it = iter(combo)
            nxt: list[tuple[int, int]] = []
            conds: list[tuple[int, int, int]] = []
            ops: list[tuple[int, int]] = list(static_ops)
            for e in entries:
                if e[0] == 0:
                    nxt.append((e[1], e[2]))
                    continue
                vconds, dst, vops = next(it)
                conds.extend(vconds)
                if dst != DEAD:
                    nxt.append((e[1], dst))
                    ops.extend(vops)
                else:
                    ops.extend(e[3])   # drop_resets
            if not nxt:
                continue   # dead under these counts: step() falls through
            dst_id = self._intern(tuple(nxt))
            out.append((tuple(conds), dst_id, tuple(ops)))
            if best is None or len(nxt) > best[0]:
                best = (len(nxt), dst_id)
        return tuple(out), best[1] if best is not None else DEAD


def build_factored_scanner(
    terminals: dict[str, Terminal],
    terminal_order: tuple[str, ...],
    budget: int | None = None,
    component_budget: int | None = None,
    counting: bool | None = None,
) -> ScannerDFA | LazyProductDFA:
    """The default (GRID_PERF_FACTORED_SCANNER, "0" = legacy eager) path
    behind dfa.build_scanner. ``budget``/``component_budget`` default (None)
    to the GRID_PERF_FACTORED_BUDGET / GRID_PERF_COMPONENT_BUDGET call-time
    reads (see _component for the component budget's 0-vs-kill-switch
    convention); ``counting`` (None = GRID_PERF_COUNTING, default off) swaps
    eligible window terminals to CountingTerminalDFA components."""
    if counting is None:
        counting = perf_flags.counting_enabled()
    if counting:
        comps = _counting_comps(terminals, terminal_order, component_budget)
    else:
        # components in terminal_order: the first GrammarInvalid from a bad
        # regex is raised for the same terminal as the eager path
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


def _counting_comps(
    terminals: dict[str, Terminal],
    terminal_order: tuple[str, ...],
    component_budget: int | None,
) -> list[TerminalDFA | LazyTerminalDFA | CountingTerminalDFA]:
    """Component fetch with counting swap-in + the global _MAX_COUNTERS
    selection (kernel v8 frame budget: at most 8 counters per product).

    Selection ranks eligible loops by largest span with the held tie-break —
    (span desc, terminal id asc, in-terminal loop order asc) — at TERMINAL
    granularity: a terminal either keeps its counting component whole or
    re-fetches as the plain expanded component (partial keeps would need
    pattern-external memo keys; deliberate simplification vs the held
    per-loop _drop_reps). Runs at product-assembly time so the pattern-keyed
    memos never see selection state."""
    comps: list = []
    counting_idx: list[int] = []
    for name in terminal_order:
        t = terminals[name]
        c = None
        if not t.is_literal:
            # counting fetch parses first — same _parse_regex, so the first
            # GrammarInvalid is raised for the same terminal as the plain path
            c = _counting_component(t.pattern, component_budget)
        if c is None:
            c = _component(t.pattern, t.is_literal, component_budget)
        else:
            counting_idx.append(len(comps))
        comps.append(c)
    total = sum(len(comps[t].counters) for t in counting_idx)
    if total > _MAX_COUNTERS:
        loops = [
            (-(n if n >= 0 else m), t, li)
            for t in counting_idx
            for li, (m, n) in enumerate(comps[t].counters)
        ]
        loops.sort()
        used = 0
        selected: set[int] = set()
        dropped: set[int] = set()
        for _negspan, t, _li in loops:
            if t in selected or t in dropped:
                continue
            k = len(comps[t].counters)
            if used + k <= _MAX_COUNTERS:
                selected.add(t)
                used += k
            else:
                dropped.add(t)
        for t in dropped:
            name = terminal_order[t]
            comps[t] = _component(
                terminals[name].pattern, terminals[name].is_literal, component_budget)
    return comps


# -- grid_core v8 lazy-scanner payload (kernel-resident lazy product) --------
#
# The kernel's LazyScanner is a transcription of LazyProductDFA (this module
# stays the executable specification). Components ship as compact immutable
# blobs — the kernel does NO regex/NFA work: byte classes, eps-folded
# per-class edge lists, accept ids and NFA accept-reachability are all
# precomputed here from artifacts the build already made. State ids on both
# sides are instance-local demand-order; only state VALUES (subset bitsets /
# sparse product tuples) determine annotations, so masks agree regardless of
# discovery order (tests/trie/test_rust_parity.py lazy leg).

def _dense_component_blob(comp: TerminalDFA) -> bytes:
    """Eager component -> kernel blob (LE): u8 kind=0, u16 n_classes,
    u32 n_states, i32 trans[n_states * n_classes], u8 accepting[n_states],
    u8 co_acc[n_states]."""
    import struct

    import numpy as np

    n_states = len(comp.trans)
    n_classes = len(comp.trans[0]) if n_states else 0
    return b"".join((
        struct.pack("<BHI", 0, n_classes, n_states),
        np.asarray(comp.trans, dtype="<i4").tobytes(),
        bytes(bytearray(comp.accepting)),
        bytes(bytearray(comp.co_acc)),
    ))


def _nfa_component_blob(comp: LazyTerminalDFA) -> bytes:
    """Over-budget component -> kernel NFA arena blob (LE): u8 kind=1,
    u16 n_classes, u32 n_nfa, u32 acc, u32 n_words, u64 reach[n_words],
    u64 start[n_words], u32 offsets[n_nfa * n_classes + 1], u32 dests[...].

    ``dests`` lists are the eps-CLOSED per-(state, class) destination sets:
    eps-closure distributes over union (subset.eps_closure_fn), so the
    kernel's subset step — union members' lists, no eps walk — lands on
    exactly the subsets LazyTerminalDFA.step interns. Raw-empty lists stay
    empty (closure only applied to non-empty dest sets), preserving the
    "no member moves -> DEAD" convention bit-for-bit."""
    import struct

    import numpy as np

    reach = comp._reach
    n_nfa = len(reach)
    n_classes = comp._n_classes
    n_words = (n_nfa + 63) // 64
    reach_words = [0] * n_words
    for q, r in enumerate(reach):
        if r:
            reach_words[q >> 6] |= 1 << (q & 63)
    start_words = [0] * n_words
    for q in comp._states[0]:
        start_words[q >> 6] |= 1 << (q & 63)
    offsets = np.zeros(n_nfa * n_classes + 1, dtype="<u4")
    dests: list[int] = []
    ec = comp._eps_closure
    ebc = comp._edge_by_class
    for q in range(n_nfa):
        per = ebc.get(q)
        base = q * n_classes
        for cls in range(n_classes):
            lst = per[cls] if per is not None else None
            if lst:
                dests.extend(sorted(ec(frozenset(lst))))
            offsets[base + cls + 1] = len(dests)
    return b"".join((
        struct.pack("<BHIII", 1, n_classes, n_nfa, comp._acc, n_words),
        np.asarray(reach_words, dtype="<u8").tobytes(),
        np.asarray(start_words, dtype="<u8").tobytes(),
        offsets.tobytes(),
        np.asarray(dests, dtype="<u4").tobytes(),
    ))


def kernel_lazy_payload(
    dfa: LazyProductDFA,
) -> tuple[list[int], list[bytes], list[list[int]]]:
    """RustWalker(lazy_scanner=...) payload for a lazy product: (global byte
    classes, component blobs in terminal order, per-component global->component
    class maps). Pure read of frozen build artifacts — never touches the
    facade's interned states, so a concurrently-walking LazyProductDFA
    serializes safely (the eps_star memo inside _eps_closure grows benignly
    under the GIL)."""
    # counting products (GRID_PERF_COUNTING) must never reach the kernel:
    # a CountingTerminalDFA's dense trans is ADVISORY on guarded cells, so
    # blobbing it would silently drop the counter guards — the forbidden
    # wrong-mask class. walk.py gates counting DFAs off the kernel; this
    # assert keeps the payload boundary honest until kernel counter frames
    # (P4 phase 2) land.
    assert not dfa.counters, \
        "counting product reached kernel_lazy_payload (kernel counter step pending)"
    blobs = [
        _nfa_component_blob(c) if isinstance(c, LazyTerminalDFA)
        else _dense_component_blob(c)
        for c in dfa.comps
    ]
    return (list(dfa._gclass_of), blobs, [list(m) for m in dfa._cmap])


def shortest_lexemes_factored(dfa: LazyProductDFA) -> dict[int, bytes]:
    """Per-component BFS: the lexicographically-least shortest accepted word
    per terminal — exactly what reserve.shortest_lexemes' union-DFA BFS
    returns (acceptance is a per-component state property; smallest-byte
    level order is lexicographic order) — without touching the product.
    Counting components BFS over (state, counts) configurations — the space
    of the expanded component, bounded by the counter caps."""
    out: dict[int, bytes] = {}
    for tid, comp in enumerate(dfa.comps):
        if not comp.co_acc[0]:
            continue   # empty language: the union BFS never sees this terminal
        if getattr(comp, "counters", ()):
            word = _shortest_word_counting(comp)
        else:
            word = _shortest_word(comp)
        if word is not None:
            out[tid] = word
    return out


def _shortest_word_counting(comp: CountingTerminalDFA) -> bytes | None:
    """_shortest_word over (control state, counts) configurations (same
    smallest-byte level order; counter caps bound the space)."""
    zero = comp.zero_counts()
    frontier: list[tuple[int, tuple[int, ...], bytes]] = [(0, zero, b"")]
    seen = {(0, zero)}
    while frontier:
        for st, _cts, path in frontier:
            if comp.accepting[st]:
                return path
        nxt: list[tuple[int, tuple[int, ...], bytes]] = []
        cof = comp.class_of
        for st, cts, path in frontier:
            for byte in range(256):
                ns, ncts = comp.step_counts(st, cts, cof[byte])
                if ns != DEAD and (ns, ncts) not in seen:
                    seen.add((ns, ncts))
                    nxt.append((ns, ncts, path + bytes([byte])))
        frontier = nxt
    return None


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
