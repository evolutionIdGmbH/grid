"""Shared subset-construction core for scanner DFAs (byte NFAs -> class DFAs).

Both scanner builders run the same four steps over a Thompson byte-NFA:
eps-closure memoization, byte-class partition refinement, per-class edge
indexing, and a FIFO subset construction over class-compressed rows. This
module holds those steps as pure helpers; the builders differ only in their
start closure and their post-passes:

- dfa.build_scanner (eager union DFA, the GRID_PERF_FACTORED_SCANNER=0
  oracle): 256-wide row expansion + accepts_all / live-mask annotation over
  the discovery order, always uncapped;
- factored._build_component (per-terminal component DFA): class-wide rows
  kept as-is + a per-state accepting bitmap, capped at
  GRID_PERF_COMPONENT_BUDGET states (SubsetBudgetExceeded -> the caller
  switches to demand-driven interning; under the cap the arrays are
  bit-identical to an uncapped run).

Determinism here is load-bearing: state ids feed genN cache keys and the
artifact store, and the factored product's materialized numbering must equal
the eager builder's state-for-state (tests/lexer/test_factored_differential).
Hence FIFO discovery, ascending-class successor order, and min-byte block
order — behavior-preserving on paper is not enough, do not reorder any of
them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

DEAD = -1

_Edges = dict[int, list[tuple[frozenset[int], int]]]


class SubsetBudgetExceeded(Exception):
    """subset_construct discovered more than ``max_states`` subset states.

    Raised only when a caller passes a cap (factored._build_component under
    GRID_PERF_COMPONENT_BUDGET); the partial arrays are discarded by the
    caller — the substring-union terminal family discovers ~2^k states
    eagerly, so there is nothing worth keeping."""


def eps_closure_fn(
    eps: dict[int, list[int]],
) -> Callable[[Iterable[int]], frozenset[int]]:
    """-> ``eps_closure(states)`` over the given eps-edge dict.

    eps-closure distributes over union: the per-state closure is computed once
    (graph reachability over eps edges, memoized), then closure(S) = union of
    the per-state closures."""
    eps_star: dict[int, frozenset[int]] = {}

    def _star(s0: int) -> frozenset[int]:
        got = eps_star.get(s0)
        if got is None:
            stack, seen = [s0], {s0}
            while stack:
                st = stack.pop()
                for nxt in eps.get(st, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            got = eps_star[s0] = frozenset(seen)
        return got

    def eps_closure(states: Iterable[int]) -> frozenset[int]:
        out: frozenset[int] = frozenset()
        for s in states:
            out |= _star(s)
        return out

    return eps_closure


def byte_classes(edges: _Edges) -> tuple[list[set[int]], list[int]]:
    """Alphabet compression: partition the 256 byte values into equivalence
    classes over the distinct edge charsets (JSON/SQL grammars have ~10-30
    classes), so the subset construction runs per CLASS; callers expand to
    byte-wide rows at the end or keep class-wide rows. Same DFA modulo state
    numbering; the dominant cost (the per-byte inner loop) drops by the
    compression factor.

    Returns ``(blocks, class_of)`` with ``blocks`` sorted by min byte —
    a deterministic class order, stable across processes (int hashing is
    PYTHONHASHSEED-independent, so the refinement sees a stable charset
    iteration order)."""
    distinct_charsets = {chars for st_edges in edges.values() for chars, _dst in st_edges}
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
    for ci_, blk in enumerate(blocks):
        for c in blk:
            class_of[c] = ci_
    return blocks, class_of


def edges_by_class(
    edges: _Edges, class_of: list[int], n_classes: int,
) -> dict[int, list[list[int] | None]]:
    """Per NFA state: class -> destination list, or None where the state has
    no edge on that class (edges evaluated once, per class)."""
    out: dict[int, list[list[int] | None]] = {}
    for st, st_edges in edges.items():
        per = out[st] = [None] * n_classes
        for chars, dst in st_edges:
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
    return out


def subset_construct(
    start_closure: frozenset[int],
    edge_by_class: dict[int, list[list[int] | None]],
    eps_closure: Callable[[Iterable[int]], frozenset[int]],
    n_classes: int,
    max_states: int | None = None,
) -> tuple[list[frozenset[int]], list[list[int]]]:
    """Rabin-Scott subset construction over class-compressed rows.

    FIFO state discovery, successors explored in ascending class index
    (``sorted(by_class.items())``) — the numbering contract shared by the
    eager builder and the factored materializer (LazyProductDFA.materialize
    reproduces this order to be equal, not just isomorphic).

    ``max_states`` (None = unbounded, the eager-builder default): raise
    SubsetBudgetExceeded once more than that many subsets exist. Checked per
    processed state, so the loop may overshoot by one state's class fanout
    before raising — irrelevant, breach output is discarded. Under the cap
    the arrays are bit-identical to the uncapped run (the check is
    read-only).

    Returns ``(order, class_rows)``: ``order[i]`` is the i-th subset state,
    ``class_rows[i][cl]`` the successor state id or DEAD. Per-state
    annotations (acceptance, live sets) are the callers' post-passes over
    ``order``."""
    ids: dict[frozenset[int], int] = {start_closure: 0}
    order = [start_closure]
    class_rows: list[list[int]] = []
    i = 0
    while i < len(order):
        if max_states is not None and len(order) > max_states:
            raise SubsetBudgetExceeded
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
        class_rows.append(row)
    return order, class_rows
