"""E8 ParserStack: persistent stack nodes + virtual-stack simulation (DESIGN.md SS6 steps 1-2).

- ``StackNode`` is an immutable, identity-hashed node (tree-of-stacks; O(1) rollback
  by keeping a reference). ``config_hash`` follows the pinned mix:
  ``H(node) = low 64 bits of BLAKE2b-128(H(parent) || u32le(state) || u32le(sym))``,
  ``H(root) = 0``; audit-only, never used for cache equality (E8).
- ``simulate(node, t)``: the NORMATIVE allowed-terminals algorithm — run the reduce
  chain on a virtual overlay until shift/accept or error. LALR rows over-approximate
  (spurious reduces), so a raw row read is never trusted (SS6 step 1).
- ``shift_terminal(node, t)``: performs reduces+shift for real, returning new nodes
  (persistent, so "scratch stack" copies are free).
"""

from __future__ import annotations

import hashlib
import struct

from grid.lalr.compile import ACCEPT, SHIFT, LALRTables


def _mix(parent_hash: int, state: int, sym: int) -> int:
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<QII", parent_hash, state & 0xFFFFFFFF, sym & 0xFFFFFFFF))
    return int.from_bytes(h.digest()[:8], "little")


class StackNode:
    """Immutable parser-stack node. Identity semantics (memo keys rely on it).

    ``kidx``/``kgen`` cache this node's grid_core intern index (kernel v4):
    parse behavior is a function of the state chain alone, so structurally
    equal chains share one kidx and the kernel's memos hit across token
    positions. Assigned lazily by MaskProducer._kidx; ``kgen`` guards against
    reset_interning, which invalidates every outstanding kidx."""

    __slots__ = ("state", "sym", "parent", "depth", "config_hash", "kidx", "kgen")

    def __init__(self, state: int, sym: int, parent: StackNode | None) -> None:
        self.state = state
        self.sym = sym
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.config_hash = _mix(0 if parent is None else parent.config_hash, state, sym)
        self.kidx = -1  # unassigned
        self.kgen = -1

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StackNode s={self.state} sym={self.sym} d={self.depth}>"


def root_node(tables: LALRTables) -> StackNode:
    return StackNode(tables.start_state, -1, None)


class _Virtual:
    """Overlay stack over a persistent chain (pops may descend into the chain)."""

    __slots__ = ("base", "overlay")

    def __init__(self, base: StackNode) -> None:
        self.base = base
        self.overlay: list[int] = []

    def top(self) -> int:
        return self.overlay[-1] if self.overlay else self.base.state

    def pop_n(self, k: int) -> bool:
        while k and self.overlay:
            self.overlay.pop()
            k -= 1
        while k:
            if self.base.parent is None:
                return False
            self.base = self.base.parent
            k -= 1
        return True

    def push(self, state: int) -> None:
        self.overlay.append(state)


def simulate(tables: LALRTables, node: StackNode, t: int) -> bool:
    """True iff terminal ``t`` (may be END) is viable from this configuration.

    Runs reduces on a virtual stack until ``t`` is shifted (or accepted for END).
    """
    v = _Virtual(node)
    for _ in range(10_000):  # cycle guard: reduce chains are finite in conflict-free tables
        act = tables.action[v.top()].get(t)
        if act is None:
            return False
        kind, arg = act
        if kind == SHIFT:
            return True
        if kind == ACCEPT:
            return True
        lhs, rhs = tables.prods[arg]
        if not v.pop_n(len(rhs)):
            return False
        nxt = tables.goto[v.top()].get(lhs)
        if nxt is None:
            return False
        v.push(nxt)
    raise AssertionError("reduce chain did not terminate")  # pragma: no cover


def allowed_terminals(tables: LALRTables, node: StackNode) -> frozenset[int]:
    """SS6 step 1: A = { t : simulate reaches a shift }. Excludes END and ignored."""
    row = tables.action[node.state]
    return frozenset(
        t for t in row
        if t != tables.end_id and simulate(tables, node, t)
    )


def eos_ok_stack(tables: LALRTables, node: StackNode) -> bool:
    """SS6 step 2, stack part: ACCEPT reachable via the reduce chain of $end."""
    return simulate(tables, node, tables.end_id)


def shift_terminal(tables: LALRTables, node: StackNode, t: int) -> StackNode | None:
    """Perform reduces then shift ``t`` for real; None if not viable (caller bug)."""
    cur = node
    for _ in range(10_000):
        act = tables.action[cur.state].get(t)
        if act is None:
            return None
        kind, arg = act
        if kind == SHIFT:
            return StackNode(arg, t, cur)
        if kind == ACCEPT:
            return cur  # only for END; accept does not push
        lhs, rhs = tables.prods[arg]
        base = cur
        for _i in range(len(rhs)):
            assert base.parent is not None, "reduce popped past root"
            base = base.parent
        nxt = tables.goto[base.state].get(lhs)
        if nxt is None:
            return None
        cur = StackNode(nxt, lhs, base)
    raise AssertionError("reduce chain did not terminate")  # pragma: no cover
