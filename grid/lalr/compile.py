"""LALR(1) table construction (DESIGN.md SS3 lalr/, E4).

Two constructions, selected per call (LALR(1) is uniquely defined, so both
produce equal LALRTables — action/goto content, conflict set, and, by the
numbering argument at _lr0_automaton, state ids):

- "lr1_merge" (default): canonical LR(1) item sets, then merge states with
  equal cores. Correctness-first; item counts scale with lookahead splits,
  quadratic-plus on shared-core/divergent-lookahead grammars (2^R member
  chains).
- "dp" (GRID_PERF_LALR_DP=1): LR(0) automaton over (prod, dot) cores plus
  exact lookaheads via the DeRemer-Pennello DR/reads/includes/lookback
  relations (TOPLAS 1982) closed by the digraph algorithm; near-linear in
  LR(0) transitions.

Conflicts raise LALRConflictError with a report of (state, terminal, actions).

Symbol numbering:
- terminal ids: the grammar's canonical terminal order, 0..T-1 (E11 requirement)
- END (``$end``): id T
- nonterminal ids: T+1.. ; the augmented start ``$accept`` is the last NT.

Tables retain per-state item cores (``state_items``) — the reserve computation
(E4a) and completion synthesis need them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from grid.errors import LALRConflictError
from grid.grammar.projection import RoleProjection

SHIFT, REDUCE, ACCEPT = 0, 1, 2


@dataclass(frozen=True)
class LALRTables:
    terminal_names: tuple[str, ...]          # index = terminal id (END last)
    nonterminal_names: tuple[str, ...]       # index - n_symbols_terminal = nt id offset
    end_id: int
    prods: tuple[tuple[int, tuple[int, ...]], ...]   # prod 0 = $accept -> start
    prod_names: tuple[str, ...]
    action: tuple[dict[int, tuple[int, int]], ...]   # [state][term] -> (kind, arg)
    goto: tuple[dict[int, int], ...]                 # [state][nt] -> state
    state_items: tuple[frozenset[tuple[int, int]], ...]  # (prod_idx, dot) per state
    start_state: int = 0
    fingerprint: str = ""
    ignored_terminal_ids: frozenset[int] = frozenset()
    literal_terminal_ids: frozenset[int] = frozenset()
    identifier_terminal_ids: frozenset[int] = field(default_factory=frozenset)

    @property
    def n_terminals(self) -> int:
        return len(self.terminal_names)

    def terminal_id(self, name: str) -> int:
        return self.terminal_names.index(name)


def _first_nullable(
    prods: list[tuple[int, tuple[int, ...]]], n_symbols: int, n_term: int
) -> tuple[dict[int, set[int]], set[int]]:
    """FIRST sets and nullability over symbol ids."""
    first: dict[int, set[int]] = {s: ({s} if s < n_term else set()) for s in range(n_symbols)}
    nullable: set[int] = set()
    changed = True
    while changed:
        changed = False
        for lhs, rhs in prods:
            if lhs not in nullable and all(s in nullable for s in rhs):
                nullable.add(lhs)
                changed = True
            before = len(first[lhs])
            for s in rhs:
                first[lhs] |= first[s]
                if s not in nullable:
                    break
            if len(first[lhs]) != before:
                changed = True
    return first, nullable


def _build_lr1_merged(
    prods: list[tuple[int, tuple[int, ...]]],
    prods_by_lhs: dict[int, list[int]],
    first: dict[int, set[int]],
    nullable: set[int],
    n_term: int,
    end_id: int,
) -> tuple[list[dict[int, int]], list[set[tuple[int, int, int]]], int]:
    """Canonical LR(1) item sets merged by core -> (trans, items, start_state).

    LALR state ids are assigned by first occurrence of each core in the LR(1)
    BFS order; items are the merged (prod, dot, lookahead) sets per state.
    """
    is_terminal = lambda s: s < n_term  # noqa: E731

    def first_seq(seq: tuple[int, ...], la: int) -> set[int]:
        out: set[int] = set()
        for s in seq:
            out |= first[s]
            if s not in nullable:
                return out
        out.add(la)
        return out

    def closure(items: frozenset[tuple[int, int, int]]) -> frozenset[tuple[int, int, int]]:
        out = set(items)
        stack = list(items)
        while stack:
            p, d, la = stack.pop()
            _lhs, rhs = prods[p]
            if d < len(rhs) and not is_terminal(rhs[d]):
                for la2 in first_seq(rhs[d + 1:], la):
                    for q in prods_by_lhs.get(rhs[d], ()):
                        item = (q, 0, la2)
                        if item not in out:
                            out.add(item)
                            stack.append(item)
        return frozenset(out)

    def goto_set(items: frozenset[tuple[int, int, int]], sym: int) -> frozenset[tuple[int, int, int]]:
        kernel = frozenset(
            (p, d + 1, la) for (p, d, la) in items if d < len(prods[p][1]) and prods[p][1][d] == sym
        )
        return closure(kernel) if kernel else frozenset()

    # canonical LR(1) states
    start = closure(frozenset({(0, 0, end_id)}))
    lr1_states: dict[frozenset, int] = {start: 0}
    order = [start]
    lr1_trans: list[dict[int, int]] = []
    i = 0
    while i < len(order):
        cur = order[i]
        i += 1
        syms = {prods[p][1][d] for (p, d, _la) in cur if d < len(prods[p][1])}
        row: dict[int, int] = {}
        for s in sorted(syms):
            nxt = goto_set(cur, s)
            if nxt not in lr1_states:
                lr1_states[nxt] = len(order)
                order.append(nxt)
            row[s] = lr1_states[nxt]
        lr1_trans.append(row)

    # merge by core -> LALR
    core_of = [frozenset((p, d) for (p, d, _la) in st) for st in order]
    core_ids: dict[frozenset, int] = {}
    merged_of: list[int] = []
    for c in core_of:
        if c not in core_ids:
            core_ids[c] = len(core_ids)
        merged_of.append(core_ids[c])
    n_states = len(core_ids)

    merged_items: list[set[tuple[int, int, int]]] = [set() for _ in range(n_states)]
    merged_trans: list[dict[int, int]] = [{} for _ in range(n_states)]
    for lr1_id, st in enumerate(order):
        m = merged_of[lr1_id]
        merged_items[m] |= st
        for sym, dst in lr1_trans[lr1_id].items():
            prev = merged_trans[m].get(sym)
            assert prev is None or prev == merged_of[dst], "core merge produced inconsistent goto"
            merged_trans[m][sym] = merged_of[dst]

    return merged_trans, merged_items, merged_of[0]


def _lr0_automaton(
    prods: list[tuple[int, tuple[int, ...]]],
    prods_by_lhs: dict[int, list[int]],
    n_term: int,
) -> tuple[list[frozenset[tuple[int, int]]], list[dict[int, int]]]:
    """LR(0) automaton over (prod, dot) items -> (closures, trans).

    States are keyed on kernels, a bijection with closed cores: closure only
    adds dot-0 items, a kernel item (p, d>=1) pins prods[p][1][d-1] to the
    incoming symbol (so distinct symbols yield disjoint kernels), and the
    start kernel {(0, 0)} recurs nowhere because $accept is in no RHS.

    State numbering matches _build_lr1_merged's first-occurrence-of-core
    order: an LR(1) state and its core enumerate the same successor symbols
    (lookaheads never add dotted symbols), so the first LR(1) state of each
    core discovers exactly the new cores this BFS discovers from that core,
    in the same sorted-symbol order, and later same-core LR(1) states
    discover none.
    """
    def closure0(kernel: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        out = set(kernel)
        stack = list(kernel)
        while stack:
            p, d = stack.pop()
            rhs = prods[p][1]
            if d < len(rhs) and rhs[d] >= n_term:
                for q in prods_by_lhs.get(rhs[d], ()):
                    item = (q, 0)
                    if item not in out:
                        out.add(item)
                        stack.append(item)
        return frozenset(out)

    start_kernel = frozenset({(0, 0)})
    states: dict[frozenset[tuple[int, int]], int] = {start_kernel: 0}
    closures = [closure0(start_kernel)]
    trans: list[dict[int, int]] = []
    i = 0
    while i < len(closures):
        cur = closures[i]
        i += 1
        syms = {prods[p][1][d] for (p, d) in cur if d < len(prods[p][1])}
        row: dict[int, int] = {}
        for s in sorted(syms):
            kernel = frozenset(
                (p, d + 1) for (p, d) in cur if d < len(prods[p][1]) and prods[p][1][d] == s
            )
            nxt = states.get(kernel)
            if nxt is None:
                nxt = states[kernel] = len(closures)
                closures.append(closure0(kernel))
            row[s] = nxt
        trans.append(row)
    return closures, trans


def _digraph(
    nodes: list[tuple[int, int]],
    edges: dict[tuple[int, int], list[tuple[int, int]]],
    base: dict[tuple[int, int], set[int]],
) -> dict[tuple[int, int], set[int]]:
    """DeRemer-Pennello digraph: least F with F(x) = base(x) | U{F(y): x->y},
    nodes of an SCC sharing one set. Iterative Tarjan-style traversal —
    relation chains scale with grammar depth, past CPython's recursion limit.
    """
    inf = len(nodes) + 1
    depth = dict.fromkeys(nodes, 0)
    f = {x: set(base[x]) for x in nodes}
    stack: list[tuple[int, int]] = []
    for root in nodes:
        if depth[root]:
            continue
        stack.append(root)
        depth[root] = len(stack)
        frames = [(root, iter(edges.get(root, ())), len(stack))]
        while frames:
            x, it, d0 = frames[-1]
            child = None
            for y in it:
                if depth[y] == 0:
                    child = y
                    break
                if depth[y] < depth[x]:
                    depth[x] = depth[y]
                f[x] |= f[y]
            if child is not None:
                stack.append(child)
                depth[child] = len(stack)
                frames.append((child, iter(edges.get(child, ())), len(stack)))
                continue
            frames.pop()
            if depth[x] == d0:
                fx = f[x]
                while True:
                    top = stack.pop()
                    depth[top] = inf
                    f[top] = fx
                    if top == x:
                        break
            if frames:
                px = frames[-1][0]
                if depth[x] < depth[px]:
                    depth[px] = depth[x]
                f[px] |= f[x]
    return f


def _dp_lookaheads(
    prods: list[tuple[int, tuple[int, ...]]],
    prods_by_lhs: dict[int, list[int]],
    closures: list[frozenset[tuple[int, int]]],
    trans: list[dict[int, int]],
    nullable: set[int],
    n_term: int,
    end_id: int,
    start_nt: int,
) -> dict[tuple[int, int], set[int]]:
    """LALR(1) lookaheads per DeRemer-Pennello: {(state, prod): terminal ids}
    for every completed item of prods 1.. ($accept reduces as ACCEPT, no LA).

    Nodes are nonterminal transitions X = (state, nt). DR(X) seeds direct
    reads ($end seeded on the start transition, mirroring the (0, 0, $end)
    start item of the LR(1) path); reads adds nullable-nonterminal chains;
    includes lifts Follow through B -> beta A gamma with gamma nullable;
    lookback attaches Follow((p', B)) to the completed item B -> omega. at
    the state omega reaches from p'.
    """
    nt_trans: list[tuple[int, int]] = []
    for q, row in enumerate(trans):
        for sym in row:
            if sym >= n_term:
                nt_trans.append((q, sym))

    dr: dict[tuple[int, int], set[int]] = {}
    reads: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x in nt_trans:
        q, a = x
        r = trans[q][a]
        seed = {t for t in trans[r] if t < n_term}
        if q == 0 and a == start_nt:
            seed.add(end_id)
        dr[x] = seed
        rd = [(r, c) for c in trans[r] if c >= n_term and c in nullable]
        if rd:
            reads[x] = rd
    read_sets = _digraph(nt_trans, reads, dr)

    includes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    lookback: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x in nt_trans:
        p_state, b = x
        for pi in prods_by_lhs.get(b, ()):
            rhs = prods[pi][1]
            s = p_state
            path = [s]
            for sym in rhs:
                s = trans[s][sym]
                path.append(s)
            lookback.setdefault((s, pi), []).append(x)
            # trailing-nullable positions: omega[i] nonterminal, omega[i+1:] =>* eps
            for i in range(len(rhs) - 1, -1, -1):
                sym = rhs[i]
                if sym >= n_term:
                    includes.setdefault((path[i], sym), []).append(x)
                if sym not in nullable:
                    break
    follow = _digraph(nt_trans, includes, read_sets)

    la: dict[tuple[int, int], set[int]] = {}
    for key, xs in lookback.items():
        acc = la.setdefault(key, set())
        for x in xs:
            acc |= follow[x]
    return la


def compile_tables(
    proj: RoleProjection,
    identifier_terminals: frozenset[str] = frozenset(),
    *,
    algorithm: str | None = None,
) -> LALRTables:
    g = proj.base
    if proj.state != "CACHED":
        raise ValueError("compile_tables requires a CACHED (built) RoleProjection")
    if algorithm is None:
        algorithm = "dp" if os.environ.get("GRID_PERF_LALR_DP", "0") == "1" else "lr1_merge"
    if algorithm not in ("lr1_merge", "dp"):
        raise ValueError(f"unknown LALR algorithm: {algorithm!r}")

    term_names = list(g.terminal_order) + ["$end"]
    tid = {n: i for i, n in enumerate(term_names)}
    end_id = tid["$end"]

    nts = sorted({p.lhs for p in proj.productions})
    nt_names = nts + ["$accept"]
    ntid = {n: len(term_names) + i for i, n in enumerate(nt_names)}

    def sym_id(s: str) -> int:
        return tid[s] if (s.isupper() or s.startswith("LIT_")) else ntid[s]

    prods: list[tuple[int, tuple[int, ...]]] = [(ntid["$accept"], (ntid[g.start],))]
    prod_names: list[str] = [f"$accept -> {g.start}"]
    for p in proj.productions:
        prods.append((ntid[p.lhs], tuple(sym_id(s) for s in p.rhs)))
        prod_names.append(f"{p.lhs} -> {' '.join(p.rhs) or 'eps'}")

    n_term = len(term_names)
    is_terminal = lambda s: s < n_term  # noqa: E731

    first, nullable = _first_nullable(prods, n_term + len(nt_names), n_term)

    prods_by_lhs: dict[int, list[int]] = {}
    for i, (lhs, _rhs) in enumerate(prods):
        prods_by_lhs.setdefault(lhs, []).append(i)

    if algorithm == "dp":
        state_closures, trans = _lr0_automaton(prods, prods_by_lhs, n_term)
        n_states = len(trans)
        start_state = 0
        la_sets = _dp_lookaheads(
            prods, prods_by_lhs, state_closures, trans, nullable, n_term, end_id, ntid[g.start]
        )

        def reduces(m: int):
            for (p, d) in sorted(state_closures[m]):
                if d != len(prods[p][1]):
                    continue
                if p == 0:
                    yield p, end_id
                else:
                    for la in sorted(la_sets.get((m, p), ())):
                        yield p, la
    else:
        trans, merged_items, start_state = _build_lr1_merged(
            prods, prods_by_lhs, first, nullable, n_term, end_id
        )
        n_states = len(trans)
        state_closures = [
            frozenset((p, d) for (p, d, _la) in merged_items[m]) for m in range(n_states)
        ]

        def reduces(m: int):
            for (p, d, la) in merged_items[m]:
                if d == len(prods[p][1]):
                    yield p, la

    action: list[dict[int, tuple[int, int]]] = [{} for _ in range(n_states)]
    goto_tbl: list[dict[int, int]] = [{} for _ in range(n_states)]
    conflicts: list[tuple[int, str, str, str]] = []

    def set_action(st: int, t: int, act: tuple[int, int]) -> None:
        cur = action[st].get(t)
        if cur is not None and cur != act:
            def fmt(a: tuple[int, int]) -> str:
                # lazy per-kind formatting: a SHIFT arg is a state id, which may
                # exceed len(prod_names) — an eager dict here indexed it anyway
                kind, arg = a
                if kind == SHIFT:
                    return f"shift {arg}"
                if kind == REDUCE:
                    return f"reduce [{prod_names[arg]}]"
                return "accept"
            conflicts.append((st, term_names[t], fmt(cur), fmt(act)))
            return
        action[st][t] = act

    for m in range(n_states):
        for sym, dst in trans[m].items():
            if is_terminal(sym):
                set_action(m, sym, (SHIFT, dst))
            else:
                goto_tbl[m][sym] = dst
        for p, la in reduces(m):
            if p == 0:
                set_action(m, end_id, (ACCEPT, 0))
            else:
                set_action(m, la, (REDUCE, p))

    if conflicts:
        raise LALRConflictError(sorted(set(conflicts)))

    ignored_ids = frozenset(tid[n] for n in g.ignored)
    literal_ids = frozenset(i for i, n in enumerate(g.terminal_order) if g.terminals[n].is_literal)
    ident_ids = frozenset(tid[n] for n in identifier_terminals if n in tid)

    return LALRTables(
        terminal_names=tuple(term_names),
        nonterminal_names=tuple(nt_names),
        end_id=end_id,
        prods=tuple(prods),
        prod_names=tuple(prod_names),
        action=tuple(action),
        goto=tuple(goto_tbl),
        state_items=tuple(state_closures),
        start_state=start_state,
        fingerprint=f"{g.fingerprint}:{proj.role_shape_hash}",
        ignored_terminal_ids=ignored_ids,
        literal_terminal_ids=literal_ids,
        identifier_terminal_ids=ident_ids,
    )
