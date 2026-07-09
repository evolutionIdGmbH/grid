"""LALR(1) table construction (DESIGN.md SS3 lalr/, E4).

Method: canonical LR(1) item sets, then merge states with equal cores -> LALR(1).
Correctness-first (grammar-sized inputs; the Rust core takes over at M4).
Conflicts raise LALRConflictError with a report of (state, terminal, actions).

Symbol numbering:
- terminal ids: the grammar's canonical terminal order, 0..T-1 (E11 requirement)
- END (``$end``): id T
- nonterminal ids: T+1.. ; the augmented start ``$accept`` is the last NT.

Tables retain per-state item cores (``state_items``) — the reserve computation
(E4a) and completion synthesis need them.
"""

from __future__ import annotations

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


def compile_tables(proj: RoleProjection, identifier_terminals: frozenset[str] = frozenset()) -> LALRTables:
    g = proj.base
    if proj.state != "CACHED":
        raise ValueError("compile_tables requires a CACHED (built) RoleProjection")

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

    # FIRST sets and nullability over symbol ids
    first: dict[int, set[int]] = {s: ({s} if is_terminal(s) else set()) for s in range(n_term + len(nt_names))}
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

    def first_seq(seq: tuple[int, ...], la: int) -> set[int]:
        out: set[int] = set()
        for s in seq:
            out |= first[s]
            if s not in nullable:
                return out
        out.add(la)
        return out

    prods_by_lhs: dict[int, list[int]] = {}
    for i, (lhs, _rhs) in enumerate(prods):
        prods_by_lhs.setdefault(lhs, []).append(i)

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
        for sym, dst in merged_trans[m].items():
            if is_terminal(sym):
                set_action(m, sym, (SHIFT, dst))
            else:
                goto_tbl[m][sym] = dst
        for (p, d, la) in merged_items[m]:
            if d == len(prods[p][1]):
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
        state_items=tuple(frozenset((p, d) for (p, d, _la) in merged_items[m]) for m in range(n_states)),
        start_state=merged_of[0],
        fingerprint=f"{g.fingerprint}:{proj.role_shape_hash}",
        ignored_terminal_ids=ignored_ids,
        literal_terminal_ids=literal_ids,
        identifier_terminal_ids=ident_ids,
    )
