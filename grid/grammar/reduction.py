"""Useless-symbol elimination and reducedness checks (DESIGN.md E1/E2).

A grammar is *reduced* iff every nonterminal is productive (derives some terminal
string) and reachable from the start symbol. E2's mandatory ``reduce()`` step uses
:func:`reduce_productions`; E1 validation uses :func:`useless_symbols` to *assert*
reducedness of authored dialect grammars.

This is the Earley/dead-end-freedom precondition: production subsetting (RBAC
projection) is exactly the operation that creates unproductive/unreachable
nonterminals (companion GUARDRAIL-REDESIGN.md SS3-L2).

Both primitives are single-pass worklists — linear in total RHS length —
replacing the original changed-flag rescan fixpoints (quadratic on deep rule
chains; spec._validate keeps useless_symbols on the compile hot path, P2).
Output sets are identical to the legacy fixpoints: both compute the same
least fixpoint, gated by corpus + randomized set-equality (Gate C).
"""

from __future__ import annotations

from grid.grammar.spec import Production


def _is_terminal(sym: str) -> bool:
    return sym.isupper() or sym.startswith("LIT_")


def productive_nonterminals(productions: list[Production]) -> set[str]:
    """Counter worklist: production i waits on its unresolved nonterminal
    RHS occurrences; when its counter hits zero, its lhs is productive.
    Registration completes over all productions before the drain starts, so
    order of productions cannot matter (same least fixpoint as the legacy
    whole-set rescan)."""
    productive: set[str] = set()
    counts = [0] * len(productions)
    waits: dict[str, list[int]] = {}
    stack: list[str] = []
    for i, p in enumerate(productions):
        n = 0
        for s in p.rhs:
            if not _is_terminal(s):
                n += 1
                waits.setdefault(s, []).append(i)
        counts[i] = n
        if n == 0 and p.lhs not in productive:
            productive.add(p.lhs)
            stack.append(p.lhs)
    while stack:
        for i in waits.get(stack.pop(), ()):
            counts[i] -= 1
            if counts[i] == 0:
                lhs = productions[i].lhs
                if lhs not in productive:
                    productive.add(lhs)
                    stack.append(lhs)
    return productive


def reachable_symbols(productions: list[Production], start: str) -> set[str]:
    """BFS/DFS from start over an lhs-indexed production map; includes
    terminals appearing in reachable rules (exactly like the legacy
    fixpoint — pushed symbols without productions are drained as no-ops)."""
    by_lhs: dict[str, list[Production]] = {}
    for p in productions:
        by_lhs.setdefault(p.lhs, []).append(p)
    reachable = {start}
    stack = [start]
    while stack:
        for p in by_lhs.get(stack.pop(), ()):
            for s in p.rhs:
                if s not in reachable:
                    reachable.add(s)
                    stack.append(s)
    return reachable


def useless_symbols(productions: list[Production], start: str) -> set[str]:
    """Nonterminals that are unproductive or unreachable (empty set == reduced)."""
    nts = {p.lhs for p in productions}
    productive = productive_nonterminals(productions)
    useless = nts - productive
    kept = [p for p in productions if p.lhs in productive and all(_is_terminal(s) or s in productive for s in p.rhs)]
    reachable = reachable_symbols(kept, start)
    useless |= {nt for nt in nts if nt not in reachable}
    return useless


def reduce_productions(productions: list[Production], start: str) -> list[Production]:
    """Standard two-pass reduction: drop unproductive, then unreachable (order matters)."""
    productive = productive_nonterminals(productions)
    kept = [
        p for p in productions
        if p.lhs in productive and all(_is_terminal(s) or s in productive for s in p.rhs)
    ]
    reachable = reachable_symbols(kept, start)
    return [p for p in kept if p.lhs in reachable]
