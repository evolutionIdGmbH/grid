"""Useless-symbol elimination and reducedness checks (DESIGN.md E1/E2).

A grammar is *reduced* iff every nonterminal is productive (derives some terminal
string) and reachable from the start symbol. E2's mandatory ``reduce()`` step uses
:func:`reduce_productions`; E1 validation uses :func:`useless_symbols` to *assert*
reducedness of authored dialect grammars.

This is the Earley/dead-end-freedom precondition: production subsetting (RBAC
projection) is exactly the operation that creates unproductive/unreachable
nonterminals (companion GUARDRAIL-REDESIGN.md SS3-L2).
"""

from __future__ import annotations

from grid.grammar.spec import Production


def _is_terminal(sym: str) -> bool:
    return sym.isupper() or sym.startswith("LIT_")


def productive_nonterminals(productions: list[Production]) -> set[str]:
    productive: set[str] = set()
    changed = True
    while changed:
        changed = False
        for p in productions:
            if p.lhs in productive:
                continue
            if all(_is_terminal(s) or s in productive for s in p.rhs):
                productive.add(p.lhs)
                changed = True
    return productive


def reachable_symbols(productions: list[Production], start: str) -> set[str]:
    reachable = {start}
    changed = True
    while changed:
        changed = False
        for p in productions:
            if p.lhs in reachable:
                for s in p.rhs:
                    if s not in reachable:
                        reachable.add(s)
                        changed = True
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
