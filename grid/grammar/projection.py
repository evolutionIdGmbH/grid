"""E2 RoleProjection: RBAC as a production subset of the dialect grammar.

Lifecycle DECLARED -> COMPOSED -> REDUCED -> VERIFIED -> CACHED (DESIGN.md SS5 E2).
Only REDUCED+VERIFIED projections reach the LALR compiler. ``verify()`` proves
L(G_role) is non-empty (productivity of the start symbol) — the G5/G6 dead-end
theorems assume it.

Terminals are never renumbered: the projection carries the parent grammar's
canonical terminal numbering (E11 cross-family cache requirement).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from grid._statecharts.engine import Statechart, load_chart
from grid.errors import EmptyLanguageError, GrammarInvalid
from grid.grammar.reduction import reduce_productions, useless_symbols
from grid.grammar.spec import DialectGrammar, Production


def _prod_key(p: Production) -> str:
    return f"{p.lhs}->{' '.join(p.rhs)}"


@dataclass
class RoleProjection:
    """A role's grammar: a subset of the dialect's productions, then reduced."""

    base: DialectGrammar
    keep: frozenset[str]                      # production keys (see role_productions)
    role_name: str = "default"
    productions: list[Production] = field(default_factory=list)
    role_shape_hash: str = ""
    _sc: Statechart = field(default_factory=lambda: Statechart(load_chart("role_projection")))

    @property
    def state(self) -> str:
        return self._sc.state

    @staticmethod
    def full(base: DialectGrammar, role_name: str = "full") -> RoleProjection:
        keys = frozenset(_prod_key(p) for p in base.productions)
        return RoleProjection(base=base, keep=keys, role_name=role_name)

    def compose(self) -> RoleProjection:
        if self.base.state != "FROZEN":
            self._sc.fire("compose_error")
            raise GrammarInvalid("projection requires a FROZEN dialect grammar")
        by_key = {_prod_key(p): p for p in self.base.productions}
        unknown = self.keep - set(by_key)
        if unknown:
            self._sc.fire("compose_error")
            raise GrammarInvalid(f"projection references unknown productions: {sorted(unknown)[:3]}")
        self.productions = [p for p in self.base.productions if _prod_key(p) in self.keep]
        self._sc.fire("compose_ok")
        return self

    def reduce(self) -> RoleProjection:
        """Mandatory useless-symbol elimination (companion SS3-L2)."""
        self.productions = reduce_productions(self.productions, self.base.start)
        self._sc.fire("reduce_ok")
        return self

    def verify(self) -> RoleProjection:
        if not any(p.lhs == self.base.start for p in self.productions):
            self._sc.fire("verify_error")
            raise EmptyLanguageError(f"role {self.role_name!r}: L(G_role) is empty")
        leftover = useless_symbols(self.productions, self.base.start)
        if leftover:  # pragma: no cover - reduce() guarantees emptiness; CI assertion
            self._sc.fire("verify_error")
            raise GrammarInvalid(f"projection not reduced after reduce(): {sorted(leftover)}")
        self._sc.fire("verify_ok")
        return self

    def register(self) -> RoleProjection:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.base.fingerprint.encode())
        for p in self.productions:
            h.update(_prod_key(p).encode())
        self.role_shape_hash = h.hexdigest()
        self._sc.fire("register")
        return self

    def build(self) -> RoleProjection:
        """compose -> reduce -> verify -> register."""
        return self.compose().reduce().verify().register()
