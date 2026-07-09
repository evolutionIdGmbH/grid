"""E16 PolicyBundle: RBAC role -> production subset of the dialect grammar.

Verb-level policy (the mask-enforceable granularity per DESIGN.md SS4.6): a role
keeps only the ``query -> X_stmt`` alternatives for its allowed verbs; the
mandatory reduction pass (E2) then eliminates the unreachable statement subtrees.
Column-level policy is post-parse (SemanticChecker) by proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grid.grammar.projection import RoleProjection, _prod_key
from grid.grammar.spec import DialectGrammar

VERB_RULES = {
    "select": "select_stmt",
    "insert": "insert_stmt",
    "update": "update_stmt",
    "delete": "delete_stmt",
}


@dataclass(frozen=True)
class PolicyBundle:
    role: str
    allowed_verbs: frozenset[str]
    table_allowlist: frozenset[str] = frozenset()
    column_allowlist: frozenset[str] = frozenset()
    _extra: dict = field(default_factory=dict, compare=False)

    @staticmethod
    def from_store(store: dict, role: str) -> PolicyBundle:
        cfg = store[role]
        return PolicyBundle(
            role=role,
            allowed_verbs=frozenset(cfg["verbs"]),
            table_allowlist=frozenset(cfg.get("tables", ())),
            column_allowlist=frozenset(cfg.get("columns", ())),
        )

    def projection(self, grammar: DialectGrammar) -> RoleProjection:
        banned_rules = {rule for verb, rule in VERB_RULES.items() if verb not in self.allowed_verbs}
        keep = frozenset(
            _prod_key(p) for p in grammar.productions
            if not (p.lhs == "query" and any(sym in banned_rules for sym in p.rhs))
        )
        return RoleProjection(base=grammar, keep=keep, role_name=self.role).build()
