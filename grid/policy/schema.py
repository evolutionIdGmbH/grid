"""E16 SchemaSnapshot: schema -> L3 identifier lexicons (DESIGN.md E3, K4)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from grid.lalr.compile import LALRTables
from grid.trie.walk import Lexicons


@dataclass(frozen=True)
class SchemaSnapshot:
    tables: tuple[tuple[str, tuple[str, ...]], ...]   # ((table, (columns...)), ...)

    @staticmethod
    def from_dict(d: dict[str, list[str]]) -> SchemaSnapshot:
        return SchemaSnapshot(tuple(sorted((t, tuple(sorted(cols))) for t, cols in d.items())))

    @property
    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for t, cols in self.tables:
            h.update(t.encode())
            for c in cols:
                h.update(c.encode())
        return h.hexdigest()

    def lexicons(self, lalr: LALRTables, policy=None) -> Lexicons:
        table_names = {t for t, _ in self.tables}
        column_names = {c for _, cols in self.tables for c in cols}
        if policy is not None:
            if policy.table_allowlist:
                table_names &= set(policy.table_allowlist)
            if policy.column_allowlist:
                column_names &= set(policy.column_allowlist)
        allowed: dict[int, set[bytes]] = {}
        for name, values in (("TABLE_NAME", table_names), ("COLUMN_NAME", column_names)):
            if name in lalr.terminal_names:
                allowed[lalr.terminal_id(name)] = {v.encode() for v in values}
        return Lexicons(allowed)
