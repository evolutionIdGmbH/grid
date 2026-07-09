"""SemanticChecker: post-parse conformance for what masks cannot enforce (SS4.6).

Column-to-table binding is not left-to-right CFG-enforceable (SELECT list precedes
FROM; alias binding is context-sensitive), so the mask guarantees verb/table
granularity and THIS layer checks, on the completed statement:
- every referenced table exists in the schema (and the policy allow-list);
- ALIAS-QUALIFIED references bind: ``t2.col`` is valid iff ``t2`` was bound by a
  ``<table> as t2`` in FROM/JOIN and ``col`` is a column of THAT table (the
  failure class observed on Spider: ``select t2.year_of_founded from orchestra
  as t1 join performance as t2 ...`` — grammatical, schema-lexicon-valid,
  semantically unbound);
- TABLE-qualified references: ``users.col`` requires ``col`` in ``users``;
- bare columns belong to at least one table referenced in the statement (not
  merely to the schema-wide union the mask uses).

Aliases are collected statement-wide before validation (SQL allows use-before-
binding in the SELECT list); nested scopes share one flat binding map (Spider's
canonical aliases are statement-unique; a scoped checker is future work).
"""

from __future__ import annotations

from dataclasses import dataclass

from grid.lalr.compile import LALRTables
from grid.lalr.stack import root_node, shift_terminal, simulate
from grid.lexer.dfa import ScannerDFA
from grid.lexer.run import LexerRun, ScanReject, scan
from grid.policy.schema import SchemaSnapshot
from grid.trie.walk import Lexicons, pick_viable


@dataclass(frozen=True)
class Violation:
    kind: str      # unknown_table | unknown_alias | column_not_in_aliased_table
    #              # | column_not_in_table | column_not_in_referenced_tables | parse_error
    lexeme: str
    detail: str


def parse_terminal_stream(
    tables: LALRTables, dfa: ScannerDFA, data: bytes, lexicons: Lexicons | None = None,
) -> list[tuple[int, bytes]] | None:
    """Contextual scan+shift of a COMPLETE statement; returns [(terminal_id, lexeme)].

    ``lexicons`` steers same-regex terminal choice (TABLE_NAME vs COLUMN_NAME vs
    ALIAS) exactly like the generation-side pick — without it, names that are
    both a table and a column of the schema can be mislabeled."""
    priority = {
        tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
        for tid in range(tables.n_terminals)
    }
    try:
        events, tail = scan(dfa, data)
    except ScanReject:
        return None
    fin = LexerRun(remainder=tail).finalize(dfa)
    if fin is None:
        return None
    all_events = list(events) + list(fin)
    segments: list[bytes] = []
    off = 0
    for ev in all_events:
        segments.append(data[off:off + ev.length])
        off += ev.length
    node = root_node(tables)
    out: list[tuple[int, bytes]] = []
    for ev, seg in zip(all_events, segments, strict=True):
        viable = frozenset(
            t for t in ev.candidates
            if t not in tables.ignored_terminal_ids and simulate(tables, node, t)
        )
        pick = pick_viable(ev, seg, viable, tables.ignored_terminal_ids, priority, lexicons)
        if pick is None:
            return None
        if pick in tables.ignored_terminal_ids:
            continue
        nxt = shift_terminal(tables, node, pick)
        if nxt is None:
            return None
        node = nxt
        out.append((pick, seg))
    return out


class SemanticChecker:
    def __init__(self, tables: LALRTables, dfa: ScannerDFA, schema: SchemaSnapshot) -> None:
        self.tables = tables
        self.dfa = dfa
        self.schema = {t: set(cols) for t, cols in schema.tables}
        names = {n: i for i, n in enumerate(tables.terminal_names)}
        self._t_table = names.get("TABLE_NAME")
        self._t_column = names.get("COLUMN_NAME")
        self._t_alias = names.get("ALIAS")
        lex: dict[int, set[bytes]] = {}
        if self._t_table is not None:
            lex[self._t_table] = {t.encode() for t in self.schema}
        if self._t_column is not None:
            lex[self._t_column] = {c.encode() for cols in self.schema.values() for c in cols}
        self._lexicons = Lexicons(lex) if lex else None

    def check(self, sql_text: str) -> list[Violation]:
        data = sql_text.encode("utf-8")
        # lexicon-steered parse labels known names precisely; names OUTSIDE the
        # schema fail its lexeme filter, so fall back to the plain parse — the
        # checker must still label and diagnose unknown identifiers
        stream = parse_terminal_stream(self.tables, self.dfa, data, self._lexicons)
        if stream is None:
            stream = parse_terminal_stream(self.tables, self.dfa, data)
        if stream is None:
            return [Violation("parse_error", sql_text, "statement does not parse")]
        out: list[Violation] = []

        # pass 1 — statement-wide alias bindings: TABLE_NAME "as" ALIAS
        bindings: dict[str, str] = {}
        for i, (tid, lx) in enumerate(stream):
            if (
                tid == self._t_table
                and i + 2 < len(stream)
                and stream[i + 1][1] == b"as"
                and stream[i + 2][0] == self._t_alias
            ):
                bindings[stream[i + 2][1].decode()] = lx.decode()

        # pass 2 — qualified references: (TABLE_NAME | ALIAS) "." COLUMN_NAME
        qualified_cols: set[int] = set()  # stream indices of qualified COLUMN_NAMEs
        for i, (_tid, lx) in enumerate(stream):
            if lx != b"." or i == 0 or i + 1 >= len(stream):
                continue
            q_tid, q_lx = stream[i - 1]
            c_tid, c_lx = stream[i + 1]
            if c_tid != self._t_column:
                continue
            qualified_cols.add(i + 1)
            col = c_lx.decode()
            if q_tid == self._t_alias:
                tbl = bindings.get(q_lx.decode())
                if tbl is None:
                    out.append(Violation(
                        "unknown_alias", q_lx.decode(),
                        f"alias {q_lx.decode()!r} is not bound in FROM/JOIN",
                    ))
                elif col not in self.schema.get(tbl, set()):
                    out.append(Violation(
                        "column_not_in_aliased_table", f"{q_lx.decode()}.{col}",
                        f"column {col!r} is not in table {tbl!r} "
                        f"(bound to alias {q_lx.decode()!r})",
                    ))
            elif q_tid == self._t_table:
                tbl = q_lx.decode()
                if tbl in self.schema and col not in self.schema[tbl]:
                    out.append(Violation(
                        "column_not_in_table", f"{tbl}.{col}",
                        f"column {col!r} is not in table {tbl!r}",
                    ))

        # pass 3 — table existence + bare columns vs the referenced-table union
        tables_used = [lx.decode() for tid, lx in stream if tid == self._t_table]
        for t in tables_used:
            if t not in self.schema:
                out.append(Violation("unknown_table", t, "table not in schema"))
        in_scope: set[str] = set()
        for t in tables_used:
            in_scope |= self.schema.get(t, set())
        for i, (tid, lx) in enumerate(stream):
            if tid == self._t_column and i not in qualified_cols:
                c = lx.decode()
                if c not in in_scope:
                    out.append(Violation(
                        "column_not_in_referenced_tables", c,
                        f"column {c!r} not in referenced tables {sorted(set(tables_used))}",
                    ))
        return out
