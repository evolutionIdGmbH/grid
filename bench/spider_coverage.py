"""Coverage oracle: grammars/sql_spider.grid vs the Spider dev gold set.

Gold queries are normalized to the grammar's canonical form (lowercase outside
string literals, double-quoted strings -> single-quoted, trailing ';' stripped)
and parsed with GRID's own contextual scan+shift discipline — the same
parse_terminal_stream mechanics the SemanticChecker uses — with per-database L3
lexicons steering the TABLE_NAME/COLUMN_NAME/ALIAS choice, plus an end-of-input
viability check. This is the viable-prefix oracle for the Spider dialect: 100% of
the 1034 dev golds parse (grammar committed at that state).

Run:  .venv-bench/bin/python bench/spider_coverage.py --spider <spider_data dir>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import compile_tables
from grid.lalr.stack import root_node, shift_terminal, simulate
from grid.lexer.dfa import build_scanner
from grid.lexer.run import LexerRun, ScanReject, scan
from grid.trie.walk import Lexicons, pick_viable

GRAMMAR = (pathlib.Path(__file__).parent.parent / "grammars" / "sql_spider.grid").read_text()


def normalize(sql: str) -> str | None:
    """Lowercase outside quotes; double->single quotes; None if unrepresentable."""
    out = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in ("'", '"'):
            j = i + 1
            while j < n and sql[j] != c:
                j += 1
            if j >= n:
                return None
            body = sql[i + 1:j]
            if "'" in body:
                return None
            out.append("'" + body + "'")
            i = j + 1
        else:
            out.append(c.lower())
            i += 1
    s = "".join(out).strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", s)


_IDENT = re.compile(r"[a-z_][a-z0-9_]*")


def db_lexicons(tables_json: list, tables) -> dict[str, Lexicons]:
    """Names outside the identifier language are dropped (see the composition
    precondition in grid/mask/producer.py — validated at guide build)."""
    t_id = tables.terminal_names.index("TABLE_NAME")
    c_id = tables.terminal_names.index("COLUMN_NAME")
    out = {}
    for db in tables_json:
        tn = {n.lower().encode() for n in db["table_names_original"] if _IDENT.fullmatch(n.lower())}
        cn = {c.lower().encode() for _t, c in db["column_names_original"]
              if c != "*" and _IDENT.fullmatch(c.lower())}
        out[db["db_id"]] = Lexicons({t_id: tn, c_id: cn})
    return out


def parse_ok(tables, dfa, priority, data: bytes, lexicons=None) -> tuple[bool, str]:
    try:
        events, tail = scan(dfa, data)
    except ScanReject as e:
        return False, f"scan: {e}"
    fin = LexerRun(remainder=tail).finalize(dfa)
    if fin is None:
        return False, "finalize"
    node = root_node(tables)
    off = 0
    for ev in list(events) + list(fin):
        seg = data[off:off + ev.length]
        off += ev.length
        viable = frozenset(
            t for t in ev.candidates
            if t not in tables.ignored_terminal_ids and simulate(tables, node, t)
        )
        pick = pick_viable(ev, seg, viable, tables.ignored_terminal_ids, priority, lexicons)
        if pick is None:
            return False, f"no-viable@{seg[:14]!r}"
        if pick in tables.ignored_terminal_ids:
            continue
        nxt = shift_terminal(tables, node, pick)
        if nxt is None:
            return False, f"shift@{seg[:14]!r}"
        node = nxt
    if not simulate(tables, node, tables.end_id):
        return False, "incomplete-at-end"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider", required=True, help="spider_data dir (dev.json, tables.json)")
    args = ap.parse_args()

    grammar = spec.load(GRAMMAR)
    proj = RoleProjection.full(grammar).build()
    tables = compile_tables(proj)
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    priority = {
        tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
        for tid in range(tables.n_terminals)
    }
    print(f"grammar OK: {tables.n_terminals} terminals, {len(tables.action)} states")

    dev = json.load(open(f"{args.spider}/dev.json"))
    lex = db_lexicons(json.load(open(f"{args.spider}/tables.json")), tables)
    fails = Counter()
    examples: dict[str, str] = {}
    skipped = ok = 0
    for ex in dev:
        norm = normalize(ex["query"])
        if norm is None:
            skipped += 1
            continue
        good, why = parse_ok(tables, dfa, priority, norm.encode(), lex.get(ex["db_id"]))
        if good:
            ok += 1
        else:
            fails[why.split("@")[0]] += 1
            examples.setdefault(why, norm[:110])
    total = len(dev) - skipped
    print(f"coverage: {ok}/{total} = {ok / total:.1%}  (skipped {skipped} un-normalizable)")
    if fails:
        print("fail buckets:", dict(fails.most_common()))
        for k, q in list(examples.items())[:10]:
            print(f"  [{k}]\n    {q}")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
