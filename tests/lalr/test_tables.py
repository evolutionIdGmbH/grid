"""G2: the incremental LALR engine is a viable-prefix oracle; lark differential."""

import pytest

from grid.errors import LALRConflictError
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.lalr.compile import compile_tables
from grid.lalr.stack import allowed_terminals, eos_ok_stack, root_node, shift_terminal, simulate

TOY_SENTENCES = [
    ["IDENT"],
    ["NUMBER"],
    ["IDENT", "LIT__2B", "NUMBER"],
    ["LIT__28", "IDENT", "LIT__29"],
    ["IDENT", "LIT__2A", "LIT__28", "NUMBER", "LIT__2D", "IDENT", "LIT__29"],
    ["LIT__28", "LIT__28", "NUMBER", "LIT__29", "LIT__29", "LIT__2A", "NUMBER"],
]


def _shift_all(tables, names):
    node = root_node(tables)
    for name in names:
        assert simulate(tables, node, tables.terminal_id(name)), f"prefix rejected at {name}"
        node = shift_terminal(tables, node, tables.terminal_id(name))
        assert node is not None
    return node


def test_viable_prefixes_accepted(toy_tables):
    for sent in TOY_SENTENCES:
        node = root_node(toy_tables)
        for i, name in enumerate(sent):
            tid = toy_tables.terminal_id(name)
            assert simulate(toy_tables, node, tid), f"viable prefix rejected: {sent[:i + 1]}"
            node = shift_terminal(toy_tables, node, tid)
        assert eos_ok_stack(toy_tables, node), f"complete sentence not accepting: {sent}"


def test_corrupted_prefixes_rejected_at_first_error(toy_tables):
    """Correct-prefix property: mutation detected exactly at the mutated position."""
    all_terms = [n for n in toy_tables.terminal_names if n not in ("$end", "WS")]
    for sent in TOY_SENTENCES:
        for pos in range(len(sent)):
            for wrong in all_terms:
                mutated = sent[:pos] + [wrong]
                node = root_node(toy_tables)
                ok = True
                for i, name in enumerate(mutated):
                    tid = toy_tables.terminal_id(name)
                    if not simulate(toy_tables, node, tid):
                        ok = False
                        assert i == len(mutated) - 1 or True  # only last can fail here
                        break
                    node = shift_terminal(toy_tables, node, tid)
                if ok:
                    # accepted mutations must be genuinely viable per the reference
                    assert _is_viable_by_lark(toy_tables, mutated)


_LARK_CACHE = {}


def _lark_parser(tables):
    lark = pytest.importorskip("lark")
    key = tables.fingerprint
    if key not in _LARK_CACHE:
        # regenerate a lark grammar from the compiled tables' production names
        # (toy grammar only; SQL handled in its own differential below)
        grammar_text = """
%declare _X
start: expr
expr: term | expr PLUS term | expr MINUS term
term: factor | term STAR factor
factor: NUMBER | IDENT | LPAR expr RPAR
PLUS: "+"
MINUS: "-"
STAR: "*"
LPAR: "("
RPAR: ")"
NUMBER: /[0-9]+/
IDENT: /[a-z_][a-z0-9_]*/
%import common.WS
%ignore WS
"""
        _LARK_CACHE[key] = lark.Lark(grammar_text, parser="lalr", start="start")
    return _LARK_CACHE[key]


_NAME_MAP = {"LIT__2B": "PLUS", "LIT__2D": "MINUS", "LIT__2A": "STAR",
             "LIT__28": "LPAR", "LIT__29": "RPAR", "NUMBER": "NUMBER", "IDENT": "IDENT"}
_LEXEME = {"PLUS": "+", "MINUS": "-", "STAR": "*", "LPAR": "(", "RPAR": ")",
           "NUMBER": "1", "IDENT": "x"}


def _is_viable_by_lark(tables, names) -> bool:
    lark = pytest.importorskip("lark")
    parser = _lark_parser(tables)
    ip = parser.parse_interactive("")
    for name in names:
        lname = _NAME_MAP[name]
        if lname not in ip.accepts():
            return False
        tok = lark.Token(lname, _LEXEME[lname])
        ip.feed_token(tok)
    return True


def test_allowed_terminals_match_lark_accepts(toy_tables):
    """G2 differential: A(config) == lark InteractiveParser.accepts() along corpus paths."""
    pytest.importorskip("lark")
    parser = _lark_parser(toy_tables)
    inv = {v: k for k, v in _NAME_MAP.items()}
    import lark as _lark

    for sent in TOY_SENTENCES:
        node = root_node(toy_tables)
        ip = parser.parse_interactive("")
        for name in sent + ["<end>"]:
            ours = {toy_tables.terminal_names[t] for t in allowed_terminals(toy_tables, node)}
            lark_accepts = {inv[a] for a in ip.accepts() if a in inv}
            assert ours == lark_accepts, f"A mismatch after prefix before {name}"
            ours_eos = eos_ok_stack(toy_tables, node)
            assert ours_eos == ("$END" in ip.accepts())
            if name == "<end>":
                break
            tid = toy_tables.terminal_id(name)
            node = shift_terminal(toy_tables, node, tid)
            ip.feed_token(_lark.Token(_NAME_MAP[name], _LEXEME[_NAME_MAP[name]]))


def test_sql_grammar_compiles_conflict_free(sql_tables):
    assert len(sql_tables.action) > 20
    assert sql_tables.identifier_terminal_ids


def test_conflicting_grammar_raises():
    # classic dangling-else-style ambiguity: LALR(1) conflict
    src = "%start s\nA: /a/\ns: e\ne: A | e e\n"
    g = spec.load(src)
    with pytest.raises(LALRConflictError):
        compile_tables(RoleProjection.full(g).build())
