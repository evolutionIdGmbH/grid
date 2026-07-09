"""The SQL-subset grammar in the three engines' native formats + replay corpus.

Language-parity caveat (recorded for the report): GRID's %ignore-WS lexer with
maximal munch and the explicit-whitespace EBNF/lark encodings agree on realistic
spaced SQL but can differ at adversarial keyword boundaries ("selectfoo" is one
identifier under maximal munch). The harness therefore replays token sequences
and counts per-engine acceptance instead of asserting mask equality.
"""

import pathlib

GRID_SQL = (pathlib.Path(__file__).parent.parent / "grammars" / "sql_subset.grid").read_text()

# XGrammar EBNF (GBNF-style). ws1 = mandatory separation at word-word boundaries.
XGRAMMAR_SQL = r"""
root ::= ws query ws ";" ws
query ::= select_stmt | insert_stmt | update_stmt | delete_stmt
select_stmt ::= "select" sel_tail
sel_tail ::= ws "*" ws1 from_part | ws1 column_list ws1 from_part
from_part ::= "from" ws1 ident where_opt limit_opt
column_list ::= ident (ws "," ws ident)*
where_opt ::= "" | ws1 "where" ws1 condition
condition ::= predicate (ws1 logic ws1 predicate)*
logic ::= "and" | "or"
predicate ::= ident ws cmp ws value | "(" ws condition ws ")"
cmp ::= "=" | "<=" | ">=" | "<>" | "<" | ">"
value ::= number | string | ident
limit_opt ::= "" | ws1 "limit" ws1 number
insert_stmt ::= "insert" ws1 "into" ws1 ident ws "(" ws column_list ws ")" ws "values" ws "(" ws value_list ws ")"
value_list ::= value (ws "," ws value)*
update_stmt ::= "update" ws1 ident ws1 "set" ws1 assign_list where_opt
assign_list ::= assign (ws "," ws assign)*
assign ::= ident ws "=" ws value
delete_stmt ::= "delete" ws1 "from" ws1 ident where_opt
ident ::= [a-z_] [a-z0-9_]*
number ::= [0-9]+
string ::= "'" [^'\n]* "'"
ws ::= [ \t\n]*
ws1 ::= [ \t\n]+
"""

# llguidance lark syntax (what Outlines 1.3's default CFG backend consumes).
# WS is one-or-more (empty-matchable terminals are illegal in every engine);
# ows = optional whitespace rule.
LLGUIDANCE_SQL = r"""
start: ows query ";" ows
query: select_stmt | insert_stmt | update_stmt | delete_stmt
select_stmt: "select" sel_tail
sel_tail: ows "*" WS from_part | WS column_list WS from_part
from_part: "from" WS IDENT where_opt limit_opt
column_list: IDENT (ows "," ows IDENT)*
where_opt: | WS "where" WS condition
condition: predicate (WS LOGIC WS predicate)*
predicate: IDENT ows CMP ows value | "(" ows condition ows ")"
value: NUMBER | STRING | IDENT
limit_opt: | WS "limit" WS NUMBER
insert_stmt: "insert" WS "into" WS IDENT ows "(" ows column_list ows ")" ows "values" ows "(" ows value_list ows ")"
value_list: value (ows "," ows value)*
update_stmt: "update" WS IDENT WS "set" WS assign_list where_opt
assign_list: assign (ows "," ows assign)*
assign: IDENT ows "=" ows value
delete_stmt: "delete" WS "from" WS IDENT where_opt
ows: | WS
LOGIC: "and" | "or"
CMP: "=" | "<=" | ">=" | "<>" | "<" | ">"
IDENT: /[a-z_][a-z0-9_]*/
NUMBER: /[0-9]+/
STRING: /'[^'\n]*'/
WS: /[ \t\n]+/
"""

CORPUS = [
    "select * from users;",
    "select id, name from users where id = 42;",
    "select email from users where id >= 10 and name = 'bob' limit 5;",
    "insert into orders (id, user_id, total) values (1, 2, 30);",
    "update users set name = 'x', email = 'y' where id = 7;",
    "delete from orders where total < 100 or user_id = 3;",
    "select id from orders where (total > 5 and user_id = 2) or id = 1;",
    "select total from orders where user_id = 12 and (total >= 100 or id <> 4) limit 20;",
]
