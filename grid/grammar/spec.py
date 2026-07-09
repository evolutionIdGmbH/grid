"""E1 DialectGrammar: grammar source format, parse/validate/freeze (DESIGN.md SS5 E1).

Format (.grid), line-oriented pure BNF:

    %start query
    %ignore WS
    WS: /[ \\t\\n]+/
    IDENT: /[a-z_][a-z0-9_]*/
    NUMBER: /[0-9]+/
    query: select_stmt ";"
    select_stmt: "select" cols "from" IDENT
    cols: "*" | col_list
    col_list: IDENT | col_list "," IDENT

- Terminals: UPPERCASE names with /regex/ patterns (subset: literals, escapes,
  [] classes with ranges/negation, ``.``, ``()``, ``|``, ``*``, ``+``, ``?``).
- Rules: lowercase names; alternatives split on ``|``; a line starting with ``|``
  continues the previous rule.
- Quoted string literals in rule bodies become anonymous literal terminals with
  priority above named terminals (keyword-vs-IDENT: longest match first, then
  literal beats named, then declaration order).
- Lexing discipline: maximal munch (DESIGN.md E7).

The canonical L1 terminal numbering (E11 requirement: projections subset
productions, never renumber terminals) is assigned at freeze in declaration
order, literals appended after named terminals in first-use order.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field

from grid._statecharts.engine import Statechart, load_chart
from grid.errors import GrammarInvalid

_TERM_DEF = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.+)$")
_RULE_DEF = re.compile(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+)$")
_RHS_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([A-Z][A-Z0-9_]*)|([a-z_][a-z0-9_]*)|(\|)')


def _literal_terminal_name(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]", lambda m: f"_{ord(m.group(0)):02X}", text)
    return f"LIT_{safe.upper()}"


@dataclass(frozen=True)
class Terminal:
    name: str
    pattern: str          # regex source (grid regex subset) or literal text
    is_literal: bool      # literal terminals: pattern is the exact text
    ignored: bool
    decl_index: int       # declaration order (priority tiebreak)

    @property
    def priority(self) -> tuple[int, int]:
        """Lower sorts first at equal match length: literals beat named terminals."""
        return (0 if self.is_literal else 1, self.decl_index)


@dataclass(frozen=True)
class Production:
    lhs: str
    rhs: tuple[str, ...]


@dataclass
class DialectGrammar:
    """E1 entity. Construct via :func:`load`; lifecycle DRAFT->PARSED->VALIDATED->FROZEN."""

    source: str
    start: str = ""
    terminals: dict[str, Terminal] = field(default_factory=dict)
    productions: list[Production] = field(default_factory=list)
    ignored: frozenset[str] = frozenset()
    fingerprint: str = ""
    terminal_order: tuple[str, ...] = ()   # canonical L1 numbering (index = terminal id)
    _sc: Statechart = field(default_factory=lambda: Statechart(load_chart("dialect_grammar")))

    @property
    def state(self) -> str:
        return self._sc.state

    @property
    def nonterminals(self) -> frozenset[str]:
        return frozenset(p.lhs for p in self.productions)

    # -- lifecycle ---------------------------------------------------------

    def parse(self) -> DialectGrammar:
        try:
            self._parse_source()
        except GrammarInvalid:
            self._sc.fire("parse_error")
            raise
        self._sc.fire("parse_ok")
        return self

    def validate(self) -> DialectGrammar:
        try:
            self._validate()
        except GrammarInvalid:
            self._sc.fire("validate_error")
            raise
        self._sc.fire("validate_ok")
        return self

    def freeze(self) -> DialectGrammar:
        named = [t for t in self.terminals.values() if not t.is_literal]
        lits = [t for t in self.terminals.values() if t.is_literal]
        ordered = sorted(named, key=lambda t: t.decl_index) + sorted(lits, key=lambda t: t.decl_index)
        self.terminal_order = tuple(t.name for t in ordered)
        self.fingerprint = self._fingerprint()
        self._sc.fire("freeze")
        return self

    # -- internals ---------------------------------------------------------

    def _parse_source(self) -> None:
        ignored: set[str] = set()
        decl = 0
        last_rule: str | None = None
        for raw_line in self.source.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("%start"):
                self.start = line.split(None, 1)[1].strip()
                continue
            if line.startswith("%ignore"):
                ignored.add(line.split(None, 1)[1].strip())
                continue
            if line.startswith("|"):
                if last_rule is None:
                    raise GrammarInvalid(f"continuation without rule: {line!r}")
                decl = self._parse_rhs(last_rule, line[1:], decl)
                continue
            m = _TERM_DEF.match(line)
            if m:
                name, pat = m.group(1), m.group(2).strip()
                if pat.startswith("/") and pat.endswith("/") and len(pat) >= 2:
                    pat = pat[1:-1]
                if name in self.terminals:
                    raise GrammarInvalid(f"duplicate terminal {name}")
                self.terminals[name] = Terminal(name, pat, is_literal=False, ignored=False, decl_index=decl)
                decl += 1
                last_rule = None
                continue
            m = _RULE_DEF.match(line)
            if m:
                last_rule = m.group(1)
                decl = self._parse_rhs(last_rule, m.group(2), decl)
                continue
            raise GrammarInvalid(f"unparseable line: {line!r}")

        if not self.start:
            raise GrammarInvalid("missing %start")
        for name in ignored:
            if name not in self.terminals:
                raise GrammarInvalid(f"%ignore references unknown terminal {name}")
            t = self.terminals[name]
            self.terminals[name] = Terminal(t.name, t.pattern, t.is_literal, True, t.decl_index)
        self.ignored = frozenset(ignored)

    def _parse_rhs(self, lhs: str, rhs_text: str, decl: int) -> int:
        alt: list[str] = []
        pos = 0
        stripped = rhs_text.strip()
        while pos < len(stripped):
            if stripped[pos].isspace():
                pos += 1
                continue
            m = _RHS_TOKEN.match(stripped, pos)
            if not m:
                raise GrammarInvalid(f"bad rhs token in rule {lhs!r} at: {stripped[pos:]!r}")
            pos = m.end()
            lit, term, rule, bar = m.group(1), m.group(2), m.group(3), m.group(4)
            if bar:
                self.productions.append(Production(lhs, tuple(alt)))
                alt = []
            elif lit is not None:
                text = re.sub(r"\\(.)", r"\1", lit)
                if not text:
                    raise GrammarInvalid(f"empty literal in rule {lhs!r}")
                name = _literal_terminal_name(text)
                if name not in self.terminals:
                    self.terminals[name] = Terminal(name, text, is_literal=True, ignored=False, decl_index=decl)
                    decl += 1
                alt.append(name)
            elif term:
                alt.append(term)
            else:
                alt.append(rule)
        self.productions.append(Production(lhs, tuple(alt)))
        return decl

    def _validate(self) -> None:
        nts = self.nonterminals
        if self.start not in nts:
            raise GrammarInvalid(f"start symbol {self.start!r} has no productions")
        for p in self.productions:
            for sym in p.rhs:
                if sym.isupper() or sym.startswith("LIT_"):
                    if sym not in self.terminals:
                        raise GrammarInvalid(f"rule {p.lhs!r} references unknown terminal {sym!r}")
                elif sym not in nts:
                    raise GrammarInvalid(f"rule {p.lhs!r} references unknown rule {sym!r}")
        for name in self.terminals:
            if self.terminals[name].ignored:
                for p in self.productions:
                    if name in p.rhs:
                        raise GrammarInvalid(f"ignored terminal {name} used in rule {p.lhs!r}")
        from grid.grammar.reduction import useless_symbols

        useless = useless_symbols(self.productions, self.start)
        if useless:
            raise GrammarInvalid(f"grammar not reduced; useless symbols: {sorted(useless)}")
        self._lint_right_recursion()

    def _lint_right_recursion(self) -> None:
        for p in self.productions:
            if len(p.rhs) >= 2 and p.rhs[-1] == p.lhs:
                warnings.warn(
                    f"L-REC01: rule {p.lhs!r} is right-recursive; prefer left recursion for lists "
                    "(affects only the per-step depth bound, DESIGN.md SS5 E1)",
                    stacklevel=3,
                )

    def _fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.start.encode())
        for name in sorted(self.terminals):
            t = self.terminals[name]
            h.update(f"T|{t.name}|{t.pattern}|{t.is_literal}|{t.ignored}".encode())
        for p in self.productions:
            h.update(f"P|{p.lhs}|{'|'.join(p.rhs)}".encode())
        h.update(("I|" + ",".join(sorted(self.ignored))).encode())
        return h.hexdigest()


def load(source: str) -> DialectGrammar:
    """Parse, validate, and freeze a dialect grammar from source text."""
    return DialectGrammar(source=source).parse().validate().freeze()
