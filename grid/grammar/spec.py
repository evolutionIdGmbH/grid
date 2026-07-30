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

from grid import perf_flags
from grid._statecharts.engine import Statechart, load_chart
from grid.errors import GrammarInvalid
from grid.grammar.parts import GrammarParts, render_text

_TERM_DEF = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.+)$")
_RULE_DEF = re.compile(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+)$")
_RHS_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([A-Z][A-Z0-9_]*)|([a-z_][a-z0-9_]*)|(\|)')
_UNESCAPE = re.compile(r"\\(.)")


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

    # -- direct emission (P2) ----------------------------------------------

    @classmethod
    def from_parts(cls, parts: GrammarParts) -> DialectGrammar:
        """Build a FROZEN grammar straight from a compiler GrammarParts
        manifest — the object twin of ``load(render_text(parts))`` with the
        text render and regex re-parse skipped.

        Replicates ONLY the _parse_source contract (decl_index assignment:
        named defs in manifest order, then literal terminals at first token
        use over the start line and rules in order; quoted-token unescaping;
        epsilon alternatives; %start/%ignore wiring). validate() and
        freeze() then run verbatim — GrammarInvalid outcomes (unproductive
        recursive schemas), L-REC01 warnings, and terminal_order numbering
        are shared code with the text path, not replicas.

        GRID_PERF_DIRECT_EMIT_CHECK=1 (CI oracle): also load the rendered
        text and assert both paths agree — on the full grammar identity for
        valid manifests, on the GrammarInvalid message otherwise. (The
        oracle arm runs validate() twice, so warning COUNTS double under
        check mode; message parity is unaffected.)"""
        g = cls(source="")
        try:
            g._build_from_parts(parts)
        except GrammarInvalid:
            g._sc.fire("parse_error")
            raise
        g._sc.fire("parse_ok")
        if perf_flags.direct_emit_check_enabled():
            return g._validate_freeze_checked(parts)
        return g.validate().freeze()

    def _validate_freeze_checked(self, parts: GrammarParts) -> DialectGrammar:
        """Check-mode tail of from_parts: text-path oracle for outcome AND
        identity parity. Raises AssertionError on any divergence."""
        try:
            oracle = load(render_text(parts))
        except GrammarInvalid as text_err:
            try:
                self.validate()
            except GrammarInvalid as obj_err:
                if str(obj_err) != str(text_err):
                    raise AssertionError(
                        "direct-emit check: divergent GrammarInvalid: "
                        f"object={obj_err} text={text_err}") from obj_err
                raise   # identical outcome; propagate the object-path error
            raise AssertionError(
                f"direct-emit check: text path invalid ({text_err}) "
                "but object path validated") from text_err
        try:
            self.validate()
        except GrammarInvalid as obj_err:
            raise AssertionError(
                f"direct-emit check: object path invalid ({obj_err}) "
                "but text path validated") from obj_err
        self.freeze()
        for attr in ("start", "ignored", "terminals", "productions",
                     "terminal_order", "fingerprint"):
            if getattr(self, attr) != getattr(oracle, attr):
                raise AssertionError(
                    f"direct-emit check: object/text divergence in {attr}")
        return self

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

    def _build_from_parts(self, parts: GrammarParts) -> None:
        """Replay _parse_source's single decl counter over the manifest.

        The parse contract being replicated (and nothing more): every named
        terminal def takes the next decl_index in manifest order; quoted
        rule-body tokens become literal terminals at FIRST use — start line
        first, then rules in manifest order, tokens left to right — with
        _parse_rhs's unescape; ''-tuples are epsilon productions; start is
        the synthetic 'start' rule and WS the sole ignored terminal (the
        emitter's fixed header, see grid/grammar/parts.py)."""
        terminals = self.terminals
        decl = 0
        for name, pattern in parts.terminal_defs:
            if name in terminals:
                raise GrammarInvalid(f"duplicate terminal {name}")
            terminals[name] = Terminal(name, pattern, is_literal=False,
                                       ignored=False, decl_index=decl)
            decl += 1
        productions = self.productions
        lit_names: dict[str, str] = {}   # raw quoted token -> terminal name
        for lhs, alts in (("start", ((parts.start_target,),)),) + parts.rules:
            for alt in alts:
                syms: list[str] = []
                for tok in alt:
                    if tok[:1] != '"':
                        syms.append(tok)
                        continue
                    name = lit_names.get(tok)
                    if name is None:
                        if len(tok) < 2 or tok[-1] != '"':
                            raise GrammarInvalid(
                                f"bad rhs token in rule {lhs!r} at: {tok!r}")
                        text = _UNESCAPE.sub(r"\1", tok[1:-1])
                        if not text:
                            raise GrammarInvalid(f"empty literal in rule {lhs!r}")
                        name = _literal_terminal_name(text)
                        if name not in terminals:
                            terminals[name] = Terminal(name, text, is_literal=True,
                                                       ignored=False, decl_index=decl)
                            decl += 1
                        lit_names[tok] = name
                    syms.append(name)
                productions.append(Production(lhs, tuple(syms)))
        self.start = "start"
        ws = terminals.get("WS")
        if ws is None:
            raise GrammarInvalid("%ignore references unknown terminal WS")
        # same in-place ignored-flag rewrite as _parse_source: dict key
        # reassignment keeps WS's original insertion position
        terminals["WS"] = Terminal(ws.name, ws.pattern, ws.is_literal, True,
                                   ws.decl_index)
        self.ignored = frozenset(("WS",))

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
