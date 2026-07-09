"""E4a ReserveTable: token-denominated min-completion costs + completion synthesis.

Costs are in MODEL TOKENS (a terminal-denominated reserve under-reserves: one
identifier can cost many tokens — DESIGN.md E4a). Per-terminal cost =
``len(greedy_tokenize(b" " + shortest_lexeme))``; the Write-span renderer joins
completion lexemes with single spaces, so the DP cost equals the rendered cost
at lexeme boundaries by construction (grammars must %ignore a whitespace
terminal — asserted at build).

``completion(node)`` is the exact minimal completion from a stack configuration:
memoized recursion over (base node identity, virtual overlay) using the per-state
item cores. In-progress keys return +inf (cycle guard); costs are non-negative so
the optimum is acyclic.

Trigger safety: minimal-completion length can grow after one more token (opening
a paren), so the guide triggers at ``budget_remaining <= len(concrete) + SAFETY``
(``grid.guide.RESERVE_SAFETY`` = 8) — recorded in the DESIGN decision log as
reserve_slack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from grid.lalr.compile import LALRTables
from grid.lalr.stack import StackNode
from grid.lexer.dfa import DEAD, ScannerDFA

INF = math.inf


def shortest_lexemes(dfa: ScannerDFA, n_terminals: int) -> dict[int, bytes]:
    """BFS per DFA state; smallest-byte tie-break for reproducibility."""
    out: dict[int, bytes] = {}
    frontier: list[tuple[int, bytes]] = [(dfa.start, b"")]
    seen = {dfa.start}
    while frontier:
        nxt: list[tuple[int, bytes]] = []
        for st, path in frontier:
            for t in dfa.accepts_all[st]:
                if t not in out:
                    out[t] = path
        for st, path in frontier:
            for byte in range(256):
                ns = dfa.trans[st][byte]
                if ns != DEAD and ns not in seen:
                    seen.add(ns)
                    nxt.append((ns, path + bytes([byte])))
        frontier = nxt
    return out


@dataclass
class ReserveTable:
    """Keyed by (grammar fingerprint, tokenizer fingerprint) — separate artifact (E4a).

    ``lexicons``: L3 identifier categories override the BFS-shortest lexeme with the
    shortest ALLOWED identifier (DESIGN E4a) — a completion must itself pass the
    identifier composition rule. An empty allow-list makes the terminal
    uncompletable (cost INF), which surfaces as EmptyLanguage-like policy errors,
    never as an illegal completion.
    """

    tables: LALRTables
    dfa: ScannerDFA
    adapter: object                       # TokenizerAdapter: greedy_tokenize
    lexicons: object | None = None        # trie.walk.Lexicons
    fingerprint: str = ""
    term_cost: dict[int, int] = field(default_factory=dict)
    term_lexeme: dict[int, bytes] = field(default_factory=dict)
    nt_cost: dict[int, float] = field(default_factory=dict)
    nt_seq: dict[int, tuple[int, ...]] = field(default_factory=dict)
    _memo: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = self.tables
        lex = shortest_lexemes(self.dfa, t.n_terminals)
        space_ok = any(
            lex.get(i) == b" " or (lex.get(i) and set(lex[i]) <= {0x20, 0x9, 0xA})
            for i in t.ignored_terminal_ids
        )
        assert space_ok, "reserve rendering requires an ignored whitespace terminal (DESIGN E4a)"
        allowed = getattr(self.lexicons, "allowed", {}) if self.lexicons is not None else {}
        for tid in range(t.n_terminals):
            if tid == t.end_id or tid in t.ignored_terminal_ids:
                continue
            if tid in allowed:
                words = allowed[tid]
                if not words:
                    continue  # uncompletable under this policy -> cost INF
                self.term_lexeme[tid] = min(words, key=lambda w: (len(w), w))
            elif tid in lex:
                self.term_lexeme[tid] = lex[tid]
            else:
                continue
            self.term_cost[tid] = len(self.adapter.greedy_tokenize(b" " + self.term_lexeme[tid]))  # type: ignore[attr-defined]
        self._compute_nt()
        self.fingerprint = f"{t.fingerprint}:{getattr(self.adapter, 'token_bytes', None) and 'tok'}"

    def _compute_nt(self) -> None:
        t = self.tables
        cost: dict[int, float] = {}
        seq: dict[int, tuple[int, ...]] = {}
        nts = {lhs for lhs, _ in t.prods}
        for nt in nts:
            cost[nt] = INF
            seq[nt] = ()
        changed = True
        while changed:
            changed = False
            for lhs, rhs in t.prods[1:]:
                c = 0.0
                s: tuple[int, ...] = ()
                ok = True
                for sym in rhs:
                    if sym < t.n_terminals:
                        tc = self.term_cost.get(sym)
                        if tc is None:
                            ok = False
                            break
                        c += tc
                        s += (sym,)
                    else:
                        if cost.get(sym, INF) == INF:
                            ok = False
                            break
                        c += cost[sym]
                        s += seq[sym]
                if ok and c < cost[lhs]:
                    cost[lhs] = c
                    seq[lhs] = s
                    changed = True
        self.nt_cost = cost
        self.nt_seq = seq

    # -- exact stack completion (SS6 step 3) ---------------------------------

    def completion(self, node: StackNode) -> tuple[float, tuple[int, ...]]:
        """Minimal (token cost, terminal sequence) completing the configuration."""
        return self._complete(node, ())

    def _complete(self, base: StackNode, overlay: tuple[tuple[int, int], ...]) -> tuple[float, tuple[int, ...]]:
        key = (base, overlay)
        cached = self._memo.get(key)
        if cached is not None:
            return cached if cached != "busy" else (INF, ())
        self._memo[key] = "busy"
        t = self.tables
        top_state = overlay[-1][0] if overlay else base.state
        best: tuple[float, tuple[int, ...]] = (INF, ())
        for (p, d) in t.state_items[top_state]:
            lhs, rhs = t.prods[p]
            if p == 0 and d == 1:
                best = min(best, (0.0, ()), key=lambda x: x[0])
                continue
            if d > len(overlay) + base.depth:
                continue
            beta = rhs[d:]
            c = 0.0
            s: tuple[int, ...] = ()
            ok = True
            for sym in beta:
                if sym < t.n_terminals:
                    tc = self.term_cost.get(sym)
                    if tc is None:
                        ok = False
                        break
                    c += tc
                    s += (sym,)
                else:
                    if self.nt_cost.get(sym, INF) == INF:
                        ok = False
                        break
                    c += self.nt_cost[sym]
                    s += self.nt_seq[sym]
            if not ok or c >= best[0]:
                continue
            nb, nov = base, list(overlay)
            pops = d
            while pops and nov:
                nov.pop()
                pops -= 1
            while pops:
                if nb.parent is None:
                    ok = False
                    break
                nb = nb.parent
                pops -= 1
            if not ok:
                continue
            origin_state = nov[-1][0] if nov else nb.state
            g = t.goto[origin_state].get(lhs)
            if g is None:
                continue
            rc, rs = self._complete(nb, tuple(nov) + ((g, lhs),))
            if c + rc < best[0]:
                best = (c + rc, s + rs)
        self._memo[key] = best
        return best

    def render(self, seq: tuple[int, ...]) -> bytes:
        """Completion bytes: space-joined shortest lexemes (space is ignored WS)."""
        return b"".join(b" " + self.term_lexeme[tid] for tid in seq)
