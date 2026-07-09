"""ReferenceGuide: the executable specification (DESIGN.md SS2, G3 oracle).

For every candidate token it re-derives viability FROM SCRATCH on the full byte
string (no trie pruning, no cache, no incremental parser state, no CI/CD split) —
a brute-force trial-parse loop over the vocabulary. Slow and obviously correct;
tests differentially bind the fast path to this.
"""

from __future__ import annotations

from grid.lalr.compile import LALRTables
from grid.lalr.stack import (
    StackNode,
    allowed_terminals,
    eos_ok_stack,
    root_node,
    shift_terminal,
    simulate,
)
from grid.lexer.dfa import DEAD, ScannerDFA
from grid.lexer.run import ScanReject, scan
from grid.trie.walk import Lexicons, pick_viable


class ReferenceGuide:
    def __init__(self, tables: LALRTables, dfa: ScannerDFA, adapter,
                 lexicons: Lexicons | None = None) -> None:
        self.tables = tables
        self.dfa = dfa
        self.adapter = adapter
        self.lexicons = lexicons
        self.eos_token_id = adapter.eos_token_id
        self._priority = {
            tid: (0 if tid in tables.literal_terminal_ids else 1, tid)
            for tid in range(tables.n_terminals)
        }

    # -- from-scratch parse of a whole byte string ---------------------------

    def _parse_prefix(self, data: bytes) -> tuple[StackNode, bytes] | None:
        """Parse data as a viable prefix; returns (stack, partial remainder) or None."""
        try:
            events, tail = scan(self.dfa, data)
        except ScanReject:
            return None
        node: StackNode | None = root_node(self.tables)
        offset = 0
        for ev in events:
            assert node is not None
            seg = data[offset:offset + ev.length]
            offset += ev.length
            viable = frozenset(
                t for t in ev.candidates
                if t not in self.tables.ignored_terminal_ids and simulate(self.tables, node, t)
            )
            pick = pick_viable(ev, seg, viable, self.tables.ignored_terminal_ids,
                               self._priority, self.lexicons)
            if pick is None:
                return None
            if pick in self.tables.ignored_terminal_ids:
                continue
            node = shift_terminal(self.tables, node, pick)
            if node is None:
                return None
        assert node is not None
        if tail:
            st = self.dfa.scan_state(tail)
            if st == DEAD:
                return None
            a_now = allowed_terminals(self.tables, node)
            ok = False
            for t in self.dfa.live[st]:
                if t in self.tables.ignored_terminal_ids:
                    ok = True
                    break
                if t in a_now and (self.lexicons is None or self.lexicons.prefix_ok(t, tail)):
                    ok = True
                    break
            if not ok:
                return None
        return node, tail

    def viable_prefix(self, data: bytes) -> bool:
        return self._parse_prefix(data) is not None

    def eos_legal(self, data: bytes) -> bool:
        """Full-string membership: scan-all + finalize + $end acceptance."""
        parsed = self._parse_prefix(data)
        if parsed is None:
            return False
        node, tail = parsed
        if tail:
            # greedy finalize of the tail (SS6 step 2 winning segmentation)
            from grid.lexer.run import LexerRun

            events = LexerRun(remainder=tail).finalize(self.dfa)
            if events is None:
                return False
            offset = 0
            cur: StackNode | None = node
            for ev in events:
                assert cur is not None
                seg = tail[offset:offset + ev.length]
                offset += ev.length
                viable = frozenset(
                    t for t in ev.candidates
                    if t not in self.tables.ignored_terminal_ids and simulate(self.tables, cur, t)
                )
                pick = pick_viable(ev, seg, viable, self.tables.ignored_terminal_ids,
                                   self._priority, self.lexicons)
                if pick is None:
                    return False
                if pick in self.tables.ignored_terminal_ids:
                    continue
                cur = shift_terminal(self.tables, cur, pick)
                if cur is None:
                    return False
            node = cur
        assert node is not None
        return eos_ok_stack(self.tables, node)

    # -- the oracle mask ------------------------------------------------------

    def valid_next_tokens(self, prefix_ids: list[int]) -> set[int]:
        """Trial-parse every vocabulary token on the detokenized prefix (G3 oracle)."""
        prefix = b"".join(self.adapter.token_bytes(t) for t in prefix_ids)
        special = getattr(self.adapter, "special_token_ids", frozenset())
        out: set[int] = set()
        for tid in self.adapter.vocabulary.values():
            if tid == self.eos_token_id:
                if self.eos_legal(prefix):
                    out.add(tid)
                continue
            if tid in special:
                continue
            data = self.adapter.token_bytes(tid)
            if data and self.viable_prefix(prefix + data):
                out.add(tid)
        return out
