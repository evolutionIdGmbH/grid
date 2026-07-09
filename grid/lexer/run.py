"""E7 LexerRun: immutable per-stream lexer status with maximal-munch advance.

The run stores only ``remainder`` — the bytes of the single in-progress partial
lexeme (every prefix whose emission was forced by a dead transition has already
been emitted). DFA state and hypotheses are derived by scanning the remainder,
which is bounded by the longest in-flight lexeme (DESIGN.md E7: a small value
object, plain copies).

Contextual emission (keyword-vs-identifier, TABLE_NAME-vs-COLUMN_NAME): maximal
munch runs over the UNION automaton; an emission event carries the full candidate
set ``accepts_all`` at the longest match, and the CALLER (guide/walk/reference —
all of which know the parser-viable terminal set) picks the winner:
highest-priority parser-viable candidate, else an ignored candidate, else reject.
Known v0 limitation (recorded): a parser-viable terminal that accepts only at a
strictly shorter position than the union-longest match is not chosen; the G3
differential surfaces any grammar where this bites.

Maximal munch means a lexeme is emitted only when a byte cannot extend the match
(forced emission) — a token ending exactly at an accepting state stays in the
remainder because a longer match may follow ("sel" | "select").
"""

from __future__ import annotations

from dataclasses import dataclass

from grid.errors import LexerHypothesisOverflow
from grid.lexer.dfa import DEAD, ScannerDFA


class ScanReject(Exception):
    """Internal signal: byte sequence is not lexable here (masked paths never see this)."""


@dataclass(frozen=True)
class EmissionEvent:
    """A forced lexeme emission: the caller chooses one terminal from ``candidates``."""

    candidates: frozenset[int]   # accepts_all at the longest match
    length: int                  # lexeme byte length


@dataclass(frozen=True)
class LexerRun:
    remainder: bytes = b""

    def state(self, dfa: ScannerDFA) -> int:
        st = dfa.scan_state(self.remainder)
        if st == DEAD:  # pragma: no cover - construction guarantees scannable remainders
            raise AssertionError("LexerRun invariant: remainder must be scannable")
        return st

    def hypotheses(self, dfa: ScannerDFA) -> frozenset[int]:
        live = dfa.live[self.state(dfa)]
        if len(live) > dfa.h_max:  # pragma: no cover - h_max is the max by construction
            raise LexerHypothesisOverflow(f"{len(live)} > H_max={dfa.h_max}")
        return live

    def at_boundary(self) -> bool:
        return not self.remainder

    def advance(self, dfa: ScannerDFA, new_bytes: bytes) -> tuple[LexerRun, tuple[EmissionEvent, ...]]:
        """Consume bytes; return (new run, forced emission events in order).

        Raises ScanReject if a byte is not lexable from the current position.
        """
        events, rem = scan(dfa, self.remainder + new_bytes)
        return LexerRun(remainder=rem), tuple(events)

    def finalize(self, dfa: ScannerDFA) -> tuple[EmissionEvent, ...] | None:
        """End-of-input: greedily segment the remainder into complete lexemes.

        Returns emission events (maximal-munch segmentation — THE winning
        hypothesis per SS6 step 2), or None if the remainder cannot be fully
        consumed (EOS illegal mid-lexeme). An empty remainder yields ().
        """
        out: list[EmissionEvent] = []
        buf = self.remainder
        i = 0
        while i < len(buf):
            st = dfa.start
            last: tuple[int, int] | None = None  # (end, state)
            j = i
            while j < len(buf):
                st = dfa.trans[st][buf[j]]
                if st == DEAD:
                    break
                j += 1
                if dfa.accept[st] != -1:
                    last = (j, st)
            if last is None:
                return None
            end, acc_state = last
            out.append(EmissionEvent(dfa.accepts_all[acc_state], end - i))
            i = end
        return tuple(out)


def scan(dfa: ScannerDFA, buf: bytes) -> tuple[list[EmissionEvent], bytes]:
    """Maximal-munch scan with forced emission; returns (events, remainder)."""
    events: list[EmissionEvent] = []
    i = 0
    while True:
        st = dfa.start
        last: tuple[int, int] | None = None  # (end, accepting state)
        j = i
        dead = False
        while j < len(buf):
            nx = dfa.trans[st][buf[j]]
            if nx == DEAD:
                dead = True
                break
            st = nx
            j += 1
            if dfa.accept[st] != -1:
                last = (j, st)
        if not dead:
            return events, buf[i:]
        if last is None:
            raise ScanReject(f"illegal byte 0x{buf[j]:02X} at offset {j}")
        end, acc_state = last
        events.append(EmissionEvent(dfa.accepts_all[acc_state], end - i))
        i = end


def choose_terminal(
    event: EmissionEvent,
    viable: frozenset[int] | set[int],
    ignored: frozenset[int],
    priority: dict[int, tuple[int, int]],
) -> int | None:
    """Contextual emission choice: highest-priority parser-viable candidate,
    else an ignored candidate, else None (reject)."""
    real = [t for t in event.candidates if t in viable]
    if real:
        return min(real, key=lambda t: priority[t])
    ign = [t for t in event.candidates if t in ignored]
    if ign:
        return min(ign, key=lambda t: priority[t])
    return None
