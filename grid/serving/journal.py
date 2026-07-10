"""ContextJournal (W4): the per-dialect record of cold-walked mask key shapes.

The journal is the memory that makes admission warmup (W5) possible: it
records, from the producer's true walk-miss path, WHICH configurations ever
had to be walked for this dialect (one grammar source, the same scope as the
registry's T2 pools), so the NEXT schema admitted over the dialect can have
them precomputed inside compile_grammar while its request legally waits in
WAITING (vLLM 0.24 holds it off-batch; verified unbounded).

Two record classes, mirroring the cold-stall anatomy:

- tier-i (``record_generic``): the exact generic/genN key of every walked
  non-identifier configuration. These keys carry no schema fingerprint by
  construction, so on a warm server the entries are already in the dialect's
  T2 pool — warmup re-registers them into the fresh template's T1/kernel
  (``MaskProducer.warm_from_t2``), killing the measured ~30 ms
  pure-registration stall. No walk is ever run for tier-i.

- tier-ii (``record_ident_context``): identifier-position A-contexts whose
  remainder was a complete lexicon word (the 11.5-14.4 ms BOUNDARY walk class,
  73% of the measured stall), abstracted over WHICH word — E11 keys these
  entries by schema fingerprint, so no cache tier can ever share them across
  schemas; the only fix is to walk (journaled A-context x the new schema's own
  lexicon words) before the request turns RUNNING. Mid-lexeme ident prefixes
  (~0.03 ms walks) are deliberately NOT journaled: they are content-dependent,
  un-enumerable, and cheap enough for the runtime prefetch pool.

The journal never influences mask content — it stores keys/contexts only; an
unused warmed entry is inert, a missed one merely walks at runtime exactly as
today. Coverage is self-healing: any context missed once is recorded on its
first cold walk and covered for every schema admitted after it.

BFS seed (plan W4, best-effort): seeding ident A-contexts by a static walk
over LALR states was evaluated and NOT shipped — the A-sets the keys embed are
``allowed_terminals(tables, node)`` over the FULL stack (reduce lookaheads
follow parent chains), so state-local enumeration cannot reproduce the exact
frozensets and unbounded stack enumeration is not "a static BFS". First-schema
misses are covered by the runtime prefetch/defer path until the journal warms
(the plan's sanctioned fallback: journal-only self-healing).

Bounded (GRID_JOURNAL_MAX, default 4096, per record class; first-seen entries
are kept — the structural contexts appear first — and overflow is dropped) and
thread-safe (records arrive from scheduler, prefetch-pool and warmup-pool
threads).
"""

from __future__ import annotations

import os
import threading


class ContextJournal:
    """Per-dialect walk-miss journal: tier-i generic keys + tier-ii ident
    A-contexts. Registry-scoped (one per grammar source, beside the T2 pool)."""

    def __init__(self, cap: int | None = None) -> None:
        if cap is None:
            cap = int(os.environ.get("GRID_JOURNAL_MAX", "4096"))
        self._cap = cap
        self._lock = threading.Lock()
        # insertion-ordered sets (dict keys): first-seen order is the warmup
        # fan-out order, so structurally-early contexts warm first
        self._generic: dict[tuple, None] = {}
        self._ident: dict[frozenset, None] = {}

    # -- recording (producer walk-miss path) --------------------------------

    def record_generic(self, key: tuple) -> None:
        """A non-identifier configuration was cold-walked under `key`
        (legacy raw generic or genN — both schema-fingerprint-free)."""
        with self._lock:
            if key not in self._generic and len(self._generic) < self._cap:
                self._generic[key] = None

    def record_ident_context(self, A: frozenset) -> None:
        """An identifier-position BOUNDARY configuration (remainder == a
        complete lexicon word) was cold-walked under allowed-terminal set `A`.
        Word-abstracted: A transfers verbatim across schemas of one dialect
        (terminals are numbered at L1 freeze; projections never renumber)."""
        with self._lock:
            if A not in self._ident and len(self._ident) < self._cap:
                self._ident[frozenset(A)] = None

    # -- planning (admission warmup) -----------------------------------------

    def plan(self, lexicons) -> tuple[list[tuple], list[tuple[bytes, frozenset]]]:
        """Warmup plan for a template with `lexicons` (grid.trie.walk.Lexicons
        or None): ``(tier_i_keys, tier_ii)`` where tier_ii enumerates
        ``(word_bytes, A)`` = every journaled ident A-context x every lexicon
        word of THIS schema whose terminal is in that A (a word whose terminal
        the context disallows can never be the pending boundary lexeme there —
        warming it would build an inert entry). Deterministic order: contexts
        first-seen, words sorted."""
        with self._lock:
            tier_i = list(self._generic)
            contexts = list(self._ident)
        tier_ii: list[tuple[bytes, frozenset]] = []
        if lexicons is not None:
            for A in contexts:
                words: set[bytes] = set()
                for tid, ws in lexicons.allowed.items():
                    if tid in A:
                        words.update(bytes(w) for w in ws)
                tier_ii.extend((w, A) for w in sorted(words))
        return tier_i, tier_ii

    @property
    def stats(self) -> dict:
        with self._lock:
            return {"generic_keys": len(self._generic),
                    "ident_contexts": len(self._ident), "cap": self._cap}
