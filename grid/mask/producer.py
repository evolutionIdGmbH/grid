"""MaskProducer: SS6 steps 1-8 orchestration (walk, cache, CD residue, EOS gate).

The cache key (E11 T1): ``(remainder bytes, sorted A signature, schema_fp?)`` —
the grammar fingerprint scopes the whole cache instance. Identifier positions
(A intersects L3 identifier categories) REQUIRE the schema fingerprint in the
key; consulting an entry whose key lacks it raises IdentifierMaskBypassError in
all builds (the identifier composition rule, DESIGN.md E3/E11).

Non-identifier positions normalize to the genN key form (``cache_key``): the
raw remainder bytes are replaced by the scanner normal form ``(p, q, v)``
whenever the lexicon-visibility guard proves the walk cannot observe the
pre-accept prefix bytes — remainders indistinguishable to the walk share one
T1/T2 entry (kill switch: GRID_GENN_KEYS=0 restores the legacy raw keys
byte-for-byte).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from grid.errors import IdentifierMaskBypassError
from grid.lalr.compile import LALRTables
from grid.lalr.stack import StackNode, allowed_terminals, shift_terminal
from grid.lexer.dfa import DEAD, ScannerDFA
from grid.mask.cache import MaskCache, MaskCacheT2, MaskEntryV7, make_entry
from grid.trie.build import TokenTrie
from grid.trie.walk import (
    CDEntry,
    Lexicons,
    _rust_walker,
    _term_words,
    _unmask,
    _words_int,
    make_verdict_kernel,
    pick_viable,
    walk,
)


def _chain(node: StackNode) -> list[int]:
    """LALR state ids root->top (the kernel's stack representation)."""
    out = [node.state]
    p = node.parent
    while p is not None:
        out.append(p.state)
        p = p.parent
    out.reverse()
    return out


def _token_table(adapter, vocab_size: int) -> tuple[bytes, np.ndarray]:
    """The E6-normative token_bytes table for the v6 kernel sessions: ONE blob
    plus int32 offsets (len vocab_size + 1). Vocab holes and specials map to
    empty slices — the kernel REJECTS empty token bytes (pinned improvement
    over the Python KeyError; the eos check short-circuits before the lookup)."""
    offs = np.zeros(vocab_size + 1, dtype=np.int32)
    parts: list[bytes] = []
    total = 0
    tb = adapter.token_bytes
    for i in range(vocab_size):
        try:
            b = tb(i)
        except (KeyError, IndexError):
            b = b""
        parts.append(b)
        total += len(b)
        offs[i + 1] = total
    return b"".join(parts), offs


class _StepMemo:
    """Per-step memo for the CD residue check: allowed sets, shifts, live sets.

    Keys use StackNode identity (nodes are immutable and interned per step)."""

    __slots__ = ("_allowed", "_shift", "_live")

    def __init__(self) -> None:
        self._allowed: dict[int, frozenset[int]] = {}
        self._shift: dict[tuple[int, int], StackNode | None] = {}
        self._live: dict[bytes, frozenset[int]] = {}

    def allowed(self, tables: LALRTables, node: StackNode) -> frozenset[int]:
        got = self._allowed.get(id(node))
        if got is None:
            got = allowed_terminals(tables, node)
            self._allowed[id(node)] = got
        return got

    def shift(self, tables: LALRTables, node: StackNode, t: int) -> StackNode | None:
        key = (id(node), t)
        if key not in self._shift:
            self._shift[key] = shift_terminal(tables, node, t)
        return self._shift[key]

    def live_of(self, dfa: ScannerDFA, remainder: bytes) -> frozenset[int]:
        got = self._live.get(remainder)
        if got is None:
            st = dfa.scan_state(remainder)
            got = dfa.live[st] if st != DEAD else frozenset()
            self._live[remainder] = got
        return got


@dataclass
class MaskProducer:
    tables: LALRTables
    dfa: ScannerDFA
    trie: TokenTrie
    vocab_size: int
    lexicons: Lexicons | None = None
    schema_fingerprint: str | None = None
    cache: MaskCache | None = None
    t2: MaskCacheT2 | None = None  # cross-template tier (DESIGN §E10); registry-scoped
    journal: object | None = None  # ContextJournal (W4); registry-scoped per dialect

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = MaskCache()
        self._validate_lexicons()
        # genN key normalization (cache_key): the lexicon-constrained terminal
        # ids are the guard alphabet — normalize only where the walk provably
        # never consults an allow-list on prefix-derived bytes. Kill switch
        # GRID_GENN_KEYS=0 restores the legacy raw generic keys exactly.
        self._LEX = frozenset() if self.lexicons is None else frozenset(self.lexicons.allowed)
        self._genn_keys = os.environ.get("GRID_GENN_KEYS", "1") != "0"
        self._priority = {
            tid: (0 if tid in self.tables.literal_terminal_ids else 1, tid)
            for tid in range(self.tables.n_terminals)
        }
        # per-node memos (nodes are immutable; keys hold nodes alive — size-capped)
        self._allowed_memo: dict[StackNode, frozenset[int]] = {}
        self._eos_memo: dict[StackNode, bool] = {}
        # grid_core verdict kernel (SS2 kernels #2/#3 + LALR simulate); None -> Python spec
        self._kernel = make_verdict_kernel(self.tables, self.dfa, self.lexicons)
        self._kernel_handles: dict[str, int] = {}  # entry_id -> registered handle
        # kernel v4 hit lookaside: (kidx, remainder) -> (entry_id, handle). The
        # pair determines the T1 key (A is a function of the chain; schema_fp is
        # fixed per producer), so an alias hit can never cross entries.
        self._entry_memo: dict[tuple[int, bytes], tuple[str, int]] = {}
        self._entry_ids: dict[int, str] = {}  # handle -> entry_id (audit records)
        self._kgen = 0  # bumped on reset_interning; StackNode.kgen must match
        self._epoch = self.cache.epoch
        # v6 kernel sessions: single-flight entry registration (pool + scheduler
        # threads may race _ensure_handle), one-time table upload flag, and the
        # fills_hit fold watermark (telemetry parity, risk (e))
        self._handle_lock = threading.Lock()
        self._v6_tables = False
        self._folded_fill_hits = 0
        # kernel v7: cold-miss materialization fully in-kernel (walk_payload +
        # register_blob, one GIL-released call each; Python keeps only a thin
        # MaskEntryV7). Read once; effective only with the kernel present.
        # Kill switch GRID_V7=0 (the default until the perf gates are green)
        # is byte-for-byte today's walk()/make_entry/register_bytes path.
        self._v7 = os.environ.get("GRID_V7", "0") != "0"

    _MEMO_CAP = 200_000
    _INTERN_CAP = 2_000_000  # kernel arena reset threshold (kidx regeneration)

    def _validate_lexicons(self) -> None:
        """The identifier composition rule's PRECONDITION, validated instead of
        assumed: every L3 allow-list word must lie in its terminal's language
        (scannable by the combined DFA and accepted as that terminal). A word
        outside it — e.g. a schema column named ``official_ratings_(millions)``
        against ``[a-z_][a-z0-9_]*`` — makes each of its prefixes pass
        prefix_ok while NO token can ever complete the lexeme: the mask goes
        empty at a viable state, breaking completeness "by theorem". Found by
        the Spider EX harness on real schema data."""
        if self.lexicons is None:
            return
        from grid.errors import GrammarInvalid

        for tid, words in self.lexicons.allowed.items():
            for w in words:
                st = self.dfa.scan_state(bytes(w))
                if st == DEAD or tid not in self.dfa.accepts_all[st]:
                    name = self.tables.terminal_names[tid]
                    raise GrammarInvalid(
                        f"L3 lexicon word {bytes(w)!r} is outside terminal {name}'s "
                        "language (identifier composition precondition; filter or "
                        "rename schema entries before building the guide)"
                    )

    # -- kernel v4 interning --------------------------------------------------

    def _kidx(self, node: StackNode) -> int:
        """This node's kernel intern index; assigned lazily, parents first.
        The walk up stops at the nearest ancestor already interned in the
        current generation, so a child costs one intern_child probe."""
        gen = self._kgen
        if node.kgen == gen:
            return node.kidx
        pending = []
        cur: StackNode | None = node
        while cur is not None and cur.kgen != gen:
            pending.append(cur)
            cur = cur.parent
        base = -1 if cur is None else cur.kidx
        kernel = self._kernel
        for nd in reversed(pending):
            base = kernel.intern_child(base, nd.state)
            nd.kidx = base
            nd.kgen = gen
        return base

    def _reset_interning(self) -> None:
        # reset_interning drops the whole kernel Memos, INCLUDING the v6
        # session-binding map (kidx-keyed) and the session status memo; live
        # kernel sessions re-intern lazily from their raw state chains.
        self._kernel.reset_interning()
        self._kgen += 1
        self._entry_memo.clear()

    def _sync_epoch(self) -> None:
        """E10 namespace rollover check: drop the (kidx, remainder) entry
        aliases AND the kernel session bindings (risk (d): v6 warm fills
        bypass _warm_handle, so a stale binding map would serve the retired
        namespace indefinitely). One int compare on the warm path."""
        cache = self.cache
        if cache.epoch != self._epoch:
            self._epoch = cache.epoch
            self._entry_memo.clear()
            if self._kernel is not None:
                self._kernel.clear_bindings()

    def shift(self, node: StackNode, t: int) -> StackNode | None:
        """SS2 lalr_advance: reduces+shift via the kernel, mirrored into
        StackNodes (config_hash and audit bookkeeping stay Python-owned);
        pure-Python shift_terminal when the kernel is absent. The final chain
        is ancestor(pops) ++ pushed frames — see grid_core advance_core."""
        if self._kernel is None:
            return shift_terminal(self.tables, node, t)
        got = self._kernel.advance_frames(self._kidx(node), t)
        if got is None:
            return None
        new_kidx, pops, frames = got
        cur: StackNode | None = node
        for _ in range(pops):
            assert cur is not None, "reduce popped past root (kernel/mirror desync)"
            cur = cur.parent
        for state, sym in frames:
            cur = StackNode(state, sym, cur)
        assert cur is not None
        cur.kidx = new_kidx
        cur.kgen = self._kgen
        return cur

    def allowed(self, node: StackNode) -> frozenset[int]:
        got = self._allowed_memo.get(node)
        if got is None:
            if self._kernel is not None:
                got = _unmask(_words_int(self._kernel.allowed_mask_at(self._kidx(node))))
            else:
                got = allowed_terminals(self.tables, node)
            if len(self._allowed_memo) > self._MEMO_CAP:
                self._allowed_memo.clear()
            self._allowed_memo[node] = got
        return got

    def eos_ok_at(self, node: StackNode) -> bool:
        got = self._eos_memo.get(node)
        if got is None:
            if self._kernel is not None:
                got = bool(self._kernel.eos_ok_at(self._kidx(node)))
            else:
                from grid.lalr.stack import eos_ok_stack

                got = eos_ok_stack(self.tables, node)
            if len(self._eos_memo) > self._MEMO_CAP:
                self._eos_memo.clear()
            self._eos_memo[node] = got
        return got

    # -- cache key (E11) ----------------------------------------------------

    def cache_key(self, remainder: bytes, A: frozenset[int]) -> tuple:
        """T1/T2 key for the configuration (remainder, A).

        Identifier positions keep the legacy schema_fp-carrying key
        byte-for-byte (E11). Non-ident positions normalize to
        ``("genN", p, q, v, sorted(A), schema_fp)`` with ``(q, l, p)`` from
        dfa.scan_with_last_accept and ``v = remainder[l:]`` verbatim when
        p >= 0 else ``b""`` — sound ONLY under the lexicon-visibility guard
        ``(live[q] | accepts_all[p]) & LEX == {}``: every terminal the walk's
        lexeme_ok/prefix_ok predicates can consult on REMAINDER-derived bytes
        lies in that set, so under the guard those checks are lexicon-inert
        and the walk future is a function of (p, q, v, A) and the lexicons.
        The lexicons stay in scope because WALKED bytes cross lexeme
        boundaries: post-boundary pending lexemes are lexeme_ok/prefix_ok
        filtered at walk time (grid/mask/cache.py make_entry, kernel
        walk_raw), so CD partitions embed schema words — hence the
        ``schema_fp`` component (None when no lexicons: dialect-wide sharing
        stays). Without it the per-dialect T2 served one schema's CD
        partition to another (50-seed fuzz counterexample: whitespace
        remainder, A={LPAREN}, '('-continuations of schema words). Guard
        failure — and a DEAD full-remainder scan, where live[q] is undefined
        — falls back to the legacy raw ("generic", ...) key. The ``v``
        component is load-bearing: b"1e"/b"1E" share (q, l, p) but requeue
        different suffix bytes (tests/mask/test_genn_keys.py). In the v2
        regime the raw FALLBACK key is schema-scoped too — ("generic", r,
        sorted(A), schema_fp) — because the same walk-time CD poisoning
        applies to byte-identical raw remainders across schemas (a fresh
        schema T2-adopting another schema's b"select" boundary entry loses
        its own words' continuations; caught by the 50-seed shared-registry
        fuzz, tests/mask/test_genn_keys.py::test_raw_fallback_is_schema_
        scoped_v2). Kill switch GRID_GENN_KEYS=0 restores the legacy keys
        byte-for-byte, INCLUDING the pre-existing unscoped fallback (and with
        it the latent pre-stage cross-schema T2 sharing hazard — documented,
        not fixed, on the kill-switch path to preserve byte-identity)."""
        ident_position = bool(A & self.tables.identifier_terminal_ids) and self.lexicons is not None
        if ident_position:
            return ("ident", remainder, tuple(sorted(A)), self.schema_fingerprint)
        if self._genn_keys:
            q, acc_len, p = self.dfa.scan_with_last_accept(remainder)
            if q != DEAD:
                vis = self.dfa.live[q] if p < 0 else self.dfa.live[q] | self.dfa.accepts_all[p]
                if not (vis & self._LEX):
                    return ("genN", p, q, remainder[acc_len:] if p >= 0 else b"",
                            tuple(sorted(A)), self.schema_fingerprint)
            return ("generic", remainder, tuple(sorted(A)), self.schema_fingerprint)
        return ("generic", remainder, tuple(sorted(A)), None)

    def set_genn_keys(self, on: bool) -> None:
        """Flip the genN key format at runtime (G10 dual-key replay and tests
        ONLY; production reads GRID_GENN_KEYS once at construction). The
        (kidx, remainder) alias memo and the kernel v6 session bindings are
        format-agnostic caches OVER cache_key results, so a flip drops them —
        no path may serve an entry resolved under the other format's key."""
        if on == self._genn_keys:
            return
        self._genn_keys = on
        self._entry_memo.clear()
        if self._kernel is not None:
            self._kernel.clear_bindings()

    def _guard_key(self, key: tuple, A: frozenset[int]) -> None:
        ident_position = bool(A & self.tables.identifier_terminal_ids) and self.lexicons is not None
        if ident_position and (key[0] != "ident" or key[3] is None):
            raise IdentifierMaskBypassError(
                "generic-IDENT cache entry consulted at an identifier position"
            )

    # -- SS6 steps 4-7 -------------------------------------------------------

    def _warm_handle(self, node: StackNode, remainder: bytes):
        """(entry_id, handle, kidx) for a T1-warm configuration, else None (a
        true miss, or no kernel). Owns the (kidx, remainder) alias memo, the
        epoch check, and the intern-cap reset — the shared front half of
        mask_hit / fill_bits_hit."""
        kernel = self._kernel
        if kernel is None:
            return None
        cache = self.cache
        assert cache is not None
        self._sync_epoch()  # namespace rollover: drop entry aliases + bindings
        kx = self._kidx(node)
        got = self._entry_memo.get((kx, remainder))
        if got is None:
            A = self.allowed(node)
            key = self.cache_key(remainder, A)
            self._guard_key(key, A)
            entry = cache.peek(key)
            if entry is None:
                return None  # true miss -> masks() walks, publishes, registers
            handle = self._ensure_handle(entry)
            got = (entry.entry_id, handle)
            if len(self._entry_memo) > self._MEMO_CAP:
                self._entry_memo.clear()
                kernel.clear_bindings()  # the kernel bind map mirrors the cap
            if len(self._entry_memo) % 4096 == 0 and kernel.intern_count() > self._INTERN_CAP:
                self._reset_interning()
                kx = self._kidx(node)
            self._entry_memo[(kx, remainder)] = got
        entry_id, handle = got
        return entry_id, handle, kx

    def mask_hit(self, node: StackNode, remainder: bytes, eos_id: int):
        """Kernel v4 warm hit: the fully assembled allowed-id buffer
        (ci ++ cd-passing ++ eos-if->=0) as an int32 array, plus the entry id —
        one FFI call once (kidx, remainder) is known. None on a T1 miss (caller
        runs the walk path) or without the kernel. Bit-identical to
        np.concatenate([ci, cd_pass, eos]) (tests/mask/test_kernel_parity.py)."""
        got = self._warm_handle(node, remainder)
        if got is None:
            return None
        entry_id, handle, kx = got
        self.cache.hits += 1  # alias hits are T1 hits (telemetry parity with get())
        buf = self._kernel.hit_pass(handle, kx, eos_id)
        return np.frombuffer(buf, dtype=np.int32), entry_id

    def fill_bits_hit(self, node: StackNode, remainder: bytes, eos_id: int,
                      out: np.ndarray) -> str | None:
        """Kernel v5 warm hit: write the packed uint32 bitmask row for this
        configuration into ``out`` in ONE FFI call (pre-packed ci bit words ++
        cd-passing bits ++ eos bit; every word of ``out`` overwritten). Returns
        the entry id, or None on a T1 miss / without the kernel (caller runs
        the walk + numpy pack path). Bit-set identical to mask_hit's id buffer
        (tests/mask/test_kernel_parity.py). This is the scheduler-side
        fill_bitmask hot path: no id materialization, no numpy packing."""
        got = self._warm_handle(node, remainder)
        if got is None:
            return None
        entry_id, handle, kx = got
        self.cache.hits += 1
        self._kernel.fill_bits(handle, kx, eos_id, out)
        return entry_id

    def peek_warm(self, node: StackNode, remainder: bytes) -> bool:
        """True when this configuration's mask can be filled without a walk —
        the serving prefetch gate (SS6 overlap: schedule background builds for
        cold successors ONLY). Never counts hit/miss telemetry."""
        if self._kernel is not None and self.cache.epoch == self._epoch \
                and (self._kidx(node), remainder) in self._entry_memo:
            return True
        A = self.allowed(node)
        assert self.cache is not None
        return self.cache.peek(self.cache_key(remainder, A)) is not None

    def masks(self, node: StackNode, remainder: bytes) -> tuple:
        """Returns (ci_tokens, cd_pass_token_ids: int32 array, entry_id). CD entries
        are checked against the live stack here (never cached). ci_tokens is a
        read-only int32 ndarray on the kernel-walk path, a tuple on the spec path."""
        entry = self._entry_for(remainder, self.allowed(node))
        cd_pass = self._check_cd_batch(entry, node)
        return entry.ci_tokens, cd_pass, entry.entry_id

    def _entry_for(self, remainder: bytes, A: frozenset[int]):
        """T1 get -> T2 handover -> walk + publish for the configuration
        (remainder, A): the node-free front half of masks(), shared with the
        kernel-session bind/prefetch paths (pure walk inputs, no StackNode).
        Counts hit/miss telemetry via cache.get, exactly as masks() did."""
        key = self.cache_key(remainder, A)
        self._guard_key(key, A)
        assert self.cache is not None
        entry = self.cache.get(key)
        if entry is None and self.t2 is not None:
            got2 = self.t2.get(key)  # T2 hit -> copy into T1, no walk (§E10)
            if got2 is not None:
                entry = self.cache.publish(got2)
        if entry is None:
            if self._v7 and self._kernel is not None:
                entry = self.cache.publish(self._v7_build(key, remainder, A))
            else:
                result = walk(
                    self.trie, self.dfa, remainder, A,
                    self.tables.ignored_terminal_ids, self._priority, self.lexicons,
                )
                if result.groups is not None:  # rust kernel: alias-expanded + sorted in-kernel
                    ci = result.ci_tokens
                    expand = None
                else:
                    ci = tuple(sorted(t for tid in result.ci_tokens for t in self.trie.expand(tid)))
                    expand = self.trie.expand
                memo = _StepMemo()
                entry = self.cache.publish(make_entry(
                    key, ci, result.cd_entries, self.vocab_size,
                    live_of=lambda rem: memo.live_of(self.dfa, rem),
                    lexicon_sensitive=self.lexicons is not None,
                    expand=expand,
                    precomputed_groups=result.groups,
                    # verdict-equivalence grouping context (mirrors the kernel key)
                    lexicons=self.lexicons,
                    ignored=self.tables.ignored_terminal_ids,
                    priority=self._priority,
                ))
            if self.t2 is not None:
                self.t2.publish(entry)
            if self.journal is not None:
                self._journal_miss(key, remainder, A)
        return entry

    def _v7_build(self, key: tuple, remainder: bytes, A: frozenset[int]) -> MaskEntryV7:
        """Kernel v7 cold build: walk_payload (rayon-capable walk + blob/ci
        serialization, GIL released) then register_blob (group build + ci
        pack + adaptive encode + BLAKE2b entry id + registration, GIL
        released, by_id-deduplicated in-kernel — the walk-twice race yields
        one handle). Python does dict inserts only. The entry_id is
        byte-identical to make_entry's for the same (key, ci, vocab)
        (tests/mask/test_v7_encode.py cross-impl vectors; the g10 dual-key
        replay is a free Python-vs-Rust differential)."""
        walker = _rust_walker(self.trie, self.dfa, self.tables.ignored_terminal_ids,
                              self._priority, self.lexicons)
        ci_bytes, blob = walker.walk_payload(bytes(remainder), _term_words(A, walker.width))
        handle, entry_id, tag, _n_groups = self._kernel.register_blob(
            blob, ci_bytes, repr(key).encode(), self.vocab_size)
        with self._handle_lock:
            self._kernel_handles.setdefault(entry_id, handle)
            self._entry_ids[handle] = entry_id
        return MaskEntryV7(entry_id, key, tag, ci_bytes, blob)

    def _journal_miss(self, key: tuple, remainder: bytes, A: frozenset[int]) -> None:
        """W4 hook (true walk-miss path only — T2 handovers were journaled at
        their own first walk): record the configuration's KEY SHAPE for
        admission warmup. Generic/genN keys are recorded verbatim (tier-i);
        identifier positions are recorded word-abstracted, and only when the
        remainder is a complete lexicon word of a terminal in A — the
        expensive BOUNDARY class (tier-ii). Mid-lexeme ident prefixes stay
        un-journaled by design (cheap, content-dependent, un-enumerable).
        Never raises: the journal is a warm-hint, not a mask dependency."""
        try:
            if key[0] != "ident":
                self.journal.record_generic(key)
                return
            lex = self.lexicons
            if lex is not None and any(
                    tid in A and remainder in words
                    for tid, words in lex.allowed.items()):
                self.journal.record_ident_context(A)
        except Exception:  # pragma: no cover - defensive (hot miss path)
            pass

    def warm_from_t2(self, keys) -> int:
        """W4/W5 tier-i warmup: adopt already-built T2 entries into THIS
        producer's T1 and register them with the kernel — no walks (a key with
        no T2 donor is skipped; tier-i never builds cold). Exactly the
        _entry_for handover (t2.get -> cache.publish -> _ensure_handle) minus
        the walk fallback. Thread-safe from warmup-pool threads: publish is
        idempotent by content hash (OBL-KEY1) and registration is
        single-flight per entry_id. Returns the number of entries warmed."""
        t2 = self.t2
        if t2 is None:
            return 0
        assert self.cache is not None
        n = 0
        for key in keys:
            if key[0] == "genN" and key[5] != self.schema_fingerprint:
                continue  # foreign-schema genN key: this producer can never
                # look it up (cache_key embeds OUR fingerprint) — registering
                # its entry would be dead weight in T1 and the kernel
            if key[0] == "generic" and key[3] is not None \
                    and key[3] != self.schema_fingerprint:
                continue  # foreign-schema scoped raw fallback key: same story
            got = t2.get(key)
            if got is None:
                continue
            entry = self.cache.publish(got)
            if self._kernel is not None:
                self._ensure_handle(entry)
            n += 1
        return n

    def _ensure_handle(self, entry) -> int:
        """Register the entry's CD groups + ci ids with the kernel once
        (content-addressed by entry_id; survives namespace rollover because
        recomputed entries hash to the same id). Rust-walk entries carry the
        registration payload verbatim (entry.kernel_groups); the words path
        below reconverts for spec-built/seeded entries and is bit-identical.
        Registration is single-flight per entry_id: a pool thread and the
        scheduler thread racing the check-then-act would otherwise register
        duplicate handles (split cd/row memos; two handles behind one v6
        binding key)."""
        handle = self._kernel_handles.get(entry.entry_id)
        if handle is not None:
            return handle
        with self._handle_lock:
            handle = self._kernel_handles.get(entry.entry_id)
            if handle is not None:
                return handle
            blob = getattr(entry, "blob", None)
            if blob is not None:
                # v7 entry (this producer post-rollover, or a foreign
                # producer's T2 handover): Rust-to-Rust import — THIS kernel
                # recomputes VEvents/tails under ITS lexicons, exactly like
                # the register_bytes-from-kernel_groups semantics below.
                handle, eid, _tag, _n = self._kernel.register_blob(
                    blob, entry.ci_bytes, repr(entry.key).encode(), self.vocab_size)
                assert eid == entry.entry_id, \
                    "v7 entry_id mismatch across implementations (bug by vectors)"
                self._kernel_handles[entry.entry_id] = handle
                self._entry_ids[handle] = entry.entry_id
                return handle
            if entry.kernel_groups is not None:
                payload = list(entry.kernel_groups)
            else:
                kw = self._kernel.width
                payload = [
                    (
                        [(_term_words(ev.candidates, kw), ev.length)
                         for ev in g.representative.events],
                        list(g.representative.segments),
                        g.representative.remainder,
                        list(g.token_ids),
                    )
                    for g in entry.cd_groups
                ]
            # ci ids cross the FFI as one i32-le buffer: per-int extraction of
            # a literal-interior giant (10k+ ids) cost 20-100 ms per (template,
            # entry) first touch — the serialized step spikes of the G8
            # adversarial arm. np.asarray + tobytes is a C loop + memcpy.
            ci = np.asarray(entry.ci_tokens, dtype=np.int32).tobytes()
            handle = self._kernel.register_bytes(payload, ci)
            self._kernel_handles[entry.entry_id] = handle
            self._entry_ids[handle] = entry.entry_id
        return handle

    def _check_cd_batch(self, entry, node: StackNode) -> np.ndarray:
        """Per-step CD residue: one stack-dependent verdict per precomputed group
        (E10 cd_groups); allowed-terminal sets and shifts memoized per node.
        Returns the passing token ids (int32, group order preserved).

        Kernel path: groups register once per entry (content-addressed by
        entry_id), then each step is one cd_pass_at(handle, kidx) call
        returning an i32-le buffer consumed zero-copy — bit-identical to the
        Python loop (tests/mask/test_kernel_parity.py)."""
        if self._kernel is not None:
            handle = self._ensure_handle(entry)
            return np.frombuffer(
                self._kernel.cd_pass_at(handle, self._kidx(node)), dtype=np.int32
            )
        memo = _StepMemo()
        out: list[int] = []
        for group in entry.cd_groups:
            if self.check_context_dependent(group.representative, node, memo):
                out.extend(group.token_ids)
        return np.asarray(out, dtype=np.int32)

    # -- kernel v6 sessions ----------------------------------------------------

    def ensure_session_tables(self, adapter) -> bool:
        """Upload the v6 session tables once per kernel: the E6-normative
        token_bytes table (the same mapping the trie was built from) and the
        scanner accept/accepts_all tables. False without a kernel — the
        caller keeps the v5 Python path."""
        if self._kernel is None:
            return False
        if self._v6_tables:
            return True
        blob, offs = _token_table(adapter, self.vocab_size)
        self._kernel.set_token_bytes(blob, offs.tobytes())
        w = self._kernel.width
        self._kernel.set_dfa_accept(
            np.asarray(self.dfa.accept, dtype=np.int32).tobytes(),
            [_term_words(s, w) for s in self.dfa.accepts_all],
        )
        self._v6_tables = True
        return True

    def session_peek_handle(self, a_words, remainder: bytes):
        """(handle | None, A) for a kernel-session configuration given its
        allowed-mask words + remainder (session_walk_inputs). The OBL-KEY1
        cache_key + _guard_key check runs HERE — bind time — mirroring the
        v5 trust model where the guard runs on alias-memo misses. peek is
        uncounted (the kernel's fills_hit carries the telemetry)."""
        A = _unmask(_words_int(a_words))
        key = self.cache_key(remainder, A)
        self._guard_key(key, A)
        assert self.cache is not None
        entry = self.cache.peek(key)
        if entry is None:
            return None, A
        return self._ensure_handle(entry), A

    def session_bind_handle(self, a_words, remainder: bytes) -> int:
        """Fill-miss path: peek-or-build the entry for this configuration and
        return its registered handle (guard enforced; a true miss walks +
        publishes synchronously and counts via cache.get, as v5's miss did)."""
        handle, A = self.session_peek_handle(a_words, remainder)
        if handle is None:
            handle = self._ensure_handle(self._entry_for(remainder, A))
        return handle

    def prefetch_build(self, remainder: bytes, A: frozenset[int]) -> None:
        """Pool-thread cold build for a kernel-session successor from pure
        (remainder, A) walk inputs — no session state is touched off the
        scheduler thread (protocol-safe by construction). Publishes to T1/T2
        and registers the entry; the scheduler thread binds at fill time.
        Also the W5 admission-warmup tier-ii primitive — on the no-kernel spec
        path (GRID_NO_RUST) it builds/publishes without registration."""
        entry = self._entry_for(remainder, A)
        if self._kernel is not None:
            self._ensure_handle(entry)

    def fold_session_stats(self) -> dict:
        """Kernel session counters, with warm fills folded into cache.hits
        (telemetry parity, risk (e)): a kernel-served fill is exactly one T1
        hit in v5 accounting. Idempotent via the fold watermark."""
        if self._kernel is None:
            return {}
        stats = dict(self._kernel.session_stats())
        new = stats["fills_hit"] - self._folded_fill_hits
        if new > 0:
            self.cache.hits += new
            self._folded_fill_hits = stats["fills_hit"]
        return stats

    # -- SS2 kernel #2: per-step CD residue check ------------------------------

    def check_context_dependent(self, e: CDEntry, node: StackNode, memo: _StepMemo | None = None) -> bool:
        memo = memo or _StepMemo()
        cur: StackNode | None = node
        for ev, seg in zip(e.events, e.segments, strict=True):
            assert cur is not None
            allowed_here = memo.allowed(self.tables, cur)
            viable = ev.candidates & allowed_here
            pick = pick_viable(ev, seg, viable, self.tables.ignored_terminal_ids,
                               self._priority, self.lexicons)
            if pick is None:
                return False
            if pick not in self.tables.ignored_terminal_ids:
                cur = memo.shift(self.tables, cur, pick)
                if cur is None:
                    return False
        if not e.remainder:
            return True
        st = self.dfa.scan_state(e.remainder)
        if st == DEAD:  # pragma: no cover
            return False
        assert cur is not None
        a_after = memo.allowed(self.tables, cur)
        for t in self.dfa.live[st]:
            if t in self.tables.ignored_terminal_ids:
                return True
            if t in a_after and (self.lexicons is None or self.lexicons.prefix_ok(t, e.remainder)):
                return True
        return False
