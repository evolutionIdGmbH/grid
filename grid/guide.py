"""GridGuide + GridState: the SS6 per-token hot path behind the Guide protocol.

Statuses (E9, all O(1)-derivable; FORCED is an instruction-level outcome, not a
status; reserve exhaustion is a budget-level trigger, not a status):
ACTIVE -> ACCEPTING/GRAMMAR_END -> COMPLETE; DEAD_END is unreachable by theorem.

Instruction tokens are torch.LongTensor (SS4.1 tensor contract).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
import torch

from grid.audit.log import EOS as K_EOS
from grid.audit.log import GENERATE as K_GENERATE
from grid.audit.log import WRITE as K_WRITE
from grid.audit.log import AuditLog
from grid.errors import DeadEndError
from grid.lalr.compile import LALRTables
from grid.lalr.reserve import INF, ReserveTable
from grid.lalr.stack import (
    StackNode,
    root_node,
)
from grid.lexer.dfa import DEAD, ScannerDFA
from grid.lexer.run import LexerRun, ScanReject
from grid.mask.producer import MaskProducer
from grid.protocols import Generate, Instruction, Write
from grid.trie.walk import Lexicons, pick_viable

ACTIVE, ACCEPTING, GRAMMAR_END, COMPLETE = "ACTIVE", "ACCEPTING", "GRAMMAR_END", "COMPLETE"

# Trigger slack >= max per-token completion inflation (decision log). The
# original 8 was calibrated on gpt2; byte-BPE whitespace blobs (Qwen2.5) can
# inflate the greedy-tokenized completion by 9+ in ONE step — observed as a
# 1-token budget overrun at G5 scale (bench/g5_scale.py, seed 910102). 16
# covers the observed worst case with margin; the G5 run asserts zero
# reserve-stopped overruns over 10k walks.
RESERVE_SAFETY = 16


@dataclass(frozen=True, eq=False)
class GridState:
    stack: StackNode
    lexer: LexerRun
    n_generated: int
    prev_token: int | None
    status: str


class GridGuide:
    """Implements the Guide protocol plus the extended CFG-guide surface (SS4.2)."""

    def __init__(
        self,
        tables: LALRTables,
        dfa: ScannerDFA,
        trie,
        adapter,
        lexicons: Lexicons | None = None,
        schema_fingerprint: str | None = None,
        reserve: ReserveTable | None = None,
        audit: AuditLog | None = None,
        max_new_tokens: int | None = None,
        j_max: int = 8,
        mask_cache=None,
        mask_t2=None,
        mask_journal=None,
        producer: MaskProducer | None = None,
    ) -> None:
        self.tables = tables
        self.dfa = dfa
        self.trie = trie
        self.adapter = adapter
        self.lexicons = lexicons
        self.eos_token_id: int = adapter.eos_token_id
        self.vocab_size: int = max(adapter.vocabulary.values()) + 1
        # `producer` (copy() path): SHARE the template's producer outright — it
        # is config-addressed throughout (T1 cache, node memos, kernel arena,
        # entry handles) and holds no per-request state, while a fresh producer
        # per request-copy means a fresh Rust kernel per request and a
        # re-REGISTRATION of every touched entry per request (literal-interior
        # entries carry 10k+ token ids; measured 20-100 ms each on the H100 —
        # the residual G8 batched-TPOT tail after the fill_bits fix).
        self.producer = producer if producer is not None else MaskProducer(
            tables=tables, dfa=dfa, trie=trie, vocab_size=self.vocab_size,
            lexicons=lexicons, schema_fingerprint=schema_fingerprint, cache=mask_cache,
            t2=mask_t2, journal=mask_journal,
        )
        self.reserve = reserve
        self.audit = audit
        self.max_new_tokens = max_new_tokens
        self.j_max = j_max
        self._priority = self.producer._priority
        self._pending: dict[int, tuple[str, str | None, int]] = {}  # id(state) -> (kind, entry, blocked)
        self._ci_arrays: dict[str, np.ndarray] = {}  # entry_id -> ci ids (int32, never mutated)
        self._ci_bits: dict[str, np.ndarray] = {}    # entry_id -> packed uint32 vocab bitmask
        self._eos_arr = np.array([self.eos_token_id], dtype=np.int32)
        self.initial_state = self._make_state(root_node(tables), LexerRun(), 0, None)

    # ---------------------------------------------------------------- status

    def _make_state(self, stack: StackNode, lexer: LexerRun, n: int, prev: int | None,
                    eos_consumed: bool = False) -> GridState:
        return GridState(stack, lexer, n, prev, self._derive_status(stack, lexer, eos_consumed))

    def _derive_status(self, stack: StackNode, lexer: LexerRun, eos_consumed: bool) -> str:
        if eos_consumed:
            return COMPLETE
        if not self._eos_ok(stack, lexer):
            return ACTIVE
        if lexer.at_boundary() and not self.producer.allowed(stack):
            return GRAMMAR_END
        return ACCEPTING

    def _eos_ok(self, stack: StackNode, lexer: LexerRun) -> bool:
        """SS6 step 2: mid-lexeme-aware end-of-input simulation."""
        node = self._finalized_node(stack, lexer)
        return node is not None and self.producer.eos_ok_at(node)

    def _finalized_node(self, stack: StackNode, lexer: LexerRun) -> StackNode | None:
        """Virtually emit the remainder's winning segmentation and shift it."""
        events = lexer.finalize(self.dfa)
        if events is None:
            return None
        node: StackNode | None = stack
        offset = 0
        for ev in events:
            assert node is not None
            seg = lexer.remainder[offset:offset + ev.length]
            offset += ev.length
            viable = ev.candidates & self.producer.allowed(node)
            pick = pick_viable(ev, seg, viable, self.tables.ignored_terminal_ids,
                               self._priority, self.lexicons)
            if pick is None:
                return None
            if pick in self.tables.ignored_terminal_ids:
                continue
            node = self.producer.shift(node, pick)
            if node is None:
                return None
        return node

    # ------------------------------------------------------------ Guide API

    def get_next_instruction(self, state: GridState) -> Instruction:
        if state.status == COMPLETE:
            return Write(torch.tensor([self.eos_token_id], dtype=torch.long))

        # SS6 step 3: budget trigger (session-level; never a bare EOS away from ACCEPT)
        if self.max_new_tokens is not None:
            remaining = self.max_new_tokens - state.n_generated
            completion = self._completion_tokens(state)
            if completion is not None and remaining <= len(completion) + RESERVE_SAFETY:
                self._pending[id(state)] = (K_WRITE, None, self.vocab_size - 1)
                return Write(torch.tensor(completion, dtype=torch.long))

        ids, entry_id = self._mask_ids(state)
        if ids.size == 0:
            raise DeadEndError(f"empty mask at step {state.n_generated} (bug by theorem)")
        blocked = self.vocab_size - len(ids)
        if len(ids) == 1:
            span = self._forced_span(state, int(ids[0]))
            self._pending[id(state)] = (K_WRITE, None, blocked)
            return Write(torch.tensor(span, dtype=torch.long))
        self._pending[id(state)] = (K_GENERATE, entry_id, blocked)
        # int32 -> long copies once (torch.tensor: hit_pass buffers are
        # read-only views, as_tensor would warn); consumers are order-free
        return Generate(torch.tensor(ids, dtype=torch.long))

    def get_next_state(self, state: GridState, token_id: int) -> GridState:
        nxt = self._advance(state, token_id, audit=True)
        assert nxt is not None, f"token {token_id} applied outside its mask (bug)"
        return nxt

    def is_final_state(self, state: GridState) -> bool:
        return self.can_terminate_state(state)

    def can_terminate_state(self, state: GridState) -> bool:
        return state.status in (ACCEPTING, GRAMMAR_END, COMPLETE)

    def must_terminate_state(self, state: GridState) -> bool:
        return state.status in (GRAMMAR_END, COMPLETE)

    def iter_valid_token_ids(self, state: GridState, candidate_token_ids) -> Iterator[int]:
        for tid in candidate_token_ids:
            if self._advance(state, int(tid), audit=False) is not None:
                yield int(tid)

    def copy(self) -> GridGuide:
        """Per-request twin (the E11 serving pattern): fresh state bookkeeping
        (_pending, audit), SHARED producer — one kernel, one registration
        space, one T1 cache per template. Per-request state lives in GridState
        and the caller's session, never in the producer."""
        return GridGuide(
            tables=self.tables, dfa=self.dfa, trie=self.trie, adapter=self.adapter,
            lexicons=self.lexicons, schema_fingerprint=self.producer.schema_fingerprint,
            reserve=self.reserve, audit=AuditLog() if self.audit is not None else None,
            max_new_tokens=self.max_new_tokens, j_max=self.j_max,
            producer=self.producer,
        )

    # ------------------------------------------------------------- internals

    def _mask_ids(self, state: GridState) -> tuple[np.ndarray, str | None]:
        """SS6 steps 1-7: ci ∪ cd-passing ∪ {eos if legal}, as an int32 array.

        Warm path (kernel v4): producer.mask_hit assembles the whole buffer in
        one hit_pass FFI call against the interned-stack memos. Miss path (and
        the no-kernel spec path): walk + publish, then numpy concatenate.
        Identifier positions admit tens of thousands of ids per step — the mask
        stays in numpy/bytes end-to-end; materializing Python ints here was the
        dominant warm-hit cost once. Order is ci (sorted) ++ cd (group order)
        ++ eos; mask consumers are order-free (set semantics / scatter indices).
        Both paths always return a fresh array, never a cached one."""
        include_eos = self.can_terminate_state(state) and state.status != COMPLETE
        hit = self.producer.mask_hit(
            state.stack, state.lexer.remainder,
            self.eos_token_id if include_eos else -1,
        )
        if hit is not None:
            return hit
        ci, cd_pass, entry_id = self.producer.masks(state.stack, state.lexer.remainder)
        base = self._ci_arrays.get(entry_id)
        if base is None:
            base = np.asarray(ci, dtype=np.int32)
            self._ci_arrays[entry_id] = base
        parts = [base, cd_pass]
        if include_eos:
            parts.append(self._eos_arr)
        return np.concatenate(parts), entry_id

    def fill_bitmask(self, state: GridState, out: np.ndarray) -> None:
        """SS2 kernel #4 semantics: write the allowed-token bitmask for `state`
        into ``out`` (uint32 words, len >= ceil(vocab_size/32)) with no token-id
        materialization. Warm path (kernel v5): producer.fill_bits_hit packs
        the whole row in ONE FFI call against the interned-stack memos — the
        vLLM scheduler-side fill is per request per step, so this must stay
        µs-scale (the numpy repack below ran per call at serving batch sizes
        and dominated G8 step time). Miss path (and the no-kernel spec path):
        walk + publish via masks(), then a one-time per-entry ci pack, cd/eos
        bits OR'd in vectorized. Both paths are bit-identical
        (tests/mask/test_kernel_parity.py)."""
        include_eos = self.can_terminate_state(state) and state.status != COMPLETE
        eos = self.eos_token_id if include_eos else -1
        if self.producer.fill_bits_hit(state.stack, state.lexer.remainder, eos, out) is not None:
            return
        ci, cd_pass, entry_id = self.producer.masks(state.stack, state.lexer.remainder)
        # masks() published the entry (walk or T2 handover) — retry the kernel
        # row fill so even the first touch of a giant skips the numpy pack
        if self.producer.fill_bits_hit(state.stack, state.lexer.remainder, eos, out) is not None:
            return
        bits = self._ci_bits.get(entry_id)
        if bits is None:
            words = (self.vocab_size + 31) // 32
            bits = np.zeros(words, dtype=np.uint32)
            if len(ci):
                idx = np.asarray(ci, dtype=np.int64)
                np.bitwise_or.at(bits, idx >> 5, (1 << (idx & 31)).astype(np.uint32))
            self._ci_bits[entry_id] = bits
        n = bits.shape[0]
        out[:n] = bits
        out[n:] = 0
        if cd_pass.size:
            idx = cd_pass.astype(np.int64)
            np.bitwise_or.at(out, idx >> 5, (1 << (idx & 31)).astype(np.uint32))
        if include_eos:
            out[self.eos_token_id >> 5] |= np.uint32(1 << (self.eos_token_id & 31))

    def is_mask_warm(self, state: GridState) -> bool:
        """True when `state`'s mask can be filled without a walk (T1-warm).
        The serving prefetch gate: background builds are scheduled only for
        cold successors, so the warm steady state never touches the pool."""
        return self.producer.peek_warm(state.stack, state.lexer.remainder)

    def _advance(self, state: GridState, token_id: int, audit: bool) -> GridState | None:
        """Apply one token (SS6 steps 11-16); None if not viable.

        Tokens applied to a state that never received an instruction (Write-span
        intermediates, processor state reconstruction) default to WRITE-kind audit
        records (no mask entry id, E14 invariant).
        """
        token_id = int(token_id)  # accept numpy scalars from mask arrays
        if audit:
            kind, entry_id, blocked = self._pending.pop(id(state), (K_WRITE, None, 0))
        else:
            kind, entry_id, blocked = self._pending.get(id(state), (K_WRITE, None, 0))
        if token_id == self.eos_token_id:
            if not self.can_terminate_state(state):
                return None
            if audit and self.audit is not None:
                self.audit.append(state.n_generated, state.stack.config_hash, None,
                                  token_id, blocked, K_EOS)
            return replace(state, prev_token=token_id, status=COMPLETE)

        if state.status == COMPLETE:
            # Pinned E9 semantics (v6 red-team §0): COMPLETE consumes only
            # (repeat-)eos — any other token is non-viable. Without this, a
            # speculative draft proposing tokens past eos re-derives a
            # non-COMPLETE status and resurrects the request.
            return None

        data = self.adapter.token_bytes(token_id)
        if not data:
            return None
        old_remainder = state.lexer.remainder
        try:
            lexer2, events = state.lexer.advance(self.dfa, data)
        except ScanReject:
            return None

        buf = old_remainder + data
        node: StackNode | None = state.stack
        offset = 0
        for ev in events:
            assert node is not None
            seg = buf[offset:offset + ev.length]
            offset += ev.length
            viable = ev.candidates & self.producer.allowed(node)
            pick = pick_viable(ev, seg, viable, self.tables.ignored_terminal_ids,
                               self._priority, self.lexicons)
            if pick is None:
                return None
            if pick in self.tables.ignored_terminal_ids:
                continue
            node = self.producer.shift(node, pick)
            if node is None:
                return None
        assert node is not None

        # partial-lexeme viability (mirror of walk classification tail rule)
        if lexer2.remainder:
            st = self.dfa.scan_state(lexer2.remainder)
            if st == DEAD:
                return None
            a_now = self.producer.allowed(node)
            ok = False
            for t in self.dfa.live[st]:
                if t in self.tables.ignored_terminal_ids:
                    ok = True
                    break
                if t in a_now and (self.lexicons is None or self.lexicons.prefix_ok(t, lexer2.remainder)):
                    ok = True
                    break
            if not ok:
                return None

        if audit and self.audit is not None:
            new_hash = node.config_hash
            self.audit.append(state.n_generated, new_hash,
                              entry_id if kind == K_GENERATE else None,
                              token_id, blocked, kind)
        return self._make_state(node, lexer2, state.n_generated + 1, token_id)

    def _forced_span(self, state: GridState, first: int) -> list[int]:
        """SS4.5: maximal chain of singleton masks, bounded by j_max."""
        span = [first]
        cur = self._advance(state, first, audit=False)
        while cur is not None and len(span) < self.j_max:
            if cur.status == COMPLETE:
                break
            ids, _ = self._mask_ids(cur)
            if len(ids) != 1 or ids[0] == self.eos_token_id:
                break
            span.append(int(ids[0]))
            cur = self._advance(cur, int(ids[0]), audit=False)
        return span

    def _completion_tokens(self, state: GridState) -> list[int] | None:
        """Concrete minimal completion: finalize remainder, stack completion, +EOS."""
        if self.reserve is None:
            return None
        pre = b""
        lexer = state.lexer
        if lexer.remainder and lexer.finalize(self.dfa) is None:
            ext = self._lexeme_extension(state)
            if ext is None:
                return None
            pre = ext
            lexer = LexerRun(remainder=lexer.remainder + ext)
        node = self._finalized_node(state.stack, lexer)
        if node is None:
            return None
        cost, seq = self.reserve.completion(node)
        if cost == INF:
            return None
        data = pre + self.reserve.render(seq)
        toks = self.adapter.greedy_tokenize(data) if data else []
        return list(toks) + [self.eos_token_id]

    def _lexeme_extension(self, state: GridState) -> bytes | None:
        """BFS: shortest byte suffix completing the current partial lexeme."""
        start = state.lexer.state(self.dfa)
        a_now = self.producer.allowed(state.stack)
        okset = a_now | self.tables.ignored_terminal_ids
        frontier: list[tuple[int, bytes]] = [(start, b"")]
        seen = {start}
        for _ in range(64):
            nxt: list[tuple[int, bytes]] = []
            for st, path in frontier:
                if path and self.dfa.accepts_all[st] & okset:
                    return path
                for byte in range(256):
                    ns = self.dfa.trans[st][byte]
                    if ns != DEAD and ns not in seen:
                        seen.add(ns)
                        nxt.append((ns, path + bytes([byte])))
            frontier = nxt
            if not frontier:
                break
        return None
