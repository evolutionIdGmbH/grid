"""vLLM V1 scheduler-side structured-output backend for GRID (M6, second slice).

Unlike the logits-processor route (grid/models/vllm_processor.py), the
scheduler-side path is fed tokens IN ORDER by vLLM's structured-output manager
(`vllm/v1/structured_output/__init__.py`) and fills a per-request row of the
batch token bitmask before sampling — the same integration point xgrammar and
llguidance use. This lifts the `async_scheduling=False` requirement of mode 2:
ordering is the manager's contract, and speculative/async paths are handled via
the interface's `rollback`/`validate_tokens`, which are natural for GRID
(states are immutable and persistent — rollback is list truncation).

`GridGrammarSession` is the vllm-free core (unit-tested anywhere).
`GridStructuredBackend` adapts it to vllm's `StructuredOutputBackend` ABC.

vLLM 0.24 has no backend plugin registry, so wiring GRID in patches THREE
sites (bench/vllm_grid_patch.py applies all three idempotently; PR-shaped):

1. `vllm/v1/structured_output/__init__.py` — the backend dispatch chain:

    elif backend == "grid":
        from grid.models.vllm_structured import GridStructuredBackend
        self.backend = GridStructuredBackend(
            self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size,
        )

2. `vllm/config/structured_outputs.py` — add "grid" to the backend choices.
3. `vllm/sampling_params.py` `_validate_structured_outputs` — a no-op branch
   for "grid" (its frontend otherwise SNIFFS grammar specs as Lark/GBNF and
   rejects both raw .grid sources and the JSON envelope before the backend is
   ever consulted; GRID validates at compile time via GrammarInvalid /
   LALRConflictError).

then per request:
    SamplingParams(structured_outputs=StructuredOutputsParams(
        grammar=<.grid source, or a JSON envelope
                 '{"grammar": "<.grid>", "schema": {table: [cols]}}'>),
        ...)  # with structured_outputs_config={"backend": "grid"}

Accepted on GPU (A10, vllm 0.24, DEFAULT async-capable scheduler) via
bench/vllm_sched_accept.py: 4/4 viable prefixes, >=1 complete — the
async_scheduling=False restriction of mode 2 does not apply to this path.

The grammar spec accepts either a raw ``.grid`` source or the JSON envelope
(adds L3 schema lexicons + fingerprint, the RBAC/schema-enforcement path).
"""

from __future__ import annotations

import json
import os

import numpy as np

from grid.guide import ACCEPTING, ACTIVE, COMPLETE, GRAMMAR_END, GridGuide
from grid.mask.producer import _chain
from grid.models.vllm_processor import _GuideRegistry

# v6 kernel-session status codes (grid_core) -> guide status strings
_STATUS = (ACTIVE, ACCEPTING, GRAMMAR_END, COMPLETE)
# session_accept flags (grid_core): 0 == REJECTED
_FLAG_OK, _FLAG_COMPLETE, _FLAG_UNBOUND = 1, 2, 4
_ST_COMPLETE = 3


def _parse_spec(grammar_spec: str) -> dict:
    """Raw .grid source, or the {"grammar": ..., "schema"?: ...} envelope."""
    s = grammar_spec.lstrip()
    if s.startswith("{"):
        try:
            spec = json.loads(grammar_spec)
        except json.JSONDecodeError:
            spec = None
        if isinstance(spec, dict) and isinstance(spec.get("grammar"), str):
            return spec
    return {"grammar": grammar_spec}


# One live batch bitmask per engine (vllm allocates it once and reuses it every
# step); converting torch row -> numpy per fill cost ~15 µs/request/step in
# torch getitem + .numpy() + .view — at batch 32 that alone is most of the gap
# to the xgrammar floor (+1.07%, bench/vllm_xgr_floor.py). Cache ONE uint32
# view of the whole tensor and hand out numpy row views instead.
_ROWS_CACHE: dict[int, tuple[object, np.ndarray]] = {}


def _np_rows(bitmask) -> np.ndarray:
    got = _ROWS_CACHE.get(id(bitmask))
    if got is None or got[0] is not bitmask:
        arr = bitmask.numpy() if hasattr(bitmask, "numpy") else np.asarray(bitmask)
        _ROWS_CACHE.clear()  # single live bitmask; also defends id() reuse
        got = (bitmask, arr.view(np.uint32))
        _ROWS_CACHE[id(bitmask)] = got
    return got[1]


class _PrefetchBuild:
    """State-shaped prefetch handle for kernel (v6) sessions: captures the
    pure walk inputs (remainder, A) on the scheduler thread at schedule time,
    so the pool thread never touches session state (protocol-safe by
    construction). Doubles as both `guide` and `state` for
    MaskPrefetcher.schedule — `_mask_ids(state)` is the build entrypoint and
    object identity keys dedup/wait/drop, exactly like a GridState."""

    __slots__ = ("producer", "remainder", "A")

    def __init__(self, producer, remainder: bytes, A) -> None:
        self.producer = producer
        self.remainder = remainder
        self.A = A

    def _mask_ids(self, _state):  # MaskPrefetcher._build protocol
        self.producer.prefetch_build(self.remainder, self.A)


class GridGrammarSession:
    """The six-method structured-output contract over a GridGuide.

    Mirrors vllm's XgrammarGrammar semantics exactly: accept_tokens consumes
    greedily and returns False on the first non-viable token (prior tokens stay
    consumed — the manager treats False as terminal for the request);
    validate_tokens returns the longest accepted prefix WITHOUT advancing;
    rollback(n) rewinds n tokens (persistent states: truncation).

    v6 kernel sessions: when the producer's grid_core kernel is present, the
    guide is audit-free, and GRID_NO_V6 is unset, per-token accept / validate /
    rollback / fill run IN-KERNEL against a session that shares the producer's
    arena/memos/entries — the Python per-token cost (lexer byte-scan, event
    shifts, status derivation, frozen-dataclass states) disappears from the
    warm path. `self.states` then stays at the initial state; the v5 path is
    fully intact for audit-enabled guides, the no-kernel spec build, and
    GRID_NO_V6=1 A/B forcing.

    With a MaskPrefetcher attached (§6 overlap contract): accept_tokens kicks a
    background build of the successor state's mask ONLY when that mask is not
    already T1-warm/bound — the cold trie walk runs with the GIL released and
    overlaps the remaining CPU scheduling work (and, under async scheduling,
    the GPU forward window); fill_bitmask waits only for the un-hidden
    remainder (recorded in prefetcher stats; G8's adversarial cold-miss arm
    measures it). Warm steady state never touches the pool: unconditional
    scheduling serialized every request's step behind the single-flight worker
    queue and GIL ping-pong (the G8 batched-TPOT pathology, H100 probe
    2026-07-09: 209 ms/step at batch 8 with zero cache misses)."""

    def __init__(self, guide: GridGuide, prefetcher=None, _force_v5: bool = False) -> None:
        self.guide = guide
        self.states = [guide.initial_state]
        self.num_processed_tokens = 0
        self.prefetcher = prefetcher
        self._kernel = None
        self._sid = None
        self._complete = False
        self._pf_target = None
        prod = guide.producer
        if (not _force_v5
                and guide.audit is None  # E14: audit-enabled guides stay on v5
                and prod._kernel is not None
                and os.environ.get("GRID_NO_V6") != "1"
                and prod.ensure_session_tables(guide.adapter)):
            self._kernel = prod._kernel
            self._sid = self._kernel.session_new(
                _chain(guide.initial_state.stack), guide.eos_token_id)

    def __del__(self):  # release the kernel session slot (best effort)
        if getattr(self, "_sid", None) is not None:
            try:
                self._kernel.session_free(self._sid)
            except Exception:
                pass

    # -- contract ------------------------------------------------------------

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        if self._sid is not None:
            return self._accept_v6(tokens)
        if self.is_terminated():
            return False
        for t in tokens:
            nxt = self.guide._advance(self.states[-1], int(t), audit=True)
            if nxt is None:
                return False
            self.states.append(nxt)
            self.num_processed_tokens += 1
        if (self.prefetcher is not None and self.states[-1].status != COMPLETE
                and not self.guide.is_mask_warm(self.states[-1])):
            self.prefetcher.schedule(self.guide, self.states[-1])
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        if self._sid is not None:
            n = self._kernel.session_validate(self._sid, [int(t) for t in tokens])
            return [int(t) for t in tokens[:n]]
        cur = self.states[-1]
        accepted: list[int] = []
        for t in tokens:
            nxt = self.guide._advance(cur, int(t), audit=False)
            if nxt is None:
                break
            accepted.append(int(t))
            cur = nxt
        return accepted

    def rollback(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        if self._sid is not None:
            self._drop_prefetch()
            self._kernel.session_rollback(self._sid, int(num_tokens))
            self.num_processed_tokens = max(0, self.num_processed_tokens - num_tokens)
            self._complete = self._kernel.session_state(self._sid)[2] == _ST_COMPLETE
            return
        if self.prefetcher is not None:
            self.prefetcher.drop(self.states[-1])
        del self.states[max(1, len(self.states) - num_tokens):]
        self.num_processed_tokens = max(0, self.num_processed_tokens - num_tokens)

    def fill_bitmask(self, bitmask, idx: int) -> None:
        pf = self.prefetcher
        if self._sid is not None:
            if pf is not None and pf._inflight and self._pf_target is not None:
                pf.wait(self._pf_target)
                self._pf_target = None
            prod = self.guide.producer
            prod._sync_epoch()  # risk (d): rollover must drop kernel bindings
            row = _np_rows(bitmask)[idx]
            if self._kernel.session_fill(self._sid, row) < 0:
                # unbound configuration: bind-time guard + peek-or-build, then
                # the retried fill is served in-kernel (miss counted by cache.get)
                a_words, remainder = self._kernel.session_walk_inputs(self._sid)
                handle = prod.session_bind_handle(a_words, remainder)
                self._kernel.session_bind(self._sid, handle)
                got = self._kernel.session_fill(self._sid, row)
                assert got >= 0, "fill after bind must hit (kernel v6 invariant)"
            return
        if pf is not None and pf._inflight:  # GIL-atomic read; empty = no lock
            pf.wait(self.states[-1])
        self.guide.fill_bitmask(self.states[-1], _np_rows(bitmask)[idx])

    def is_terminated(self) -> bool:
        if self._sid is not None:
            return self._complete
        return self.states[-1].status == COMPLETE

    def reset(self) -> None:
        if self._sid is not None:
            self._drop_prefetch()
            self._kernel.session_reset(self._sid)
            self.num_processed_tokens = 0
            self._complete = False
            return
        if self.prefetcher is not None:
            self.prefetcher.drop(self.states[-1])
        del self.states[1:]
        self.num_processed_tokens = 0

    # -- v6 internals ----------------------------------------------------------

    def _accept_v6(self, tokens: list[int]) -> bool:
        if self._complete:
            return False
        accept = self._kernel.session_accept
        sid = self._sid
        flags = 0
        for t in tokens:
            flags = accept(sid, int(t))
            if flags == 0:
                return False
            self.num_processed_tokens += 1
            if flags & _FLAG_COMPLETE:
                self._complete = True
        if flags & _FLAG_UNBOUND and not self._complete:
            self._bind_or_schedule()
        return True

    def _bind_or_schedule(self) -> None:
        """Warm gate for the successor's fill: T1-warm -> bind now (the kernel
        serves every later fill); cold -> schedule the walk on the prefetch
        pool from captured pure inputs. Replaces v5's is_mask_warm gate."""
        prod = self.guide.producer
        a_words, remainder = self._kernel.session_walk_inputs(self._sid)
        handle, A = prod.session_peek_handle(a_words, remainder)
        if handle is not None:
            self._kernel.session_bind(self._sid, handle)
        elif self.prefetcher is not None:
            self._drop_prefetch()  # supersede any unconsumed older build
            self._pf_target = _PrefetchBuild(prod, remainder, A)
            self.prefetcher.schedule(self._pf_target, self._pf_target)

    def _drop_prefetch(self) -> None:
        if self.prefetcher is not None and self._pf_target is not None:
            self.prefetcher.drop(self._pf_target)
            self._pf_target = None


try:  # pragma: no cover - exercised on vllm hosts (next runner session)
    from dataclasses import dataclass

    import torch
    from vllm.v1.structured_output.backend_types import (
        StructuredOutputBackend,
        StructuredOutputGrammar,
        StructuredOutputOptions,
    )

    @dataclass
    class _GridGrammar(StructuredOutputGrammar):
        session: GridGrammarSession

        def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
            return self.session.accept_tokens(request_id, tokens)

        def validate_tokens(self, tokens: list[int]) -> list[int]:
            return self.session.validate_tokens(tokens)

        def rollback(self, num_tokens: int) -> None:
            self.session.rollback(num_tokens)

        def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
            self.session.fill_bitmask(bitmask, idx)

        def is_terminated(self) -> bool:
            return self.session.is_terminated()

        def reset(self):
            self.session.reset()

    @dataclass
    class GridStructuredBackend(StructuredOutputBackend):
        """vllm_config / tokenizer / vocab_size fields per the base dataclass."""

        def __post_init__(self) -> None:
            from grid.models.hf_adapter import HFTokenizerAdapter
            from grid.serving import MaskPrefetcher

            self._registry = _GuideRegistry(HFTokenizerAdapter(self.tokenizer))
            # 4 workers: cold walks release the GIL, so concurrent cold
            # successors (heterogeneous batches, adversarial cold-miss) overlap
            # instead of queueing; the warm path never schedules (see
            # GridGrammarSession.accept_tokens)
            self._prefetcher = MaskPrefetcher(max_workers=4)

        def compile_grammar(
            self, request_type: StructuredOutputOptions, grammar_spec: str
        ) -> StructuredOutputGrammar:
            if request_type != StructuredOutputOptions.GRAMMAR:
                raise ValueError(
                    f"grid backend supports GRAMMAR requests only, got {request_type}"
                )
            guide = self._registry.guide_for(_parse_spec(grammar_spec))
            return _GridGrammar(
                session=GridGrammarSession(guide, prefetcher=self._prefetcher)
            )

        def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
            return torch.zeros(
                (max_num_seqs, (self.vocab_size + 31) // 32), dtype=torch.int32
            )

        def destroy(self):
            if getattr(self, "_prefetcher", None) is not None:
                self._prefetcher.shutdown()
                self._prefetcher = None
            self._registry = None

except ImportError:  # vllm not installed: the session/core stays importable
    GridStructuredBackend = None  # type: ignore[assignment]
