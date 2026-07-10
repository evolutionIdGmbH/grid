"""vLLM V1 backend (M6, mode 2): GridGuide behind vllm's LogitsProcessor plugin.

Mode-2 semantics (DESIGN.md SS4.5, LESSONS 2.3): a logits processor can only
mask, never append — jump-forward ``Write`` spans degrade to singleton masks
(no model-call savings, soundness intact), and the token-denominated reserve
cannot append completions, so the termination guarantee weakens to "EOS only at
ACCEPT" (budget truncation is vLLM's ``max_tokens`` cutoff, reported by vLLM).

Integration contract (vllm >= 0.24, V1 engine):
- REQUIRES the synchronous scheduler: ``LLM(..., async_scheduling=False)``.
  Under async scheduling the CPU prepares step k+1 while the GPU computes
  step k, so logits processors observe ``-1`` placeholders where the previous
  token belongs — a sequence-stateful mask would run one step stale and
  desync (observed, then verified on the H100 smoke). Scheduler-integrated
  masking (the route vLLM's native structured output takes) is the M6
  follow-up that lifts this.
- pass the class to the engine:  ``LLM(model, logits_processors=[GridVLLMLogitsProcessor], async_scheduling=False)``
- activate per request: ``SamplingParams(extra_args={"grid": {
      "grammar": <.grid source str>,
      "schema": {table: [columns...]}   # optional -> L3 lexicons + fingerprint
  }})``
- statefulness uses vLLM's documented live-reference contract: the
  ``output_tok_ids`` list in each ``BatchUpdate.added`` tuple is the request's
  RUNNING output list; the tracker advances the GridState along its tail before
  every forward pass.

Requests sharing (grammar, schema) share compiled artifacts AND the write-back
mask cache: one template guide per fingerprint, ``template.copy()`` per request
(the E11 serving pattern — cross-request warm hits are the design's payoff).

The core logic lives in the vllm-free ``GridRequestTracker`` (unit-tested on
any host); the thin ``GridVLLMLogitsProcessor`` adapter binds it to the vLLM
ABC and exists only when vllm is importable.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings

import torch

from grid.guide import COMPLETE, GridGuide

# ------------------------------------------------------------------ core


class _GuideRegistry:
    """(grammar, schema) fingerprint -> template guide; requests get copies
    that share the template's PRODUCER — one Rust kernel, one entry
    registration space, one write-back mask cache across requests (a
    per-request producer re-registered every touched entry per request:
    20-100 ms per literal-interior entry, the G8 batched-TPOT tail).

    Template builds run under E17 single-flight semantics
    (grid/serving/singleflight.py): vLLM compiles grammars on an executor, so
    N concurrent requests for one fingerprint get ONE build, shared results,
    the same exception on failure, and a negatively-cached FAILED slot (no
    recompile storms on known-bad specs) — the G8 concurrent-cold-start gate."""

    def __init__(self, adapter) -> None:
        from grid.serving import SingleFlight
        from grid.trie.build import build_trie

        self.adapter = adapter
        self.trie = build_trie(adapter)
        self._flight = SingleFlight(failed_ttl_s=30.0)
        # T2 pools (DESIGN §E10), one per dialect: templates of DIFFERENT
        # schemas/roles over one grammar share schema-independent entries —
        # a fresh schema starts with the literal-interior giants already warm
        # (the G8 adversarial cold-miss cost). The adapter is fixed per
        # registry, so the grammar source alone scopes a pool.
        self._t2_pools: dict[str, object] = {}
        # ContextJournals (W4), same per-dialect scope as the T2 pools: every
        # producer of a dialect records its cold walks into ONE journal, so
        # admission warmup (W5) can precompute a fresh schema's configurations
        # while its request waits off-batch. Keys/contexts only, never masks.
        self._journals: dict[str, object] = {}

    @property
    def stats(self) -> dict:
        return self._flight.stats

    def guide_for(self, spec: dict) -> GridGuide:
        grammar_src = spec["grammar"]
        schema = spec.get("schema")
        # optional verb-RBAC: an allowed-verb list projects the grammar to only
        # those statement kinds (select/insert/update/delete) before compile —
        # the mask-enforceable RBAC granularity (DESIGN.md SS4.6). Without it the
        # full grammar is used (all verbs), which is a silent verb-RBAC hole for
        # role-scoped serving (found by the G6(b) prompt suite).
        verbs = spec.get("verbs")
        verbs = tuple(sorted(verbs)) if verbs else None
        key = hashlib.blake2b(
            (grammar_src + "\x00" + json.dumps(schema, sort_keys=True)
             + "\x00" + repr(verbs)).encode(),
            digest_size=12,
        ).hexdigest()
        template = self._flight.get_or_build(key, lambda: self._build(grammar_src, schema, verbs))
        return template.copy()  # fresh state bookkeeping, SHARED mask cache

    def _build(self, grammar_src: str, schema, verbs=None) -> GridGuide:
        from grid.grammar import spec as gspec
        from grid.grammar.projection import RoleProjection
        from grid.lalr.compile import compile_tables
        from grid.lexer.dfa import build_scanner
        from grid.mask.cache import MaskCacheT2
        from grid.serving import ContextJournal

        dialect = hashlib.blake2b(grammar_src.encode(), digest_size=12).hexdigest()
        t2 = self._t2_pools.setdefault(dialect, MaskCacheT2())
        # GRID_ADMIT_WARM=0 is the W4+W5 kill switch: no journal is wired at
        # all (producer.journal stays None — the walk-miss path is exactly
        # today's), and admission_warmup no-ops independently. Read at template
        # build, like the producer's own one-shot env reads.
        journal = None if os.environ.get("GRID_ADMIT_WARM", "1") == "0" \
            else self._journals.setdefault(dialect, ContextJournal())

        grammar = gspec.load(grammar_src)
        if verbs:
            from grid.policy.bundle import PolicyBundle

            proj = PolicyBundle(role="serving", allowed_verbs=frozenset(verbs)).projection(grammar)
        else:
            proj = RoleProjection.full(grammar).build()
        lexicons = fingerprint = None
        if schema:
            from grid.policy.schema import SchemaSnapshot

            snap = SchemaSnapshot.from_dict(schema)
            tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
            lexicons = snap.lexicons(tables)
            fingerprint = snap.fingerprint
        else:
            tables = compile_tables(proj)
        dfa = build_scanner(grammar.terminals, grammar.terminal_order)
        return GridGuide(
            tables=tables, dfa=dfa, trie=self.trie, adapter=self.adapter,
            lexicons=lexicons, schema_fingerprint=fingerprint,
            reserve=None, audit=None,  # mode 2: no appends, audit opt-in later
            mask_t2=t2, mask_journal=journal,
        )


class GridRequestTracker:
    """Per-slot grammar state for a batch; vllm-free (the unit-tested core).

    For each tracked slot: advance the GridState along the live output list's
    unseen tail, then expose the next-step allowed token ids. A token outside
    the mask (another processor interfered, or speculative decode divergence)
    deactivates constraint for that request with a warning — sound generation
    cannot be guaranteed once the state desyncs (documented mode-2 boundary).
    """

    def __init__(self, registry: _GuideRegistry) -> None:
        self.registry = registry
        self.reqs: dict[int, dict] = {}  # slot -> {guide, state, out, seen}

    # -- lifecycle ---------------------------------------------------------

    def add(self, slot: int, grid_spec: dict, output_tok_ids: list[int]) -> None:
        guide = self.registry.guide_for(grid_spec)
        self.reqs[slot] = {
            "guide": guide, "state": guide.initial_state,
            "out": output_tok_ids, "seen": 0,
        }

    def remove(self, slot: int) -> None:
        self.reqs.pop(slot, None)

    def move(self, a: int, b: int, swap: bool) -> None:
        ea, eb = self.reqs.pop(a, None), self.reqs.pop(b, None)
        if ea is not None:
            self.reqs[b] = ea
        if eb is not None and swap:
            self.reqs[a] = eb

    # -- per-step ------------------------------------------------------------

    def masks(self) -> dict[int, list[int]]:
        """Advance every tracked request along its live output tail, then
        return {slot: allowed token ids for the NEXT step}. Mode-2 rules:
        Write spans -> first token only; COMPLETE -> EOS only."""
        from grid.protocols import Write

        out: dict[int, list[int]] = {}
        for slot in list(self.reqs):
            r = self.reqs[slot]
            guide: GridGuide = r["guide"]
            state = r["state"]
            toks = r["out"]
            while r["seen"] < len(toks):
                tok = int(toks[r["seen"]])
                if tok < 0:
                    # vLLM async-scheduling placeholder (-1): the slot is
                    # reserved but not yet sampled; it is overwritten in place
                    # later, so stop here and re-read on the next update_state
                    break
                # audit=True pops the guide's per-instruction _pending entry
                # (audit log is None in mode 2, so nothing is appended);
                # audit=False would leak one entry per generated token
                nxt = guide._advance(state, tok, audit=True)
                if nxt is None:
                    warnings.warn(
                        f"grid/vllm: token {tok} outside the mask for slot {slot}; "
                        "deactivating constraint for this request (state desync)",
                        stacklevel=2,
                    )
                    self.remove(slot)
                    state = None
                    break
                state = nxt
                r["seen"] += 1
            if state is None:
                continue
            r["state"] = state
            if state.status == COMPLETE:
                out[slot] = [guide.eos_token_id]
                continue
            instr = guide.get_next_instruction(state)
            ids = [int(t) for t in instr.tokens]
            out[slot] = ids[:1] if isinstance(instr, Write) else ids
        return out


def apply_masks(logits: torch.Tensor, masks: dict[int, list[int]]) -> torch.Tensor:
    """Scatter -inf outside each slot's allowed ids (in place)."""
    for slot, ids in masks.items():
        row = logits[slot]
        idx = torch.as_tensor(ids, dtype=torch.long, device=logits.device)
        keep = row.index_select(0, idx).clone()
        row.fill_(float("-inf"))
        row.index_copy_(0, idx, keep)
    return logits


# ------------------------------------------------------------------ vllm shim

try:  # pragma: no cover - exercised on vllm hosts (the box smoke), not in CI
    from vllm.v1.sample.logits_processor import (
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )

    class GridVLLMLogitsProcessor(LogitsProcessor):
        """The vLLM V1 adapter around GridRequestTracker (see module doc)."""

        def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
            self.device = device
            self._tracker: GridRequestTracker | None = None
            self._tok_name = vllm_config.model_config.tokenizer
            self._masks: dict[int, list[int]] = {}

        @classmethod
        def validate_params(cls, sampling_params) -> None:
            spec = (sampling_params.extra_args or {}).get("grid")
            if spec is None:
                return
            if not isinstance(spec, dict) or not isinstance(spec.get("grammar"), str):
                raise ValueError("extra_args['grid'] must be {'grammar': str, 'schema'?: dict}")
            schema = spec.get("schema")
            if schema is not None and not isinstance(schema, dict):
                raise ValueError("extra_args['grid']['schema'] must be {table: [columns]}")

        def _ensure_tracker(self) -> GridRequestTracker:
            if self._tracker is None:
                from transformers import AutoTokenizer

                from grid.models.hf_adapter import HFTokenizerAdapter

                adapter = HFTokenizerAdapter(AutoTokenizer.from_pretrained(self._tok_name))
                self._tracker = GridRequestTracker(_GuideRegistry(adapter))
            return self._tracker

        def is_argmax_invariant(self) -> bool:
            return False  # hard masking changes argmax by design

        def update_state(self, batch_update: BatchUpdate | None) -> None:
            if batch_update is not None:
                tracker = self._ensure_tracker()
                for slot in batch_update.removed:
                    tracker.remove(slot)
                for slot, params, _prompt, output_tok_ids in batch_update.added:
                    spec = (getattr(params, "extra_args", None) or {}).get("grid")
                    if spec is None:
                        tracker.remove(slot)  # slot may be reused by an unconstrained req
                        continue
                    tracker.add(slot, spec, output_tok_ids)
                for a, b, direct in batch_update.moved:
                    tracker.move(a, b, swap=direct == MoveDirectionality.SWAP)
            self._masks = self._tracker.masks() if self._tracker else {}

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if self._masks:
                apply_masks(logits, self._masks)
            return logits

except ImportError:  # vllm not installed: the tracker/core stays importable
    GridVLLMLogitsProcessor = None  # type: ignore[assignment]
