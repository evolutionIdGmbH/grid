"""SS4.4/SS4.5: the tool-family generation adapter + the GRID-owned decode loop.

Mode 1 (GRID-owned loop, local models): Write spans are appended WITHOUT forward
passes; the guide is advanced (and audited) once per appended token. Mode 2
(processor-only, e.g. vLLM) lives in GridLogitsProcessor (singleton-mask degrade).

E15 GenerationSession: INIT -> PROMPT_ENCODED -> STREAMING -> STOPPED(reason);
``processor.finish()`` fires on every stop; INV-OUT1 (non-error stops parse) is
asserted in debug mode by the E2E tests. ``stop_at`` policy: rejected in sql
mode, allowed in cfg mode (excluded from INV-OUT1, flagged in the audit seal).
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import torch

from grid.guide import COMPLETE
from grid.protocols import Generate, Write
from grid.samplers import GreedySampler, MultinomialSampler


@dataclass(frozen=True)
class GenerationParameters:
    max_tokens: int | None
    stop_at: list[str] | None
    seed: int | None


@dataclass(frozen=True)
class SamplingParameters:
    sampler: str
    num_samples: int
    top_p: float | None
    top_k: int | None
    temperature: float | None


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stop_reason: str
    audit: object | None = None

    def __str__(self) -> str:  # adapter callers mostly want the text
        return self.text


class GridSequenceGeneratorAdapter:
    """Tool-family adapter contract: __call__(prompts, max_tokens=None, stop_at=None, seed=None) + .stream()."""

    def __init__(self, model, logits_processor, sampler, mode: str = "cfg") -> None:
        self.model = model
        self.logits_processor = logits_processor
        self.mode = mode
        if isinstance(sampler, MultinomialSampler):
            self.sampling_params = SamplingParameters(
                "multinomial", sampler.samples, sampler.top_p, sampler.top_k, sampler.temperature
            )
        elif isinstance(sampler, GreedySampler):
            self.sampling_params = SamplingParameters("greedy", sampler.samples, None, None, 0.0)
        else:
            raise TypeError(f"unsupported sampler {type(sampler)}")
        self.sampler = sampler

    def prepare_generation_parameters(self, max_tokens, stop_at, seed) -> GenerationParameters:
        if isinstance(stop_at, str):
            stop_at = [stop_at]
        return GenerationParameters(max_tokens, stop_at, seed)

    # -- GRID-owned loop (mode 1) --------------------------------------------

    def __call__(self, prompts, max_tokens: int | None = None, stop_at=None, seed: int | None = None):
        if self.mode == "sql" and stop_at is not None:
            raise ValueError("stop_at unsupported in sql mode (INV-OUT1); use cfg mode")
        if isinstance(prompts, list):
            return [self(p, max_tokens=max_tokens, stop_at=stop_at, seed=seed) for p in prompts]
        params = self.prepare_generation_parameters(max_tokens, stop_at, seed)
        return self._generate_one(prompts, params)

    def stream(self, prompt: str, max_tokens: int | None = None, seed: int | None = None):
        result = self._generate_one(prompt, self.prepare_generation_parameters(max_tokens, None, seed))
        tokenizer = self.model.tokenizer
        for tid in result.token_ids:
            yield tokenizer.decode([tid])[0]

    def _generate_one(self, prompt: str, params: GenerationParameters) -> GenerationResult:
        processor = copy(self.logits_processor)  # adapter contract: clone via copy.copy per generation
        guide = processor.guide
        if params.max_tokens is not None:
            guide.max_new_tokens = params.max_tokens

        tokenizer = self.model.tokenizer
        prompt_ids, _ = tokenizer.encode(prompt)
        ids = list(prompt_ids)
        rng = torch.Generator()
        rng.manual_seed(params.seed if params.seed is not None else 0)

        state = guide.initial_state
        out: list[int] = []
        stop_reason = "ERROR"
        try:
            while True:
                instr = guide.get_next_instruction(state)
                if isinstance(instr, Write):
                    span = [int(t) for t in instr.tokens]
                    budget_write = guide.max_new_tokens is not None and (
                        span and span[-1] == guide.eos_token_id and len(span) > 1
                    )
                    for t in span:
                        state = guide.get_next_state(state, t)
                        out.append(t)
                        if state.status == COMPLETE:
                            break
                    if state.status == COMPLETE:
                        stop_reason = (
                            "MAX_TOKENS_WITH_JUMP_COMPLETE" if budget_write else "EOS_ACCEPT"
                        )
                        break
                    continue
                assert isinstance(instr, Generate) and instr.tokens is not None
                logits = self.model(ids + out)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask[torch.as_tensor(instr.tokens, dtype=torch.long)] = False
                masked = logits.masked_fill(mask, float("-inf"))
                tok_ids, _anc, _w = self.sampler(masked.unsqueeze(0), torch.zeros(1), rng)
                t = int(tok_ids[0, 0])
                state = guide.get_next_state(state, t)
                out.append(t)
                if state.status == COMPLETE:
                    stop_reason = "EOS_ACCEPT"
                    break
                if params.stop_at and self.mode == "cfg":
                    text_so_far = tokenizer.decode(out)[0]
                    if any(s in text_so_far for s in params.stop_at):
                        stop_reason = "STOP_SEQUENCE"
                        break
        finally:
            processor.finish()
            if guide.audit is not None and not guide.audit.sealed:
                guide.audit.seal(
                    stop_reason,
                    {"grammar": guide.tables.fingerprint, "trie": guide.trie.tokenizer_fingerprint},
                    flags={"mode": self.mode},
                )

        text = tokenizer.decode([t for t in out if t != guide.eos_token_id])[0]
        return GenerationResult(text=text, token_ids=out, stop_reason=stop_reason, audit=guide.audit)
