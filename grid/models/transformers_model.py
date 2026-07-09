"""Transformers model wrapper for the GRID-owned decode loop (SS4.5 mode 1)."""

from __future__ import annotations

import torch

from grid.models.hf_adapter import HFTokenizerAdapter


class TransformersModel:
    """Callable(ids) -> last-position logits; .tokenizer is the GRID adapter."""

    def __init__(self, model, adapter: HFTokenizerAdapter, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.tokenizer = adapter
        self.device = device
        self.vocab_size = max(adapter.vocabulary.values()) + 1

    @staticmethod
    def from_pretrained(name: str, device: str = "cpu", dtype=None) -> TransformersModel:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        return TransformersModel(model, HFTokenizerAdapter(tok), device=device)

    @torch.no_grad()
    def __call__(self, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id]
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits = self.model(ids).logits[0, -1, :].float().cpu()
        # model head may be larger/smaller than tokenizer vocab; align
        if logits.shape[0] < self.vocab_size:
            pad = torch.full((self.vocab_size - logits.shape[0],), float("-inf"))
            logits = torch.cat([logits, pad])
        return logits[: self.vocab_size]
