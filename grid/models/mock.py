"""MockModel: deterministic seeded logits for guide/adapter tests (mini-G5)."""

from __future__ import annotations

import torch

from grid.models.tokenizer_adapter import MockTokenizer


class MockModel:
    def __init__(self, tokenizer: MockTokenizer, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.vocab_size = max(tokenizer.vocabulary.values()) + 1
        self.seed = seed

    def __call__(self, token_ids: list[int]) -> torch.Tensor:
        g = torch.Generator().manual_seed(
            (self.seed * 1_000_003 + len(token_ids) * 7919 + (token_ids[-1] if token_ids else 0)) % (2**31)
        )
        return torch.randn(self.vocab_size, generator=g)
