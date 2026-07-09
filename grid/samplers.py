"""Samplers (SS4.1 contract): return the 3-tuple
(next_token_ids (n,1), ancestors (n,), weights (n,))."""

from __future__ import annotations

import torch


class GreedySampler:
    def __init__(self) -> None:
        self.samples = 1

    def __call__(self, next_token_logits: torch.Tensor, sequence_weights: torch.Tensor, _rng):
        logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        ids = torch.argmax(logprobs, dim=-1, keepdim=True)
        ancestors = torch.arange(next_token_logits.shape[0])
        weights = sequence_weights + torch.gather(logprobs, 1, ids).squeeze(-1)
        return ids, ancestors, weights


class MultinomialSampler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.samples = 1
        self.temperature = temperature
        self.top_k = None
        self.top_p = None

    def __call__(self, next_token_logits: torch.Tensor, sequence_weights: torch.Tensor, rng):
        scaled = next_token_logits / max(self.temperature, 1e-6)
        probs = torch.nn.functional.softmax(scaled, dim=-1)
        ids = torch.multinomial(probs, num_samples=1, generator=rng)
        logprobs = torch.nn.functional.log_softmax(scaled, dim=-1)
        ancestors = torch.arange(next_token_logits.shape[0])
        weights = sequence_weights + torch.gather(logprobs, 1, ids).squeeze(-1)
        return ids, ancestors, weights


def greedy() -> GreedySampler:
    return GreedySampler()


def multinomial(temperature: float = 1.0) -> MultinomialSampler:
    return MultinomialSampler(temperature=temperature)
