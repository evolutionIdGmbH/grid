"""GRID protocol shapes (DESIGN.md SS4.1) — the interface convention shared by
our internal generation tools, defined here and conformance-tested under
tests/protocols/.

Normative contracts:
- Guides emit ``Write``/``Generate`` instructions whose ``tokens`` are
  ``torch.LongTensor``; processors additionally normalize with ``torch.as_tensor``
  at the boundary as a safety net.
- ``Generate(None)`` means "all tokens allowed"; GRID guides never emit it
  (processors handle it defensively by skipping masking for that row).
- Samplers return the 3-tuple ``(next_token_ids (n,1), ancestors (n,), weights (n,))``.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import torch


@dataclass(frozen=True)
class Write:
    """Write instruction: append this token sequence without sampling."""

    tokens: torch.Tensor


@dataclass(frozen=True)
class Generate:
    """Generate instruction: sample from exactly these tokens.

    ``None`` means "all tokens allowed". GRID guides never emit ``Generate(None)``
    (DESIGN.md SS4.1); processors handle it defensively by skipping masking.
    """

    tokens: torch.Tensor | None


Instruction = Write | Generate


@runtime_checkable
class Guide(Protocol):
    """Base generation-guide protocol of the GRID tool family."""

    initial_state: Any

    def get_next_instruction(self, state: Any) -> Instruction: ...

    def get_next_state(self, state: Any, token_id: int) -> Any: ...

    def is_final_state(self, state: Any) -> bool: ...

    def copy(self) -> Guide: ...


@runtime_checkable
class Tokenizer(Hashable, Protocol):
    """Tokenizer protocol of the GRID tool family."""

    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: dict[str, int]
    special_tokens: set[str]

    def encode(self, prompt: str | list[str]) -> tuple[Any, Any]: ...

    def decode(self, token_ids: Any) -> list[str]: ...

    def convert_token_to_string(self, token: str) -> str: ...


@runtime_checkable
class Sampler(Protocol):
    """Sampler protocol: NORMATIVE return is the 3-tuple
    ``(next_token_ids (n_seqs, 1), ancestors (n_seqs,), weights (n_seqs,))``."""

    samples: int

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        sequence_weights: torch.Tensor,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
