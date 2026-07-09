"""G0: the protocol shapes in grid/protocols.py match the tool-family convention.

Self-contained: expected signatures are stated here as the normative record —
no third-party sources are read or shipped."""

import dataclasses
import inspect

import pytest
import torch

import grid.protocols as P

GUIDE_METHODS = {
    "get_next_instruction": ["self", "state"],
    "get_next_state": ["self", "state", "token_id"],
    "is_final_state": ["self", "state"],
    "copy": ["self"],
}

TOKENIZER_ATTRS = {"eos_token", "eos_token_id", "pad_token_id", "vocabulary", "special_tokens"}
TOKENIZER_METHODS = {"encode", "decode", "convert_token_to_string"}


def test_guide_protocol_shape():
    for name, args in GUIDE_METHODS.items():
        fn = getattr(P.Guide, name)
        assert [p for p in inspect.signature(fn).parameters] == args, name
    assert "initial_state" in P.Guide.__annotations__


def test_write_generate_shape():
    assert list(P.Write.__dataclass_fields__) == ["tokens"]
    assert list(P.Generate.__dataclass_fields__) == ["tokens"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        P.Write(tokens=torch.tensor([1])).tokens = None  # type: ignore[misc]


def test_tokenizer_protocol_shape():
    assert TOKENIZER_ATTRS <= set(P.Tokenizer.__annotations__)
    for name in TOKENIZER_METHODS:
        assert callable(getattr(P.Tokenizer, name))


def test_sampler_protocol_shape():
    assert "samples" in P.Sampler.__annotations__
    args = [p for p in inspect.signature(P.Sampler.__call__).parameters]
    assert args == ["self", "next_token_logits", "sequence_weights", "rng"]


def test_grid_guide_satisfies_protocol_and_tensor_contract(toy_source, toy_tokenizer):
    """The tensor contract: instruction.tokens.to(device) must work on every
    instruction a GRID guide emits (SS4.1)."""
    from grid.generate import build_guide

    guide = build_guide(toy_source, toy_tokenizer)
    assert isinstance(guide, P.Guide)
    instr = guide.get_next_instruction(guide.initial_state)
    moved = instr.tokens.to("cpu", non_blocking=True)
    assert moved.dtype == torch.long and len(moved) > 0


def test_grid_sampler_return_shape():
    from grid.samplers import greedy, multinomial

    for sampler in (greedy(), multinomial(0.7)):
        logits = torch.randn(2, 16)
        ids, ancestors, weights = sampler(logits, torch.zeros(2), torch.Generator().manual_seed(0))
        assert ids.shape == (2, 1) and ancestors.shape == (2,) and weights.shape == (2,)
