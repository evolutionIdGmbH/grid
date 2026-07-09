"""GridGrammarSession (scheduler-side structured-output core) + kernel #4's
fill_bitmask — vllm-free. The bitmask must be bit-exact with _mask_ids at every
walked state; accept/validate/rollback follow vllm's XgrammarGrammar semantics.
"""

import random

import numpy as np
import pytest
import torch

from grid.guide import COMPLETE
from grid.models.vllm_structured import GridGrammarSession, _parse_spec


@pytest.fixture
def guide(toy_source, toy_tokenizer):
    from grid.generate import build_guide

    return build_guide(toy_source, toy_tokenizer)


def _bits_to_ids(words: np.ndarray, vocab: int) -> set[int]:
    out = set()
    for w, word in enumerate(words.tolist()):
        b = 0
        while word:
            if word & 1:
                out.add(w * 32 + b)
            word >>= 1
            b += 1
    return {t for t in out if t < vocab}


def test_fill_bitmask_matches_mask_ids_along_walk(guide):
    rng = random.Random(5)
    words = (guide.vocab_size + 31) // 32
    out = np.empty(words, dtype=np.uint32)
    state = guide.initial_state
    for _ in range(20):
        ids, _ = guide._mask_ids(state)
        guide.fill_bitmask(state, out)
        assert _bits_to_ids(out, guide.vocab_size) == {int(t) for t in ids}
        if state.status == "COMPLETE":
            break
        pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
        state = guide.get_next_state(state, pick)


def test_session_accept_validate_rollback(guide):
    s = GridGrammarSession(guide)
    ids, _ = guide._mask_ids(s.states[-1])
    first = int(next(t for t in ids if t != guide.eos_token_id))
    bad = next(t for t in range(guide.vocab_size) if not bool((ids == t).any()))

    # validate does not advance
    assert s.validate_tokens([first, bad]) == [first]
    assert s.num_processed_tokens == 0 and len(s.states) == 1

    # accept advances; a bad token fails after consuming the good prefix
    assert s.accept_tokens("r0", [first]) is True
    assert s.num_processed_tokens == 1
    assert s.accept_tokens("r0", [bad]) is False

    # rollback truncates persistent states
    s.rollback(1)
    assert s.num_processed_tokens == 0 and len(s.states) == 1

    s.reset()
    assert len(s.states) == 1 and not s.is_terminated()


def test_session_terminates_and_refuses_after_eos(guide):
    rng = random.Random(9)
    s = GridGrammarSession(guide)
    # shadow the session with a GridState walked in lockstep: v6 kernel
    # sessions do not extend s.states, so picks come from the shadow's mask
    st = guide.initial_state
    for _ in range(64):
        ids, _ = guide._mask_ids(st)
        pick = guide.eos_token_id if guide.eos_token_id in ids else rng.choice(ids)
        assert s.accept_tokens("r1", [int(pick)])
        st = guide.get_next_state(st, int(pick))
        if s.is_terminated():
            break
    assert s.is_terminated()
    assert st.status == COMPLETE
    assert s.accept_tokens("r1", [0]) is False  # terminated sessions refuse


def test_fill_bitmask_through_torch_int32_row(guide):
    """The vllm bitmask is an int32 torch tensor; the uint32 view must write
    through (the exact fill path the backend uses)."""
    words = (guide.vocab_size + 31) // 32
    bm = torch.zeros((2, words), dtype=torch.int32)
    s = GridGrammarSession(guide)
    s.fill_bitmask(bm, 1)
    ids, _ = guide._mask_ids(s.states[-1])
    got = _bits_to_ids(bm[1].numpy().view(np.uint32), guide.vocab_size)
    assert got == {int(t) for t in ids}
    assert bm[0].abs().sum() == 0  # other rows untouched


def test_parse_spec_envelope_and_raw():
    assert _parse_spec('{"grammar": "x", "schema": {"t": ["c"]}}')["schema"] == {"t": ["c"]}
    assert _parse_spec("%start s\ns: \"a\"")["grammar"].startswith("%start")
    assert _parse_spec("{ not json")["grammar"].startswith("{ not json")
