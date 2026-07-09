"""E6 on a real tokenizer (network + transformers): enabled with GRID_HF_TESTS=1."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GRID_HF_TESTS") != "1",
    reason="set GRID_HF_TESTS=1 to run real-tokenizer tests (network)",
)


@pytest.fixture(scope="module")
def gpt2():
    transformers = pytest.importorskip("transformers")
    from grid.models.hf_adapter import HFTokenizerAdapter

    return HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained("gpt2"))


def test_byte_complete(gpt2):
    from grid.models.tokenizer_adapter import verify_byte_complete

    assert verify_byte_complete(gpt2)


def test_token_bytes_roundtrip(gpt2):
    """decode(ids) == concatenated token_bytes for plain ascii text (E6 canonicality)."""
    text = "select name from users where id = 42;"
    ids, _ = gpt2.encode(text)
    joined = b"".join(gpt2.token_bytes(t) for t in ids)
    assert joined.decode("utf-8") == text
    assert gpt2.decode(ids)[0] == text


def test_space_marker_inversion(gpt2):
    tid = gpt2.vocabulary["Ġselect"]
    assert gpt2.token_bytes(tid) == b" select"


def test_real_vocab_guide_differential_slice(gpt2, sql_source):
    """Mini-G3 on the real 50k vocab: fast mask == oracle at a handful of states."""
    import random

    from grid._reference.guide import ReferenceGuide
    from grid.generate import build_guide
    from grid.guide import COMPLETE

    guide = build_guide(sql_source, gpt2)
    ref = ReferenceGuide(guide.tables, guide.dfa, gpt2)
    rng = random.Random(1)
    state = guide.initial_state
    prefix: list[int] = []
    for _ in range(4):  # oracle is O(vocab * scan) per state: keep the slice small
        fast, _ = guide._mask_ids(state)
        oracle = ref.valid_next_tokens(prefix)
        assert set(fast) == oracle
        tok = rng.choice(sorted(set(fast) - {guide.eos_token_id}))
        state = guide.get_next_state(state, tok)
        prefix.append(tok)
        if state.status == COMPLETE:
            break
