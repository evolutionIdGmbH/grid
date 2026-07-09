"""E13/SS4.3: anchoring, __copy__ isolation, finish semantics, Write degrade, batch."""

from copy import copy

import pytest
import torch

from grid.errors import ProcessorReuseError
from grid.generate import build_guide
from grid.processors import GridLogitsProcessor
from grid.protocols import Generate


@pytest.fixture()
def proc(toy_source, toy_tokenizer):
    guide = build_guide(toy_source, toy_tokenizer)
    return GridLogitsProcessor(toy_tokenizer, guide)


def test_anchoring_excludes_prompt(proc, toy_tokenizer):
    """Prompt ids (arbitrary text) must never be fed to the guide."""
    prompt_ids, _ = toy_tokenizer.encode("Write me an expression: ")  # not grammatical
    ids = torch.tensor([list(prompt_ids)])
    logits = torch.zeros(1, proc.guide.vocab_size)
    out = proc.process_logits(ids, logits.clone())
    assert proc._seq_start_idx == len(prompt_ids)
    allowed = (out[0] != float("-inf")).nonzero().flatten().tolist()
    # start-of-grammar mask: must include 'foo' but not '+'
    assert toy_tokenizer.vocabulary["foo"] in allowed
    assert toy_tokenizer.vocabulary["+"] not in allowed


def test_copy_isolates_state(proc):
    p2 = copy(proc)
    assert p2 is not proc
    assert p2.guide is not proc.guide
    assert p2._guide_states is not proc._guide_states
    assert p2._seq_start_idx is None


def test_sequential_generations_do_not_share_state(toy_source, toy_tokenizer, toy_model):
    from grid import generate
    from grid.samplers import multinomial

    g = generate.cfg(toy_model, toy_source, sampler=multinomial(1.0))
    r1 = g("", max_tokens=25, seed=1)
    r2 = g("", max_tokens=25, seed=1)  # second call: fresh processor via copy()
    assert r1.text == r2.text


def test_finish_then_reuse_raises(proc):
    ids = torch.tensor([[0]])
    logits = torch.zeros(1, proc.guide.vocab_size)
    proc.process_logits(ids, logits.clone())
    proc.finish()
    with pytest.raises(ProcessorReuseError):
        proc.process_logits(ids, logits.clone())


def test_generate_none_skips_row(proc):
    class _State:
        status = "ACTIVE"

    class NullGuide:
        initial_state = _State()

        def get_next_instruction(self, state):
            return Generate(None)

        def get_next_state(self, state, token_id):
            return _State()

        def is_final_state(self, state):
            return False

        def copy(self):
            return self

    p = GridLogitsProcessor(proc.tokenizer, NullGuide())
    logits = torch.randn(1, 16)
    out = p.process_logits(torch.tensor([[1, 2]]), logits.clone())
    assert torch.equal(out, logits)  # untouched: Generate(None) row skipped


def test_write_degrades_to_singleton_in_processor_mode(toy_source, toy_tokenizer):
    """SS4.5 mode 2: never the pinned span-union."""
    guide = build_guide(toy_source, toy_tokenizer)
    proc = GridLogitsProcessor(toy_tokenizer, guide)
    # drive to a state where the instruction is a Write span: budget trigger
    guide.max_new_tokens = 1  # immediately at/below reserve -> Write completion
    ids = torch.tensor([[0]])
    logits = torch.zeros(1, guide.vocab_size)
    out = proc.process_logits(ids, logits.clone())
    allowed = (out[0] != float("-inf")).nonzero().flatten().tolist()
    assert len(allowed) == 1  # only forced_ids[0], not the whole span


def test_batch_rows_independent(toy_source, toy_tokenizer):
    guide = build_guide(toy_source, toy_tokenizer)
    proc = GridLogitsProcessor(toy_tokenizer, guide)
    foo = toy_tokenizer.vocabulary["foo"]
    plus = toy_tokenizer.vocabulary["+"]
    logits = torch.zeros(2, guide.vocab_size)
    proc.process_logits(torch.tensor([[0], [0]]), logits.clone())  # anchor: prompt = [0]
    proc.process_logits(torch.tensor([[0, foo], [0, foo]]), logits.clone())
    out = proc.process_logits(torch.tensor([[0, foo, plus], [0, foo, foo]]), logits.clone())
    a0 = (out[0] != float("-inf")).nonzero().flatten().tolist()
    a1 = (out[1] != float("-inf")).nonzero().flatten().tolist()
    assert a0 != a1  # 'foo+' vs 'foofoo' -> different masks
    assert toy_tokenizer.vocabulary["("] in a0  # after '+': factor may start
    assert toy_tokenizer.vocabulary["("] not in a1  # inside identifier: no '('
