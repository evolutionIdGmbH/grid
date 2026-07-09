"""SS9: statechart tests generated from the YAML source of truth, plus entity observers."""

import pytest

from grid._statecharts.engine import Statechart, all_chart_names, load_chart
from grid.errors import IllegalTransition, ProcessorReuseError


@pytest.mark.parametrize("name", all_chart_names())
def test_chart_wellformed(name):
    spec = load_chart(name)
    assert spec.initial in spec.states
    for (frm, _trig), to in spec.transitions.items():
        assert frm in spec.states and to in spec.states
        assert frm not in spec.terminal  # engine validates too; belt and braces


@pytest.mark.parametrize("name", all_chart_names())
def test_allowed_transitions_fire(name):
    spec = load_chart(name)
    for (frm, trig), to in spec.transitions.items():
        sc = Statechart(spec, state=frm)
        assert sc.fire(trig) == to


@pytest.mark.parametrize("name", all_chart_names())
def test_unlisted_transitions_raise(name):
    spec = load_chart(name)
    triggers = {t for (_f, t) in spec.transitions}
    for state in spec.states:
        for trig in triggers | {"__nonsense__"}:
            if (state, trig) in spec.transitions:
                continue
            sc = Statechart(spec, state=state)
            with pytest.raises(IllegalTransition):
                sc.fire(trig)


def test_processor_lifecycle_observer(toy_source, toy_tokenizer):
    """E13 observer test: FRESH -> ANCHORED -> FINISHED; reuse raises."""
    import torch

    from grid.generate import build_guide
    from grid.processors import GridLogitsProcessor

    guide = build_guide(toy_source, toy_tokenizer)
    proc = GridLogitsProcessor(toy_tokenizer, guide)
    assert proc._sc.state == "FRESH"
    ids = torch.tensor([[0, 1]])
    logits = torch.zeros(1, guide.vocab_size)
    proc.process_logits(ids, logits.clone())
    assert proc._sc.state == "ANCHORED"
    proc.finish()
    assert proc._sc.state == "FINISHED"
    with pytest.raises(ProcessorReuseError):
        proc.process_logits(ids, logits.clone())
