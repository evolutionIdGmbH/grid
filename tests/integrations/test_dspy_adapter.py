"""GridJSONAdapter: build-time enforceability + recorded residue.

The compile core runs dependency-free; adapter-shape tests skip unless dspy
is installed (it is an integration dependency, never grid's).
"""

import pydantic
import pytest

from grid.integrations.dspy_adapter import (
    SignatureNotEnforceable,
    compile_schema_for_model,
)


class Judgment(pydantic.BaseModel):
    verdict: str  # free string: enforceable shape
    confidence: float
    tags: list[str]


class Constrained(pydantic.BaseModel):
    # multipleOf is a recorded (not mask-enforced) numeric constraint
    score: float = pydantic.Field(multiple_of=0.25)


def test_pydantic_model_compiles_with_empty_residue():
    src, recorded = compile_schema_for_model(Judgment)
    assert "%start" in src
    assert recorded == set()


def test_recorded_residue_is_named():
    src, recorded = compile_schema_for_model(Constrained)
    assert "%start" in src
    assert recorded, "multipleOf must surface in the recorded set"
    assert any("multipleOf" in r for r in recorded)


def test_strict_mode_is_a_build_error():
    with pytest.raises(SignatureNotEnforceable):
        compile_schema_for_model(Constrained, strict=True)


def test_plain_schema_accepted_too():
    src, recorded = compile_schema_for_model(
        {"type": "object", "properties": {"a": {"type": "integer"}},
         "required": ["a"], "additionalProperties": False}
    )
    assert "%start" in src and recorded == set()


# ---------------------------------------------------------------- dspy shape


def test_adapter_compiles_signature_and_scopes_residue():
    dspy = pytest.importorskip("dspy")
    from grid.integrations.dspy_adapter import GridJSONAdapter

    class Extract(dspy.Signature):
        """Extract a judgment."""

        text: str = dspy.InputField()
        verdict: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    ad = GridJSONAdapter()
    src, recorded = ad.compile_signature(Extract)
    assert "%start" in src
    assert ad.recorded_for(Extract) == recorded
    # cache: same object back
    assert ad.compile_signature(Extract)[0] is src


def test_strict_gate_fires_on_surviving_constraints():
    dspy = pytest.importorskip("dspy")
    from grid.integrations.dspy_adapter import (
        GridJSONAdapter,
        SignatureNotEnforceable,
    )

    class Tagged(dspy.Signature):
        text: str = dspy.InputField()
        tags: set[str] = dspy.OutputField()  # -> uniqueItems (recorded)

    assert GridJSONAdapter().recorded_for(Tagged) == {"uniqueItems"}
    with pytest.raises(SignatureNotEnforceable):
        GridJSONAdapter(strict=True).compile_signature(Tagged)


def test_server_mode_injects_grammar(monkeypatch):
    dspy = pytest.importorskip("dspy")
    from grid.integrations.dspy_adapter import GridJSONAdapter

    class Extract(dspy.Signature):
        text: str = dspy.InputField()
        verdict: str = dspy.OutputField()

    ad = GridJSONAdapter(mode="server")
    seen = {}

    def fake_super_call(lm, lm_kwargs, signature, demos, inputs):
        seen.update(lm_kwargs)
        return [{"verdict": "ok"}]

    monkeypatch.setattr(
        "dspy.adapters.json_adapter.JSONAdapter.__call__",
        lambda self, lm, lm_kwargs, signature, demos, inputs:
            fake_super_call(lm, lm_kwargs, signature, demos, inputs),
    )
    ad(None, {}, Extract, [], {"text": "x"})
    assert "guided_grammar" in seen.get("extra_body", {})
    assert seen["extra_body"]["guided_decoding_backend"] == "grid"
