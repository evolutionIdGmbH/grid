"""GRID adapter for DSPy typed pipelines (docs/integrations-plan.md item 4).

DSPy signatures declare every output field's type before the request - the
ideal GRID workload: few schemas x many calls. This adapter compiles the
signature's response schema through ``grid.jsonschema`` at the same point
DSPy builds its structured-output model, which buys three things the plain
``JSONAdapter`` cannot express:

- **Build-time enforceability.** ``strict=True`` turns a signature GRID
  cannot fully mask-enforce into an error when the program is compiled (the
  first call for that signature, or eagerly via :func:`assert_enforceable`)
  rather than a runtime surprise.
- **The recorded residue.** Per signature, the names of any constraints that
  are accepted but not mask-enforced (``recorded_for``), so downstream
  validation is scoped to exactly that set - usually empty for
  pydantic-derived schemas.
- **Server-side masking.** ``mode="server"`` attaches the compiled ``.grid``
  grammar to the request (vLLM guided-decoding shape) for GRID-enabled
  servers; ``mode="client"`` (default) changes no request and works against
  any OpenAI-compatible endpoint, relying on the provider's own structured
  outputs while GRID supplies the build-time guarantee and the residue.

The compile core (:func:`compile_schema_for_model`) has no DSPy dependency;
``GridJSONAdapter`` requires ``dspy`` at import time.

Scope caveat (measured against dspy 3.2): DSPy's signature-to-model
derivation DROPS pydantic ``Field`` metadata (``multiple_of``, bounds,
patterns) before any backend sees it - those constraints silently vanish
for every structured-output backend, not just GRID. The strict gate and
the residue therefore cover what survives derivation: ``Literal``/enum
annotations, nested models, and container semantics (e.g. ``set[str]`` ->
``uniqueItems``, which GRID records). Constraints you need enforced must
live in the annotation itself, not in Field kwargs.
"""

from __future__ import annotations

from typing import Any

from grid.jsonschema import Unsupported, compile_schema_with_paths

__all__ = [
    "GridJSONAdapter",
    "SignatureNotEnforceable",
    "assert_enforceable",
    "compile_schema_for_model",
    "compile_schema_paths_for_model",
]


class SignatureNotEnforceable(Unsupported):
    """A DSPy signature whose schema GRID cannot fully mask-enforce, raised
    under strict mode at program build (never mid-request)."""


def compile_schema_for_model(model_or_schema: Any, *, strict: bool = False):
    """(grid_source, recorded) for a pydantic model class or a JSON schema.

    ``recorded`` names every constraint present but not mask-enforced -
    the set downstream validation should be scoped to. ``strict=True``
    re-raises GRID's Unsupported as :class:`SignatureNotEnforceable`,
    with the offending instance path when the compiler located one
    ("strict: uniqueItems at $.tags").
    """
    src, recorded, _paths = compile_schema_paths_for_model(
        model_or_schema, strict=strict)
    return src, recorded


def compile_schema_paths_for_model(model_or_schema: Any, *,
                                   strict: bool = False):
    """(grid_source, recorded, {instance path: names}) — the path-qualified
    twin of :func:`compile_schema_for_model`. Paths are instance-shaped
    ("$.tags"), i.e. where in the OUTPUT downstream validation should look;
    hash-consed shared subschemas report their first-seen path."""
    schema = (
        model_or_schema.model_json_schema()
        if hasattr(model_or_schema, "model_json_schema")
        else model_or_schema
    )
    try:
        return compile_schema_with_paths(schema, strict=strict)
    except Unsupported as e:
        path = getattr(e, "path", None)
        msg = f"{e} at {path}" if path else str(e)
        raise SignatureNotEnforceable(msg) from e


try:  # dspy is an integration dependency, never grid's
    import dspy  # noqa: F401
    from dspy.adapters import JSONAdapter as _JSONAdapter
    from dspy.adapters.json_adapter import _get_structured_outputs_response_format

    _HAVE_DSPY = True
except Exception:  # pragma: no cover - exercised via importorskip in tests
    _HAVE_DSPY = False


if _HAVE_DSPY:

    class GridJSONAdapter(_JSONAdapter):
        """Drop-in ``dspy.adapters.JSONAdapter`` with GRID compilation.

        mode="client" (default): request unchanged; GRID provides the
        strict build gate and the recorded residue. mode="server": also
        attaches the compiled grammar for GRID-enabled servers via
        ``extra_body`` (vLLM guided-decoding shape).
        """

        def __init__(self, *args: Any, strict: bool = False,
                     mode: str = "client", **kwargs: Any) -> None:
            if mode not in ("client", "server"):
                raise ValueError(f"mode must be client|server, got {mode!r}")
            super().__init__(*args, **kwargs)
            self.strict = strict
            self.mode = mode
            self._compiled: dict[Any, tuple[str, set[str]]] = {}

        # -- grid surface ------------------------------------------------
        def compile_signature(self, signature) -> tuple[str, set[str]]:
            """Compile (and cache) the signature's response schema."""
            src, recorded, _paths = self._compile_full(signature)
            return src, recorded

        def _compile_full(self, signature):
            got = self._compiled.get(signature)
            if got is None:
                model = _get_structured_outputs_response_format(
                    signature, self.use_native_function_calling
                )
                got = compile_schema_paths_for_model(model, strict=self.strict)
                self._compiled[signature] = got
            return got

        def recorded_for(self, signature) -> set[str]:
            """Constraint names GRID accepted but does not mask-enforce for
            this signature - scope any extra validation to exactly these."""
            return set(self._compile_full(signature)[1])

        def recorded_paths_for(self, signature) -> dict[str, set[str]]:
            """Path-qualified residue: {"$.tags": {"uniqueItems"}} - which
            output field each unenforced constraint lives on, so validators
            can be generated per field rather than hunted by hand."""
            return {k: set(v) for k, v in self._compile_full(signature)[2].items()}

        # -- dspy hook ---------------------------------------------------
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            src, _recorded = self.compile_signature(signature)
            if self.mode == "server":
                extra = dict(lm_kwargs.get("extra_body") or {})
                extra.setdefault("guided_grammar", src)
                extra.setdefault("guided_decoding_backend", "grid")
                lm_kwargs["extra_body"] = extra
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

        async def acall(self, lm, lm_kwargs, signature, demos, inputs):
            src, _recorded = self.compile_signature(signature)
            if self.mode == "server":
                extra = dict(lm_kwargs.get("extra_body") or {})
                extra.setdefault("guided_grammar", src)
                extra.setdefault("guided_decoding_backend", "grid")
                lm_kwargs["extra_body"] = extra
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)


def assert_enforceable(program: Any, adapter: GridJSONAdapter) -> dict[str, set[str]]:
    """Eagerly compile every signature in a DSPy program under the adapter.

    Returns {predictor_name: recorded_names}. Under a strict adapter this
    raises :class:`SignatureNotEnforceable` at build; under default mode it
    returns the residue map so callers can decide. Program build is the
    right place to learn a signature cannot be enforced - not production.
    """
    if not _HAVE_DSPY:
        raise ImportError("dspy is required for assert_enforceable")
    out: dict[str, set[str]] = {}
    for name, pred in program.named_predictors():
        out[name] = adapter.recorded_for(pred.signature)
    return out
