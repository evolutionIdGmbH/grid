"""S1 patch site 5 (bench/vllm_grid_patch.py::patch_scheduler_jump):
dry-run against a vendored copy of the vLLM update_from_output
structured-output accept region.

HONESTY NOTE (differs from test_patch4): the site-4 vendored region is
verbatim from the 0.24.0 sdist; this one is a RECONSTRUCTION of the
long-stable upstream update_from_output shape (accept_tokens call + the
drafter's validate_tokens spec block) — the 0.24.0 sdist is not on this
workstation. The patch function fail-louds (SystemExit) on a mismatched
anchor, and test_full_sdist_dry_run_if_available re-runs against the real
tree whenever one is present (mandatory on the first box session; the S1
step-3 probe may still relocate the injection point entirely).

Asserted here: the guard lands exactly once directly after accept_tokens,
INSIDE the should_advance block (16-space statements) and BEFORE the
drafter's spec_token_ids assignment (drafter proposals must keep winning);
idempotent by marker; SystemExit on a missing anchor; default-None getattr
shape; the patched region stays valid Python. Plus static assertions on
the upstream PR draft (bench/vllm_upstream_jump_tokens.patch).
"""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent

# Reconstructed update_from_output region (see module docstring): the
# structured-output accept, the nans bookkeeping, and the drafter spec
# block that must stay downstream of the injected guard.
_VENDORED_REGION = '''\
            if new_token_ids and self.structured_output_manager.should_advance(request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # spec_token_ids comes from the model runner output
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    assert metadata is not None and metadata.grammar is not None
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]
'''

_SCAFFOLD = (
    "class _S:\n"
    "    def update_from_output(self):\n"
    "        for req_id in ids:\n"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location(
        "vllm_grid_patch", ROOT / "bench" / "vllm_grid_patch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no top-level vllm import: safe anywhere
    return mod


@pytest.fixture(scope="module")
def patcher():
    return _load_patcher()


@pytest.fixture()
def sched_file(tmp_path):
    p = tmp_path / "scheduler.py"
    p.write_text(_VENDORED_REGION)
    return p


def test_guard_lands_after_accept_before_spec_block(patcher, sched_file):
    assert patcher.patch_scheduler_jump(sched_file) is True
    out = sched_file.read_text()
    assert out.count(patcher._SITE5_MARKER) == 1
    assert out.count(patcher._SITE5_GUARD) == 1
    # placement: after the accept_tokens call, before the drafter's
    # validate_tokens spec block (drafter assignment must stay downstream
    # so proposer behavior is unchanged)
    assert out.index("grammar.accept_tokens(") \
        < out.index(patcher._SITE5_MARKER) \
        < out.index("# spec_token_ids comes from the model runner output")
    # nothing but the guard was inserted; removal restores the original
    assert out.replace(patcher._SITE5_GUARD, "", 1) == _VENDORED_REGION


def test_guard_indentation_is_should_advance_block_level(patcher, sched_file):
    """Statements at 16 spaces (inside `if ... should_advance(request):`),
    the conditional assignment at 20 — never at the loop level, where
    structured_output_request could be None for non-structured requests."""
    patcher.patch_scheduler_jump(sched_file)
    for ln in (ln for ln in patcher._SITE5_GUARD.splitlines() if ln):
        assert ln.startswith(" " * 16), ln
    assert "                _grid_jump = getattr(" in patcher._SITE5_GUARD
    assert "                    request.spec_token_ids = _grid_jump()\n" \
        in patcher._SITE5_GUARD


def test_guard_is_default_none_getattr_shape(patcher):
    """Backends without jump_tokens must be skipped entirely: getattr
    default None + callable check, mirroring the site-4 is_ready shape."""
    g = patcher._SITE5_GUARD
    assert '"jump_tokens", None)' in g
    assert "if callable(_grid_jump):" in g


def test_patched_region_is_valid_python(patcher, sched_file):
    patcher.patch_scheduler_jump(sched_file)
    body = "".join("        " + ln + "\n" if ln.strip() else "\n"
                   for ln in sched_file.read_text().splitlines())
    ast.parse(_SCAFFOLD + body)


def test_second_run_is_idempotent(patcher, sched_file):
    assert patcher.patch_scheduler_jump(sched_file) is True
    once = sched_file.read_text()
    assert patcher.patch_scheduler_jump(sched_file) is False
    assert sched_file.read_text() == once, "twice must equal once"


def test_missing_anchor_exits_nonzero(patcher, tmp_path):
    p = tmp_path / "scheduler.py"
    p.write_text("def update_from_output(self):\n    pass  # layout changed\n")
    with pytest.raises(SystemExit) as ei:
        patcher.patch_scheduler_jump(p)
    assert ei.value.code not in (0, None)
    assert p.read_text().count(patcher._SITE5_MARKER) == 0


def test_sites_4_and_5_compose(patcher, tmp_path):
    """Both scheduler patches target the same file; applying site 5 to a
    site-4-patched file (and vice versa) must not disturb the other."""
    import test_patch4

    p = tmp_path / "scheduler.py"
    p.write_text(test_patch4._VENDORED_REGION + _VENDORED_REGION)
    assert patcher.patch_scheduler_defer(p) is True
    assert patcher.patch_scheduler_jump(p) is True
    out = p.read_text()
    assert out.count(patcher._SITE4_MARKER) == 1
    assert out.count(patcher._SITE5_MARKER) == 1
    assert patcher.patch_scheduler_defer(p) is False
    assert patcher.patch_scheduler_jump(p) is False


def test_full_sdist_dry_run_if_available(patcher, tmp_path):
    """MANDATORY on the first box session: when a fetched 0.24.0 sdist is
    around (GRID_VLLM_SDIST or /tmp), dry-run site 5 against the REAL full
    scheduler.py — anchor unique, idempotent, still-valid Python. Locally
    this skips (the anchor is reconstructed, see module docstring)."""
    import os
    rel = "vllm-0.24.0/vllm/v1/core/sched/scheduler.py"
    candidates = [os.environ.get("GRID_VLLM_SDIST", "")]
    for depth in ("", "*/", "*/*/", "*/*/*/", "*/*/*/*/"):
        candidates += [str(p) for p in pathlib.Path("/tmp").glob(depth + rel)]
    real = next((c for c in candidates if c and pathlib.Path(c).is_file()), None)
    if real is None:
        pytest.skip("no vLLM 0.24.0 sdist scheduler.py available")
    p = tmp_path / "scheduler.py"
    p.write_text(pathlib.Path(real).read_text())
    assert patcher.patch_scheduler_jump(p) is True
    out = p.read_text()
    assert out.count(patcher._SITE5_MARKER) == 1
    assert patcher.patch_scheduler_jump(p) is False
    ast.parse(out)  # the patched scheduler must stay valid Python


# --------------------------------------------------- upstream PR draft (S1)


def test_upstream_draft_has_hook_and_injection():
    """The non-gating upstream deliverable: StructuredOutputGrammar
    .jump_tokens() default [] + the update_from_output injection, with the
    drafter-wins ordering documented."""
    draft = (ROOT / "bench" / "vllm_upstream_jump_tokens.patch").read_text()
    assert "def jump_tokens(self) -> list[int]:" in draft
    assert "return []" in draft, "default-[] hook"
    assert "backend_types.py" in draft and "scheduler.py" in draft
    assert "grammar.jump_tokens()" in draft, "scheduler injection"
    assert "state-neutral" in draft
    hunk = draft.split("core/sched/scheduler.py")[-1]
    assert "+                    request.spec_token_ids = jump" in hunk
    assert "validate_tokens" in draft, "drafter-wins ordering documented"
