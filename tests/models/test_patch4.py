"""W7 patch site 4 (bench/vllm_grid_patch.py::patch_scheduler_defer):
dry-run against a VENDORED verbatim copy of the vLLM 0.24.0 scheduler.py
RUNNING-loop region (vllm is not installed locally — the patch function
takes a path, so it is fully testable without it).

Asserted: the guard lands exactly once at the verified-unique anchor (after
the next_decode_eligible_step defer, before the defer_prefills defer, inside
the while loop, at loop-body indentation); the second run is a no-op
(idempotent by marker); a missing anchor is a SystemExit (fail-loud, the
established sites-1-3 pattern); the guard text keeps the default-True
getattr shape (non-grid backends and GRID_DEFER=0 make it a no-op). Plus
static assertions on the upstream PR draft (bench/vllm_upstream_is_ready
.patch): is_ready() default True, the RUNNING-loop guard, a starvation cap.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent

# Verbatim from the vLLM 0.24.0 sdist, vllm/v1/core/sched/scheduler.py lines
# 429-460 — the RUNNING loop head with the anchor block (unique at line 451)
# and its two neighbouring defer blocks.
_VENDORED_REGION = '''\
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue
'''


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


def test_guard_lands_at_anchor_once(patcher, sched_file):
    assert patcher.patch_scheduler_defer(sched_file) is True
    out = sched_file.read_text()
    assert out.count(patcher._SITE4_MARKER) == 1
    assert out.count(patcher._SITE4_GUARD) == 1
    # placement: after the next_decode_eligible_step block, before the
    # defer_prefills block — i.e. INSIDE the while loop
    assert out.index("next_decode_eligible_step") \
        < out.index(patcher._SITE4_MARKER) \
        < out.index("if defer_prefills and request.is_prefill_chunk:")
    assert out.index("while req_index < len(self.running)") \
        < out.index(patcher._SITE4_MARKER)
    # nothing but the guard was inserted; removal restores the original
    assert out.replace(patcher._SITE4_GUARD, "", 1) == _VENDORED_REGION


def test_guard_indentation_is_loop_body_level(patcher, sched_file):
    patcher.patch_scheduler_defer(sched_file)
    guard_lines = [ln for ln in patcher._SITE4_GUARD.splitlines() if ln]
    for ln in guard_lines:
        assert ln.startswith(" " * 12) and not ln.startswith(" " * 24), ln
    # statements sit at 12, the skip body at 16/20 — same shape as the
    # neighbouring defers
    assert "            _grid_so_req = request.structured_output_request" \
        in patcher._SITE4_GUARD
    assert "                    req_index += 1\n" \
        "                    continue\n" in patcher._SITE4_GUARD


def test_guard_is_default_true_getattr_shape(patcher):
    """Non-grid backends (no is_ready attr) and GRID_DEFER=0 (is_ready
    returns True) must make the guard a no-op: skip ONLY on a callable
    is_ready returning False."""
    g = patcher._SITE4_GUARD
    assert 'getattr(_grid_so_req.grammar, "is_ready", None)' in g
    assert "if callable(_grid_is_ready) and not _grid_is_ready():" in g
    assert "_grid_so_req is not None and _grid_so_req.grammar is not None" in g


def test_second_run_is_idempotent(patcher, sched_file):
    assert patcher.patch_scheduler_defer(sched_file) is True
    once = sched_file.read_text()
    assert patcher.patch_scheduler_defer(sched_file) is False
    assert sched_file.read_text() == once, "twice must equal once"


def test_missing_anchor_exits_nonzero(patcher, tmp_path):
    p = tmp_path / "scheduler.py"
    p.write_text("def schedule(self):\n    pass  # layout changed\n")
    with pytest.raises(SystemExit) as ei:
        patcher.patch_scheduler_defer(p)
    assert ei.value.code not in (0, None)
    assert p.read_text().count("grid mask-readiness defer") == 0


def test_full_sdist_dry_run_if_available(patcher, tmp_path):
    """Bonus: when a fetched 0.24.0 sdist is around (GRID_VLLM_SDIST or the
    session scratchpad), dry-run the patch against the REAL full
    scheduler.py — anchor unique, idempotent."""
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
    assert patcher.patch_scheduler_defer(p) is True
    out = p.read_text()
    assert out.count(patcher._SITE4_MARKER) == 1
    assert patcher.patch_scheduler_defer(p) is False
    import ast
    ast.parse(out)  # the patched scheduler must stay valid Python


# --------------------------------------------------- upstream PR draft (W7)


def test_upstream_draft_has_hook_guard_and_cap():
    """The non-gating upstream deliverable: StructuredOutputGrammar
    .is_ready() default True + the RUNNING-loop guard + a starvation cap."""
    draft = (ROOT / "bench" / "vllm_upstream_is_ready.patch").read_text()
    assert "def is_ready(self) -> bool:" in draft
    assert "return True" in draft, "default-True hook"
    assert "backend_types.py" in draft and "scheduler.py" in draft
    assert "so_request.grammar.is_ready()" in draft, "RUNNING-loop guard"
    assert "GRAMMAR_DEFER_CAP_S" in draft, "starvation cap"
    guard_hunk = draft.split("core/sched/scheduler.py")[-1]
    assert "+                    if now < deadline:" in guard_hunk
    assert "+                        req_index += 1" in guard_hunk
