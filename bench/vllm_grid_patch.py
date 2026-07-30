"""Patch an installed vLLM (0.24.x) to register GRID as a structured-output
backend. Five sites, all idempotent; see grid/models/vllm_structured.py for
the rationale. Run inside the venv that has vllm installed:

    .venv/bin/python bench/vllm_grid_patch.py

Site 4 (W6/W7 defer chassis) adds the RUNNING-loop mask-readiness guard to
the V1 scheduler: a request whose grammar reports `is_ready()` False is
skipped for the round exactly like the `next_decode_eligible_step` defer at
the same position — absent from num_scheduled_tokens => absent from
structured_output_request_ids => no bitmask row, no sampled token, KV blocks
intact. Default-True getattr shape: non-grid backends (no is_ready attr) and
GRID_DEFER=0 (is_ready always True) make the guard a no-op. The long-term
upstream path is bench/vllm_upstream_is_ready.patch (StructuredOutputGrammar
.is_ready() default True + this guard + a scheduler-side starvation cap),
which retires this site.

Site 5 (S1 jump-forward) injects the grammar's forced continuation
(singleton-mask chain, GridGrammarSession.jump_tokens) as the request's
draft tokens right after the scheduler's update_from_output accept_tokens
call: the next step verifies the whole run under per-position bitmasks in
ONE forward pass — the spec-decode compute shape, with certain acceptance
because a forced position's bitmask admits exactly one token. Default-None
getattr shape: non-grid backends (no jump_tokens attr) skip; GRID_JUMP=0
sessions return [] (no-op). ANCHOR CAVEAT — unlike sites 1-4 (verified
against the 0.24.0 sdist), the site-5 anchor is written against vLLM's
long-stable upstream update_from_output shape and has NOT been dry-run
against the pinned 0.24.0 tree from this workstation; the first box session
must run tests/models/test_patch5.py::test_full_sdist_dry_run_if_available
(or just this script — a mismatched anchor is a loud SystemExit, never a
silent mispatch), and the S1 step-3 probe may still relocate the injection
if 0.24 hard-couples spec-token handling to a configured drafter. Upstream
path: bench/vllm_upstream_jump_tokens.patch (jump_tokens() default [] +
this injection), riding the same PR vehicle as is_ready.
"""

from __future__ import annotations

import pathlib
import re
import sys

_SITE4_MARKER = "grid mask-readiness defer"

# the current_step/next_decode_eligible_step defer block — unique at line 451
# of the vLLM 0.24.0 sdist scheduler.py (verified)
_SITE4_ANCHOR = (
    "            if self.current_step < request.next_decode_eligible_step:\n"
    "                # V2+PP+async: enforce `pp_size` steps between same-req decodes\n"
    "                # to match worker-side sampled-tokens broadcast slot ring cadence.\n"
    "                req_index += 1\n"
    "                continue\n"
)

_SITE4_GUARD = (
    "\n"
    "            # grid mask-readiness defer: a RUNNING structured request whose\n"
    "            # grammar reports its next mask is not ready (cold build in\n"
    "            # flight) is skipped for this round, same shape as the defer\n"
    "            # above — absent from num_scheduled_tokens => no bitmask row,\n"
    "            # no sampled token, KV blocks intact. Starvation is bounded by\n"
    "            # the grammar's own time cap (GRID_DEFER_MS, default 100 ms):\n"
    "            # on expiry is_ready() returns True and the next fill BLOCKS\n"
    "            # on the exact mask (never approximated). Backends without an\n"
    "            # is_ready attr (and GRID_DEFER=0) default to ready: no-op.\n"
    "            _grid_so_req = request.structured_output_request\n"
    "            if _grid_so_req is not None and _grid_so_req.grammar is not None:\n"
    '                _grid_is_ready = getattr(_grid_so_req.grammar, "is_ready", None)\n'
    "                if callable(_grid_is_ready) and not _grid_is_ready():\n"
    "                    req_index += 1\n"
    "                    continue\n"
)


def patch_scheduler_defer(sched: pathlib.Path) -> bool:
    """Site 4, callable in isolation (tests dry-run it against a vendored
    copy of the 0.24.0 scheduler region — vllm itself need not be
    installed). Returns True when the file was modified, False when the
    marker shows it is already patched; SystemExit when the anchor is
    missing or ambiguous (vllm layout changed)."""
    src = sched.read_text()
    if _SITE4_MARKER in src:
        return False
    if src.count(_SITE4_ANCHOR) != 1:
        sys.exit(f"anchor not found in {sched}; vllm layout changed")
    sched.write_text(src.replace(_SITE4_ANCHOR, _SITE4_ANCHOR + _SITE4_GUARD, 1))
    return True


_SITE5_MARKER = "grid jump-forward"

# update_from_output's structured-output accept call — the two-line call
# statement, not the surrounding if/comment (comments churn across vLLM
# versions; the call shape has been stable since V1 structured outputs
# landed). Re-verify uniqueness on the box tree before trusting a run.
_SITE5_ANCHOR = (
    "                request.structured_output_request.grammar.accept_tokens("
    "  # type: ignore[union-attr]\n"
    "                    req_id, new_token_ids)\n"
)

_SITE5_GUARD = (
    "\n"
    "                # grid jump-forward (site 5): install the grammar's forced\n"
    "                # continuation (singleton-mask chain) as this request's\n"
    "                # draft tokens — the next step verifies the whole run under\n"
    "                # per-position bitmasks in one forward pass. At a forced\n"
    "                # position the bitmask admits exactly one token, so every\n"
    "                # span token is accepted with certainty under any sampler\n"
    "                # (parity-exact, never approximate). jump_tokens() is\n"
    "                # state-neutral, so the drafter's validate_tokens flow\n"
    "                # below composes; when a proposer ran, its assignment\n"
    "                # overwrites this one (drafter wins). Backends without a\n"
    "                # jump_tokens attr skip (default-None getattr shape);\n"
    "                # GRID_JUMP=0 sessions return [] — assigning that to the\n"
    "                # already-consumed spec_token_ids also clears stale spans\n"
    "                # between steps when no drafter is configured.\n"
    '                _grid_jump = getattr(\n'
    '                    request.structured_output_request.grammar,\n'
    '                    "jump_tokens", None)\n'
    "                if callable(_grid_jump):\n"
    "                    request.spec_token_ids = _grid_jump()\n"
)


def patch_scheduler_jump(sched: pathlib.Path) -> bool:
    """Site 5, callable in isolation (tests dry-run it against a vendored
    region). Same discipline as site 4: idempotent by marker, SystemExit on
    a missing/ambiguous anchor. The guard lands INSIDE the should_advance
    block, directly after accept_tokens and BEFORE the drafter's
    spec_token_ids assignment (drafter proposals must keep winning)."""
    src = sched.read_text()
    if _SITE5_MARKER in src:
        return False
    if src.count(_SITE5_ANCHOR) != 1:
        sys.exit(f"site-5 anchor not found in {sched}; vllm layout changed")
    sched.write_text(src.replace(_SITE5_ANCHOR, _SITE5_ANCHOR + _SITE5_GUARD, 1))
    return True


def main() -> None:
    import vllm

    base = pathlib.Path(vllm.__file__).parent

    # 1. backend dispatch chain
    so_init = base / "v1" / "structured_output" / "__init__.py"
    src = so_init.read_text()
    if "GridStructuredBackend" not in src:
        patch = (
            '            elif backend == "grid":\n'
            "                from grid.models.vllm_structured import GridStructuredBackend\n"
            "\n"
            "                self.backend = GridStructuredBackend(\n"
            "                    self.vllm_config,\n"
            "                    tokenizer=self.tokenizer,\n"
            "                    vocab_size=vocab_size,\n"
            "                )\n"
        )
        anchor = '            elif backend == "guidance":'
        if anchor not in src:
            sys.exit(f"anchor not found in {so_init}; vllm layout changed")
        so_init.write_text(src.replace(anchor, patch + anchor, 1))
        print(f"patched {so_init}")
    else:
        print("dispatch chain: already patched")

    # 2. backend choices literal
    cfg = base / "config" / "structured_outputs.py"
    if cfg.exists():
        s = cfg.read_text()
        if '"grid"' not in s:
            s2, n = re.subn(r'("xgrammar")(\s*,)', r'\1, "grid"\2', s, count=1)
            if n:
                cfg.write_text(s2)
                print(f"patched {cfg}")
        else:
            print("backend choices: already patched")

    # 3. frontend validation dispatch (otherwise grammar specs are sniffed as
    #    Lark/GBNF and rejected before the backend is consulted)
    sp = base / "sampling_params.py"
    s = sp.read_text()
    if 'backend == "grid"' not in s:
        anchor = '        elif backend == "outlines":'
        patch = (
            '        elif backend == "grid":\n'
            "            pass  # grid validates at compile time"
            " (GrammarInvalid/LALRConflictError)\n"
        )
        if anchor not in s:
            sys.exit(f"anchor not found in {sp}; vllm layout changed")
        sp.write_text(s.replace(anchor, patch + anchor, 1))
        print(f"patched {sp}")
    else:
        print("validation dispatch: already patched")

    # 4. RUNNING-loop mask-readiness defer (W6/W7 skip-a-round chassis)
    sched = base / "v1" / "core" / "sched" / "scheduler.py"
    if patch_scheduler_defer(sched):
        print(f"patched {sched}")
    else:
        print("mask-readiness defer: already patched")

    # 5. jump-forward draft injection (S1; no-op unless GRID_JUMP=1)
    if patch_scheduler_jump(sched):
        print(f"patched {sched} (site 5)")
    else:
        print("jump-forward injection: already patched")


if __name__ == "__main__":
    main()
