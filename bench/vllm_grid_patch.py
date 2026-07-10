"""Patch an installed vLLM (0.24.x) to register GRID as a structured-output
backend. Four sites, all idempotent; see grid/models/vllm_structured.py for
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


if __name__ == "__main__":
    main()
