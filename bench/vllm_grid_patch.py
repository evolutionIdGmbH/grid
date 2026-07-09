"""Patch an installed vLLM (0.24.x) to register GRID as a structured-output
backend. Three sites, all idempotent; see grid/models/vllm_structured.py for
the rationale. Run inside the venv that has vllm installed:

    .venv/bin/python bench/vllm_grid_patch.py
"""

from __future__ import annotations

import pathlib
import re
import sys


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


if __name__ == "__main__":
    main()
