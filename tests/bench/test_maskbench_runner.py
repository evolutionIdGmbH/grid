"""bench/maskbench_grid.py process_file — the stale-extras clear (E4 step 5).

Verified corpus artifact this pins down: engines set `extra` only on compile
SUCCESS, and process_file's `finally: status.update(engine.extra)` therefore
stamped the PREVIOUS schema's n_terminals/kernel/ignored_features onto every
timeout/compile-error status (tmp/mb-grid-v030rc2's timeout records carry a
neighbor's fields). New runs must emit non-ok statuses with no extras at all.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))

import maskbench_grid  # noqa: E402


class _DecliningEngine:
    """Engine whose PREVIOUS schema left extras behind and whose current
    compile declares an error — the exact stale-write shape."""

    def __init__(self):
        self.extra = {"n_terminals": 145, "kernel": True, "ignored_features": []}

    def compile_grammar(self, schema):
        raise ValueError("declared decline")


def test_non_ok_status_carries_no_stale_extras(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"schema": {"type": "object"}, "tests": []}))
    status = maskbench_grid.process_file(_DecliningEngine(), str(f), time_limit=5)
    assert "compile_error" in status
    for k in ("n_terminals", "kernel", "ignored_features"):
        assert k not in status
