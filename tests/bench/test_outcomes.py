"""bench/perfbench/outcomes.py — the outcome classifier's unit gates.

Two layers (E4 plan):
- synthetic fixtures for every marker shape, including the two malformed-record
  shapes that produced real published-number retractions (the F1 unmarked
  partial, the marker-less at-cap status);
- the full real v0.2.5 corpus (tmp/mb-grid-final, 11,306 statuses) with exact
  bucket counts, cross-checked against the perfbench manifest — skipped where
  the corpus dirs aren't present (they are machine-local, gitignored).
"""

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "bench" / "perfbench"))

import outcomes  # noqa: E402

MB_FINAL = ROOT / "tmp" / "mb-grid-final"
MB_RC2 = ROOT / "tmp" / "mb-grid-v030rc2"


# ------------------------------------------------------- phase-record shapes


def _completed(mode=None, **over):
    rec = {
        "file": "x.json",
        "flags": {},
        "phases": {p: 1 for p in outcomes.COMPILE_PHASES},
        "stats": {},
        "running": None,
    }
    if mode == "first_mask":
        rec["mode"] = "first_mask"
        rec["phases"].update({p: 1 for p in outcomes.FIRST_MASK_PHASES})
    rec.update(over)
    return rec


def test_completed_compile_record_is_ok():
    assert outcomes.classify_phase_record(_completed()) == "ok"


def test_completed_first_mask_record_is_ok():
    assert outcomes.classify_phase_record(_completed(mode="first_mask")) == "ok"


def test_unmarked_partial_is_incomplete_never_ok():
    # the F1-retraction shape: child killed mid-scanner by an interrupted
    # parent, no timeout_s/rc marker — read as a ~0.0s "fix" by the ad-hoc
    # scripts this module replaces
    rec = {"phases": {"schema_compile": 5, "spec_load": 2}, "running": "scanner"}
    assert outcomes.classify_phase_record(rec) == "incomplete"


def test_timeout_marker_attributes_in_flight_phase():
    rec = {"phases": {"schema_compile": 5}, "running": "scanner", "timeout_s": 120}
    assert outcomes.classify_phase_record(rec) == "timeout@scanner"


def test_timeout_marker_without_running_phase():
    # parent found no readable child record at kill time
    assert outcomes.classify_phase_record({"phases": {}, "timeout_s": 120}) == "timeout@?"


def test_declared_error_wins_over_expected_nonzero_rc():
    # declared outcomes re-raise after flushing, so rc is also present
    rec = {"phases": {}, "running": "schema_compile", "error": "Unsupported", "rc": 1}
    assert outcomes.classify_phase_record(rec) == "declared:Unsupported"


def test_timeout_beats_error_pessimistically():
    # a record that might not have terminated is never counted as terminated
    rec = {"phases": {}, "running": "lalr", "error": "Unsupported", "timeout_s": 120}
    assert outcomes.classify_phase_record(rec) == "timeout@lalr"


def test_rc_without_error_is_crash():
    rec = {"phases": {"schema_compile": 5}, "running": "spec_load", "rc": -9}
    assert outcomes.classify_phase_record(rec) == "crash"


def test_first_mask_record_missing_first_mask_phase_is_malformed():
    # record-shape gate (E4 gate d): a "completed" --first-mask record without
    # the first_mask phase must never classify ok
    rec = _completed(mode="first_mask")
    del rec["phases"]["first_mask"]
    assert outcomes.classify_phase_record(rec) == "malformed"


def test_compile_record_missing_phase_is_malformed():
    rec = _completed()
    del rec["phases"]["scanner"]
    assert outcomes.classify_phase_record(rec) == "malformed"


def test_unreadable_record_is_malformed():
    assert outcomes.classify_phase_record({}) == "malformed"


# ------------------------------------------------------------ status shapes

CAP_US = 120 * 1_000_000


def test_status_ok():
    assert outcomes.classify_status({"ttfm_us": 7300}, CAP_US) == "ok"


def test_status_compile_error_class():
    s = {"ttfm_us": 0, "compile_error": "Unsupported: patternProperties on x"}
    assert outcomes.classify_status(s, CAP_US) == "compile_error:Unsupported"


def test_status_explicit_timeout_marker():
    s = {"ttfm_us": CAP_US, "timeout": "compile"}
    assert outcomes.classify_status(s, CAP_US) == "timeout"


def test_status_marker_less_at_cap_is_timeout():
    # belt-and-braces: ranked-at-cap without a marker (older status dirs)
    assert outcomes.classify_status({"ttfm_us": CAP_US}, CAP_US) == "timeout"
    assert outcomes.classify_status({"ttfm_us": CAP_US - 1}, CAP_US) == "ok"


def test_extras_never_read_off_non_ok_status():
    # the stale-write artifact: maskbench's `finally: status.update(extra)`
    # gave timeout records the PREVIOUS schema's extras (rc2's timeout records
    # carry a neighbor's n_terminals/kernel)
    stale = {"ttfm_us": CAP_US, "timeout": "compile",
             "n_terminals": 145, "kernel": True, "ignored_features": []}
    assert outcomes.extras(stale, CAP_US) == {}
    ok = {"ttfm_us": 7300, "n_terminals": 12, "kernel": True}
    assert outcomes.extras(ok, CAP_US) == {"n_terminals": 12, "kernel": True}


# ------------------------------------------------------------------ compare


@pytest.mark.parametrize("base,new,verdict", [
    ("ok", "ok", "unchanged"),
    ("timeout", "timeout@scanner", "unchanged"),          # families match
    ("timeout", "timeout@first_mask", "unchanged"),       # first-mask-era timeout
    ("timeout", "ok", "improved"),                        # sanctioned: no oracle
    ("timeout", "declared:Unsupported", "improved"),      # sanctioned direction
    ("compile_error:Unsupported", "declared:Unsupported", "unchanged"),
    ("ok", "timeout@scanner", "GATE"),                    # regression
    ("ok", "declared:Unsupported", "GATE"),               # ok -> anything
    ("compile_error:Unsupported", "declared:LALRConflictError", "GATE"),
    ("compile_error:Unsupported", "ok", "GATE"),          # unsanctioned coverage flip
    ("timeout", "incomplete", "GATE"),                    # never passes
    ("ok", "malformed", "GATE"),
    ("timeout", "crash", "GATE"),
])
def test_compare_oracle_rule(base, new, verdict):
    assert outcomes.compare_outcome(base, new) == verdict


def test_strict_identity_gate_fails_improvements():
    assert outcomes.compare_outcome("timeout", "ok", strict=True) == "GATE"
    assert outcomes.compare_outcome("timeout", "timeout@lalr", strict=True) == "unchanged"


# ------------------------------------------------- real-corpus bucket gates


@pytest.mark.skipif(not MB_FINAL.is_dir(), reason="v0.2.5 corpus dir not on this machine")
def test_mb_grid_final_exact_buckets():
    out = outcomes.classify_dir(str(MB_FINAL))
    assert len(out) == 11306
    fams = [outcomes.outcome_family(o) for o in out.values()]
    n_ok = fams.count("ok")
    n_to = fams.count("timeout")
    n_declared = sum(1 for f in fams if f.startswith("declared:"))
    assert (n_declared, n_to, n_ok) == (668, 16, 10622)
    # ranked = ok + timeouts-at-cap: the manifest's population cross-check
    manifest = json.load(open(ROOT / "bench" / "perfbench" / "manifest.json"))
    assert n_ok + n_to == manifest["n_schemas_ranked"] == 10638
    # nothing silently non-terminating in a frozen status dir
    assert not [f for f in fams if f in ("crash", "incomplete", "malformed")]


@pytest.mark.skipif(not MB_FINAL.is_dir(), reason="v0.2.5 corpus dir not on this machine")
def test_mb_grid_final_timeouts_all_carry_markers():
    # documents that the at-cap fallback is belt-and-braces there, not load-bearing
    statuses, cap_us = outcomes.load_status_dir(str(MB_FINAL))
    assert cap_us == 120 * 1_000_000
    at_cap = [s for s in statuses.values()
              if outcomes.classify_status(s, cap_us) == "timeout"]
    assert len(at_cap) == 16 and all("timeout" in s for s in at_cap)


@pytest.mark.skipif(not MB_RC2.is_dir(), reason="v0.3.0rc2 corpus dir not on this machine")
def test_mb_grid_v030rc2_exact_buckets():
    out = outcomes.classify_dir(str(MB_RC2))
    assert len(out) == 11306
    fams = [outcomes.outcome_family(o) for o in out.values()]
    assert (sum(1 for f in fams if f.startswith("declared:")),
            fams.count("timeout"), fams.count("ok")) == (635, 7, 10664)
