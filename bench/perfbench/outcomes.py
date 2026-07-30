"""Outcome-aware classification for perfbench phase records and maskbench statuses.

The reusable replacement for the ad-hoc per-investigation scripts that produced
the F1 retraction (BAKEOFF.md): unmarked partial records from killed children
read as ~0.0s "fixes". Two record dialects, one shared outcome vocabulary:

- phase records (bench/perfbench/profile_phases.py out dirs, ``*.phases.json``):
  ``classify_phase_record(rec)`` ->
  ``ok | declared:<ErrorName> | timeout@<phase> | crash | incomplete | malformed``
- maskbench statuses (bench/maskbench_grid.py out dirs, ``<id>.json`` +
  ``_meta.json``): ``classify_status(status, cap_us)`` ->
  ``ok | compile_error:<class> | timeout``

Classification rules (each defends against a verified corpus artifact):

- ``incomplete`` (running != null, no timeout_s/rc marker) is NEVER ok — the
  exact F1 failure mode (killed children fossilized as sub-second fixes).
- Marker precedence is pessimistic: ``timeout_s`` beats ``error`` beats ``rc``.
  A record that might not have terminated is never counted as terminated.
- ``timeout`` for statuses is belt-and-braces: the explicit marker OR
  ``ttfm_us >= cap_us`` (cap from the dir's ``_meta.json`` time_limit). All 16
  tmp/mb-grid-final cap records carry ``timeout: "compile"``, but older status
  dirs may not.
- A completed first-mask phase record (``mode: "first_mask"``) missing any of
  the first-mask phases is ``malformed``, never ok (record-shape gate).
- Informational extras (n_terminals/kernel/ignored_features) are NEVER read
  off a non-ok status: maskbench's ``finally: status.update(engine.extra)``
  wrote the PREVIOUS schema's extras onto timeout/compile-error records in
  every dir produced before the extras-clear fix, and the frozen v0.2.5 dirs
  keep the stale fields forever (tmp/mb-grid-v030rc2's DataConnector-era
  timeout records carry a neighbor's n_terminals/kernel). Use ``extras()``.

Compare mode joins a regenerated leg dir against a baseline status dir under
the diff_hashcons oracle rule: baseline timeouts have no oracle, so
timeout -> ok/declared is the sanctioned direction ("improved"); ok -> anything,
any declared-class change (including declared -> ok), and crash/incomplete/
malformed anywhere are gate failures. ``--strict`` turns sanctioned
improvements into failures too (the flag-OFF identity gate).

Usage:
    python bench/perfbench/outcomes.py classify <dir>
    python bench/perfbench/outcomes.py compare <leg_dir> <baseline_dir> \
        [--cap-s N] [--strict]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

COMPILE_PHASES = ("schema_compile", "spec_load", "projection", "lalr", "scanner")
FIRST_MASK_PHASES = ("trie", "guide", "first_mask", "prefix_masks")

# status keys written by maskbench engines' `extra` dict — informational only,
# and stale on non-ok records produced before the extras-clear fix
EXTRA_KEYS = ("n_terminals", "kernel", "ignored_features")


# ------------------------------------------------------------------ classify


def classify_phase_record(rec: dict) -> str:
    """Outcome class of one profile_phases record (see module docstring)."""
    if not isinstance(rec, dict) or not isinstance(rec.get("phases"), dict):
        return "malformed"
    if "timeout_s" in rec:
        # parent killed the child past its deadline; the phase in flight at
        # kill time is `running` (the child flushes before starting each phase)
        return f"timeout@{rec.get('running') or '?'}"
    if rec.get("error"):
        # declared outcome (Unsupported, LALRConflictError, ...): the child
        # flushed the error then re-raised, so `rc` is also present — the
        # error marker wins over the expected nonzero exit
        return f"declared:{rec['error']}"
    if rec.get("rc"):
        return "crash"
    if rec.get("running") is not None:
        return "incomplete"
    expected = COMPILE_PHASES + (
        FIRST_MASK_PHASES if rec.get("mode") == "first_mask" else ()
    )
    if any(p not in rec["phases"] for p in expected):
        return "malformed"
    return "ok"


def classify_status(status: dict, cap_us: int | None) -> str:
    """Outcome class of one maskbench status (see module docstring)."""
    if "compile_error" in status:
        # "<ExcName>: <msg>" -> class <ExcName> (maskbench writes
        # f"{type(e).__name__}: {e}")
        return "compile_error:" + status["compile_error"].split(":", 1)[0].strip()
    if "timeout" in status:
        return "timeout"
    if cap_us is not None and status.get("ttfm_us", 0) >= cap_us:
        return "timeout"  # marker-less record ranked at cap
    return "ok"


def extras(status: dict, cap_us: int | None = None) -> dict:
    """The engine-informational fields of a status — {} unless the status is
    ok. On non-ok records these fields are the PREVIOUS schema's (stale-write
    artifact, see module docstring) and must never be attributed."""
    if classify_status(status, cap_us) != "ok":
        return {}
    return {k: status[k] for k in EXTRA_KEYS if k in status}


# ------------------------------------------------------------------- compare


def outcome_family(outcome: str) -> str:
    """Joint vocabulary for cross-dialect compare: phase-record and status
    outcomes reduce to ok | declared:<class> | timeout | crash | incomplete |
    malformed (timeout keeps no phase detail — v0.2.5 compile timeouts and
    first-mask-era timeouts are the same "did not terminate inside the cap"
    family for gating; the matrix still shows the phase)."""
    if outcome.startswith("compile_error:"):
        return "declared:" + outcome[len("compile_error:"):]
    if outcome.startswith("timeout"):
        return "timeout"
    return outcome


def compare_outcome(base: str, new: str, strict: bool = False) -> str:
    """Verdict for one schema: unchanged | improved | GATE.

    Oracle rule (diff_hashcons.py): a baseline timeout has no oracle, so any
    terminating outcome (ok or declared) is sanctioned -> "improved". Baseline
    ok must stay ok; a declared class must stay that class (declared -> ok is
    an unsanctioned coverage change, surfaced for explicit adjudication, not
    silently passed). crash/incomplete/malformed never pass. strict=True
    (flag-OFF identity gate) fails sanctioned improvements too."""
    b, n = outcome_family(base), outcome_family(new)
    if n in ("crash", "incomplete", "malformed"):
        return "GATE"
    if b == n:
        return "unchanged"
    if b == "timeout" and (n == "ok" or n.startswith("declared:")):
        return "GATE" if strict else "improved"
    return "GATE"


# ------------------------------------------------------------------- loading


def load_phase_dir(d: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.phases.json"))):
        sid = os.path.basename(f)[: -len(".phases.json")]
        try:
            with open(f) as fh:
                out[sid] = json.load(fh)
        except (OSError, ValueError):
            out[sid] = {}  # unreadable record -> malformed
    return out


def load_status_dir(d: str) -> tuple[dict[str, dict], int | None]:
    out: dict[str, dict] = {}
    cap_us = None
    mf = os.path.join(d, "_meta.json")
    if os.path.exists(mf):
        with open(mf) as fh:
            meta = json.load(fh)
        if "time_limit" in meta:
            cap_us = int(meta["time_limit"]) * 1_000_000
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(f) == "_meta.json":
            continue
        sid = os.path.basename(f)[: -len(".json")]
        with open(f) as fh:
            out[sid] = json.load(fh)
    return out, cap_us


def is_phase_dir(d: str) -> bool:
    return bool(glob.glob(os.path.join(d, "*.phases.json")))


def classify_dir(d: str, cap_us: int | None = None) -> dict[str, str]:
    """sid -> outcome for either dialect (auto-detected)."""
    if is_phase_dir(d):
        return {sid: classify_phase_record(r) for sid, r in load_phase_dir(d).items()}
    statuses, meta_cap = load_status_dir(d)
    cap = cap_us if cap_us is not None else meta_cap
    return {sid: classify_status(s, cap) for sid, s in statuses.items()}


# ---------------------------------------------------------------------- CLI


def _cmd_classify(d: str, cap_us: int | None) -> int:
    outcomes = classify_dir(d, cap_us)
    counts = Counter(outcome_family(o) for o in outcomes.values())
    print(f"{d}: {len(outcomes)} records")
    for fam, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:40s} {n}")
    bad = {s: o for s, o in outcomes.items()
           if outcome_family(o) in ("crash", "incomplete", "malformed")}
    for sid, o in sorted(bad.items()):
        print(f"  !! {o:12s} {sid}")
    return 1 if bad else 0


def _cmd_compare(leg_dir: str, base_dir: str, cap_us: int | None, strict: bool) -> int:
    new = classify_dir(leg_dir, cap_us)
    base = classify_dir(base_dir, cap_us)
    rows: list[tuple[str, str, str, str]] = []
    for sid in sorted(new):
        b = base.get(sid)
        if b is None:
            rows.append((sid, "-", new[sid], "no-baseline"))
            continue
        rows.append((sid, b, new[sid], compare_outcome(b, new[sid], strict=strict)))
    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'schema':{w}s}  {'baseline':28s} {'leg':28s} verdict")
    for sid, b, n, v in rows:
        mark = "  " if v in ("unchanged",) else ("+ " if v == "improved" else "!!")
        print(f"{mark}{sid:{w}s} {b:28s} {n:28s} {v}")
    counts = Counter(v for _, _, _, v in rows)
    print(f"\n{len(rows)} schemas: " + ", ".join(f"{v}={n}" for v, n in sorted(counts.items())))
    return 1 if counts.get("GATE") else 0


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("classify", help="bucket counts for one record dir")
    p.add_argument("dir")
    p.add_argument("--cap-s", type=int, default=None,
                   help="timeout cap override (default: the dir's _meta.json)")
    p = sub.add_parser("compare", help="outcome matrix: leg dir vs baseline dir")
    p.add_argument("leg_dir")
    p.add_argument("baseline_dir")
    p.add_argument("--cap-s", type=int, default=None)
    p.add_argument("--strict", action="store_true",
                   help="identity gate: sanctioned improvements also fail")
    args = ap.parse_args(argv)
    cap_us = args.cap_s * 1_000_000 if args.cap_s else None
    if args.cmd == "classify":
        return _cmd_classify(args.dir, cap_us)
    return _cmd_compare(args.leg_dir, args.baseline_dir, cap_us, args.strict)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
