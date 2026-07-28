"""GRID_PERF_LALR_DP corpus differential + A/B timing (CANDIDATES.md #6).

Parent mode walks manifest sets and runs one child subprocess per schema
(timeout-guarded, resume-supported, modeled on profile_phases.py). The child
compiles through projection once, then runs compile_tables twice — DP first
(the fast path), legacy second — streaming each phase record to disk the
moment it finishes, so a parent kill during the legacy build still leaves a
comparable DP timing ("legacy-incomparable": the candidate's win condition,
not a failure). Ship gate: zero mismatches.

Per-schema results:
    equal             both built, LALRTables field-by-field equal
    conflict_parity   both raised LALRConflictError, equal normalized sets
    skip:<Error>      pipeline failed before compile_tables (both paths moot)
    MISMATCH:<fields> / CONFLICT_MISMATCH / CLASS_MISMATCH:...   gate failures

Usage:
    python bench/perfbench/diff_lalr.py \
        --group ttfm_capped:120 --group stratified_200:60 \
        --jobs 4 --out tmp/perfbench-diff-lalr
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "jsb-src", "data")


def _norm_conflicts(report: list) -> set:
    return {(st, term, frozenset((c, a))) for (st, term, c, a) in report}


def child(schema_file: str, out_file: str) -> None:
    rec: dict = {"file": schema_file, "phases": {}, "result": None}

    def flush() -> None:
        with open(out_file, "w") as f:
            f.write(json.dumps(rec, indent=1))

    def phase(name: str, fn):
        rec["running"] = name
        flush()
        t0 = time.monotonic()
        val = fn()
        rec["phases"][name] = round((time.monotonic() - t0) * 1e6)
        return val

    with open(schema_file) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]  # wrapped maskbench layout

    from grid.errors import LALRConflictError
    from grid.grammar import spec
    from grid.grammar.projection import RoleProjection
    from grid.jsonschema import compile_json_schema
    from grid.lalr.compile import compile_tables

    try:
        src, _recorded = phase("schema_compile", lambda: compile_json_schema(schema))
        grammar = phase("spec_load", lambda: spec.load(src))
        proj = phase("projection", lambda: RoleProjection.full(grammar).build())
    except Exception as e:  # pipeline failure upstream of the candidate
        rec["running"] = None
        rec["result"] = f"skip:{type(e).__name__}"
        flush()
        return

    def build(algorithm: str):
        try:
            return compile_tables(proj, algorithm=algorithm), None
        except LALRConflictError as e:
            return None, e

    dp_tables, dp_err = phase("lalr_dp", lambda: build("dp"))
    legacy_tables, legacy_err = phase("lalr_legacy", lambda: build("lr1_merge"))
    rec["running"] = None

    if dp_tables is not None and legacy_tables is not None:
        diffs = [
            f.name for f in dataclasses.fields(legacy_tables)
            if getattr(legacy_tables, f.name) != getattr(dp_tables, f.name)
        ]
        rec["result"] = "equal" if not diffs else f"MISMATCH:{','.join(diffs)}"
        rec["states"] = len(legacy_tables.action)
    elif dp_err is not None and legacy_err is not None:
        same = _norm_conflicts(dp_err.report) == _norm_conflicts(legacy_err.report)
        rec["result"] = "conflict_parity" if same else "CONFLICT_MISMATCH"
    else:
        rec["result"] = f"CLASS_MISMATCH:dp={'conflict' if dp_err else 'ok'},legacy={'conflict' if legacy_err else 'ok'}"
    flush()


def schema_path(schema_id: str) -> str | None:
    # manifest ids without a split prefix (BFCL_*, JME_*) have no file in the
    # jsb-src checkout; callers skip those rather than fail the run
    if "---" not in schema_id:
        return None
    split, name = schema_id.split("---", 1)
    path = os.path.join(DATA_DIR, split, name + ".json")
    return path if os.path.exists(path) else None


def parent(groups: list[str], jobs: int, out_dir: str) -> None:
    with open(MANIFEST) as f:
        sets = json.load(f)["sets"]
    os.makedirs(out_dir, exist_ok=True)

    work: list[tuple[str, int]] = []
    seen: set[str] = set()
    for g in groups:
        name, timeout_s = g.split(":")
        for sid in sets[name]:
            if sid not in seen:
                seen.add(sid)
                work.append((sid, int(timeout_s)))

    print(f"{len(work)} schemas queued, jobs={jobs}", flush=True)
    running: list[tuple[subprocess.Popen, str, float, int]] = []
    queue = list(reversed(work))
    done = 0

    def out_path(sid: str) -> str:
        return os.path.join(out_dir, sid + ".diff.json")

    while queue or running:
        while queue and len(running) < jobs:
            sid, timeout_s = queue.pop()
            if os.path.exists(out_path(sid)):  # resume support
                done += 1
                continue
            spath = schema_path(sid)
            if spath is None:
                done += 1
                with open(out_path(sid), "w") as f:
                    f.write(json.dumps({"file": None, "phases": {}, "result": "skip:missing_data"}))
                continue
            p = subprocess.Popen(
                [sys.executable, __file__, "--child", spath, out_path(sid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running.append((p, sid, time.monotonic(), timeout_s))
        time.sleep(0.2)
        still = []
        for p, sid, t0, timeout_s in running:
            if p.poll() is not None:
                done += 1
                print(f"[{done}/{len(work)}] {sid} rc={p.returncode}", flush=True)
            elif time.monotonic() - t0 > timeout_s:
                p.kill()
                p.wait()
                done += 1
                try:
                    with open(out_path(sid)) as f:
                        rec = json.load(f)
                except Exception:
                    rec = {"phases": {}}
                rec["timeout_s"] = timeout_s
                with open(out_path(sid), "w") as f:
                    f.write(json.dumps(rec, indent=1))
                print(f"[{done}/{len(work)}] {sid} TIMEOUT in {rec.get('running')}", flush=True)
            else:
                still.append((p, sid, t0, timeout_s))
        running = still

    summarize(out_dir)


def summarize(out_dir: str) -> None:
    import glob

    counts: dict[str, int] = {}
    dp_us = legacy_us = 0
    mismatches: list[str] = []
    unverified: list[str] = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*.diff.json"))):
        with open(f) as fh:
            rec = json.load(fh)
        result = rec.get("result")
        if result is None:
            # parent kill: legacy-incomparable iff DP already finished
            if "lalr_dp" in rec.get("phases", {}):
                result = "legacy_timeout_dp_ok"
            else:
                result = f"timeout_in:{rec.get('running')}"
        key = result.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        if key in ("MISMATCH", "CONFLICT_MISMATCH", "CLASS_MISMATCH"):
            mismatches.append(f"{os.path.basename(f)}: {result}")
        elif key == "timeout_in":
            unverified.append(f"{os.path.basename(f)}: {result}")
        dp_us += rec.get("phases", {}).get("lalr_dp", 0)
        legacy_us += rec.get("phases", {}).get("lalr_legacy", 0)
    print(f"\nresults: {counts}")
    print(f"lalr totals (completed builds only): dp {dp_us / 1e6:.1f}s, legacy {legacy_us / 1e6:.1f}s")
    if unverified:
        # not a correctness verdict either way — these block a default flip
        print("UNVERIFIED (timed out before both builds finished):")
        for line in unverified:
            print(" ", line)
    if mismatches:
        print("GATE FAILURES:")
        for line in mismatches:
            print(" ", line)
        sys.exit(1)
    print("gate: zero mismatches")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("SCHEMA", "OUT"))
    ap.add_argument("--group", action="append", default=[], help="set_name:timeout_s")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="tmp/perfbench-diff-lalr")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(*args.child)
    elif args.summarize_only:
        summarize(args.out)
    else:
        parent(args.group, args.jobs, args.out)


if __name__ == "__main__":
    main()
