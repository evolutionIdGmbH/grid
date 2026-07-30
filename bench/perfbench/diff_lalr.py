"""GRID_PERF_LALR_DP corpus differential + A/B timing (CANDIDATES.md #6).

Parent mode walks manifest sets and runs one child subprocess per schema
(timeout-guarded, resume-supported, modeled on profile_phases.py). The child
compiles through projection once, then runs compile_tables twice — DP first
(the fast path), legacy second — streaming each phase record to disk the
moment it finishes, so a parent kill during the legacy build still leaves a
comparable DP timing ("legacy-incomparable": the candidate's win condition,
not a failure). Ship gate: zero mismatches.

Both childs record the construction size counters (compile_tables
stats out-param: dp lr0_states/lr0_items, legacy lr1_states/lr1_items) —
the P5 LALR-budget calibration data (items materialized is the budget's
work unit).

Per-schema results:
    equal             both built, LALRTables field-by-field equal
    conflict_parity   both raised LALRConflictError, equal normalized sets
    budget_parity     both raised LALRBudgetExceeded (class equality only:
                      fire counts differ by construction)
    budget_legacy_only  sanctioned asymmetry — the lr1_merge oracle fired
                      where dp completed (dp defines shipped outcomes)
    skip:<Error>      pipeline failed before compile_tables (both paths moot)
    MISMATCH:<fields> / CONFLICT_MISMATCH / CLASS_MISMATCH:... /
    BUDGET_MISMATCH:... (dp fired, legacy did not)   gate failures

Counts mode (--counts) swaps in a dp-only child that replays the SHIPPED
maskbench build sequence — compile_json_schema_grammar, RoleProjection,
compile_tables(dp), LALR-conflict retry with unify_string_values=True —
recording per-leg counters and outcome (ok | conflict | declared:<cls> |
skip:<cls>). This is the corpus-wide budget-calibration sweep: every leg it
reports "ok"/"conflict" for is a build the budget must never fire on.
--ids-from-dir <status_dir> takes the schema list from an existing
maskbench status dir (e.g. tmp/mb-grid-v030rc2) instead of manifest groups.

Usage:
    python bench/perfbench/diff_lalr.py \
        --group ttfm_capped:120 --group stratified_200:60 \
        --jobs 4 --out tmp/perfbench-diff-lalr
    python bench/perfbench/diff_lalr.py --counts \
        --ids-from-dir tmp/mb-grid-v030rc2 --timeout 120 \
        --jobs 10 --out tmp/perfbench-lalr-counts
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
# the flat maskbench checkout: ALL 11,306 corpus ids as <id>.json, including
# the split-less ones (BFCL_*, JME_*, MCPspec_*, Synthesized_*) that have no
# file under DATA_DIR's <split>/<name> layout
MB_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "tmp", "jsb-src", "maskbench", "data")


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

    from grid.errors import LALRBudgetExceeded, LALRConflictError
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
        stats: dict = {}
        try:
            result = compile_tables(proj, algorithm=algorithm, stats=stats), None
        except (LALRBudgetExceeded, LALRConflictError) as e:
            result = None, e
        # counters land in the record before the next flush (the following
        # phase's start-flush), so a parent kill during the legacy build
        # still leaves the finished dp counts on disk
        rec.setdefault("lalr", {})[algorithm] = stats
        if isinstance(result[1], LALRBudgetExceeded):
            stats["budget_fire"] = {"states": result[1].states, "items": result[1].items}
        return result

    dp_tables, dp_err = phase("lalr_dp", lambda: build("dp"))
    legacy_tables, legacy_err = phase("lalr_legacy", lambda: build("lr1_merge"))
    rec["running"] = None

    dp_budget = isinstance(dp_err, LALRBudgetExceeded)
    legacy_budget = isinstance(legacy_err, LALRBudgetExceeded)
    if dp_tables is not None and legacy_tables is not None:
        diffs = [
            f.name for f in dataclasses.fields(legacy_tables)
            if getattr(legacy_tables, f.name) != getattr(dp_tables, f.name)
        ]
        rec["result"] = "equal" if not diffs else f"MISMATCH:{','.join(diffs)}"
        rec["states"] = len(legacy_tables.action)
    elif dp_budget and legacy_budget:
        # over-budget scoping: class equality only (fire counts differ by
        # construction — LR(1) materializes >= LR(0) items)
        rec["result"] = "budget_parity"
    elif legacy_budget and not dp_budget:
        # sanctioned asymmetry: the oracle may fire where dp completes; dp
        # defines shipped outcomes (grid/lalr/compile.py module docstring)
        rec["result"] = "budget_legacy_only"
    elif dp_budget:
        # LR(1) >= LR(0) items makes this impossible; a gate failure if seen
        rec["result"] = "BUDGET_MISMATCH:dp_fired_legacy_did_not"
    elif dp_err is not None and legacy_err is not None:
        same = _norm_conflicts(dp_err.report) == _norm_conflicts(legacy_err.report)
        rec["result"] = "conflict_parity" if same else "CONFLICT_MISMATCH"
    else:
        rec["result"] = f"CLASS_MISMATCH:dp={'conflict' if dp_err else 'ok'},legacy={'conflict' if legacy_err else 'ok'}"
    flush()


def counts_child(schema_file: str, out_file: str) -> None:
    """dp-only calibration child: the SHIPPED maskbench build sequence
    (initial build, then the LALR-conflict retry with
    unify_string_values=True), construction counters per leg."""
    rec: dict = {"file": schema_file, "mode": "counts", "legs": [], "result": None}

    def flush() -> None:
        with open(out_file, "w") as f:
            f.write(json.dumps(rec, indent=1))

    with open(schema_file) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]  # wrapped maskbench layout

    from grid import perf_flags
    from grid.errors import LALRConflictError
    from grid.grammar.projection import RoleProjection
    from grid.jsonschema import compile_json_schema_grammar
    from grid.lalr.compile import compile_tables

    def leg(unify: bool) -> str:
        entry: dict = {"unify": unify}
        rec["legs"].append(entry)
        rec["running"] = f"leg_unify{int(unify)}"
        flush()
        stats: dict = {}
        entry["stats"] = stats
        t0 = time.monotonic()
        try:
            grammar, _recorded = compile_json_schema_grammar(
                schema, unify_string_values=unify)
            proj = (RoleProjection.full_built(grammar)
                    if perf_flags.direct_emit_enabled()
                    else RoleProjection.full(grammar).build())
        except Exception as e:  # upstream of compile_tables: both legs moot
            entry["outcome"] = f"skip:{type(e).__name__}"
            entry["front_us"] = round((time.monotonic() - t0) * 1e6)
            return entry["outcome"]
        entry["front_us"] = round((time.monotonic() - t0) * 1e6)
        t1 = time.monotonic()
        try:
            compile_tables(proj, algorithm="dp", stats=stats)
            entry["outcome"] = "ok"
        except LALRConflictError:
            # stats is already populated: conflicts are a fill-stage outcome,
            # after construction — a completed build the budget must respect
            entry["outcome"] = "conflict"
        except Exception as e:  # declared decline (LALRBudgetExceeded, ...)
            entry["outcome"] = f"declared:{type(e).__name__}"
            for k in ("states", "items"):
                v = getattr(e, k, None)
                if isinstance(v, int):
                    entry[f"declared_{k}"] = v
        entry["lalr_us"] = round((time.monotonic() - t1) * 1e6)
        return entry["outcome"]

    out = leg(False)
    if out == "conflict":
        # maskbench GridEngine's conflict retry, once; a second "conflict" is
        # the honest final outcome (rc2's declared:LALRConflictError bucket)
        out = leg(True)
    rec["running"] = None
    rec["result"] = out
    flush()


def schema_path(schema_id: str) -> str | None:
    # split-form ids resolve through the historical <split>/<name> layout
    # first (unchanged resolution for every id earlier runs covered), then
    # any id falls back to the flat maskbench checkout
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        path = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(path):
            return path
    path = os.path.join(MB_DATA_DIR, schema_id + ".json")
    return path if os.path.exists(path) else None


def build_work(groups: list[str], ids_from_dir: str | None, timeout: int) -> list[tuple[str, int]]:
    work: list[tuple[str, int]] = []
    seen: set[str] = set()
    if groups:
        with open(MANIFEST) as f:
            sets = json.load(f)["sets"]
        for g in groups:
            name, timeout_s = g.split(":")
            for sid in sets[name]:
                if sid not in seen:
                    seen.add(sid)
                    work.append((sid, int(timeout_s)))
    if ids_from_dir:
        import glob

        for f in sorted(glob.glob(os.path.join(ids_from_dir, "*.json"))):
            base = os.path.basename(f)
            if base == "_meta.json":
                continue
            sid = base[: -len(".json")]
            if sid not in seen:
                seen.add(sid)
                work.append((sid, timeout))
    return work


def parent(work: list[tuple[str, int]], jobs: int, out_dir: str, counts: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    suffix = ".counts.json" if counts else ".diff.json"
    child_flag = "--counts-child" if counts else "--child"

    print(f"{len(work)} schemas queued, jobs={jobs}", flush=True)
    running: list[tuple[subprocess.Popen, str, float, int]] = []
    queue = list(reversed(work))
    done = 0

    def out_path(sid: str) -> str:
        return os.path.join(out_dir, sid + suffix)

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
                [sys.executable, __file__, child_flag, spath, out_path(sid)],
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

    (summarize_counts if counts else summarize)(out_dir)


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
        if key in ("MISMATCH", "CONFLICT_MISMATCH", "CLASS_MISMATCH", "BUDGET_MISMATCH"):
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


def summarize_counts(out_dir: str) -> None:
    """Calibration digest: every completed dp construction (ok or conflict —
    builds the budget must never fire on), items/states distribution, top
    completers, declared fires, and killed/incomplete records (never counted
    as completers — the F1 lesson)."""
    import glob

    results: dict[str, int] = {}
    completed: list[tuple[int, int, int, str, str, bool]] = []
    declines: list[str] = []
    killed: list[str] = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*.counts.json"))):
        sid = os.path.basename(f)[: -len(".counts.json")]
        with open(f) as fh:
            rec = json.load(fh)
        if "timeout_s" in rec or rec.get("running"):
            killed.append(f"{sid}: killed in {rec.get('running')}")
            results["timeout"] = results.get("timeout", 0) + 1
            continue
        result = rec.get("result") or "malformed"
        key = result.split(":")[0]
        results[key] = results.get(key, 0) + 1
        for entry in rec.get("legs", []):
            out = entry.get("outcome", "")
            st = entry.get("stats") or {}
            if out in ("ok", "conflict") and "lr0_items" in st:
                completed.append((st["lr0_items"], st["lr0_states"],
                                  entry.get("lalr_us", 0), sid, out,
                                  bool(entry.get("unify"))))
            elif out.startswith("declared:"):
                declines.append(
                    f"{sid} unify={int(bool(entry.get('unify')))}: {out} "
                    f"items={entry.get('declared_items')} "
                    f"states={entry.get('declared_states')}")
    print(f"\nresults: {results}")
    print(f"completed dp constructions (budget must never fire on these): {len(completed)}")
    if completed:
        completed.sort(reverse=True)
        items = sorted(c[0] for c in completed)

        def pct(p: float) -> int:
            return items[min(len(items) - 1, int(p / 100 * len(items)))]

        print(f"lr0_items p50={pct(50):,} p99={pct(99):,} max={items[-1]:,}")
        print("top completers by lr0_items:")
        for it, stt, us, sid, out, unify in completed[:12]:
            print(f"  {it:>10,} items {stt:>9,} states {us / 1e6:7.2f}s "
                  f"{out:8s} unify={int(unify)} {sid}")
    if declines:
        print(f"declared declines ({len(declines)}):")
        for line in declines:
            print(" ", line)
    if killed:
        print(f"killed/incomplete ({len(killed)}):")
        for line in killed[:12]:
            print(" ", line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("SCHEMA", "OUT"))
    ap.add_argument("--counts-child", nargs=2, metavar=("SCHEMA", "OUT"))
    ap.add_argument("--counts", action="store_true",
                    help="dp-only maskbench-sequence calibration sweep")
    ap.add_argument("--group", action="append", default=[], help="set_name:timeout_s")
    ap.add_argument("--ids-from-dir", default=None,
                    help="derive schema ids from a maskbench status dir")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-schema timeout for --ids-from-dir entries")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="tmp/perfbench-diff-lalr")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(*args.child)
    elif args.counts_child:
        counts_child(*args.counts_child)
    elif args.summarize_only:
        (summarize_counts if args.counts else summarize)(args.out)
    else:
        work = build_work(args.group, args.ids_from_dir, args.timeout)
        parent(work, args.jobs, args.out, counts=args.counts)


if __name__ == "__main__":
    main()
