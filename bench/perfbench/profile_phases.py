"""Per-phase TTFM attribution over perfbench task sets (DESIGN.md prerequisite).

Parent mode walks manifest sets and runs one child subprocess per schema
(timeout-guarded; a timeout still attributes, because the child streams each
phase record to disk the moment it finishes, so the phase in flight at kill
time is the last+1). Child mode runs the five compile phases:

    schema_compile -> spec_load -> projection -> lalr -> scanner

and records size stats alongside timings (terminals, grammar source bytes,
DFA states, peak RSS) so cost drivers can be correlated, not guessed.

Usage:
    python bench/perfbench/profile_phases.py \
        --group ttfm_capped:75:0:1 --group ttfm_tail_1pct:120:40:1 \
        --group stratified_200:30:0:7 --jobs 3 --out tmp/perfbench-profile

Group syntax: name:timeout_s:limit:stride (limit 0 = whole set).
Attribution shares tolerate --jobs 3 on a big-core host; publishable
absolute numbers require --jobs 1.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "jsb-src", "data")

# measure the tree this script lives in: the venv's grid install points at the
# main checkout, so without this pin a worktree run silently profiles main
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PHASES = ("schema_compile", "spec_load", "projection", "lalr", "scanner")


def child(schema_file: str, out_file: str) -> None:
    # flags snapshot: every leg is self-describing (the F1/F2 investigations
    # were blocked because leg env was unrecoverable from the records)
    rec: dict = {
        "file": schema_file,
        "flags": {k: v for k, v in os.environ.items() if k.startswith("GRID_PERF_")},
        "phases": {},
        "stats": {},
    }

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

    try:
        with open(schema_file) as f:
            schema = json.load(f)
        if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
            schema = schema["schema"]  # wrapped maskbench layout

        from grid import perf_flags
        from grid.grammar import spec
        from grid.grammar.projection import RoleProjection
        from grid.jsonschema import compile_json_schema
        from grid.lalr.compile import compile_tables
        from grid.lexer.dfa import build_scanner

        if perf_flags.direct_emit_enabled():
            # GRID_PERF_DIRECT_EMIT leg: the store-off sequence of
            # compile_json_schema_grammar, split at the same joints so the
            # record format (and A/B comparability) is unchanged —
            # schema_compile = manifest build (no text render), spec_load =
            # object build + shared validate/freeze, projection = trusted
            # full_built. The flags snapshot above self-describes the leg.
            from grid.jsonschema.compiler import compile_schema_parts

            parts, recorded = phase(
                "schema_compile", lambda: compile_schema_parts(schema))
            rec["stats"]["recorded"] = len(recorded)
            rec["stats"]["direct_emit"] = True
            grammar = phase(
                "spec_load", lambda: spec.DialectGrammar.from_parts(parts))
            rec["stats"]["terminals"] = len(grammar.terminals)
            proj = phase("projection", lambda: RoleProjection.full_built(grammar))
        else:
            src, recorded = phase("schema_compile", lambda: compile_json_schema(schema))
            rec["stats"]["src_bytes"] = len(src)
            rec["stats"]["recorded"] = len(recorded)
            grammar = phase("spec_load", lambda: spec.load(src))
            rec["stats"]["terminals"] = len(grammar.terminals)
            proj = phase("projection", lambda: RoleProjection.full(grammar).build())
        tables = phase("lalr", lambda: compile_tables(proj))
        rec["stats"]["lalr_states"] = len(getattr(tables, "action", ()) or ())
        dfa = phase("scanner", lambda: build_scanner(grammar.terminals, grammar.terminal_order))
        if getattr(dfa, "lazy", False):
            # GRID_PERF_FACTORED_SCANNER over-budget regime: LazyProductDFA has no
            # dense trans; report the product states materialized so far
            rec["stats"]["dfa_states"] = len(dfa._states)
            rec["stats"]["dfa_lazy"] = True
        else:
            rec["stats"]["dfa_states"] = len(dfa.trans)
    except Exception as e:
        # declared outcomes (Unsupported, LALRConflictError, ...) become a
        # first-class record category instead of an anonymous nonzero exit
        rec["error"] = type(e).__name__
        rec["error_msg"] = str(e)[:200]
        flush()
        raise
    rec["running"] = None
    rec["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20))
    flush()


def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    # remaining ids (BFCL_*, JME_*, ...) are maskbench-layout files named by
    # their full id (same fallback as diff_hashcons/diff_scanner_digest)
    return os.path.join(DATA_DIR, "..", "maskbench", "data",
                        schema_id + ".json")


def parent(groups: list[str], jobs: int, out_dir: str) -> None:
    with open(MANIFEST) as f:
        sets = json.load(f)["sets"]
    os.makedirs(out_dir, exist_ok=True)

    work: list[tuple[str, int]] = []  # (schema_id, timeout_s)
    seen: set[str] = set()
    for g in groups:
        name, timeout_s, limit, stride = g.split(":")
        ids = sets[name][:: int(stride) or 1]
        if int(limit):
            ids = ids[: int(limit)]
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                work.append((sid, int(timeout_s)))

    print(f"{len(work)} schemas queued, jobs={jobs}", flush=True)
    running: list[tuple[subprocess.Popen, str, float, int]] = []
    queue = list(reversed(work))
    done = 0

    def out_path(sid: str) -> str:
        return os.path.join(out_dir, sid + ".phases.json")

    while queue or running:
        while queue and len(running) < jobs:
            sid, timeout_s = queue.pop()
            if os.path.exists(out_path(sid)):  # resume support
                try:
                    with open(out_path(sid)) as f:
                        prev = json.load(f)
                except Exception:
                    prev = {"running": "?"}  # unreadable = treat as partial
                if prev.get("running") is None or "timeout_s" in prev or "rc" in prev:
                    done += 1
                    continue
                # unmarked partial from an interrupted parent: requeue, never
                # fossilize (partials read as ~0.0s legs in summaries)
                os.remove(out_path(sid))
            p = subprocess.Popen(
                [sys.executable, __file__, "--child", schema_path(sid), out_path(sid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running.append((p, sid, time.monotonic(), timeout_s))
        time.sleep(0.2)
        still = []
        for p, sid, t0, timeout_s in running:
            if p.poll() is not None:
                done += 1
                if p.returncode != 0:
                    # annotate exactly like the timeout branch: a crashed or
                    # killed child must never be mistakable for a completed one
                    try:
                        with open(out_path(sid)) as f:
                            rec = json.load(f)
                    except Exception:
                        rec = {"phases": {}}
                    rec["rc"] = p.returncode
                    with open(out_path(sid), "w") as f:
                        f.write(json.dumps(rec, indent=1))
                print(f"[{done}/{len(work)}] {sid} rc={p.returncode}", flush=True)
            elif time.monotonic() - t0 > timeout_s:
                p.kill()
                p.wait()
                done += 1
                # mark the timeout in the record the child left behind
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
    import glob as g

    totals = dict.fromkeys(PHASES, 0)
    timeouts: dict[str, int] = {}
    incomplete: dict[str, int] = {}
    n_done = 0
    for f in g.glob(os.path.join(out_dir, "*.phases.json")):
        with open(f) as fh:
            rec = json.load(fh)
        if rec.get("running") is None and "error" not in rec and "rc" not in rec:
            # completed = child reached the final flush; a pre-first-phase
            # crash also lacks 'running' but carries error/rc
            n_done += 1
        elif "timeout_s" in rec or "rc" in rec:
            ph = rec.get("running") or "?"
            timeouts[ph] = timeouts.get(ph, 0) + 1
        else:
            # unmarked partial (interrupted parent): its phase sums are not
            # attributions — exclude from totals entirely
            ph = rec.get("running") or "?"
            incomplete[ph] = incomplete.get(ph, 0) + 1
            continue
        for k, v in rec.get("phases", {}).items():
            totals[k] += v
    tot = sum(totals.values()) or 1
    print(f"\ncompleted: {n_done}; timeouts/crashed by in-flight phase: {timeouts}")
    if incomplete:
        print(f"incomplete (unmarked, will re-run on resume) by in-flight phase: {incomplete}")
    for ph in PHASES:
        print(f"  {ph:15s} {totals[ph] / 1e6:9.1f}s  {100 * totals[ph] / tot:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--group", action="append", required=True, help="name:timeout_s:limit:stride")
        ap.add_argument("--jobs", type=int, default=1)
        ap.add_argument("--out", required=True)
        ap.add_argument("--summarize-only", action="store_true")
        args = ap.parse_args()
        if args.summarize_only:
            summarize(args.out)
        else:
            parent(args.group, args.jobs, args.out)
