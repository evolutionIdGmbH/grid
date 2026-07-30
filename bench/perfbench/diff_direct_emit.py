"""GRID_PERF_DIRECT_EMIT corpus differential gate (P2, CANDIDATES id 10).

Per schema, flag-off oracle (compile text -> spec.load) vs flag-on object
path (compile_schema_parts -> DialectGrammar.from_parts), both arms in ONE
process: compile emission is PYTHONHASHSEED-sensitive on a handful of
corpus schemas (set-iteration order reaches rule/terminal numbering), so
cross-process text comparison needs a pinned seed — same-process arms share
one by construction. Workers scrub GRID_PERF_DIRECT_EMIT*, and
GRID_PERF_ARTIFACT_STORE: a warm store would serve both arms the same text
and make the differential vacuous.

Per-schema assertions, in gate order (first divergence labels the record):

  bucket           ok / unsupported / grammar_invalid / exception class —
                   message included (declared outcomes are load-bearing)
  terminal_order   PRIMARY grammar check: spec._fingerprint hashes SORTED
                   terminal names without decl_index, so fingerprint
                   equality alone cannot catch a literal first-use-order
                   bug that renumbers terminal ids -> masks -> kernel
                   T1/T2 keys under an identical fingerprint
  fingerprint / terminals / productions / start / ignored
  recorded         recorded-unenforced set parity
  lrec01           L-REC01 warning-count parity
  tables           (--tables only) role_shape_hash equality of
                   full().build() vs full_built(), LALRTables.fingerprint
                   equality — the keys of kernel configurations and T1/T2

Usage:
    python bench/perfbench/diff_direct_emit.py \
        --sets ttfm_capped,ttfm_tail_1pct,stratified_200,tbm_tail_100 \
        --tables [--maskbench-corpus 1] [--timeout 20] [--jobs 6] \
        --out tmp/diff-direct-emit
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import warnings

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.environ.get(
    "GRID_JSB_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "tmp",
                 "jsb-src", "data"))
MB_DATA_DIR = os.path.join(DATA_DIR, "..", "maskbench", "data")

# measure the tree this script lives in, not the venv's editable install
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class _Timeout(BaseException):
    """SIGALRM payload; BaseException so no library except-clause eats it."""


def run_arm(schema, flag_on: bool, timeout_s: int, tables: bool) -> dict:
    from grid.errors import GrammarInvalid
    from grid.jsonschema import compile_json_schema_grammar
    from grid.jsonschema.compiler import Unsupported

    os.environ["GRID_PERF_DIRECT_EMIT"] = "1" if flag_on else "0"

    def _alarm(*a):
        raise _Timeout()

    old_h = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    t0 = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g, recorded = compile_json_schema_grammar(schema)
        arm = {
            "bucket": "ok",
            "grammar": g,
            "recorded": sorted(recorded),
            "lrec01": sum(1 for x in w if "L-REC01" in str(x.message)),
            "secs": round(time.monotonic() - t0, 3),
        }
        if tables:
            from grid.grammar.projection import RoleProjection
            from grid.lalr.compile import compile_tables

            proj = RoleProjection.full_built(g) if flag_on \
                else RoleProjection.full(g).build()
            arm["role_shape_hash"] = proj.role_shape_hash
            arm["tables_fp"] = compile_tables(proj).fingerprint
        return arm
    except Unsupported as e:
        return {"bucket": "unsupported", "msg": str(e),
                "secs": round(time.monotonic() - t0, 3)}
    except GrammarInvalid as e:
        return {"bucket": "grammar_invalid", "msg": str(e),
                "secs": round(time.monotonic() - t0, 3)}
    except _Timeout:
        return {"bucket": "timeout", "secs": timeout_s}
    except Exception as e:
        return {"bucket": type(e).__name__, "msg": str(e)[:200],
                "secs": round(time.monotonic() - t0, 3)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)


def compare(schema_file: str, timeout_s: int, tables: bool) -> dict:
    with open(schema_file) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]
    old = run_arm(schema, False, timeout_s, tables)
    new = run_arm(schema, True, timeout_s, tables)
    rec: dict = {"file": os.path.basename(schema_file),
                 "old_secs": old["secs"], "new_secs": new["secs"]}
    if old["bucket"] == "timeout":
        rec["status"] = f"old_timeout_new_{new['bucket']}"
        return rec
    if new["bucket"] != old["bucket"]:
        rec["status"] = "FLIP_bucket"
        rec["old"], rec["new"] = old["bucket"], new["bucket"]
        return rec
    if old["bucket"] != "ok":
        if old.get("msg") != new.get("msg"):
            rec["status"] = "FLIP_msg"
            rec["old"], rec["new"] = old.get("msg"), new.get("msg")
        else:
            rec["status"] = f"equal_{old['bucket']}"
        return rec
    ga, gb = old["grammar"], new["grammar"]
    for check, attr in (
        ("terminal_order", "terminal_order"),  # PRIMARY, see module doc
        ("fingerprint", "fingerprint"),
        ("terminals", "terminals"),
        ("productions", "productions"),
        ("start", "start"),
        ("ignored", "ignored"),
    ):
        va, vb = getattr(ga, attr), getattr(gb, attr)
        if va != vb:
            rec["status"] = f"FLIP_{check}"
            rec["old"], rec["new"] = repr(va)[:300], repr(vb)[:300]
            return rec
    if old["recorded"] != new["recorded"]:
        rec["status"] = "FLIP_recorded"
        rec["old"], rec["new"] = old["recorded"], new["recorded"]
        return rec
    if old["lrec01"] != new["lrec01"]:
        rec["status"] = "FLIP_lrec01"
        rec["old"], rec["new"] = old["lrec01"], new["lrec01"]
        return rec
    if tables:
        for key in ("role_shape_hash", "tables_fp"):
            if old.get(key) != new.get(key):
                rec["status"] = f"FLIP_{key}"
                rec["old"], rec["new"] = old.get(key), new.get(key)
                return rec
    rec["status"] = "equal"
    rec["fingerprint"] = ga.fingerprint
    return rec


def worker(list_file: str, out_file: str, timeout_s: int,
           tables: bool) -> None:
    # scrub: the differential owns the emit flag; a warm artifact store
    # would serve both arms identical text and make the gate vacuous
    for k in ("GRID_PERF_DIRECT_EMIT", "GRID_PERF_DIRECT_EMIT_CHECK",
              "GRID_PERF_ARTIFACT_STORE"):
        os.environ.pop(k, None)
    with open(list_file) as f:
        files = json.load(f)
    with open(out_file, "w") as out:
        for sf in files:
            try:
                rec = compare(sf, timeout_s, tables)
            except Exception as e:   # harness failure, not a compile outcome
                rec = {"file": sf, "status": "HARNESS_ERROR",
                       "msg": f"{type(e).__name__}: {e}"}
            out.write(json.dumps(rec) + "\n")
            out.flush()


def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    # remaining ids (BFCL_*, JME_*, Synthesized---*, ...) are
    # maskbench-layout files named by their full id
    return os.path.join(MB_DATA_DIR, schema_id + ".json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("LIST", "OUT"))
    ap.add_argument("--sets", default="")
    ap.add_argument("--maskbench-corpus", type=int, default=0,
                    help="also add every Nth schema of the full 11.3k "
                         "maskbench data dir (1 = all)")
    ap.add_argument("--tables", action="store_true",
                    help="also gate role_shape_hash + LALRTables.fingerprint")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="tmp/diff-direct-emit")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1], args.timeout, args.tables)
        return

    files: list[str] = []
    seen: set[str] = set()
    if args.sets:
        with open(MANIFEST) as f:
            sets = json.load(f)["sets"]
        for name in args.sets.split(","):
            for sid in sets[name]:
                p = schema_path(sid)
                if p not in seen:
                    seen.add(p)
                    files.append(p)
    if args.maskbench_corpus:
        allf = sorted(
            os.path.join(MB_DATA_DIR, fn)
            for fn in os.listdir(MB_DATA_DIR) if fn.endswith(".json"))
        for p in allf[::args.maskbench_corpus]:
            if p not in seen:
                seen.add(p)
                files.append(p)

    os.makedirs(args.out, exist_ok=True)
    chunks = [files[i::args.jobs] for i in range(args.jobs)]
    procs = []
    for i, chunk in enumerate(chunks):
        lf = os.path.join(args.out, f"chunk{i}.json")
        of = os.path.join(args.out, f"chunk{i}.jsonl")
        with open(lf, "w") as f:
            json.dump(chunk, f)
        cmd = [sys.executable, __file__, "--worker", lf, of,
               "--timeout", str(args.timeout)]
        if args.tables:
            cmd.append("--tables")
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()

    counts: dict[str, int] = {}
    bad = []
    for i in range(args.jobs):
        with open(os.path.join(args.out, f"chunk{i}.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                if rec["status"].startswith(("FLIP", "HARNESS")):
                    bad.append(rec)
    print(f"n={len(files)}", json.dumps(counts, indent=1, sort_keys=True))
    for rec in bad[:40]:
        print("BAD:", json.dumps(rec)[:400])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
