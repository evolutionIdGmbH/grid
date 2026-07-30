"""Store-warm corpus differential gate (S3 step 10).

Per schema, the FULL store-wired compile chain (schema_src -> spec.load ->
projection -> load_or_compile_tables -> load_or_build_scanner) plus a
deterministic guide drive, digested:

  src       sha of the compiled .grid source + recorded set
  tables    LALRTables fingerprint + identifier ids + action-table sha
  scanner   dense ScannerDFA sha, or the bounded lazy probe
            (diff_scanner_digest.digest_dense/digest_lazy)
  drive     per-step (entry_id, mask id count, boundary ids) along K steps of
            follow-the-first-allowed-token from the initial state (MockTokenizer
            byte-fallback vocab) — the mask entry_id parity arm
  error     class + message text when any stage raises (error parity: failed
            builds never put, so warm runs must fail identically)

Three legs, one corpus order, workers scrub GRID_PERF_* then apply leg env:

  off        GRID_PERF_ARTIFACT_STORE=0                      (baseline)
  cold       store on, EMPTY shared store (populates it)
  warm       store on, the store the cold leg just populated  (redeploy)

Gate: every digest equal across all three legs (compare exits nonzero on any
divergence). Usage:

    python bench/perfbench/diff_store_warm.py \
        --sets stratified_200,p3_family,ttfm_capped --timeout 120 --jobs 3 \
        --out tmp/store-diff
    # runs off -> cold -> warm, then compares off/cold, off/warm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")
DATA_DIR = os.environ.get(
    "GRID_JSB_DATA", os.path.join(HERE, "..", "..", "tmp", "jsb-src", "data"))

_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DRIVE_STEPS = 6


class _Timeout(BaseException):
    """SIGALRM payload; BaseException so no library except-clause eats it."""


def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, "..", "maskbench", "data", schema_id + ".json")


# ---------------------------------------------------------------- record


def _scanner_digests():
    """digest_dense/digest_lazy from diff_scanner_digest.py (bench is not a
    package: load by path, once)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "diff_scanner_digest", os.path.join(HERE, "diff_scanner_digest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.digest_dense, mod.digest_lazy


def unit_record(path: str, timeout_s: int, adapter) -> dict:
    digest_dense, digest_lazy = _scanner_digests()
    from grid.generate import build_guide
    from grid.grammar import spec
    from grid.grammar.projection import RoleProjection
    from grid.guide import COMPLETE
    from grid.jsonschema import compile_json_schema
    from grid.lexer.dfa import ScannerDFA
    from grid.serving.artifact_store import (
        load_or_build_scanner,
        load_or_compile_tables,
    )

    rec: dict = {"stage": "schema"}

    def _alarm(*_a):
        raise _Timeout()

    old_h = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    t0 = time.monotonic()
    try:
        try:
            with open(path) as f:
                schema = json.load(f)
            if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
                schema = schema["schema"]
            src, recorded = compile_json_schema(schema)
            rec["src"] = hashlib.sha256(
                (src + "\x00" + ",".join(sorted(recorded))).encode()).hexdigest()
            rec["stage"] = "grammar"
            grammar = spec.load(src)
            rec["stage"] = "tables"
            proj = RoleProjection.full(grammar).build()
            tables = load_or_compile_tables(proj)
            rec["tables"] = hashlib.sha256(repr(
                (tables.fingerprint, sorted(tables.identifier_terminal_ids),
                 tables.action)).encode()).hexdigest()
            rec["stage"] = "scanner"
            dfa = load_or_build_scanner(grammar)
            rec["scanner"] = (digest_dense(dfa) if isinstance(dfa, ScannerDFA)
                              else digest_lazy(dfa))
            rec["stage"] = "drive"
            guide = build_guide(src, adapter)
            st = guide.initial_state
            steps = []
            for _ in range(_DRIVE_STEPS):
                ids, entry_id = guide._mask_ids(st)
                ids = [int(t) for t in ids]
                steps.append((entry_id, len(ids), ids[:8], ids[-8:]))
                if st.status == COMPLETE or not ids:
                    break
                st = guide.get_next_state(st, ids[0])
            rec["drive"] = hashlib.sha256(repr(steps).encode()).hexdigest()
        except _Timeout:
            raise
        except Exception as e:  # declared outcomes are first-class records
            rec["error"] = f"{type(e).__name__}:{str(e)[:300]}"
        return rec
    except _Timeout:
        rec["timeout"] = rec.pop("stage")
        return rec
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)
        rec.pop("stage", None)
        rec["secs"] = round(time.monotonic() - t0, 3)


def worker(list_file: str, out_file: str, timeout_s: int,
           flags: list[str]) -> None:
    for k in [k for k in os.environ if k.startswith("GRID_PERF_")]:
        del os.environ[k]
    for f in flags:
        name, _, value = f.partition("=")
        os.environ[name] = value
    from grid.models.tokenizer_adapter import MockTokenizer

    adapter = MockTokenizer()  # byte-fallback complete: every schema drivable
    with open(list_file) as f:
        units = json.load(f)
    with open(out_file, "w") as out:
        for sid, path in units:
            try:
                rec = unit_record(path, timeout_s, adapter)
            except Exception as e:  # harness failure, not a build outcome
                rec = {"harness_error": f"{type(e).__name__}: {e}"}
            rec["id"] = sid
            out.write(json.dumps(rec) + "\n")
            out.flush()


# ---------------------------------------------------------------- parent


def load_run(out_dir: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("chunk") and fn.endswith(".jsonl"):
            with open(os.path.join(out_dir, fn)) as f:
                for line in f:
                    rec = json.loads(line)
                    by_id[rec["id"]] = rec
    return by_id


_FIELDS = ("src", "tables", "scanner", "drive", "error", "timeout",
           "harness_error")


def compare(dir_a: str, dir_b: str, label: str) -> int:
    a, b = load_run(dir_a), load_run(dir_b)
    bad = 0
    for missing, which in ((set(a) - set(b), dir_b), (set(b) - set(a), dir_a)):
        for uid in sorted(missing):
            print(f"MISSING in {which}: {uid}")
            bad += 1
    for uid in sorted(set(a) & set(b)):
        ra, rb = a[uid], b[uid]
        diffs = [(f, ra.get(f), rb.get(f)) for f in _FIELDS
                 if ra.get(f) != rb.get(f)]
        if diffs:
            bad += 1
            print(f"DIFF {uid}")
            for f, va, vb in diffs:
                print(f"  {f}: {str(va)[:160]}  !=  {str(vb)[:160]}")
    n = len(set(a) & set(b))
    print(f"{label}: " + (f"FAIL {bad} divergent/missing of {n}" if bad
                          else f"OK {n} units identical"))
    return bad


def run_leg(units: list, leg_dir: str, jobs: int, timeout_s: int,
            flags: list[str]) -> None:
    os.makedirs(leg_dir, exist_ok=True)
    chunks = [units[i::jobs] for i in range(jobs)]
    procs = []
    for i, chunk in enumerate(chunks):
        lf = os.path.join(leg_dir, f"chunk{i}.json")
        of = os.path.join(leg_dir, f"chunk{i}.jsonl")
        with open(lf, "w") as f:
            json.dump(chunk, f)
        cmd = [sys.executable, __file__, "--worker", lf, of,
               "--timeout", str(timeout_s)]
        for fl in flags:
            cmd += ["--flag", fl]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("LIST", "OUT"))
    ap.add_argument("--sets", default="stratified_200")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="tmp/store-diff")
    ap.add_argument("--flag", action="append", default=[])
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1], args.timeout, args.flag)
        return

    off_dir = os.path.join(args.out, "off")
    cold_dir = os.path.join(args.out, "cold")
    warm_dir = os.path.join(args.out, "warm")
    if not args.compare_only:
        with open(MANIFEST) as f:
            sets = json.load(f)["sets"]
        units, seen = [], set()
        for name in args.sets.split(","):
            for sid in sets[name]:
                if sid not in seen:
                    seen.add(sid)
                    units.append([sid, schema_path(sid)])
        store = os.path.join(args.out, "store")
        shutil.rmtree(store, ignore_errors=True)
        os.makedirs(store, exist_ok=True)
        print(f"{len(units)} units x 3 legs (off, cold, warm), jobs={args.jobs}",
              flush=True)
        run_leg(units, off_dir, args.jobs, args.timeout,
                ["GRID_PERF_ARTIFACT_STORE=0"])
        on = ["GRID_PERF_ARTIFACT_STORE=1", f"GRID_CACHE_DIR={store}"]
        run_leg(units, cold_dir, args.jobs, args.timeout, on)
        run_leg(units, warm_dir, args.jobs, args.timeout, on)

    bad = compare(off_dir, cold_dir, "off-vs-cold")
    bad += compare(off_dir, warm_dir, "off-vs-warm")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
