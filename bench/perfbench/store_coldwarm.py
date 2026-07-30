"""Artifact-store cold/warm redeploy harness (S3 step 9; DESIGN.md protocol).

Every measurement runs in a FRESH child process — same-process "warm" reads
conflate in-memory memos (registry single-flight, factored._COMPONENTS,
T1/T2) with the store and measure ~0.16 ms where a true redeploy pays
spec-load + unpickle. Three compile scenarios per schema, each its own child:

    C  flag-off baseline               (GRID_PERF_ARTIFACT_STORE=0)
    A  flag-on, EMPTY per-schema store (cold compile + persist cost)
    B  flag-on, the SAME store         (the redeploy warm hit)

A and B are reported separately, never blended (SELECTION.md #8). The
summary reports per-scenario phase totals (p50/p99/max), the per-schema
store size, and the DEDUPLICATED deployment footprint (store keys are
content-derived, so equal (namespace, key) across schemas is one file).
Verify-flag legs are gone as of Wave A: E3 deleted GRID_PERF_NFA_LIVE and
the epoch defaults are production defaults, so the production leg IS the
default leg (re-baselining the old verify-inflated p99 happens here for
free).

The journal TBM protocol (--tbm) simulates a redeployment of a serving
population (spider SQL dialect + schema lexicons, mock adapter, kernel walk
when grid_core is importable):

    serve          drive N requests, journals accrue, flush to the store
    redeploy-warm  fresh process, journal RESTORED -> admission_warmup
                   off-batch -> first-request per-state mask-time
                   distribution
    redeploy-cold  fresh process, journal namespace disabled (empty journal,
                   warmup runs but has nothing) -> same drive/distribution

Compile artifacts are store-warm in BOTH redeploy legs, isolating the
journal's contribution to the first-request TBM tail.

Usage:
    python bench/perfbench/store_coldwarm.py \
        --group stratified_200:60:0:4 --group p3_family:300:0:1 \
        --group ttfm_capped:300:0:1 --tbm --jobs 1 --out tmp/store-coldwarm
    python bench/perfbench/store_coldwarm.py --summarize-only --out ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")
# GRID_JSB_DATA: corpus root override (worktrees don't carry tmp/jsb-src)
DATA_DIR = os.environ.get("GRID_JSB_DATA") or os.path.join(
    HERE, "..", "..", "tmp", "jsb-src", "data")

_ROOT = os.path.dirname(os.path.dirname(HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PHASES = ("schema_compile", "spec_load", "projection", "lalr", "scanner")

TBM_SCHEMA = {"employees": ["id", "name"], "orders": ["total", "qty"]}
TBM_TEXTS = [
    b"select name from employees",
    b"select total from orders where qty=1",
    b"select id from employees where id=2",
    b"select name,id from employees where id=42",
]
TBM_TOKENS = (
    "select", "sel", "ect", "insert", "update", "delete", "from", "where",
    "and", "or", "limit", "into", "values", "set", " ", "*", ",", ";", "=",
    "<", ">", "(", ")", "employees", "orders", "qty", "total", "name", "id",
    " from ", " where ", "select ", "'x'", "'", "1", "2", "42", "0",
)


# --------------------------------------------------------------- children


def compile_child(schema_file: str, out_file: str) -> None:
    """One store-wired compile chain, phases streamed to disk (a timeout
    still attributes: the phase in flight is running at kill time)."""
    rec: dict = {
        "file": schema_file,
        "flags": {k: v for k, v in os.environ.items()
                  if k.startswith(("GRID_PERF_", "GRID_CACHE_DIR"))},
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
            schema = schema["schema"]

        from grid.grammar import spec
        from grid.grammar.projection import RoleProjection
        from grid.jsonschema import compile_json_schema
        from grid.serving.artifact_store import (
            load_or_build_scanner,
            load_or_compile_tables,
        )

        src, recorded = phase("schema_compile", lambda: compile_json_schema(schema))
        rec["stats"]["src_bytes"] = len(src)
        grammar = phase("spec_load", lambda: spec.load(src))
        rec["stats"]["terminals"] = len(grammar.terminal_order)
        proj = phase("projection", lambda: RoleProjection.full(grammar).build())
        phase("lalr", lambda: load_or_compile_tables(proj))
        dfa = phase("scanner", lambda: load_or_build_scanner(grammar))
        rec["stats"]["dfa_lazy"] = bool(getattr(dfa, "lazy", False))
    except Exception as e:
        rec["error"] = type(e).__name__
        rec["error_msg"] = str(e)[:200]
        flush()
        raise
    rec["running"] = None
    flush()


def tbm_child(role: str, out_file: str) -> None:
    """serve | redeploy-warm | redeploy-cold (see module doc). The parent
    sets GRID_CACHE_DIR / GRID_ADMIT_WARM / store flags; redeploy-cold
    additionally gets GRID_PERF_STORE_JOURNAL=0."""
    from concurrent.futures import ThreadPoolExecutor

    from grid.models.tokenizer_adapter import MockTokenizer
    from grid.models.vllm_processor import _GuideRegistry
    from grid.models.vllm_structured import admission_warmup

    spider = open(os.path.join(_ROOT, "grammars", "sql_spider.grid")).read()
    tok = MockTokenizer(extra_tokens=TBM_TOKENS)
    rec: dict = {"role": role, "flags": {
        k: v for k, v in os.environ.items()
        if k.startswith(("GRID_PERF_", "GRID_ADMIT", "GRID_CACHE_DIR"))}}

    t0 = time.monotonic()
    reg = _GuideRegistry(tok)
    guide = reg.guide_for({"grammar": spider, "schema": TBM_SCHEMA})
    rec["build_ms"] = round((time.monotonic() - t0) * 1e3, 3)
    journal = guide.producer.journal
    rec["journal_at_build"] = journal.stats if journal is not None else None

    if role == "serve":
        for text in TBM_TEXTS:
            st = guide.initial_state
            guide._mask_ids(st)
            for t in tok.greedy_tokenize(text):
                st = guide.get_next_state(st, int(t))
                guide._mask_ids(st)
        reg.flush_journals()
        rec["journal_after"] = journal.stats if journal is not None else None
    else:
        with ThreadPoolExecutor(2) as pool:
            t0 = time.monotonic()
            rec["warmup"] = admission_warmup(guide, pool)
            rec["warmup_wall_ms"] = round((time.monotonic() - t0) * 1e3, 3)
        # first request = the first drive after admission; per-state mask time
        per_state_us: list[float] = []
        text = TBM_TEXTS[0]
        st = guide.initial_state
        t0 = time.monotonic()
        guide._mask_ids(st)
        per_state_us.append((time.monotonic() - t0) * 1e6)
        for t in tok.greedy_tokenize(text):
            st = guide.get_next_state(st, int(t))
            t0 = time.monotonic()
            guide._mask_ids(st)
            per_state_us.append((time.monotonic() - t0) * 1e6)
        rec["first_request_us"] = [round(x, 1) for x in per_state_us]
        # steady state contrast: the remaining requests
        rest_us: list[float] = []
        for text in TBM_TEXTS[1:]:
            st = guide.initial_state
            for t in tok.greedy_tokenize(text):
                st = guide.get_next_state(st, int(t))
                t0 = time.monotonic()
                guide._mask_ids(st)
                rest_us.append((time.monotonic() - t0) * 1e6)
        rec["later_requests_us"] = [round(x, 1) for x in rest_us]

    with open(out_file, "w") as f:
        f.write(json.dumps(rec, indent=1))


# ----------------------------------------------------------------- parent


def schema_path(schema_id: str) -> str:
    if "---" in schema_id:
        split, name = schema_id.split("---", 1)
        p = os.path.join(DATA_DIR, split, name + ".json")
        if os.path.exists(p):
            return p
    return os.path.join(DATA_DIR, "..", "maskbench", "data", schema_id + ".json")


def _run_child(sid: str, scenario: str, env: dict, timeout_s: int,
               out_path: str) -> None:
    try:
        subprocess.run(
            [sys.executable, __file__, "--child", schema_path(sid), out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        try:
            with open(out_path) as f:
                rec = json.load(f)
        except Exception:
            rec = {"phases": {}}
        rec["timeout_s"] = timeout_s
        with open(out_path, "w") as f:
            f.write(json.dumps(rec, indent=1))


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def parent(groups: list[str], out_dir: str, run_tbm: bool) -> None:
    with open(MANIFEST) as f:
        sets = json.load(f)["sets"]
    os.makedirs(out_dir, exist_ok=True)
    stores_dir = os.path.join(out_dir, "stores")
    os.makedirs(stores_dir, exist_ok=True)

    work: list[tuple[str, int]] = []
    seen: set[str] = set()
    group_of: dict[str, str] = {}
    for g in groups:
        name, timeout_s, limit, stride = g.split(":")
        ids = sets[name][:: int(stride) or 1]
        if int(limit):
            ids = ids[: int(limit)]
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                work.append((sid, int(timeout_s)))
                group_of[sid] = name
    with open(os.path.join(out_dir, "groups.json"), "w") as f:
        json.dump(group_of, f, indent=1)

    base_env = {k: v for k, v in os.environ.items()
                if not k.startswith("GRID_")}
    base_env["PYTHONPATH"] = _ROOT

    print(f"{len(work)} schemas x 3 scenarios, sequential (jobs=1 for "
          "publishable absolutes)", flush=True)
    for i, (sid, timeout_s) in enumerate(work):
        store = os.path.join(stores_dir, sid)
        out = lambda sc: os.path.join(out_dir, f"{sid}.{sc}.json")  # noqa: E731
        if os.path.exists(out("B")):
            print(f"[{i + 1}/{len(work)}] {sid} (resume: done)", flush=True)
            continue
        shutil.rmtree(store, ignore_errors=True)
        os.makedirs(store, exist_ok=True)
        t0 = time.monotonic()
        # C: flag-off baseline
        env = dict(base_env, GRID_PERF_ARTIFACT_STORE="0")
        _run_child(sid, "C", env, timeout_s, out("C"))
        # A: flag-on, empty store (cold + persist)
        env = dict(base_env, GRID_PERF_ARTIFACT_STORE="1", GRID_CACHE_DIR=store)
        _run_child(sid, "A", env, timeout_s, out("A"))
        size = _dir_bytes(store)
        # B: flag-on, populated store (redeploy warm)
        _run_child(sid, "B", env, timeout_s, out("B"))
        try:
            with open(out("B")) as f:
                rec = json.load(f)
            rec["store_bytes"] = size
            with open(out("B"), "w") as f:
                f.write(json.dumps(rec, indent=1))
        except Exception:
            pass
        print(f"[{i + 1}/{len(work)}] {sid} ({time.monotonic() - t0:.1f}s)",
              flush=True)

    if run_tbm:
        tbm_dir = os.path.join(out_dir, "tbm-store")
        for leg, extra in (
            ("serve", {}),
            ("redeploy-warm", {}),
            ("redeploy-cold", {"GRID_PERF_STORE_JOURNAL": "0"}),
        ):
            env = dict(base_env, GRID_PERF_ARTIFACT_STORE="1",
                       GRID_CACHE_DIR=tbm_dir, GRID_ADMIT_WARM="1", **extra)
            if leg == "serve":
                shutil.rmtree(tbm_dir, ignore_errors=True)
                os.makedirs(tbm_dir, exist_ok=True)
            print(f"tbm: {leg}", flush=True)
            subprocess.run(
                [sys.executable, __file__, "--tbm-child", leg,
                 os.path.join(out_dir, f"tbm.{leg}.json")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=600,
            )

    summarize(out_dir)


# -------------------------------------------------------------- summarize


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))
    return xs[idx]


def summarize(out_dir: str) -> None:
    import glob as g

    try:
        with open(os.path.join(out_dir, "groups.json")) as f:
            group_of = json.load(f)
    except Exception:
        group_of = {}

    by_group: dict[str, dict[str, list]] = {}
    dedup: dict[tuple[str, str], int] = {}
    per_store: list[int] = []
    n_timeout = {"A": 0, "B": 0, "C": 0}
    for path in sorted(g.glob(os.path.join(out_dir, "*.?.json"))):
        sid, sc = os.path.basename(path)[:-5].rsplit(".", 1)
        with open(path) as f:
            rec = json.load(f)
        grp = group_of.get(sid, "?")
        slot = by_group.setdefault(grp, {"A": [], "B": [], "C": []})
        if "timeout_s" in rec or rec.get("running") is not None:
            n_timeout[sc] += 1
            slot[sc].append(None)  # timed out: excluded from percentiles,
            continue                # counted so A/B/C cardinality matches
        slot[sc].append(sum(rec.get("phases", {}).values()) / 1e3)
        if sc == "B" and "store_bytes" in rec:
            per_store.append(rec["store_bytes"])

    for sid_store in g.glob(os.path.join(out_dir, "stores", "*")):
        for root, _dirs, files in os.walk(sid_store):
            for name in files:
                if not name.endswith(".bin"):
                    continue
                ns = os.path.basename(root)
                try:
                    size = os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
                dedup[(ns, name)] = size

    print("\n== compile scenarios (ms/schema totals; fresh process each; "
          "timeouts excluded from percentiles, counted below)")
    hdr = (f"{'group':18s} {'n':>4s} " + " ".join(
        f"{sc + '-' + q:>10s}" for sc in ("C", "A", "B")
        for q in ("p50", "p99", "max")))
    print(hdr)
    for grp, slots in sorted(by_group.items()):
        n = len(slots["C"])
        row = f"{grp:18s} {n:4d} "
        for sc in ("C", "A", "B"):
            xs = [x for x in slots[sc] if x is not None]
            row += (f" {_pct(xs, 50):9.1f} {_pct(xs, 99):9.1f} "
                    f"{max(xs) if xs else float('nan'):9.1f}")
        print(row)
    print(f"timeouts/incomplete per scenario: {n_timeout}")
    if per_store:
        print(f"\nper-schema store size: p50 {_pct([float(x) for x in per_store], 50) / 1024:.0f} KiB, "
              f"max {max(per_store) / 1024 / 1024:.1f} MiB")
    if dedup:
        by_ns: dict[str, int] = {}
        for (ns, _), size in dedup.items():
            by_ns[ns] = by_ns.get(ns, 0) + size
        total = sum(dedup.values())
        print(f"deduplicated deployment footprint ({len(dedup)} unique "
              f"entries): {total / 1024 / 1024:.1f} MiB "
              + " ".join(f"{ns}={sz / 1024 / 1024:.1f}M" for ns, sz in sorted(by_ns.items())))

    for leg in ("serve", "redeploy-warm", "redeploy-cold"):
        path = os.path.join(out_dir, f"tbm.{leg}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rec = json.load(f)
        if leg == "serve":
            print(f"\n== journal TBM protocol (spider dialect + lexicons)\n"
                  f"serve: build {rec.get('build_ms')} ms, journal "
                  f"{rec.get('journal_after')}", flush=True)
            continue
        first = rec.get("first_request_us", [])
        later = rec.get("later_requests_us", [])
        wu = rec.get("warmup", {})
        print(f"{leg}: build {rec.get('build_ms')} ms; warmup tier_i={wu.get('tier_i')} "
              f"tier_ii={wu.get('tier_ii')} wall={rec.get('warmup_wall_ms')} ms; "
              f"first-request mask us p50={_pct(first, 50):.0f} "
              f"p99={_pct(first, 99):.0f} max={max(first) if first else float('nan'):.0f}; "
              f"later p50={_pct(later, 50):.0f} max={max(later) if later else float('nan'):.0f}")

    print("\npage-cache note: warm-cache B (same host, files just written). "
          "Post-purge B is a separate sensitivity run (macOS: `purge`).")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        compile_child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--tbm-child":
        tbm_child(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--group", action="append", default=[],
                        help="name:timeout_s:limit:stride")
        ap.add_argument("--out", required=True)
        ap.add_argument("--tbm", action="store_true")
        ap.add_argument("--summarize-only", action="store_true")
        args = ap.parse_args()
        if args.summarize_only:
            summarize(args.out)
        else:
            parent(args.group, args.out, args.tbm)
