"""GRID_PERF_HASHCONS corpus differential gate (0.3.x candidate #17).

Compares compile_schema flag-off vs flag-on over jsb-src schemas: outcome
bucket (compiled / Unsupported message / exception class), recorded set, and
.grid text — byte-equal, else start-anchored rule isomorphism (DAG sharing
lets the compiler's id() memo skip duplicate rule families legacy built and
deduped, renumbering later rules; language and masks are unchanged, which is
what the isomorphism check certifies). ANY divergence on a schema the legacy
path completes is a gate failure; legacy timeouts have no oracle and are
reported by their new-arm outcome (the sanctioned improvements).

Usage:
    python bench/perfbench/diff_hashcons.py --components norm,dedupe \
        --sets ttfm_capped,ttfm_tail_1pct,stratified_200,tbm_tail_100 \
        [--all-corpus N] [--timeout 20] [--jobs 4] --out tmp/diff-hashcons
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

MANIFEST = os.path.join(os.path.dirname(__file__), "manifest.json")
DATA_DIR = os.environ.get(
    "GRID_JSB_DATA",
    os.path.join(os.path.dirname(__file__), "..", "..", "tmp",
                 "jsb-src", "data"))


class _Timeout(BaseException):
    """SIGALRM payload; BaseException so no library except-clause eats it."""


def grammar_isomorphic(a: str, b: str) -> bool:
    """Start-anchored parallel-walk isomorphism modulo renaming/reordering
    that provably cannot change the language or the masks:

    - rules: bijective renaming, alts equal token-wise in order (DAG
      sharing renumbers the compiler's _n counter);
    - K/E literal terminals: renaming by pattern, order-insensitive within
      the block — distinct exact literals can never tie on a lexeme, and
      the emitter's fixed K < E < S < STRING block order keeps every
      cross-class (is_literal, decl_index) priority relation;
    - S (constrained-regex) terminals: patterns must match IN ORDER —
      overlapping regexes resolve ties by decl_index, so their order is
      load-bearing; names map positionally;
    - %-header and STRING/NUMBER/INT lines byte-equal."""
    pa, pb = _split_grammar(a), _split_grammar(b)
    if pa is None or pb is None:
        return False
    (head_a, k_a, e_a, s_a, rules_a, start_a) = pa
    (head_b, k_b, e_b, s_b, rules_b, start_b) = pb
    if head_a != head_b or len(rules_a) != len(rules_b):
        return False
    term_ren: dict[str, str] = {}
    for lits_a, lits_b in ((k_a, k_b), (e_a, e_b)):
        if len(lits_a) != len(lits_b):
            return False
        by_pat = {pat: name for name, pat in lits_b.items()}
        if len(by_pat) != len(lits_b):
            return False
        for name, pat in lits_a.items():
            got = by_pat.get(pat)
            if got is None:
                return False
            term_ren[name] = got
    if len(s_a) != len(s_b) or \
            [pat for _, pat in s_a] != [pat for _, pat in s_b]:
        return False
    for (na, _), (nb, _) in zip(s_a, s_b):
        term_ren[na] = nb

    ren = {start_a: start_b}
    queue = [(start_a, start_b)]
    while queue:
        na, nb = queue.pop()
        alts_a, alts_b = rules_a.get(na), rules_b.get(nb)
        if alts_a is None or alts_b is None or len(alts_a) != len(alts_b):
            return False
        for alt_a, alt_b in zip(alts_a, alts_b):
            ta, tb = alt_a.split(), alt_b.split()
            if len(ta) != len(tb):
                return False
            for x, y in zip(ta, tb):
                xr, yr = x in rules_a, y in rules_b
                if xr != yr:
                    return False
                if not xr:
                    if term_ren.get(x, x) != y:
                        return False
                    continue
                got = ren.get(x)
                if got is None:
                    ren[x] = y
                    queue.append((x, y))
                elif got != y:
                    return False
    # emission prunes to reachable-from-start, so the walk must cover
    # everything, and the renaming must be bijective
    return len(ren) == len(rules_a) and len(set(ren.values())) == len(rules_b)


def _split_grammar(src: str):
    """-> (fixed-header tuple, K name->pattern, E name->pattern,
    S [(name, pattern)] in emitted order, {rule: [alts]}, start) or None."""
    head: list[str] = []
    kmap: dict[str, str] = {}
    emap: dict[str, str] = {}
    slist: list[tuple[str, str]] = []
    rules: dict[str, list[str]] = {}
    start_target = None
    for line in src.splitlines():
        if line.startswith("start: "):
            start_target = line[len("start: "):].strip()
            continue
        name, sep, rhs = line.partition(": ")
        if not sep:
            head.append(line)
        elif rhs.startswith("/"):
            if name.startswith("%") or name in ("WS", "STRING", "NUMBER",
                                                "INT"):
                head.append(line)
            elif name.startswith("K") and name[1:].isdigit():
                kmap[name] = rhs
            elif name.startswith("E") and name[1:].isdigit():
                emap[name] = rhs
            else:                       # S<n> and NEVER: order-sensitive
                slist.append((name, rhs))
        elif name.startswith("%"):
            head.append(line)
        else:
            rules[name] = [alt.strip() for alt in rhs.split(" | ")]
    if start_target is None or start_target not in rules:
        return None
    return tuple(head), kmap, emap, slist, rules, start_target


def run_arm(schema, components: frozenset[str], timeout_s: int) -> dict:
    from grid.jsonschema.compiler import Unsupported, compile_schema

    def _alarm(*a):
        raise _Timeout()

    old_h = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    t0 = time.monotonic()
    try:
        src, recorded = compile_schema(schema, hashcons=components)
        return {"bucket": "ok", "src": src, "recorded": sorted(recorded),
                "secs": round(time.monotonic() - t0, 3)}
    except Unsupported as e:
        return {"bucket": "unsupported", "msg": str(e),
                "secs": round(time.monotonic() - t0, 3)}
    except _Timeout:
        return {"bucket": "timeout", "secs": timeout_s}
    except Exception as e:
        return {"bucket": type(e).__name__, "msg": str(e)[:200],
                "secs": round(time.monotonic() - t0, 3)}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_h)


def compare(schema_file: str, components: frozenset[str],
            timeout_s: int) -> dict:
    with open(schema_file) as f:
        schema = json.load(f)
    if isinstance(schema, dict) and "schema" in schema and "tests" in schema:
        schema = schema["schema"]
    old = run_arm(schema, frozenset(), timeout_s)
    new = run_arm(schema, components, timeout_s)
    rec: dict = {"file": schema_file,
                 "old_secs": old["secs"], "new_secs": new["secs"]}
    if old["bucket"] == "timeout":
        rec["status"] = f"old_timeout_new_{new['bucket']}"
        return rec
    if new["bucket"] != old["bucket"]:
        rec["status"] = "FLIP_bucket"
        rec["old"], rec["new"] = old["bucket"], new["bucket"]
        return rec
    if old["bucket"] == "ok":
        if old["recorded"] != new["recorded"]:
            rec["status"] = "FLIP_recorded"
            rec["old"], rec["new"] = old["recorded"], new["recorded"]
        elif old["src"] == new["src"]:
            rec["status"] = "equal"
        elif grammar_isomorphic(old["src"], new["src"]):
            rec["status"] = "iso"
        else:
            rec["status"] = "FLIP_text"
        return rec
    if old.get("msg") != new.get("msg"):
        rec["status"] = "FLIP_msg"
        rec["old"], rec["new"] = old.get("msg"), new.get("msg")
    else:
        rec["status"] = f"equal_{old['bucket']}"
    return rec


def worker(list_file: str, out_file: str, components: str,
           timeout_s: int) -> None:
    comps = frozenset(c for c in components.split(",") if c)
    with open(list_file) as f:
        files = json.load(f)
    with open(out_file, "w") as out:
        for sf in files:
            try:
                rec = compare(sf, comps, timeout_s)
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
    return os.path.join(DATA_DIR, "..", "maskbench", "data",
                        schema_id + ".json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("LIST", "OUT"))
    ap.add_argument("--components", default="norm,dedupe")
    ap.add_argument("--sets", default="")
    ap.add_argument("--all-corpus", type=int, default=0,
                    help="also add every Nth schema of the full data dir")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="tmp/diff-hashcons")
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1], args.components, args.timeout)
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
    if args.all_corpus:
        allf = []
        for split in sorted(os.listdir(DATA_DIR)):
            d = os.path.join(DATA_DIR, split)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".json"):
                    allf.append(os.path.join(d, fn))
        for p in allf[::args.all_corpus]:
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
        procs.append(subprocess.Popen(
            [sys.executable, __file__, "--worker", lf, of,
             "--components", args.components, "--timeout", str(args.timeout)]))
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
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
