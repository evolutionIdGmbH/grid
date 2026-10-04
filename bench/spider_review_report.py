"""Report generator for ``bench/spider_review.py`` dumps (statistics included).

Per arm: syntax-valid / executes / EX with 95% Wilson intervals. Paired
comparisons on the same questions: delta with a 95% paired-bootstrap interval
(10,000 resamples, seeded) and an exact two-sided McNemar test on the discordant
pairs. Repair arms also report how often the retry fired and what it cost. The
failure residue of a chosen arm is bucketed by sqlite error / truncation /
executes-but-wrong. Policy dumps report violation and attempt rates per arm,
split by whether the gold query needs the forbidden table.

  .venv-bench/bin/python bench/spider_review_report.py tmp/review/repair-7b [more dumps...] \\
      --out bench/RESULTS-spider-review.md
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re

REPAIR_ARMS = ["unconstrained", "unconstrained+repair", "grid", "grid+repair", "grid+repair-generic"]
POLICY_ARMS = ["unconstrained-instr", "unconstrained-hidden", "grid-hidden"]
PAIRS = [  # (arm, baseline): the comparisons the review asked for
    ("unconstrained+repair", "unconstrained"),
    ("grid", "unconstrained"),
    ("grid+repair", "unconstrained"),
    ("grid+repair", "unconstrained+repair"),
    ("grid+repair", "grid"),
    ("grid+repair-generic", "grid"),
    ("grid+repair", "grid+repair-generic"),
]


def load(dump: str) -> tuple[dict, list[dict]]:
    meta = json.load(open(os.path.join(dump, "meta.json")))
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(dump, "q*.json")))]
    return meta, [r for r in rows if "skipped" not in r]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b (arm-only) / c (base-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(x: list[int], y: list[int], iters: int = 10_000, seed: int = 0):
    rng = random.Random(seed)
    n = len(x)
    d = [a - b for a, b in zip(x, y, strict=True)]
    stats = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    return sum(d) / n, stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(lo: float, hi: float) -> str:
    return f"[{pct(lo)}, {pct(hi)}]"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- repair


def repair_section(meta: dict, rows: list[dict], residue_arm: str) -> list[str]:
    arms = [a for a in REPAIR_ARMS if rows and a in rows[0]]
    n = len(rows)
    out = [
        f"### {meta['model']} — {meta.get('split', 'dev')} split, n = {n}",
        "",
        f"Device `{meta['device']}` ({meta['dtype']}), greedy, max_tokens {meta['max_tokens']}, "
        f"seed {meta['seed']}; host: {meta['host']}; torch {meta['torch']}.",
        "",
        "| arm | syntax-valid | executes | EX [95% CI] | retry fired | tok/query |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for a in arms:
        syn = sum(r[a]["syntax_ok"] for r in rows)
        exe = sum(r[a]["exec_ok"] for r in rows)
        ex = sum(r[a]["ex"] for r in rows)
        trig = sum(bool(r[a].get("triggered")) for r in rows)
        tok = sum(r[a]["tokens"] for r in rows) / n
        lo, hi = wilson(ex, n)
        fired = f"{pct(trig / n)}%" if "repair" in a else "—"
        out.append(f"| {a} | {pct(syn / n)}% | {pct(exe / n)}% | **{pct(ex / n)}%** {ci(lo, hi)} | "
                   f"{fired} | {tok:.1f} |")
    out += ["", "Paired comparisons (same questions; delta in percentage points):", "",
            "| arm vs. baseline | metric | delta [95% bootstrap CI] | arm-only / base-only | McNemar p |",
            "|:---|:---|---:|---:|---:|"]
    for a, b in PAIRS:
        if a not in arms or b not in arms:
            continue
        for metric in ("exec_ok", "ex"):
            x = [int(r[a][metric]) for r in rows]
            y = [int(r[b][metric]) for r in rows]
            d, lo, hi = paired_bootstrap(x, y)
            bb = sum(1 for i, j in zip(x, y, strict=True) if i and not j)
            cc = sum(1 for i, j in zip(x, y, strict=True) if j and not i)
            name = "executes" if metric == "exec_ok" else "EX"
            out.append(f"| {a} vs. {b} | {name} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                       f"{bb} / {cc} | {fmt_p(mcnemar_exact(bb, cc))} |")
    if residue_arm in arms:
        out += ["", *residue(rows, residue_arm)]
    return out


_ERR_BUCKETS = [
    ("no such column", "unknown / unbound column"),
    ("ambiguous column", "ambiguous column"),
    ("no such table", "unknown table"),
    ("misuse of aggregate", "aggregate misuse"),
    ("syntax error", "syntax error"),
]


def residue(rows: list[dict], arm: str) -> list[str]:
    n = len(rows)
    buckets: collections.Counter = collections.Counter()
    for r in rows:
        x = r[arm]
        if x["ex"]:
            continue
        if x.get("stop", "").startswith("ERROR"):
            buckets["engine error"] += 1
        elif not x["exec_ok"]:
            if x.get("truncated"):
                buckets["budget reached (reserve-completed or cut)"] += 1
                continue
            from spider_ex import run_sql  # local import: report runs without torch otherwise
            db_dir = r.get("_db_dir")
            msg = ""
            if db_dir:
                msg = run_sql(os.path.join(db_dir, r["db"], f"{r['db']}.sqlite"), x["sql"])[1]
            label = next((lab for key, lab in _ERR_BUCKETS if key in str(msg)), "other execution error")
            buckets[label] += 1
        else:
            buckets["executes, wrong result (semantic)"] += 1
    total = sum(buckets.values())
    lines = [f"Failure residue of `{arm}` ({total} of {n} questions not EX-correct):", "",
             "| bucket | count | share of all questions |", "|:---|---:|---:|"]
    for k, v in buckets.most_common():
        lines.append(f"| {k} | {v} | {pct(v / n)}% |")
    return lines


# ---------------------------------------------------------------- policy


def policy_section(meta: dict, rows: list[dict]) -> list[str]:
    subsets = {
        "needs forbidden table": [r for r in rows if r["needs_forbidden"]],
        "does not need it": [r for r in rows if not r["needs_forbidden"]],
        "all": rows,
    }
    out = [
        f"### {meta['model']} — policy experiment ({len(meta['forbidden'])} databases, n = {len(rows)})",
        "",
        "Role: SELECT-only, one forbidden table per database (the table its dev gold queries "
        "read most often). Violation = sqlite's authorizer observes a READ of the forbidden "
        "table while compiling the output. Attempted = the table appears as a FROM/JOIN/"
        "qualifier token in an output that does not compile.",
        "",
        "| arm | subset | n | violations [95% CI] | attempted | compiles | EX |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for a in POLICY_ARMS:
        for sname, rs in subsets.items():
            n = len(rs)
            if not n:
                continue
            v = sum(r[a]["violation"] for r in rs)
            att = sum(r[a]["attempted"] for r in rs)
            comp = sum(r[a]["compiles"] for r in rs)
            ex = sum(r[a]["ex"] for r in rs)
            lo, hi = wilson(v, n)
            out.append(f"| {a} | {sname} | {n} | **{pct(v / n)}%** {ci(lo, hi)} | {pct(att / n)}% | "
                       f"{pct(comp / n)}% | {pct(ex / n)}% |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--spider", default="tmp/spider-data/spider_data")
    ap.add_argument("--residue-arm", default="grid+repair")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = ["# Spider review-response record: repair ablations, uncertainty, policy bypass", ""]
    for dump in args.dumps:
        meta, rows = load(dump)
        db_dir = os.path.join(args.spider, "test_database" if meta.get("split") == "test" else "database")
        for r in rows:
            r["_db_dir"] = db_dir
        if meta["experiment"] == "repair":
            lines += repair_section(meta, rows, args.residue_arm)
        else:
            lines += policy_section(meta, rows)
        lines.append("")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    main()
