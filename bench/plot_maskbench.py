"""Render the MaskBench hero chart (TBM avg + TTFM avg, log-scale us) from the
runner's output dirs — the same style as guidance-ai/maskbench's plots/hero.png.

Run:  .venv-bench/bin/python bench/plot_maskbench.py tmp/mb-grid tmp/mb-llg tmp/mb-xgr \\
          --out bench/maskbench.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from maskbench_grid import aggregate  # noqa: E402

COLORS = {
    "GRID": "#d95f02",
    "llguidance": "#111111",
    "XGrammar (compliant)": "#26b3cd",
}
FALLBACK = ["#7570b3", "#1b9e77", "#e7298a"]


def human(us: float) -> str:
    if us < 1_000:
        return f"{us:.0f}µs"
    if us < 1_000_000:
        v = us / 1_000
        return f"{v:.1f}ms" if v < 10 else f"{v:.0f}ms"
    v = us / 1_000_000
    return f"{v:.1f}s" if v < 10 else f"{v:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="runner output dirs (one per engine)")
    ap.add_argument("--out", default="bench/maskbench.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggs = [aggregate(d) for d in args.dirs]
    metrics = [("TBM avg", "tbm_avg"), ("TTFM avg", "ttfm_avg")]

    fig, ax = plt.subplots(figsize=(13, 3.2 + 0.42 * len(aggs)), dpi=150)
    bar_h, group_gap = 0.8, 1.2
    yticks, ylabels = [], []
    xmax = 0.0
    y = 0.0
    for label, key in metrics:
        vals = [a[key] for a in aggs]
        best = min(vals)
        ys = [y - i * bar_h for i in range(len(aggs))]
        for i, (a, v) in enumerate(zip(aggs, vals, strict=True)):
            color = COLORS.get(a["engine"], FALLBACK[i % len(FALLBACK)])
            ax.barh(ys[i], v, height=bar_h * 0.92, color=color, zorder=3)
            mult = v / best
            mult_s = "1x" if mult < 1.005 else (f"{mult:.1f}x" if mult < 10 else f"{mult:,.0f}x")
            ax.text(v / 1.12, ys[i], mult_s, va="center", ha="right",
                    color="white", fontsize=10, fontweight="bold", zorder=4)
            ax.text(v * 1.12, ys[i], human(v), va="center", ha="left",
                    color="black", fontsize=10, zorder=4)
            xmax = max(xmax, v)
        yticks.append(sum(ys) / len(ys))
        ylabels.append(label)
        y = ys[-1] - bar_h - group_gap

    ax.set_xscale("log")
    ax.set_xlim(min(a[k] for a in aggs for _, k in metrics) / 2.2, xmax * 6)
    ax.set_yticks(yticks, ylabels, fontsize=12)
    ax.set_xlabel("Time (log scale, microseconds)", fontsize=12)
    ax.set_title("Time To First Mask (TTFM) and Time Between Masks (TBM)", fontsize=14)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", which="major", color="#dddddd", zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=COLORS.get(a["engine"], FALLBACK[i % len(FALLBACK)]))
               for i, a in enumerate(aggs)]
    ax.legend(handles, [a["engine"] for a in aggs], loc="upper right",
              fontsize=11, framealpha=0.95)

    meta = f"{aggs[0]['schemas']} schemas, stratified MaskBench sample | local dev host (unpinned)"
    fig.text(0.99, 0.01, meta, ha="right", va="bottom", fontsize=8, color="#777777")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"chart -> {args.out}")


if __name__ == "__main__":
    main()
