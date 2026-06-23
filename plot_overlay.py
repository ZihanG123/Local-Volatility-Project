"""
plot_overlay.py
===============

Make presentation plots from a results JSON produced by run_batch.py.

Output: a grid of subplots, one panel per day. Each panel shows V/C'' = sigma^2
on the y-axis against the absolute strike K on the x-axis, with the baseline
calibration and the CMA-ES optimized curve overlaid. Each panel marks the spot
S0 with a dashed vertical line.

Usage:
    python3 plot_overlay.py results_<timestamp>.json
    python3 plot_overlay.py results_<timestamp>.json --out-dir figures
    python3 plot_overlay.py results_<timestamp>.json --cols 6
"""

import argparse
import csv as _csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str):
    with open(path) as f:
        blob = json.load(f)
    out = []
    for day, data in blob["days"].items():
        if "error" in data:
            print(f"  skipping {day} (error: {data['error']})")
            continue
        if "K_grid" not in data and "moneyness" not in data:
            continue
        out.append((day, data))
    return sorted(out), blob.get("config", {})


def plot_per_day_grid(results, out_path, ncols: int = 5, ymax_pct: float = 98.0):
    n = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.0 * ncols, 2.3 * nrows),
                              squeeze=False)
    for k, (day, data) in enumerate(results):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        if "K_grid" in data:
            K = np.asarray(data["K_grid"])
        else:
            K = np.asarray(data["moneyness"]) * float(data["S0"])
        base = np.asarray(data["vcpp_baseline"])
        opt  = np.asarray(data["vcpp_optimized"])
        ax.plot(K, base, color="tomato",    lw=0.9, alpha=0.75, label="baseline")
        ax.plot(K, opt,  color="royalblue", lw=1.2, alpha=0.95, label="optimized")
        ax.axvline(data["S0"], color="gray", ls="--", lw=0.8, alpha=0.7)
        both = np.concatenate([base, opt])
        cap = float(np.percentile(both, ymax_pct))
        floor = float(both.min())
        ax.set_ylim(max(0, floor * 0.9), cap * 1.05)
        s = day
        date_str = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        impr = data.get("improvement_pct", 0.0)
        ax.set_title(f"{date_str}   J {impr:+.1f}%", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(loc="upper right", fontsize=6)
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Strike $K$", fontsize=9)
    for row in axes:
        row[0].set_ylabel(r"$V/C''(K) = \sigma^2$", fontsize=9)
    fig.suptitle(f"CMA-ES local-variance smoothing  -  {n} SPX days",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary_csv(results, out_path):
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["day", "S0", "R1", "R2",
                    "J_masked_baseline", "J_masked_optimized", "improvement_pct",
                    "max_jump_baseline", "max_jump_optimized",
                    "mean_jump_baseline", "mean_jump_optimized",
                    "feasible", "elapsed_sec"])
        for day, d in results:
            w.writerow([
                day, f"{d['S0']:.2f}", d['R1'], d['R2'],
                f"{d['J_masked_baseline']:.6e}",
                f"{d['J_masked_optimized']:.6e}",
                f"{d['improvement_pct']:.2f}",
                f"{d['max_jump_baseline']:.4f}",
                f"{d['max_jump_optimized']:.4f}",
                f"{d['mean_jump_baseline']:.4f}",
                f"{d['mean_jump_optimized']:.4f}",
                d['feasible'],
                f"{d.get('elapsed_sec', 0):.1f}",
            ])
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", help="results JSON from run_batch.py")
    p.add_argument("--out-dir", default="figures",
                   help="directory for output figures (default: figures/)")
    p.add_argument("--cols", type=int, default=5,
                   help="number of columns in the panel grid (default: 5)")
    args = p.parse_args()
    results, _ = load_results(args.results)
    if not results:
        print("no successful days in results file", file=sys.stderr)
        sys.exit(1)
    print(f"loaded {len(results)} days ({results[0][0]} .. {results[-1][0]})")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_per_day_grid(results, os.path.join(args.out_dir, "vcpp_grid.png"),
                       ncols=args.cols)
    write_summary_csv(results, os.path.join(args.out_dir, "summary.csv"))
    print("done.")


if __name__ == "__main__":
    main()
