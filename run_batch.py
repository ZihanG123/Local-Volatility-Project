"""
run_batch.py
============

Run the CMA-ES sigma-only optimization on a list of days from `processed_data/`
and save the final V/C'' = sigma^2 curves to a single results file.

Typical use:
    python3 run_batch.py                       # default: 6 representative days
    python3 run_batch.py --days 20110103 20110207 20110315
    python3 run_batch.py --all                 # every day in processed_data/

Output: results_<timestamp>.json with one entry per day containing
    - day, S0, R1, R2
    - K grid (fine), V/C'' baseline, V/C'' optimized
    - in-range jump stats (max, mean) before and after
    - masked J before and after
    - feasibility, runtime
This is the file `plot_overlay.py` reads.

Design notes
------------
* Each day's CMA-ES is a 4-stage schedule by default -- shorter than the
  7-stage research run in the notebook, because we're now testing across
  ~6 days rather than tuning one. You can pass --full to use the 7-stage.
* No per-stage plots are saved; only the final V/C'' curve per day.
* Days are run sequentially; the script writes the JSON incrementally so a
  crash mid-run leaves valid partial results.
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

# Silence the harmless exp-overflow warnings from the LVG bracketing
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(over="ignore", invalid="ignore", divide="ignore")

# Import the optimization module (must be alongside this script)
import lvg_optim as L


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SCHEDULE = [
    # (initial_step, max_generations) -- shorter than the notebook's 7-stage,
    # enough to converge well per day. Use --full for the long schedule.
    (0.30, 250),
    (0.10, 250),
    (0.05, 350),
    (0.02, 400),
]

# Long schedule: matches the proven 7-stage from the old auto-RP notebook.
# Step sizes shrink geometrically while generations GROW -- the tail spends
# most of the compute at very small step sizes refining the last few percent.
FULL_SCHEDULE = [
    (0.30,   300),
    (0.10,   300),
    (0.05,   300),
    (0.01,   500),
    (0.005,  500),
    (0.001,  1000),
    (0.0005, 1000),
]

# Multi-start noise-refinement: after the main schedule, perturb theta with
# tiny Gaussian noise and run a fine CMA-ES from each perturbation. Accept
# only if it improves. The noise lets it escape narrow local minima that
# the deterministic descent settled into. Disabled by default; use --refine.
N_REFINE_TRIALS    = 20
REFINE_NOISE_SCALE = 1e-4
REFINE_STEP_SIZE   = 1e-4
REFINE_MAX_GEN     = 1000

PENALTY_WEIGHT = 1e8
N_GRID_OVERLAY = 400        # points on the fine K grid for V/C''(K) overlay
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
# Per-day pipeline
# ---------------------------------------------------------------------------

def load_day(data_root: str, day: str):
    """Load Quotes.csv, theta_init_data.json, summary.json for one day."""
    folder = os.path.join(data_root, day)
    with open(os.path.join(folder, "Quotes.csv")) as f:
        rows = [(float(r[0]), float(r[1]), float(r[2])) for r in csv.reader(f)]
    strikes = np.array([r[0] for r in rows])
    asks    = np.array([r[1] for r in rows])
    bids    = np.array([r[2] for r in rows])
    with open(os.path.join(folder, "theta_init_data.json")) as f:
        d = json.load(f)
    with open(os.path.join(folder, "summary.json")) as f:
        s = json.load(f)

    R1 = len(d["initial_nus1"])
    R2 = len(d["initial_nus2"])
    S0 = float(d["S0"])
    Kbar = float(d["Kbar"])

    theta = np.concatenate([
        np.asarray(d["initial_nus1"], float),
        L.inv_softplus(np.asarray(d["initial_sigs1"], float)),
        np.asarray(d["initial_nus2"], float),
        L.inv_softplus(np.asarray(d["initial_sigs2"], float)),
    ])
    return {
        "theta": theta, "R1": R1, "R2": R2, "S0": S0, "Kbar": Kbar,
        "strikes": strikes, "bids": bids, "asks": asks,
        "summary": s,
    }


def sample_VCpp_on_grid(theta, R1, R2, S0, K_grid):
    """Evaluate V/C'' = sigma^2(K) at points K_grid (in absolute strike).

    sigma^2 is piecewise constant: between nu_j and nu_{j+1} it equals
    sigma_j^2. We just look up which interval each K falls in.
    """
    sig1 = L.softplus(theta[R1:2*R1]) + 1e-6
    sig2 = L.softplus(theta[2*R1+R2:2*R1+2*R2]) + 1e-6
    nus1 = theta[:R1]                       # left wing nodes (ascending, < S0)
    nus2 = theta[2*R1:2*R1+R2]              # right wing nodes (ascending, > S0)

    # sigma at K:  left wing intervals are (nus1[j], nus1[j+1]) with sigma sig1[j]
    #              for K in (nus1[-1], nus2[0])  use bridge: pick sig1[-1] for K<=S0, sig2[0] otherwise
    #              right wing intervals are (nus2[j], nus2[j+1]) with sigma sig2[j]
    out = np.empty_like(K_grid, dtype=float)
    for i, K in enumerate(K_grid):
        if K <= nus1[0]:
            out[i] = sig1[0]**2
        elif K < nus1[-1]:
            j = int(np.searchsorted(nus1, K) - 1)
            out[i] = sig1[max(j, 0)]**2
        elif K < S0:
            out[i] = sig1[-1]**2
        elif K < nus2[0]:
            out[i] = sig2[0]**2
        elif K >= nus2[-1]:
            out[i] = sig2[-1]**2
        else:
            j = int(np.searchsorted(nus2, K) - 1)
            out[i] = sig2[max(j, 0)]**2
    return out


def _node_positions(theta, R1, R2, S0):
    nus1 = theta[:R1]; nus2 = theta[2*R1:2*R1+R2]
    return np.concatenate([nus1[1:R1], [S0], nus2[0:R2-1]])


def _jump_stats(theta, R1, R2, S0, K_lo, K_hi):
    """Max and mean |Δσ²| over nodes inside [K_lo, K_hi]."""
    j = L._sigma2_jump_vector(theta, R1, R2)
    pos = _node_positions(theta, R1, R2, S0)
    m = (pos >= K_lo) & (pos <= K_hi)
    if not np.any(m):
        return 0.0, 0.0
    a = np.abs(j[m])
    return float(np.max(a)), float(np.mean(a))


def optimize_one_day(d: dict, schedule: list, verbose: bool = False,
                     mask_moneyness_lo: float = 0.85,
                     mask_moneyness_hi: float = 1.10,
                     refine: bool = False) -> dict:
    """Run the CMA-ES schedule on one day's calibration. Returns final result.

    Mask: by default we restrict the smoothness objective to nodes whose strike
    is within [mask_moneyness_lo * S0, mask_moneyness_hi * S0]. Real SPX data
    quotes strikes far below ATM (e.g. K=300 when S0=1262) where sigma^2 has
    to be enormous to fit the deep-ITM intrinsic value -- including those in
    the objective lets one giant jump dominate everything else. The band
    [0.85, 1.10] focuses on the part of the smile that traders actually use.

    refine: if True, after the main schedule run N_REFINE_TRIALS additional
    short CMA-ES calls from tiny-noise-perturbed starting points (step size
    REFINE_STEP_SIZE, REFINE_MAX_GEN generations each). Each trial is
    accepted only if it improves over the current best. This is the
    multi-start phase from the old auto-RP notebook -- it lets the
    optimizer escape narrow local minima the deterministic descent
    settled into. Adds ~N_REFINE_TRIALS * a few seconds per day.
    """
    theta_init = d["theta"]
    R1, R2, S0, Kbar = d["R1"], d["R2"], d["S0"], d["Kbar"]
    strikes, bids, asks = d["strikes"], d["bids"], d["asks"]
    # mask in absolute strike, intersected with available quote range
    K_lo = max(float(strikes.min()), mask_moneyness_lo * S0)
    K_hi = min(float(strikes.max()), mask_moneyness_hi * S0)

    Jb_masked = L.calculate_J_smooth(theta_init, R1, R2, S0, K_lo, K_hi)
    max_b, mean_b = _jump_stats(theta_init, R1, R2, S0, K_lo, K_hi)

    t0 = time.time()
    current = theta_init.copy()
    for stage, (step, n_gen) in enumerate(schedule, start=1):
        res = L.cmaes_optimize_sigmas(
            theta_init=current, R1=R1, R2=R2, S0=S0, Kbar=Kbar,
            theta_baseline=current,
            market_strikes=strikes, bid_prices=bids, ask_prices=asks,
            penalty_weight=PENALTY_WEIGHT, strike_lo=K_lo, strike_hi=K_hi,
            objective_mode="L2",
            initial_step_size=step, max_generations=n_gen,
            convergence_tol=1e-10, random_seed=RANDOM_SEED + stage,
            verbose=verbose, save_plot=False,
        )
        current = res["best_theta"]
    elapsed_schedule = time.time() - t0

    # ── Multi-start noise-refinement phase ─────────────────────────────────
    # After the deterministic schedule, perturb theta with tiny noise and run
    # a fine CMA-ES from each perturbation. Accept only if it improves. This
    # is what shook tier-2 days loose in the old auto-RP run.
    refine_trials_accepted = 0
    if refine:
        t1 = time.time()
        J_best = L.calculate_J_smooth(current, R1, R2, S0, K_lo, K_hi)
        best_theta_so_far = current.copy()
        rng = np.random.default_rng(RANDOM_SEED + 9000)
        for trial in range(N_REFINE_TRIALS):
            noise = rng.standard_normal(len(current)) * REFINE_NOISE_SCALE
            theta_perturbed = current + noise
            res_t = L.cmaes_optimize_sigmas(
                theta_init=theta_perturbed, R1=R1, R2=R2, S0=S0, Kbar=Kbar,
                theta_baseline=current,
                market_strikes=strikes, bid_prices=bids, ask_prices=asks,
                penalty_weight=PENALTY_WEIGHT, strike_lo=K_lo, strike_hi=K_hi,
                objective_mode="L2",
                initial_step_size=REFINE_STEP_SIZE,
                max_generations=REFINE_MAX_GEN,
                convergence_tol=1e-10, random_seed=10_000 + trial,
                verbose=False, save_plot=False,
            )
            J_trial = L.calculate_J_smooth(res_t["best_theta"], R1, R2, S0, K_lo, K_hi)
            if J_trial < J_best:
                J_best = J_trial
                best_theta_so_far = res_t["best_theta"].copy()
                refine_trials_accepted += 1
        current = best_theta_so_far
        elapsed_refine = time.time() - t1
        elapsed = elapsed_schedule + elapsed_refine
        if verbose:
            print(f"  refine: {refine_trials_accepted}/{N_REFINE_TRIALS} trials improved "
                  f"({elapsed_refine:.1f}s)")
    else:
        elapsed = elapsed_schedule

    Jf_masked = L.calculate_J_smooth(current, R1, R2, S0, K_lo, K_hi)
    max_f, mean_f = _jump_stats(current, R1, R2, S0, K_lo, K_hi)
    feasible = L.is_feasible(current, R1, R2, S0, Kbar, strikes, bids, asks)

    # sample V/C'' on a common fine grid in moneyness K/S0
    moneyness = np.linspace(K_lo / S0, K_hi / S0, N_GRID_OVERLAY)
    K_grid = moneyness * S0
    vcpp_base = sample_VCpp_on_grid(theta_init, R1, R2, S0, K_grid)
    vcpp_opt  = sample_VCpp_on_grid(current,    R1, R2, S0, K_grid)

    return {
        "S0":           S0,
        "Kbar":         Kbar,
        "R1":           R1,
        "R2":           R2,
        "K_lo":         K_lo,
        "K_hi":         K_hi,
        "K_grid":       K_grid.tolist(),           # absolute strike (for plotting)
        "moneyness":    moneyness.tolist(),        # K / S0 (kept for reference)
        "vcpp_baseline":  vcpp_base.tolist(),      # raw sigma^2
        "vcpp_optimized": vcpp_opt.tolist(),
        "J_masked_baseline":  float(Jb_masked),
        "J_masked_optimized": float(Jf_masked),
        "improvement_pct":    100.0*(Jb_masked - Jf_masked)/Jb_masked if Jb_masked > 0 else 0.0,
        "max_jump_baseline":   max_b,
        "mean_jump_baseline":  mean_b,
        "max_jump_optimized":  max_f,
        "mean_jump_optimized": mean_f,
        "feasible":     bool(feasible),
        "elapsed_sec":  float(elapsed),
        "refine_trials_accepted": int(refine_trials_accepted),
    }


# ---------------------------------------------------------------------------
# Day selection
# ---------------------------------------------------------------------------

def list_available_days(data_root: str) -> list[str]:
    """All YYYYMMDD subfolders under data_root that have the three required files."""
    if not os.path.isdir(data_root):
        return []
    days = []
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if not os.path.isdir(d):
            continue
        if not name.isdigit() or len(name) != 8:
            continue
        if all(os.path.isfile(os.path.join(d, f))
               for f in ("Quotes.csv", "theta_init_data.json", "summary.json")):
            days.append(name)
    return days


def pick_representative(days: list[str], n: int) -> list[str]:
    """Evenly sample `n` days from the sorted list, always including endpoints."""
    if n >= len(days):
        return days
    if n <= 1:
        return [days[0]]
    idx = np.linspace(0, len(days) - 1, n).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    return [days[i] for i in idx]


def pick_random(days: list[str], n: int, seed: int | None = None) -> list[str]:
    """Randomly sample `n` days, sorted by date. Reproducible if seed given."""
    if n >= len(days):
        return days
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(days), size=n, replace=False)
    return sorted(days[i] for i in idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="processed_data",
                   help="folder containing per-day subfolders (default: processed_data)")
    p.add_argument("--out", default=None,
                   help="output JSON file (default: results_<timestamp>.json)")
    p.add_argument("--days", nargs="+", default=None,
                   help="explicit list of YYYYMMDD days to run")
    p.add_argument("--all", action="store_true",
                   help="run all available days")
    p.add_argument("--n", type=int, default=6,
                   help="if not --all/--days, run this many representative days (default: 6)")
    p.add_argument("--random", action="store_true",
                   help="pick --n days randomly (default: evenly spaced)")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for --random (default: nondeterministic)")
    p.add_argument("--full", action="store_true",
                   help="use the 7-stage long schedule instead of the 4-stage default")
    p.add_argument("--refine", action="store_true",
                   help=f"after the main schedule, run {N_REFINE_TRIALS} noise-perturbed "
                        f"refinement trials per day (step size {REFINE_STEP_SIZE}, "
                        f"{REFINE_MAX_GEN} gens each); accept only if improved. "
                        f"Adds ~30-60s/day. Helps tier-2 days escape local minima.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    available = list_available_days(args.data_root)
    if not available:
        print(f"ERROR: no day folders found under {args.data_root}/", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(available)} available days: "
          f"{available[0]} .. {available[-1]}")

    if args.days:
        days = [d for d in args.days if d in available]
        missing = set(args.days) - set(days)
        if missing:
            print(f"WARNING: missing day folders: {sorted(missing)}")
    elif args.all:
        days = available
    else:
        if args.random:
            days = pick_random(available, args.n, seed=args.seed)
        else:
            days = pick_representative(available, args.n)

    schedule = FULL_SCHEDULE if args.full else DEFAULT_SCHEDULE
    refine_str = f" + {N_REFINE_TRIALS}-trial refinement" if args.refine else ""
    print(f"running {len(days)} days with {'FULL' if args.full else 'default'} "
          f"schedule ({len(schedule)} stages){refine_str}:")
    for d in days:
        print(f"  {d}")

    out_path = args.out or f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results = {"days": {}, "config": {
        "data_root": args.data_root,
        "schedule": [list(s) for s in schedule],
        "penalty_weight": PENALTY_WEIGHT,
        "n_grid_overlay": N_GRID_OVERLAY,
        "random_seed": RANDOM_SEED,
    }}

    t_start = time.time()
    for k, day in enumerate(days, start=1):
        print(f"\n[{k:>3}/{len(days)}] day {day} ...", flush=True)
        try:
            d = load_day(args.data_root, day)
            r = optimize_one_day(d, schedule, verbose=args.verbose, refine=args.refine)
            results["days"][day] = r
            print(f"  J masked: {r['J_masked_baseline']:.2e} -> {r['J_masked_optimized']:.2e}  "
                  f"({r['improvement_pct']:+.2f}%)  feasible={r['feasible']}  "
                  f"elapsed={r['elapsed_sec']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results["days"][day] = {"error": f"{type(e).__name__}: {e}"}

        # incremental save so a crash mid-run doesn't lose work
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    total = time.time() - t_start
    n_ok = sum(1 for d, r in results["days"].items() if "error" not in r)
    print(f"\nDONE: {n_ok}/{len(days)} days succeeded in {total/60:.1f} min")
    print(f"results saved to {out_path}")


if __name__ == "__main__":
    main()
