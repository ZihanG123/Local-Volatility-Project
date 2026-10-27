"""
Runs the multi-maturity pipeline on one trading day and builds the surface.

Mirrors MultiMaturity.test_date_index, but returns the pieces instead of only
plotting, and keeps the time to expiry for each maturity.

    python3 run_surface.py                       # 3 maturities, date index 4
    python3 run_surface.py --n-mat 4 --index 10
    python3 run_surface.py --no-optimize         # surface from the raw calibration
"""

import argparse

import numpy as np
import pandas as pd

from calibration import load_theta
from data_utils import get_spx_close, process_date_exdate
from initial_point import find_initial_point, smart_K_union, find_bad_constraints
from optimization import coordinateDescent
import surface_lvg as S


def prepare_day(dataframe, index, n_mat):
    date_counts = dataframe.groupby("date")["exdate"].nunique()
    valid_dates = sorted(date_counts[date_counts > 0].index.tolist())
    target_date = valid_dates[index]
    exdates = sorted(dataframe[dataframe["date"] == target_date]["exdate"].unique())[:n_mat]
    S0 = get_spx_close(target_date)
    print(f"date {target_date}   S0 {S0:.2f}   maturities {list(exdates)}")

    bid_ask_compiled, strikes = [], []
    for exdate in exdates:
        K, C_l, C_h = process_date_exdate(dataframe, target_date, exdate, S0)
        bid_ask_compiled.append(np.column_stack([K, C_l, C_h]))
        strikes.append(K)

    eps, tolerance = 50, 1e-8
    while eps >= 0:
        K_array = smart_K_union(strikes, eps=eps)
        Kbar = np.max(K_array[:, -1]) + 20
        try:
            initial_point = find_initial_point(S0, K_array, bid_ask_compiled, Kbar)
            break
        except ValueError:
            if eps == 0:
                eps = -1
            eps = eps // 2
            if eps < 0:
                bid_ask_compiled, _, tolerance = find_bad_constraints(
                    S0, K_array, bid_ask_compiled, Kbar, tolerance)
                eps = 50
                strikes = [b[:, 0] for b in bid_ask_compiled]
    print(f"union grid {K_array.shape}   Kbar {Kbar:.1f}")

    sigs_nus_compiled, Ts = [], []
    prev_sigs, prev_nus = None, None
    d0 = pd.to_datetime(target_date, format="%Y%m%d")
    for i, exdate in enumerate(exdates):
        sigs, nus = load_theta(S0, initial_point[i, :], K_array[i, :], 0, Kbar,
                               prev_sigs=prev_sigs, prev_nus=prev_nus)
        prev_sigs, prev_nus = sigs, nus
        sigs_nus_compiled.append((sigs, nus))
        bd = len(pd.bdate_range(d0, pd.to_datetime(exdate, format="%Y%m%d"))) - 1
        Ts.append(bd / 252.0)
        print(f"  maturity {i+1}: {bd:>3} bdays  T={Ts[-1]:.4f}y  "
              f"{len(sigs)} sigmas  {len(nus)} nodes")

    return S0, Kbar, sigs_nus_compiled, bid_ask_compiled, Ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="SPX_opt_2011_2012.csv")
    ap.add_argument("--index", type=int, default=4)
    ap.add_argument("--n-mat", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--no-optimize", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--time-var", default="sqrt", choices=["T", "sqrt", "log"],
                    help="variable the cubic is fitted in")
    ap.add_argument("--slope-smooth", type=int, default=0,
                    help="strikes to average knot slopes over (0 = off)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    S0, Kbar, sn, ba, Ts = prepare_day(df, args.index, args.n_mat)

    klo = max(b[:, 0].min() for b in ba)
    khi = min(b[:, 0].max() for b in ba)

    print("\n--- RAW calibration: which time variable? ---")
    print(f"{'time var':<10}{'min d2C/dK2':>14}{'violations':>12}"
          f"{'min dC/dT':>14}{'knot jump':>12}")
    for tv in ("T", "sqrt", "log"):
        c = S.check_surface(S.build_surface(sn, S0, Ts, k_lo=klo, k_hi=khi,
                                            time_var=tv, slope_smooth=args.slope_smooth), verbose=False)
        print(f"{tv:<10}{c['min_d2CdK2']:>14.3e}"
              f"{c['n_bad_d2K']:>7}/{c['n_grid']:<5}"
              f"{c['min_dCdT']:>14.3e}{c['max_dCdT_jump_at_knots']:>12.2e}")

    print(f"\n--- surface from the RAW calibration (time_var={args.time_var}) ---")
    surf0 = S.build_surface(sn, S0, Ts, k_lo=klo, k_hi=khi, time_var=args.time_var,
                           slope_smooth=args.slope_smooth)
    chk0 = S.check_surface(surf0)
    print(f"  blend would give a jump of {S.compare_to_blend(surf0):.3e}")

    if args.no_optimize:
        return

    print(f"\n--- optimising ({args.iterations} iterations) ---")
    sn = coordinateDescent(S0, sn, ba, num_iterations=args.iterations,
                           num_indices=max(1, len(sn[0][0]) // 10))

    print("\n--- surface from the OPTIMISED smiles ---")
    surf1 = S.build_surface(sn, S0, Ts, k_lo=klo, k_hi=khi,
                            time_var=args.time_var,
                           slope_smooth=args.slope_smooth)
    chk1 = S.check_surface(surf1)

    print(f"\n{'':<24}{'raw':>14}{'optimised':>14}")
    for k in ("min_dCdT", "min_d2CdK2", "max_dCdT_jump_at_knots"):
        print(f"{k:<24}{chk0[k]:>14.4e}{chk1[k]:>14.4e}")

    if args.plot:
        import matplotlib.pyplot as plt
        S.plot_surface(surf1)
        plt.show()


if __name__ == "__main__":
    main()
