"""
Locates where the interpolated surface loses convexity in K.

    python3 diagnose_convexity.py --csv ../SPX_opt_2011_2012.csv --n-mat 3
"""

import argparse

import numpy as np
import pandas as pd

import surface_lvg as S
from optimization import coordinateDescent
from run_surface import prepare_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="SPX_opt_2011_2012.csv")
    ap.add_argument("--index", type=int, default=4)
    ap.add_argument("--n-mat", type=int, default=3)
    ap.add_argument("--time-var", default="sqrt", choices=["T", "sqrt", "log"])
    ap.add_argument("--slope-smooth", type=int, default=0,
                    help="strikes to average knot slopes over (0 = off)")
    ap.add_argument("--n-strikes", type=int, default=2400)
    ap.add_argument("--optimize", action="store_true",
                    help="smooth the smiles first; the violations sit on nodes "
                         "where sigma jumps, which is what smoothing removes")
    ap.add_argument("--iterations", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    S0, Kbar, sn, ba, Ts = prepare_day(df, args.index, args.n_mat)

    def sigma_jumps(sn):
        r = [max(max(s[i+1]/max(s[i], 1e-30), s[i]/max(s[i+1], 1e-30))
                 for i in range(len(s)-1)) for s, _ in sn]
        return r

    print(f"\nworst sigma ratio between adjacent intervals, per maturity: "
          f"{[f'{v:.1f}' for v in sigma_jumps(sn)]}")

    if args.optimize:
        print(f"optimising ({args.iterations} iterations)...")
        sn = coordinateDescent(S0, sn, ba, num_iterations=args.iterations,
                               num_indices=max(1, len(sn[0][0]) // 10))
        print(f"after:  {[f'{v:.1f}' for v in sigma_jumps(sn)]}")

    klo = max(b[:, 0].min() for b in ba)
    khi = min(b[:, 0].max() for b in ba)
    surf = S.build_surface(sn, S0, Ts, k_lo=klo, k_hi=khi,
                           time_var=args.time_var, n_strikes=args.n_strikes)
    print(f"\ntime_var={args.time_var}  n_strikes={args.n_strikes}  "
          f"optimised={args.optimize}  slope_smooth={args.slope_smooth}")

    Tg = np.linspace(Ts[0], Ts[-1], 250)
    Kg, d2K = S.second_derivative_K(surf, Tg)
    bad = d2K < 0

    print(f"\nviolations {bad.sum()} of {d2K.size}")
    print(f"d2C/dK2: min {d2K.min():.3e}   median {np.median(d2K):.3e}   "
          f"max {d2K.max():.3e}")

    # each calibrated smile is convex by construction, so anything here comes
    # from the interpolation
    print("\nat the calibrated maturities (should be clean):")
    for T, lab in zip(Ts, [f"T{i+1}" for i in range(len(Ts))]):
        _, d2 = S.second_derivative_K(surf, T)
        print(f"  {lab}  T={T:.4f}   min d2C/dK2 {d2.min():+.3e}   "
              f"violations {(d2 < 0).sum()}")

    if not bad.any():
        return

    ti, ki = np.where(bad)
    print(f"\nwhere the violations sit:")
    print(f"  T    from {Tg[ti].min():.4f} to {Tg[ti].max():.4f}   "
          f"(knots at {', '.join(f'{t:.4f}' for t in Ts)})")
    print(f"  K    from {Kg[ki].min():.1f} to {Kg[ki].max():.1f}   "
          f"log(K/S0) {np.log(Kg[ki].min()/S0):+.3f} to "
          f"{np.log(Kg[ki].max()/S0):+.3f}")

    # which interval between maturities
    print("\n  by interval:")
    for a, b in zip(Ts[:-1], Ts[1:]):
        m = (Tg[ti] >= a) & (Tg[ti] <= b)
        print(f"    T in [{a:.4f}, {b:.4f}]: {m.sum():>5} violations")

    # worst point, with the four neighbouring strikes
    t0, k0 = np.unravel_index(np.argmin(d2K), d2K.shape)
    print(f"\n  worst at T={Tg[t0]:.4f}, K={Kg[k0]:.1f} "
          f"(log(K/S0)={np.log(Kg[k0]/S0):+.3f}), d2C/dK2={d2K[t0,k0]:.3e}")
    C = S.surface_C(surf, Tg[t0])[0]
    lo = max(k0 - 1, 0)
    print(f"  prices nearby: " +
          "  ".join(f"K={surf['K'][j]:.1f} C={C[j]:.6f}"
                    for j in range(lo, min(lo + 4, C.size))))

    scale = np.median(np.abs(d2K[t0]))
    print(f"\n  |worst| / median |d2C/dK2| at that T = "
          f"{abs(d2K[t0,k0]) / max(scale, 1e-300):.1f}")

    # V'' jumps at every partition node and the maturities have different
    # nodes, so interpolating mixes one smile's left-of-jump with another's
    # right-of-jump
    Kbad = Kg[k0]
    print(f"\n  partition nodes near K={Kbad:.2f}:")
    for i, (sigs, nus) in enumerate(sn):
        j = int(np.argmin(np.abs(nus - Kbad)))
        lo = max(j - 1, 0)
        hi = min(j, len(sigs) - 1)
        print(f"    maturity {i+1}: nearest node {nus[j]:.3f} "
              f"(gap {nus[j]-Kbad:+.3f})   sigma {sigs[lo]:.3f} -> {sigs[hi]:.3f}"
              f"   ratio {sigs[hi]/max(sigs[lo],1e-30):.3f}")

    nodes_near = [np.sum(np.abs(nus - Kbad) < 1.0) for _, nus in sn]
    print(f"  nodes within 1.0 of the worst strike, per maturity: {nodes_near}")

    allnodes = np.unique(np.concatenate([nus for _, nus in sn]))
    dist = np.abs(Kg[ki][:, None] - allnodes[None, :]).min(axis=1)
    h = Kg[1] - Kg[0]
    print(f"  distance from a violating strike to the nearest node: "
          f"median {np.median(dist):.3f}, max {dist.max():.3f}  (grid step {h:.3f})")


if __name__ == "__main__":
    main()
