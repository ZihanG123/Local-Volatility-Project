"""
Interpolates the calibrated smiles through time to give C(T,K).

A blend C(T,K) = (1-w(T)) Cbar^(i-1) + w(T) Cbar^i has dC/dT = w'(T) * spread,
so matching dC/dT across a knot would force consecutive calendar spreads to be
proportional in K. They are not, so no w(T) gives a smooth surface -- the
weight must vary with K, i.e. interpolate each strike separately through time.
PCHIP does that and is monotone, so dC/dT >= 0 and C is C^1 in T come free.
Convexity in K does NOT; check_surface tests it.

time_var is the variable the cubic is fitted in. Listed expiries are roughly
geometrically spaced, so in log T the knots are near-even, the cubic bends
less, and convexity survives where it fails in T. Any increasing x(T) keeps
monotonicity and C^1, so this is a free choice.

Interpolates the time value, so C >= (S0-K)^+ holds exactly at every T.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline
from scipy.ndimage import uniform_filter1d

from calibration import load_c_values, V_K


TIME_VARS = {"T": (lambda T: T, lambda x: x, lambda T: np.ones_like(T)),
             "sqrt": (np.sqrt, lambda x: x ** 2,
                      lambda T: 0.5 / np.sqrt(np.maximum(T, 1e-300))),
             "log": (np.log, np.exp, lambda T: 1.0 / np.maximum(T, 1e-300))}


def monotone_slopes(x, V, smooth=0):
    """PCHIP knot slopes, averaged across strikes, clipped back inside the
    monotonicity bound. smooth=0 is plain PCHIP."""
    d = PchipInterpolator(x, V, axis=0).derivative()(x)
    if smooth and smooth > 1:
        d = uniform_filter1d(d, size=int(smooth), axis=1, mode="nearest")

    delta = np.diff(V, axis=0) / np.diff(x)[:, None]
    lo, hi = np.empty_like(d), np.empty_like(d)
    lo[0], hi[0] = np.minimum(0, 3 * delta[0]), np.maximum(0, 3 * delta[0])
    lo[-1], hi[-1] = np.minimum(0, 3 * delta[-1]), np.maximum(0, 3 * delta[-1])
    if d.shape[0] > 2:
        a, b = delta[:-1], delta[1:]
        m = 3 * np.minimum(np.abs(a), np.abs(b)) * np.sign(a)
        same = np.sign(a) == np.sign(b)          # extremum -> slope 0
        lo[1:-1] = np.where(same, np.minimum(0, m), 0.0)
        hi[1:-1] = np.where(same, np.maximum(0, m), 0.0)
    return np.clip(d, lo, hi)


def build_surface(sigs_nus_compiled, S0, Ts, n_strikes=600, k_lo=None, k_hi=None,
                  time_var="sqrt", slope_smooth=0):
    Ts = np.asarray(Ts, dtype=float)
    if Ts.size != len(sigs_nus_compiled):
        raise ValueError("need one T per maturity")
    if np.any(np.diff(Ts) <= 0):
        raise ValueError("Ts must be strictly increasing")

    L = max(nus[0] for _, nus in sigs_nus_compiled)
    Kbar = min(nus[-1] for _, nus in sigs_nus_compiled)
    klo = L if k_lo is None else max(k_lo, L)
    khi = Kbar if k_hi is None else min(k_hi, Kbar)
    Kg = np.linspace(klo, khi, n_strikes)

    V = np.empty((len(sigs_nus_compiled), Kg.size))
    for i, (sigs, nus) in enumerate(sigs_nus_compiled):
        c1, c2 = load_c_values(S0, sigs, nus)
        V[i] = V_K(S0, sigs, nus, c1, c2, Kg, type="V")

    worst = np.diff(V, axis=0).min()
    if worst < -1e-9:
        # a crossing in the inputs propagates straight into the surface
        print(f"WARNING: input smiles cross, min V^i - V^(i-1) = {worst:.3e}")

    if time_var not in TIME_VARS:
        raise ValueError(f"time_var must be one of {sorted(TIME_VARS)}")
    fwd, _, dxdT = TIME_VARS[time_var]
    xs = fwd(Ts)
    P = (PchipInterpolator(xs, V, axis=0) if not slope_smooth else
         CubicHermiteSpline(xs, V, monotone_slopes(xs, V, slope_smooth), axis=0))
    return {"T": Ts, "K": Kg, "S0": S0, "L": L, "Kbar": Kbar, "V": V,
            "time_var": time_var, "fwd": fwd, "dxdT": dxdT,
            "slope_smooth": slope_smooth, "P": P}


def surface_V(surf, T):
    return surf["P"](surf["fwd"](np.atleast_1d(T)))


def surface_C(surf, T):
    return surface_V(surf, T) + np.maximum(surf["S0"] - surf["K"], 0.0)


def surface_dCdT(surf, T):
    # intrinsic does not depend on T; chain rule through x(T)
    T = np.atleast_1d(T)
    return surf["P"].derivative()(surf["fwd"](T)) * surf["dxdT"](T)[:, None]


def second_derivative_K(surf, T):
    Kg = surf["K"]
    h = Kg[1] - Kg[0]
    C = surface_C(surf, T)
    return Kg[1:-1], (C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]) / h ** 2


def local_vol(surf, T):
    """Dupire a^2 = 2 (dC/dT) / (d2C/dK2)."""
    Kg, d2K = second_derivative_K(surf, T)
    dCdT = surface_dCdT(surf, T)[:, 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return Kg, 2.0 * dCdT / d2K


def check_surface(surf, nT=250, verbose=True):
    """No-arbitrage on a dense (T,K) grid. Reports minima, not pass/fail, so a
    marginal surface shows as marginal."""
    Tg = np.linspace(surf["T"][0], surf["T"][-1], nT)
    Kg, S0 = surf["K"], surf["S0"]

    V = surface_V(surf, Tg)
    C = V + np.maximum(S0 - Kg, 0.0)
    dCdT = surface_dCdT(surf, Tg)
    _, d2K = second_derivative_K(surf, Tg)
    dK = np.diff(C, axis=1) / (Kg[1] - Kg[0])

    jumps = [float(np.abs(surface_dCdT(surf, t + 1e-7)
                          - surface_dCdT(surf, t - 1e-7)).max())
             for t in surf["T"][1:-1]]

    out = {"min_dCdT": float(dCdT.min()),
           "min_d2CdK2": float(d2K.min()),
           "max_dCdK": float(dK.max()),
           "min_time_value": float(V.min()),
           "max_C_over_S0": float((C / S0).max()),
           "n_bad_d2K": int((d2K < 0).sum()),
           "n_grid": int(d2K.size),
           "max_dCdT_jump_at_knots": max(jumps) if jumps else 0.0}

    if verbose:
        print(f"  calendar     min dC/dT    {out['min_dCdT']:+.4e}   (>= 0)")
        print(f"  density      min d2C/dK2  {out['min_d2CdK2']:+.4e}   (>= 0)"
              f"   violations {out['n_bad_d2K']}/{out['n_grid']}")
        print(f"  monotone     max dC/dK    {out['max_dCdK']:+.4e}   (<= 0)")
        print(f"  intrinsic    min V        {out['min_time_value']:+.4e}   (>= 0)")
        print(f"  upper bound  max C/S0     {out['max_C_over_S0']:.4f}   (<= 1)")
        print(f"  smoothness   max |dC/dT| jump at knots  "
              f"{out['max_dCdT_jump_at_knots']:.3e}")
    return out


def compare_to_blend(surf):
    """max |dC/dT| jump at the knots under the T-only exponential blend."""
    T, V, S0, Kg = surf["T"], surf["V"], surf["S0"], surf["K"]
    C = V + np.maximum(S0 - Kg, 0.0)
    worst = 0.0
    for i in range(1, len(T) - 1):
        left = np.exp(T[i]) * (C[i] - C[i - 1]) / (np.exp(T[i]) - np.exp(T[i - 1]))
        right = np.exp(T[i]) * (C[i + 1] - C[i]) / (np.exp(T[i + 1]) - np.exp(T[i]))
        worst = max(worst, float(np.abs(right - left).max()))
    return worst


def plot_surface(surf, n_slice=5):
    import matplotlib.pyplot as plt
    S0, Kg = surf["S0"], surf["K"]
    lm = np.log(np.maximum(Kg, 1e-9) / S0)
    Tg = np.linspace(surf["T"][0], surf["T"][-1], 200)
    slices = np.linspace(surf["T"][0], surf["T"][-1], n_slice)
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, n_slice))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    for T, c in zip(slices, cols):
        ax[0, 0].plot(lm, surface_C(surf, T)[0], color=c, lw=1.1, label=f"T={T:.2f}")
        k, a2 = local_vol(surf, T)
        ax[0, 1].plot(np.log(np.maximum(k, 1e-9) / S0),
                      np.sqrt(np.clip(a2[0], 0, None)), color=c, lw=1.1,
                      label=f"T={T:.2f}")
    ax[0, 0].set_ylabel("C(T,K)")
    ax[0, 0].set_title("price slices")
    ax[0, 1].set_ylabel("local vol")
    ax[0, 1].set_title(f"Dupire local vol  (fitted in {surf['time_var']} T)")

    ax[1, 0].plot(Tg, surface_dCdT(surf, Tg).min(axis=1), lw=1.2)
    ax[1, 0].axhline(0, color="k", ls="--", lw=0.8)
    ax[1, 0].set_ylabel("min dC/dT over K")
    ax[1, 0].set_title("calendar condition")

    _, d2K = second_derivative_K(surf, Tg)
    ax[1, 1].semilogy(Tg, np.clip(d2K.min(axis=1), 1e-18, None), lw=1.2)
    ax[1, 1].set_ylabel("min d2C/dK2 over K")
    ax[1, 1].set_title("density positive -- not inherited, must be checked")

    for a in (ax[1, 0], ax[1, 1]):
        a.set_xlabel("T")
        for t in surf["T"]:
            a.axvline(t, color="0.7", lw=0.6)
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xlabel("log(K/S0)")
        a.axvline(0, color="0.6", lw=0.6)
        a.legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=.3)
    fig.tight_layout()
    return fig


def sample_sigs_nus(M=3, R1=20, R2=20, S0=1270.0, Kbar=2000.0, L=0.0,
                    sigma_step=0.15, seed=0):
    """Fixture, runs without data. V(S0) grows with a uniform scaling of
    sigma, so scaling by (1 + sigma_step*i) orders the maturities."""
    rng = np.random.default_rng(seed)
    nus = np.concatenate([np.linspace(L, S0, R1 + 1),
                          np.linspace(S0, Kbar, R2 + 1)[1:]])
    base = np.linspace(90.0, 120.0, R1 + R2)
    base += 14.0 * np.sin(np.linspace(0, 7, R1 + R2))
    base += rng.normal(0, 3.0, R1 + R2)
    return [(base * (1.0 + sigma_step * i), nus.copy()) for i in range(M)]
