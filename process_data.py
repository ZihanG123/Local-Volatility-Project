"""
process_data.py
================

End-to-end data pipeline for the LVG / CMA-ES smoothness research project.

Inputs
------
SPX_opt_2011_2012.csv  -- OptionMetrics option chain (raw)

Outputs
-------
processed_data/<YYYYMMDD>/
    Quotes.csv                 strike, ask, bid (adjusted, calls only)
    ArbFree_calls_strikes.csv  arb-free call prices on the same grid (from 5a)
    theta_init_data.json       initial LVG calibration (from 5b)
    summary.json               S0, Kbar, D0T, F0T, target/actual expiry, n_strikes

The pipeline per day
--------------------
1. filter the OM chain to a single 6-month expiry (~126 trading days ahead,
   US federal holidays accounted for; pick the closest listed exdate)
2. drop stale rows (bid==0 or volume+OI==0)
3. estimate D0T and F0T from put-call parity (OLS on mid-quotes)
4. build adjusted call bid/ask using the A/B min-max identity from the
   LVG slides (page 22): combine call and put quotes into the tightest
   call-only envelope (B <= C_hat <= A)
5. set S0 := F0T (work in the forward measure -> K_hat = K)
6. run prob5a  (linear programming + analytic-center smoothing)
7. run prob5b  (LVG interpolation -> nus, sigmas)
8. write per-day folder

Notes
-----
* prob5a and prob5b logic is refactored into functions here. The original
  notebooks remain on disk; this file is the canonical source from here on.
* Days with too few strikes (< 15 valid C/P pairs) are skipped and logged.
* All operations are deterministic; running twice produces identical output.
"""

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, linprog, minimize


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    spx_csv: str            = "SPX_opt_2011_2012.csv"
    out_root: str           = "processed_data"
    start_date: int         = 20110103           # first trading day to process
    n_days: int             = 200                # contiguous trading days
    expiry_target_bdays: int = 126               # ~6 months in business days
    min_pairs: int          = 15                 # skip days with fewer call/put pairs
    Kbar: float             = 2000.0             # upper LVG boundary (slides use this)
    # convexity / smoothing tolerances (carried over from the original 5a/5b)
    analytic_center_method: str = "trust-constr"


# =============================================================================
# Trading-day helpers (US federal holidays via numpy.busday_offset)
# =============================================================================
# A small fixed list is enough for 2011-2012; numpy can do the heavy lifting.
# Source: US federal holidays observed on weekdays in 2011 + 2012.

US_HOLIDAYS = np.array([
    # 2011
    "2011-01-17", "2011-02-21", "2011-04-22", "2011-05-30",
    "2011-07-04", "2011-09-05", "2011-11-24", "2011-12-26",
    # 2012
    "2012-01-02", "2012-01-16", "2012-02-20", "2012-04-06",
    "2012-05-28", "2012-07-04", "2012-09-03", "2012-11-22",
    "2012-12-25",
], dtype="datetime64[D]")

US_BD = np.busdaycalendar(holidays=US_HOLIDAYS)


def _to_d64(yyyymmdd: int) -> np.datetime64:
    s = str(int(yyyymmdd))
    return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}", "D")


def _to_yyyymmdd(d: np.datetime64) -> int:
    return int(str(d).replace("-", ""))


def get_trading_days(start: int, n: int) -> list[int]:
    """`n` consecutive US trading days starting at `start` (inclusive)."""
    d = _to_d64(start)
    days = []
    while len(days) < n:
        if np.is_busday(d, busdaycal=US_BD):
            days.append(_to_yyyymmdd(d))
        d = d + np.timedelta64(1, "D")
    return days


def target_expiry_date(trade_date: int, n_bdays: int) -> int:
    """The trading date exactly `n_bdays` business days ahead, as YYYYMMDD."""
    d = _to_d64(trade_date)
    target = np.busday_offset(d, n_bdays, roll="forward", busdaycal=US_BD)
    return _to_yyyymmdd(target)


# =============================================================================
# OM data loading & filtering
# =============================================================================

def load_spx_csv(path: str) -> pd.DataFrame:
    """Load the OptionMetrics chain; normalize types/columns."""
    df = pd.read_csv(path, usecols=[
        "date", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "volume", "open_interest",
        "impl_volatility",
    ])
    # OM stores strike * 1000
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df = df.drop(columns=["strike_price"])
    # ensure ints
    df["date"]   = df["date"].astype(int)
    df["exdate"] = df["exdate"].astype(int)
    return df


def _is_third_friday(d_yyyymmdd: int) -> bool:
    """SPX standard monthly expiries are the 3rd Friday of each month.
    SPX *cash-settled* monthlies actually settle on the 3rd Friday's
    Saturday in OptionMetrics, so we also accept the day after 3rd Friday.
    """
    import datetime as _dt
    s = str(int(d_yyyymmdd))
    py_d = _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    # Friday = 4, Saturday = 5
    if py_d.weekday() == 4 and 15 <= py_d.day <= 21:
        return True
    if py_d.weekday() == 5 and 16 <= py_d.day <= 22:
        return True
    return False


def pick_expiry(df_day: pd.DataFrame, trade_date: int, target_bdays: int,
                prefer_monthly: bool = True):
    """Pick the listed exdate closest to `target_bdays` business days out.

    SPX has standard monthly (3rd Friday) expiries plus quarterlies and weeklies;
    monthly options are by far the most liquid 6m out, so we prefer them when
    available within a reasonable distance.

    Returns (exdate_yyyymmdd, T_actual_bdays) or None if no usable expiry.
    """
    expiries = sorted(df_day["exdate"].unique())
    if not expiries:
        return None

    # candidate set: business-day distance from trade_date, filter very-short
    candidates = []
    for ex in expiries:
        dist = int(np.busday_count(_to_d64(trade_date), _to_d64(ex), busdaycal=US_BD))
        if dist < 20:
            continue
        candidates.append((ex, dist))
    if not candidates:
        return None

    if prefer_monthly:
        monthlies = [(ex, d) for (ex, d) in candidates if _is_third_friday(ex)]
        if monthlies:
            # within 25 business days of target -> monthly wins outright
            best = min(monthlies, key=lambda x: abs(x[1] - target_bdays))
            if abs(best[1] - target_bdays) <= 25:
                return best
    # fall back: closest in distance regardless of type
    return min(candidates, key=lambda x: abs(x[1] - target_bdays))


def filter_strikes(df_day_expiry: pd.DataFrame) -> pd.DataFrame:
    """Drop stale quotes; pair calls with puts at each strike.

    Returns a DataFrame indexed by strike with columns:
        C_b, C_a (call bid/ask), P_b, P_a (put bid/ask).
    """
    # stale-quote filter: bid==0 OR (volume==0 and open_interest==0)
    df = df_day_expiry[
        (df_day_expiry["best_bid"] > 0) &
        ((df_day_expiry["volume"] > 0) | (df_day_expiry["open_interest"] > 0))
    ].copy()

    calls = df[df["cp_flag"] == "C"][["strike", "best_bid", "best_offer"]]
    puts  = df[df["cp_flag"] == "P"][["strike", "best_bid", "best_offer"]]
    calls = calls.rename(columns={"best_bid": "C_b", "best_offer": "C_a"})
    puts  = puts.rename(columns={"best_bid": "P_b", "best_offer": "P_a"})

    # if duplicate strikes (multiple option ids), take the tightest spread
    calls = calls.sort_values("strike").groupby("strike", as_index=False).agg(
        C_b=("C_b", "max"), C_a=("C_a", "min")
    )
    puts = puts.sort_values("strike").groupby("strike", as_index=False).agg(
        P_b=("P_b", "max"), P_a=("P_a", "min")
    )

    merged = pd.merge(calls, puts, on="strike", how="inner")
    # discard pairs where the bid >= ask after dedup
    merged = merged[(merged["C_a"] > merged["C_b"]) & (merged["P_a"] > merged["P_b"])]
    return merged.reset_index(drop=True)


# =============================================================================
# Put-call parity regression (least squares on mid-quotes)
# =============================================================================

def estimate_DF(pairs: pd.DataFrame):
    """OLS:  (C_mid - P_mid) = A + B * K.
    Then D0T := -B, F0T := A / D0T.
    Returns (D0T, F0T, resid_max).
    """
    K = pairs["strike"].to_numpy()
    Cm = 0.5 * (pairs["C_b"] + pairs["C_a"]).to_numpy()
    Pm = 0.5 * (pairs["P_b"] + pairs["P_a"]).to_numpy()
    y = Cm - Pm
    # design matrix [1, K]
    X = np.column_stack([np.ones_like(K), K])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A_int, B_slope = coef
    D0T = -B_slope
    if D0T <= 0:
        raise ValueError(f"non-positive discount factor D0T={D0T:.4f} -- bad parity fit")
    F0T = A_int / D0T
    resid = y - X @ coef
    return float(D0T), float(F0T), float(np.max(np.abs(resid)))


# =============================================================================
# Adjusted call envelope (A, B from slide 22)
# =============================================================================

def build_adjusted_calls(pairs: pd.DataFrame, D0T: float, F0T: float, S0: float):
    """Build adjusted call bid/ask envelopes via:
        A(K_hat) := min[ (S0/(D F)) * C_a(K),  (S0/(D F)) * P_a(K) + S0 - K_hat ]
        B(K_hat) := max[ (S0/(D F)) * C_b(K),  (S0/(D F)) * P_b(K) + S0 - K_hat ]
    where K_hat = K * F0T/S0.

    With our normalization S0 := F0T, K_hat = K. Returns a DataFrame with
    columns: K_hat, A (adj ask), B (adj bid).
    """
    K = pairs["strike"].to_numpy()
    K_hat = K * F0T / S0          # exactly K when S0=F0T
    scale = S0 / (D0T * F0T)

    adj_C_a = scale * pairs["C_a"].to_numpy()
    adj_C_b = scale * pairs["C_b"].to_numpy()
    adj_P_a = scale * pairs["P_a"].to_numpy()
    adj_P_b = scale * pairs["P_b"].to_numpy()

    A = np.minimum(adj_C_a, adj_P_a + S0 - K_hat)   # tightest ask
    B = np.maximum(adj_C_b, adj_P_b + S0 - K_hat)   # tightest bid

    out = pd.DataFrame({"K_hat": K_hat, "A": A, "B": B})
    # discard inverted envelopes (would indicate arb in the quotes)
    out = out[out["A"] > out["B"]].reset_index(drop=True)
    return out


# =============================================================================
# prob5a: LP feasibility + analytic-center smoothing
# =============================================================================
# Refactored from Hwk_prob5a_sol__3_.ipynb. Logic identical; we just take K,
# C_h, C_l, S as function arguments rather than reading Quotes.csv globally.

def run_prob5a(K: list[float], C_h: list[float], C_l: list[float],
               S: float, Kbar: float = 2000.0):
    """Find arbitrage-free call prices at strikes K, bracketed by C_l <= C <= C_h.

    Returns the analytic-center call vector (same length as K) -- this is the
    `call` array the original notebook produced and that prob5b reads.
    """
    N = len(K)
    A = np.zeros((N - 1 + N - 2 + 4, N))
    b = np.zeros((N - 1 + N - 2 + 4, 1))
    lb = np.zeros((N, 1))
    ub = np.zeros((N, 1))

    L = 0
    Kbar_eff = max(Kbar, K[-1] + 100)

    # monotonicity in strike: C_{i+1} - C_i <= 0
    for i in range(0, N - 1):
        A[i, i] = -1.0
        A[i, i + 1] = 1.0

    # convexity in strike (discrete second-difference >= 0)
    for i in range(1, N - 1):
        A[N - 1 + i - 1, i - 1] = -1.0 / (K[i] - K[i - 1])
        A[N - 1 + i - 1, i]     =  1.0 / (K[i] - K[i - 1]) + 1.0 / (K[i + 1] - K[i])
        A[N - 1 + i - 1, i + 1] = -1.0 / (K[i + 1] - K[i])

    # box bounds: per-strike bid/ask AND intrinsic-value floor
    for i in range(0, N):
        lb[i] = max([0.0, S - K[i], C_l[i]])
        ub[i] = min([S, C_h[i]])

    # boundary constraints at L (deep ITM) and Kbar_eff (deep OTM)
    A[N - 1 + N - 2, 0] = -1
    b[N - 1 + N - 2, 0] = K[0] - L - S

    A[N - 1 + N - 2 + 1, N - 1] = -1

    A[N - 1 + N - 2 + 2, 0] =  1.0 / (K[0] - L) + 1.0 / (K[1] - K[0])
    A[N - 1 + N - 2 + 2, 1] = -1.0 / (K[1] - K[0])
    b[N - 1 + N - 2 + 2, 0] = S / (K[0] - L)

    A[N - 1 + N - 2 + 3, N - 2] = -1.0 / (K[N - 1] - K[N - 2])
    A[N - 1 + N - 2 + 3, N - 1] =  1.0 / (Kbar_eff - K[N - 1]) + 1.0 / (K[N - 1] - K[N - 2])

    A = np.concatenate((A, np.identity(N), -np.identity(N)), axis=0)
    b = np.concatenate((b, ub, -lb), axis=0)
    b = b.T[0]

    # ── Chebyshev-center warm start ───────────────────────────────────────
    # Instead of starting analytic-center optimization from a polytope vertex
    # (where the log-barrier objective is -inf), we first find the center of
    # the largest inscribed ball in {x : Ax <= b}. This is the Chebyshev
    # center, a single LP:
    #     max  r
    #     s.t. A_i x + ||A_i||_2 * r <= b_i  for all i
    #          r >= 0
    # The resulting x has guaranteed positive slack to every constraint,
    # which makes a much better starting point for the analytic-center
    # log-barrier optimization than a vertex from a c=0 feasibility LP.
    row_norms = np.linalg.norm(A, axis=1)
    A_chev = np.hstack([A, row_norms.reshape(-1, 1)])   # N+1 columns
    c_chev = np.zeros(N + 1); c_chev[-1] = -1.0          # max r = -min(-r)
    bounds_chev = [(None, None)] * N + [(0.0, None)]     # r >= 0
    res_chev = linprog(c_chev, A_ub=A_chev, b_ub=b,
                       bounds=bounds_chev, method="highs")
    if not res_chev.success:
        # fallback: ordinary feasibility (vertex) LP
        c = np.zeros(N)
        res = linprog(c, A, b, method="highs")
        if not res.success:
            raise RuntimeError(f"prob5a linprog failed: {res.message}")
        x0 = np.asarray(res.x)
    else:
        x0 = np.asarray(res_chev.x[:N])
        r_star = float(res_chev.x[N])
        if r_star <= 0:
            # polytope degenerate -- fall back to vertex
            c = np.zeros(N)
            res = linprog(c, A, b, method="highs")
            x0 = np.asarray(res.x)

    # analytic-center smoothing (from the Chebyshev-center warm start)
    def objective(x):
        slack = b - A @ x
        # use sum of log slacks rather than product (numerically stable)
        # equivalent to the original -product up to monotone transform
        slack = np.clip(slack, 1e-12, None)
        return -float(np.sum(np.log(slack)))

    result = minimize(
        objective, x0, method="trust-constr",
        constraints=LinearConstraint(A, lb=-np.inf, ub=b),
        options={"maxiter": 500, "gtol": 1e-8},
    )
    if not result.success:
        # fall back to the Chebyshev center if smoothing fails -- still arb-free
        return x0, Kbar_eff
    return np.asarray(result.x, dtype=float), Kbar_eff


# =============================================================================
# prob5b: LVG interpolation -> initial (nus, sigmas)
# =============================================================================
# Refactored from Hwk_prob5b_sol__2_.ipynb. Same logic.

def _Atilde(sigma, A_, B_, w, K_):
    return 0.5 * (A_ - sigma * B_) * np.exp(-(w - K_) / sigma) + \
           0.5 * (A_ + sigma * B_) * np.exp( (w - K_) / sigma)


def _Btilde(sigma, A_, B_, w, K_):
    return -(0.5 / sigma) * (A_ - sigma * B_) * np.exp(-(w - K_) / sigma) + \
            (0.5 / sigma) * (A_ + sigma * B_) * np.exp( (w - K_) / sigma)


def _findSigmaHat(A_, B_, w, K1, K2, V):
    eps = 1.0e-8
    def f(sigma):
        return _Atilde(sigma, A_, B_, w, K1) + _Btilde(sigma, A_, B_, w, K1) * (K2 - w) - V
    low, high = eps, 1.0
    n_grow = 0
    while f(high) > 0 and n_grow < 60:
        high *= 2.0; n_grow += 1
    while high - low > eps:
        mid = 0.5 * (low + high)
        if f(mid) >= 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _sigmaTilde(sigma, A_, B_, w, K1, K2, V):
    eps = 1.0e-8
    At = _Atilde(sigma, A_, B_, w, K1)
    Bt = _Btilde(sigma, A_, B_, w, K1)
    def f(st):
        return 0.5 * (At + st * Bt) * np.exp( (K2 - w) / st) + \
               0.5 * (At - st * Bt) * np.exp(-(K2 - w) / st) - V
    low, high = eps, 1.0
    n_grow = 0
    while f(high) > 0 and n_grow < 60:
        high *= 2.0; n_grow += 1
    while high - low > eps:
        mid = 0.5 * (low + high)
        if f(mid) >= 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _findSigma(A_, B_, w, K1, K2, V, B1, sigma_h):
    eps = 1.0e-8
    def f(sigma):
        st = _sigmaTilde(sigma, A_, B_, w, K1, K2, V)
        At = _Atilde(sigma, A_, B_, w, K1)
        Bt = _Btilde(sigma, A_, B_, w, K1)
        return B1 - (0.5 / st) * (At + st * Bt) * np.exp( (K2 - w) / st) \
                  + (0.5 / st) * (At - st * Bt) * np.exp(-(K2 - w) / st)
    low, high = sigma_h, sigma_h + 1
    n_grow = 0
    while f(high) > 0 and n_grow < 60:
        high *= 2.0; n_grow += 1
    while high - low > eps:
        mid = 0.5 * (low + high)
        if f(mid) >= 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _interpolate(K_wing, V_wing):
    """Inner loop of prob5b: piecewise sigma fitting on one wing.

    Returns (nu_list, omega_list, kappa_list, B_last). Same structure as the
    original notebook.
    """
    N = len(K_wing)
    nu = []; kappa = []; omega = []
    A_ = V_wing[0]
    delta = 0.5
    B_ = delta * (V_wing[1] - V_wing[0]) / (K_wing[1] - K_wing[0])
    for i in range(N - 2):
        K1t = K_wing[i]; K2t = K_wing[i + 1]
        V1t = V_wing[i]; V2t = V_wing[i + 1]
        B1 = delta * (V_wing[i + 2] - V2t) / (K_wing[i + 2] - K2t) + \
             (1.0 - delta) * (V2t - V1t) / (K2t - K1t)
        w = (V2t + B_ * K1t - A_ - B1 * K2t) / (B_ - B1)
        sigma_h = _findSigmaHat(A_, B_, w, K1t, K2t, V2t)
        sigma   = _findSigma(A_, B_, w, K1t, K2t, V2t, B1, sigma_h)
        sigma_t = _sigmaTilde(sigma, A_, B_, w, K1t, K2t, V2t)
        At = _Atilde(sigma, A_, B_, w, K1t)
        Bt = _Btilde(sigma, A_, B_, w, K1t)
        nu.append(K1t); nu.append(w)
        omega.append(1.0 / sigma); omega.append(1.0 / sigma_t)
        kappa.append([0.5 * (A_ + sigma * B_), 0.5 * (A_ - sigma * B_)])
        kappa.append([0.5 * (At + sigma_t * Bt), 0.5 * (At - sigma_t * Bt)])
        A_ = V2t
        B_ = B1
    return nu, omega, kappa, B_


def run_prob5b(call: np.ndarray, K: list[float], S: float, Kbar: float):
    """LVG interpolation. Returns dict ready for theta_init_data.json."""
    C = list(call)
    N = len(C)

    # find the strike interval straddling S
    i0 = 0
    for i in range(N - 1):
        if (K[i] < S) and (S < K[i + 1]):
            i0 = i

    # Initial value at S, bracketed below/above by the local interpolants
    Lt1 = C[i0]     + (S - K[i0])     * (C[i0]     - C[i0 - 1]) / (K[i0]     - K[i0 - 1])
    Lt2 = C[i0 + 1] + (S - K[i0 + 1]) * (C[i0 + 2] - C[i0 + 1]) / (K[i0 + 2] - K[i0 + 1])
    Lt  = max(Lt1, Lt2)
    Ut  = C[i0] * (K[i0 + 1] - S) / (K[i0 + 1] - K[i0]) + \
          C[i0 + 1] * (S - K[i0]) / (K[i0 + 1] - K[i0])
    delta = 0.5
    Vtemp = delta * Lt + (1 - delta) * Ut

    # left wing: K -> S, V(K) = C(K) - max(S-K, 0) on the put side
    K1 = [0]; V1 = [0]
    for i in range(N):
        if K[i] < S:
            K1.append(K[i]); V1.append(C[i] - S + K[i])
    K1.append(S); V1.append(Vtemp)

    # right wing: Kbar - K, reversed
    K2 = [0]; V2 = [0]
    for i in range(N):
        if K[N - 1 - i] > S:
            K2.append(Kbar - K[N - 1 - i]); V2.append(C[N - 1 - i])
    K2.append(Kbar - S); V2.append(Vtemp)

    nu1, omega1, kappa1, B1_last = _interpolate(K1, V1)
    nu2, omega2, kappa2, B2_last = _interpolate(K2, V2)

    # final-touch step (slopes at the wing tips matched to each other)
    rho = 0.5 + 0.5 * (V2[-2] - V2[-1]) / (K2[-1] - K2[-2]) + \
                0.5 * (V1[-1] - V1[-2]) / (K1[-1] - K1[-2])

    def _append_final(nu_l, omega_l, kappa_l, K_wing, V_wing, B_last, B1):
        K1t = K_wing[-2]; K2t = K_wing[-1]
        V1t = V_wing[-2]; V2t = V_wing[-1]
        A_ = V1t; B_ = B_last
        w  = (V2t + B_ * K1t - A_ - B1 * K2t) / (B_ - B1)
        sigma_h = _findSigmaHat(A_, B_, w, K1t, K2t, V2t)
        sigma   = _findSigma(A_, B_, w, K1t, K2t, V2t, B1, sigma_h)
        sigma_t = _sigmaTilde(sigma, A_, B_, w, K1t, K2t, V2t)
        At = _Atilde(sigma, A_, B_, w, K1t)
        Bt = _Btilde(sigma, A_, B_, w, K1t)
        nu_l.append(K1t); nu_l.append(w)
        omega_l.append(1.0 / sigma); omega_l.append(1.0 / sigma_t)
        kappa_l.append([0.5 * (A_ + sigma * B_), 0.5 * (A_ - sigma * B_)])
        kappa_l.append([0.5 * (At + sigma_t * Bt), 0.5 * (At - sigma_t * Bt)])

    _append_final(nu1, omega1, kappa1, K1, V1, B1_last, rho)
    _append_final(nu2, omega2, kappa2, K2, V2, B2_last, 1 - rho)

    # Convert prob5b output to the CMA-ES theta_init format.
    # This is the canonical bridge from Hwk_prob5b_sol.ipynb cell 27:
    # left wing is straightforward (already in absolute strikes, ascending);
    # right wing was built in reflected coords (Kbar - K), so we drop the
    # reflected origin nu2[0] (which maps to absolute K=Kbar -- handled by
    # the explicit Kbar appended below), reverse the rest, and un-reflect.
    # Sigmas just reverse, no skip.
    initial_nus1  = list(nu1)
    initial_sigs1 = [1.0 / w for w in omega1]
    initial_nus2  = [Kbar - x for x in reversed(nu2[1:])] + [Kbar]
    initial_sigs2 = [1.0 / w for w in reversed(omega2)]

    return {
        "S0":   float(S), "Kbar": float(Kbar),
        "initial_nus1":  [float(x) for x in initial_nus1],
        "initial_sigs1": [float(x) for x in initial_sigs1],
        "initial_nus2":  [float(x) for x in initial_nus2],
        "initial_sigs2": [float(x) for x in initial_sigs2],
    }


# =============================================================================
# Per-day orchestration
# =============================================================================

def process_day(df_day_expiry: pd.DataFrame, trade_date: int, exdate: int,
                T_actual: int, cfg: Config, out_dir: str) -> dict:
    """Run the full pipeline for one (date, expiry) and write outputs."""
    pairs = filter_strikes(df_day_expiry)
    if len(pairs) < cfg.min_pairs:
        return {"status": "skip", "reason": f"only {len(pairs)} valid pairs", "n_pairs": len(pairs)}

    # put-call parity -> D, F
    D0T, F0T, resid_max = estimate_DF(pairs)

    # work in the forward measure: S0 := F0T
    S0 = F0T

    adj = build_adjusted_calls(pairs, D0T, F0T, S0)
    if len(adj) < cfg.min_pairs:
        return {"status": "skip", "reason": f"only {len(adj)} valid adjusted envelopes"}

    # write Quotes.csv (strike, ask, bid)
    os.makedirs(out_dir, exist_ok=True)
    quotes_path = os.path.join(out_dir, "Quotes.csv")
    adj_sorted = adj.sort_values("K_hat").reset_index(drop=True)
    with open(quotes_path, "w", newline="") as f:
        w = csv.writer(f)
        for _, row in adj_sorted.iterrows():
            w.writerow([f"{row['K_hat']:.4f}", f"{row['A']:.4f}", f"{row['B']:.4f}"])

    # prob5a
    K  = adj_sorted["K_hat"].tolist()
    Ch = adj_sorted["A"].tolist()
    Cl = adj_sorted["B"].tolist()
    try:
        call, Kbar_eff = run_prob5a(K, Ch, Cl, S0, cfg.Kbar)
    except Exception as e:
        return {"status": "fail_5a", "reason": str(e), "n_strikes": len(K)}

    # write ArbFree_calls_strikes.csv (matches original 5b reader: C, K)
    afpath = os.path.join(out_dir, "ArbFree_calls_strikes.csv")
    with open(afpath, "w", newline="") as f:
        w = csv.writer(f)
        for c_val, k_val in zip(call, K):
            w.writerow([f"{c_val:.6f}", f"{k_val:.6f}"])

    # prob5b
    try:
        theta_init = run_prob5b(call, K, S0, Kbar_eff)
    except Exception as e:
        return {"status": "fail_5b", "reason": str(e), "n_strikes": len(K)}

    # write theta_init_data.json
    json_path = os.path.join(out_dir, "theta_init_data.json")
    with open(json_path, "w") as f:
        json.dump(theta_init, f, indent=2)

    # summary
    summary = {
        "trade_date": int(trade_date),
        "exdate":     int(exdate),
        "T_target_bdays": int(cfg.expiry_target_bdays),
        "T_actual_bdays": int(T_actual),
        "S0":  float(S0), "Kbar": float(Kbar_eff),
        "D0T": float(D0T), "F0T": float(F0T),
        "n_strikes":    int(len(K)),
        "n_left_nodes": int(len(theta_init["initial_nus1"])),
        "n_right_nodes": int(len(theta_init["initial_nus2"])),
        "parity_resid_max": float(resid_max),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return {"status": "ok", **summary}


# =============================================================================
# Main driver
# =============================================================================

def main(cfg: Optional[Config] = None):
    if cfg is None:
        cfg = Config()
    print(f"Loading {cfg.spx_csv} ...")
    df = load_spx_csv(cfg.spx_csv)
    print(f"  loaded {len(df):,} rows")

    days = get_trading_days(cfg.start_date, cfg.n_days)
    print(f"  processing {len(days)} trading days "
          f"({days[0]} .. {days[-1]})")

    by_date = {d: g for d, g in df.groupby("date")}

    statuses = []
    for k, d in enumerate(days, start=1):
        if d not in by_date:
            print(f"[{k:>3}/{len(days)}] {d}  SKIP (no OM rows)")
            statuses.append({"date": d, "status": "skip", "reason": "no rows"})
            continue
        df_d = by_date[d]
        exp = pick_expiry(df_d, d, cfg.expiry_target_bdays)
        if exp is None:
            print(f"[{k:>3}/{len(days)}] {d}  SKIP (no usable expiry)")
            statuses.append({"date": d, "status": "skip", "reason": "no expiry"})
            continue
        exdate, T_actual = exp
        df_de = df_d[df_d["exdate"] == exdate]
        out_dir = os.path.join(cfg.out_root, str(d))
        try:
            res = process_day(df_de, d, exdate, T_actual, cfg, out_dir)
        except Exception as e:
            res = {"status": "fail", "reason": f"unhandled: {e}"}
        status = res.get("status", "?")
        if status == "ok":
            print(f"[{k:>3}/{len(days)}] {d} -> exp {exdate} ({T_actual}bd)  "
                  f"S0={res['S0']:.2f} D={res['D0T']:.5f} "
                  f"N={res['n_strikes']:>3} nu1={res['n_left_nodes']:>3} nu2={res['n_right_nodes']:>3}")
        else:
            print(f"[{k:>3}/{len(days)}] {d}  {status}: {res.get('reason','')}")
        statuses.append({"date": d, **res})

    # roll-up
    os.makedirs(cfg.out_root, exist_ok=True)
    with open(os.path.join(cfg.out_root, "_index.json"), "w") as f:
        json.dump(statuses, f, indent=2)
    n_ok = sum(1 for s in statuses if s.get("status") == "ok")
    print(f"\nDONE: {n_ok}/{len(statuses)} days processed successfully")
    print(f"Per-day folders written under: {cfg.out_root}/")


if __name__ == "__main__":
    main()
