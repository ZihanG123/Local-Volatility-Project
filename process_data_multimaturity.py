"""
process_data_multimaturity.py
=============================

Multi-maturity extension of process_data.py.

WHAT IS NEW RELATIVE TO process_data.py
---------------------------------------
The single-maturity pipeline handled one expiry per trading day.  This file
handles M expiries simultaneously and produces a calendar-arbitrage-free
family of smiles, following Carr & Nadtochiy (2017), Section IV-B / IV-C.

Three changes, in pipeline order:

1.  NORMALIZATION (build_adjusted_calls_common)
    Each expiry has its own (D_i, F_i).  We pick ONE common S0 and map

        kappa_i = K * S0 / F_i          (adjusted / "discounted" strike)
        Chat_i  = C * S0 / (D_i * F_i)  (adjusted price)

    so every maturity becomes a call on a martingale started at the SAME S0,
    with the SAME intrinsic (S0 - kappa)^+.  Note this is the reciprocal of
    the K_hat formula in process_data.py; the two agree only when S0 = F0T,
    which is why the old form was fine for a single maturity.  Derivation:
    put-call parity C - P = D(F - K); multiply by S0/(D F) to get
    Chat - Phat = S0 - K*S0/F, so kappa = K*S0/F.

2.  UNION GRID + JOINT LP (build_union_grid, run_prob5a_joint)
    Because the shift S0/F_i differs by maturity, the adjusted strikes no
    longer line up.  We take the union of all adjusted strikes and solve ONE
    analytic-center problem over all M x N price variables at once:
        - per maturity: monotonicity, convexity, boundary rows (as before)
        - box: real bid/ask where a quote exists for that (maturity, strike);
          the trivial no-arbitrage band (S0-kappa)^+ <= C <= S0 where it does not
        - NEW cross-maturity rows: Chat_{i}(kappa) >= Chat_{i-1}(kappa)
    Same smoothed-log barrier + eps homotopy as before, now sparse.
    The calendar ordering ("smiles lie above each other") is enforced HERE.

3.  MULTI-SMILE LVG INTERPOLATION (_interpolate, with prev_wing)
    Carr-Nadtochiy Theorem 16, Step 1.3 / Figure 6.  Each maturity's time
    value V^i must dominate V^{i-1} EVERYWHERE, not just at grid points.
    Before extending V^i across [K_j, K_{j+1}] we test whether the tangent
    line from (K_j, A) with slope B would fall under V^{i-1}:

        A + (K_{j+1} - K_j) * B  <  V^{i-1}(K_{j+1})   ?

    If not (always the case when V^{i-1} == 0), we run the ORIGINAL code path
    unchanged -- Step 1.3.a, two partition points {w, K_{j+1}}.
    If so, we bisect for the crossing point y, place a partition point at y
    with sigma_bar solving
        Atilde(sigma_bar) + Btilde(sigma_bar) * (K_{j+1} - y) = V^{i-1}(K_{j+1})
    and restart the step from (y, Atilde, Btilde).  Lemma 23 guarantees that
    restart lands in case (a), so the interval gets three partition points
    {y, w', K_{j+1}}.  This is Step 1.3.b, the right panel of Figure 6.

REGRESSION PROPERTY
-------------------
With maturity_targets_bdays = [126] (a single expiry) and S0 set to that
maturity's forward, every code path here reduces to process_data.py exactly:
V^{i-1} == 0 makes the y-test always false, the union grid is that maturity's
own strike grid, and the joint LP has no cross-maturity rows.  Use that to
regression-test against the existing processed_data/ output before trusting
any multi-maturity number.

Outputs
-------
processed_data_mm/<YYYYMMDD>/
    summary.json          S0, Kbar, union grid size, per-maturity D/F/T
    union_grid.csv        the shared adjusted-strike partition
    maturity_<k>/
        Quotes.csv        kappa, ask, bid  (NaN where no quote exists)
        ArbFree.csv       Chat, kappa  (joint LP output on the union grid)
        theta_init.json   initial_nus1/sigs1/nus2/sigs2, S0, Kbar, T, t_star
"""

import csv
import json
import math
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog, minimize


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    spx_csv: str             = "SPX_opt_2011_2012.csv"
    out_root: str            = "processed_data_mm"
    start_date: int          = 20110103
    n_days: int              = 5
    # Target maturities in business days.  The paper's numerical section used
    # 2/7/27/47/67 working days; we avoid the very short end where the parity
    # regression gets noisy.  One listed expiry is chosen per target.
    maturity_targets_bdays: list = field(default_factory=lambda: [21, 42, 63, 126, 189])
    min_pairs: int           = 15      # per maturity, before it is dropped
    min_maturities: int      = 2       # skip the day if fewer survive
    Kbar: float              = 2000.0
    # Union grid: adjusted strikes closer than this are merged into one node.
    # Without this, two maturities whose shifts differ by 0.01 would each
    # contribute a partition point and the grid would double for no reason.
    union_tol: float         = 0.50
    # Where a maturity has no quote at a union strike, use the trivial
    # no-arbitrage band (S0-kappa)^+ <= C <= S0.  This is what the professor's
    # note specifies.  Setting this False instead interpolates that maturity's
    # own bid/ask curves across the gap and widens by synthetic_widen -- much
    # better conditioned for the analytic center, but not the paper's recipe.
    synthetic_band_trivial: bool = True
    synthetic_widen: float   = 3.0
    # Barrier start.  'zeros' reproduces process_data.py exactly; 'midbox'
    # starts from the middle of each box and usually converges in fewer steps.
    x0_mode: str             = "zeros"
    eps_schedule: list       = field(default_factory=lambda: [
        1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])
    # The joint system has M*N variables instead of N, so the per-eps budget
    # has to grow with it.  200 (the single-maturity value) leaves the barrier
    # short of the polytope and produces small negative slacks.
    maxiter_per_eps: int     = 2000
    # If the barrier still ends outside the polytope, retry with this many
    # extra passes before giving up.
    n_retries: int           = 2


# =============================================================================
# Trading-day helpers  (unchanged from process_data.py)
# =============================================================================

US_HOLIDAYS = np.array([
    "2011-01-17", "2011-02-21", "2011-04-22", "2011-05-30",
    "2011-07-04", "2011-09-05", "2011-11-24", "2011-12-26",
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


def get_trading_days(start: int, n: int) -> list:
    d = _to_d64(start)
    days = []
    while len(days) < n:
        if np.is_busday(d, busdaycal=US_BD):
            days.append(_to_yyyymmdd(d))
        d = d + np.timedelta64(1, "D")
    return days


def bdays_to_expiry(trade_date: int, exdate: int) -> int:
    return int(np.busday_count(_to_d64(trade_date), _to_d64(exdate), busdaycal=US_BD))


def _is_third_friday(d_yyyymmdd: int) -> bool:
    import datetime as _dt
    s = str(int(d_yyyymmdd))
    py_d = _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    if py_d.weekday() == 4 and 15 <= py_d.day <= 21:
        return True
    if py_d.weekday() == 5 and 16 <= py_d.day <= 22:
        return True
    return False


# =============================================================================
# Loading / filtering  (unchanged, except pick_expiries is now plural)
# =============================================================================

def load_spx_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[
        "date", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "volume", "open_interest",
        "impl_volatility",
    ])
    df["strike"] = df["strike_price"].astype(float) / 1000.0
    df = df.drop(columns=["strike_price"])
    df["date"] = df["date"].astype(int)
    df["exdate"] = df["exdate"].astype(int)
    return df


def pick_expiries(df_day: pd.DataFrame, trade_date: int, targets_bdays: list,
                  prefer_monthly: bool = True, min_bdays: int = 5) -> list:
    """One listed expiry per target, nearest in business days, deduplicated.

    Returns [(exdate, T_bdays), ...] sorted by T ascending.  Monthlies win when
    they are within 25 business days of the target, same rule as the
    single-maturity pick_expiry.
    """
    expiries = sorted(df_day["exdate"].unique())
    candidates = []
    for ex in expiries:
        dist = bdays_to_expiry(trade_date, ex)
        if dist < min_bdays:
            continue
        candidates.append((int(ex), dist))
    if not candidates:
        return []

    chosen = {}
    for tgt in targets_bdays:
        best = None
        if prefer_monthly:
            monthlies = [(ex, d) for (ex, d) in candidates if _is_third_friday(ex)]
            if monthlies:
                cand = min(monthlies, key=lambda x: abs(x[1] - tgt))
                if abs(cand[1] - tgt) <= 25:
                    best = cand
        if best is None:
            best = min(candidates, key=lambda x: abs(x[1] - tgt))
        chosen[best[0]] = best[1]

    return sorted([(ex, d) for ex, d in chosen.items()], key=lambda x: x[1])


def filter_strikes(df_day_expiry: pd.DataFrame) -> pd.DataFrame:
    df = df_day_expiry[
        (df_day_expiry["best_bid"] > 0) &
        ((df_day_expiry["volume"] > 0) | (df_day_expiry["open_interest"] > 0))
    ].copy()

    calls = df[df["cp_flag"] == "C"][["strike", "best_bid", "best_offer"]]
    puts = df[df["cp_flag"] == "P"][["strike", "best_bid", "best_offer"]]
    calls = calls.rename(columns={"best_bid": "C_b", "best_offer": "C_a"})
    puts = puts.rename(columns={"best_bid": "P_b", "best_offer": "P_a"})

    calls = calls.sort_values("strike").groupby("strike", as_index=False).agg(
        C_b=("C_b", "max"), C_a=("C_a", "min"))
    puts = puts.sort_values("strike").groupby("strike", as_index=False).agg(
        P_b=("P_b", "max"), P_a=("P_a", "min"))

    merged = pd.merge(calls, puts, on="strike", how="inner")
    merged = merged[(merged["C_a"] > merged["C_b"]) & (merged["P_a"] > merged["P_b"])]
    return merged.reset_index(drop=True)


def estimate_DF(pairs: pd.DataFrame):
    """OLS (C_mid - P_mid) = a + b*K;  D = -b,  F = a/D."""
    K = pairs["strike"].to_numpy()
    Cm = 0.5 * (pairs["C_b"] + pairs["C_a"]).to_numpy()
    Pm = 0.5 * (pairs["P_b"] + pairs["P_a"]).to_numpy()
    y = Cm - Pm
    X = np.column_stack([np.ones_like(K), K])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_int, b_slope = coef
    D = -b_slope
    if D <= 0:
        raise ValueError(f"non-positive discount factor D={D:.4f} -- bad parity fit")
    F = a_int / D
    resid = y - X @ coef
    return float(D), float(F), float(np.max(np.abs(resid)))


# =============================================================================
# Adjusted call envelope, generalized to a common S0
# =============================================================================

def build_adjusted_calls_common(pairs: pd.DataFrame, D: float, F: float,
                                S0: float) -> pd.DataFrame:
    """Map one maturity's quotes onto the common martingale scale.

        kappa = K * S0 / F
        scale = S0 / (D * F)
        A(kappa) = min[ scale*C_a,  scale*P_a + S0 - kappa ]     (tightest ask)
        B(kappa) = max[ scale*C_b,  scale*P_b + S0 - kappa ]     (tightest bid)

    With S0 = F this is exactly build_adjusted_calls from process_data.py.
    """
    K = pairs["strike"].to_numpy()
    kappa = K * S0 / F
    scale = S0 / (D * F)

    A = np.minimum(scale * pairs["C_a"].to_numpy(),
                   scale * pairs["P_a"].to_numpy() + S0 - kappa)
    B = np.maximum(scale * pairs["C_b"].to_numpy(),
                   scale * pairs["P_b"].to_numpy() + S0 - kappa)

    out = pd.DataFrame({"kappa": kappa, "A": A, "B": B})
    out = out[out["A"] > out["B"]].reset_index(drop=True)
    return out.sort_values("kappa").reset_index(drop=True)


# =============================================================================
# Union grid
# =============================================================================

def build_union_grid(kappa_lists: list, tol: float) -> np.ndarray:
    """Union of all maturities' adjusted strikes, merging nodes closer than tol.

    Merging matters: two maturities whose discount shifts differ by a fraction
    of a point would otherwise each contribute a partition point, doubling the
    grid and the LVG partition for no informational gain.
    """
    allk = np.sort(np.concatenate([np.asarray(x, dtype=float) for x in kappa_lists]))
    keep = [allk[0]]
    for k in allk[1:]:
        if k - keep[-1] > tol:
            keep.append(k)
    return np.asarray(keep)


def map_to_union(kappa: np.ndarray, grid: np.ndarray, tol: float) -> np.ndarray:
    """Index of the nearest union node for each kappa; -1 if none within tol."""
    idx = np.searchsorted(grid, kappa)
    out = np.full(len(kappa), -1, dtype=int)
    for j, (k, i) in enumerate(zip(kappa, idx)):
        best, bestd = -1, np.inf
        for cand in (i - 1, i, i + 1):
            if 0 <= cand < len(grid):
                d = abs(grid[cand] - k)
                if d < bestd:
                    best, bestd = cand, d
        if bestd <= tol:
            out[j] = best
    return out


def assemble_boxes(grid: np.ndarray, per_mat_adj: list, S0: float,
                   tol: float, trivial: bool, widen: float):
    """Per-maturity (lb, ub) vectors on the union grid.

    Where a real quote exists:      lb = max(0, S0-kappa, bid),  ub = min(S0, ask)
    Where it does not, and trivial: lb = max(0, S0-kappa),       ub = S0
    Where it does not, and not trivial: linearly interpolate that maturity's own
    bid/ask curves in kappa and widen by `widen` times the local half-spread.
    """
    N = len(grid)
    intrinsic = np.maximum(S0 - grid, 0.0)
    lbs, ubs, has_quote = [], [], []
    merge_conflicts = []

    for adj in per_mat_adj:
        kap = adj["kappa"].to_numpy()
        ask = adj["A"].to_numpy()
        bid = adj["B"].to_numpy()
        pos = map_to_union(kap, grid, tol)

        q_ask = np.full(N, np.nan)
        q_bid = np.full(N, np.nan)
        # Two quoted strikes can collapse onto one union node when they are
        # within union_tol of each other.  Intersecting their boxes (tightest
        # ask, tightest bid) can INVERT when the quotes disagree -- that was
        # the fail_box mode.  Fall back to the union of the boxes in that
        # case, which is still a valid no-arbitrage bracket for both.
        n_merge_conflict = 0
        for j, p in enumerate(pos):
            if p < 0:
                continue
            if np.isnan(q_ask[p]):
                q_ask[p], q_bid[p] = ask[j], bid[j]
                continue
            tight_a = min(q_ask[p], ask[j])
            tight_b = max(q_bid[p], bid[j])
            if tight_a > tight_b:
                q_ask[p], q_bid[p] = tight_a, tight_b
            else:
                q_ask[p] = max(q_ask[p], ask[j])
                q_bid[p] = min(q_bid[p], bid[j])
                n_merge_conflict += 1
        if n_merge_conflict:
            merge_conflicts.append(n_merge_conflict)

        hq = ~np.isnan(q_ask)
        lb = intrinsic.copy()
        ub = np.full(N, S0)

        if trivial:
            lb[hq] = np.maximum(lb[hq], q_bid[hq])
            ub[hq] = np.minimum(ub[hq], q_ask[hq])
        else:
            mid = 0.5 * (q_ask + q_bid)
            half = 0.5 * (q_ask - q_bid)
            gi = np.where(hq)[0]
            mid_i = np.interp(grid, grid[gi], mid[gi])
            half_i = np.interp(grid, grid[gi], half[gi]) * widen
            lb = np.maximum(lb, mid_i - half_i)
            ub = np.minimum(ub, mid_i + half_i)
            lb[hq] = np.maximum(intrinsic[hq], q_bid[hq])
            ub[hq] = np.minimum(S0, q_ask[hq])
            lb = np.maximum(lb, intrinsic)
            ub = np.minimum(ub, S0)

        lbs.append(lb)
        ubs.append(ub)
        has_quote.append(hq)

    # Last-resort clamp: if a box is still inverted (e.g. the quoted ask sits
    # below intrinsic, which happens with stale deep-ITM quotes), widen it to
    # the trivial no-arbitrage band rather than failing the whole day.
    n_widened = 0
    for i in range(len(lbs)):
        bad = lbs[i] > ubs[i]
        if bad.any():
            n_widened += int(bad.sum())
            lbs[i][bad] = intrinsic[bad]
            ubs[i][bad] = S0
            has_quote[i][bad] = False
    return lbs, ubs, has_quote, {"merge_conflicts": int(sum(merge_conflicts)),
                                 "boxes_widened": n_widened}


# =============================================================================
# Joint analytic-center problem  (Section 2, extended across maturities)
# =============================================================================

def _maturity_block_rows(grid: np.ndarray, S0: float, Kbar_eff: float,
                         L: float = 0.0):
    """Rows of A x <= b for ONE maturity, in local column indices 0..N-1.

    Identical in content to run_prob5a in process_data.py:
      (N-1) monotonicity, (N-2) convexity, 4 boundary rows.
    Returned as (row_idx, col_idx, val, b) triplets with rows numbered from 0.
    """
    N = len(grid)
    K = grid
    rows, cols, vals = [], [], []
    b = []
    r = 0

    # monotonicity: C_{i+1} - C_i <= 0
    for i in range(N - 1):
        rows += [r, r]; cols += [i, i + 1]; vals += [-1.0, 1.0]
        b.append(0.0); r += 1

    # convexity: discrete second difference >= 0  (written as <= 0 form)
    for i in range(1, N - 1):
        dl = K[i] - K[i - 1]
        dr = K[i + 1] - K[i]
        rows += [r, r, r]
        cols += [i - 1, i, i + 1]
        vals += [-1.0 / dl, 1.0 / dl + 1.0 / dr, -1.0 / dr]
        b.append(0.0); r += 1

    # boundary at L: chord slope from (L, S0-L) to (K_0, C_0) is >= -1
    rows += [r]; cols += [0]; vals += [-1.0]
    b.append(K[0] - L - S0); r += 1

    # C_{N-1} >= 0
    rows += [r]; cols += [N - 1]; vals += [-1.0]
    b.append(0.0); r += 1

    # convexity at the first strike, using (L, S0-L) as the left neighbour
    rows += [r, r]; cols += [0, 1]
    vals += [1.0 / (K[0] - L) + 1.0 / (K[1] - K[0]), -1.0 / (K[1] - K[0])]
    b.append(S0 / (K[0] - L)); r += 1

    # convexity at the last strike, using (Kbar_eff, 0) as the right neighbour
    rows += [r, r]; cols += [N - 2, N - 1]
    vals += [-1.0 / (K[N - 1] - K[N - 2]),
             1.0 / (Kbar_eff - K[N - 1]) + 1.0 / (K[N - 1] - K[N - 2])]
    b.append(0.0); r += 1

    return rows, cols, vals, np.asarray(b), r


def build_joint_system(grid: np.ndarray, lbs: list, ubs: list, S0: float,
                       Kbar_eff: float, enforce_calendar: bool = True):
    """Assemble the full sparse (A, b) for M maturities on the shared grid.

    Variable order is x[i*N + m] = Chat_i(grid[m]).
    Row blocks:
        per-maturity shape rows      (monotonicity / convexity / boundary)
        box rows                     ( x <= ub  and  -x <= -lb )
        calendar rows                ( x_{i-1,m} - x_{i,m} <= 0 )
    """
    M = len(lbs)
    N = len(grid)
    R, C, V = [], [], []
    bvec = []
    roff = 0

    br, bc, bv, bb, nrow = _maturity_block_rows(grid, S0, Kbar_eff)
    for i in range(M):
        coff = i * N
        R += [r + roff for r in br]
        C += [c + coff for c in bc]
        V += list(bv)
        bvec.append(bb)
        roff += nrow

    # box rows
    for i in range(M):
        coff = i * N
        for m in range(N):
            R.append(roff); C.append(coff + m); V.append(1.0)
            roff += 1
        bvec.append(ubs[i])
    for i in range(M):
        coff = i * N
        for m in range(N):
            R.append(roff); C.append(coff + m); V.append(-1.0)
            roff += 1
        bvec.append(-lbs[i])

    # calendar rows: earlier maturity must not exceed the later one
    n_cal = 0
    if enforce_calendar and M > 1:
        for i in range(1, M):
            prev = (i - 1) * N
            cur = i * N
            for m in range(N):
                R += [roff, roff]
                C += [prev + m, cur + m]
                V += [1.0, -1.0]
                roff += 1
                n_cal += 1
        bvec.append(np.zeros(n_cal))

    A = sp.csr_matrix((V, (R, C)), shape=(roff, M * N))
    b = np.concatenate(bvec)
    return A, b, n_cal


def solve_analytic_center(A, b, x0, eps_schedule, maxiter):
    """Smoothed-log barrier with eps homotopy.  Same construction as before.

    phi_eps(s) = log(s)                                        for s >= eps
               = log(eps) + u - u^2/2,  u = (s-eps)/eps         for s <  eps
    C^2 at s = eps, finite everywhere, so we can start outside the polytope.
    """
    def obj(x, eps):
        s = b - A @ x
        above = s >= eps
        out = np.empty_like(s)
        out[above] = np.log(s[above])
        u = (s[~above] - eps) / eps
        out[~above] = np.log(eps) + u - 0.5 * u * u
        return -float(np.sum(out))

    def grad(x, eps):
        s = b - A @ x
        above = s >= eps
        d = np.empty_like(s)
        d[above] = 1.0 / s[above]
        u = (s[~above] - eps) / eps
        d[~above] = (1.0 - u) / eps
        return A.T @ d

    x = np.asarray(x0, dtype=float)
    for eps in eps_schedule:
        res = minimize(obj, x, args=(eps,), jac=grad, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-9, "ftol": 1e-12})
        x = np.asarray(res.x, dtype=float)
    return x


def run_prob5a_joint(grid: np.ndarray, lbs: list, ubs: list, S0: float,
                     Kbar: float, cfg: Config):
    """Joint analytic center.  Returns (prices [M, N], diagnostics dict)."""
    M, N = len(lbs), len(grid)
    Kbar_eff = max(Kbar, float(grid[-1]) + 100.0)
    A, b, n_cal = build_joint_system(grid, lbs, ubs, S0, Kbar_eff)

    # Feasibility probe.  The barrier cannot tell you the polytope is empty --
    # it will just return whatever minimizes the smoothed objective.  A cheap
    # LP up front turns a silent garbage result into a clear failure, and tells
    # you WHICH constraints are the problem via the residual breakdown.
    lp = linprog(np.zeros(M * N), A_ub=A, b_ub=b, bounds=[(None, None)] * (M * N),
                 method="highs")
    feasible = bool(lp.success)

    if cfg.x0_mode == "midbox":
        x0 = np.concatenate([0.5 * (lbs[i] + ubs[i]) for i in range(M)])
    else:
        x0 = np.zeros(M * N)

    maxit = cfg.maxiter_per_eps
    x = solve_analytic_center(A, b, x0, cfg.eps_schedule, maxit)
    for _ in range(cfg.n_retries):
        if np.min(b - A @ x) >= -1e-9:
            break
        # warm-start from where we are and give L-BFGS more room; the barrier
        # is convex, so this only ever moves us further inside
        maxit *= 4
        x = solve_analytic_center(A, b, x, cfg.eps_schedule, maxit)

    slack = b - A @ x
    diag = {
        "feasible_lp": feasible,
        "n_rows": int(A.shape[0]),
        "n_cols": int(A.shape[1]),
        "n_calendar_rows": int(n_cal),
        "min_slack": float(slack.min()),
        "n_violated": int(np.sum(slack < -1e-6)),
        "n_near_zero_slack": int(np.sum(np.abs(slack) < 1e-8)),
    }
    if n_cal > 0:
        cal_slack = slack[-n_cal:]
        diag["min_calendar_slack"] = float(cal_slack.min())
        diag["n_calendar_tight"] = int(np.sum(cal_slack < 1e-6))

    if not feasible:
        raise RuntimeError(
            "joint polytope is empty (LP feasibility probe failed) -- "
            "a later maturity's quotes are inconsistent with an earlier one")

    if diag["n_violated"] > 0:
        raise RuntimeError(
            f"joint analytic center left {diag['n_violated']} rows violated "
            f"(min slack {diag['min_slack']:.3e}); "
            f"LP feasibility was {feasible}")

    return x.reshape(M, N), diag


# =============================================================================
# LVG piecewise-exponential machinery
# =============================================================================
# A wing is (nu, omega, kappa) with omega_j = 1/sigma_j and, for x in
# [nu_j, nu_{j+1}),
#     V(x) = kappa_j[0] * exp(+(x-nu_j)*omega_j)
#          + kappa_j[1] * exp(-(x-nu_j)*omega_j)
# The original _interpolate already builds kappa but run_prob5b threw it away.
# We need it now: V^{i-1} has to be evaluable at arbitrary points, not just at
# the grid, to compute the crossing point y.

def _wing_locate(wing, x: float):
    """(piece index, offset from that piece's anchor).  Clamped at both ends.

    Left of the first anchor we clamp the offset to 0 rather than returning 0
    outright: for a full wing nu[0] is the wing origin where V == 0 anyway, but
    for a sub-wing (as built inside _extend_interval) the first piece carries a
    nonzero value and must be evaluated, not zeroed.
    """
    nu, omega, kappa = wing
    if x <= nu[0]:
        return 0, 0.0
    j = min(bisect_right(nu, x) - 1, len(nu) - 1)
    return j, x - nu[j]


def eval_wing(wing, x: float) -> float:
    _, omega, kappa = wing
    j, d = _wing_locate(wing, x)
    om = omega[j]
    return kappa[j][0] * math.exp(d * om) + kappa[j][1] * math.exp(-d * om)


def eval_wing_deriv(wing, x: float) -> float:
    _, omega, kappa = wing
    j, d = _wing_locate(wing, x)
    om = omega[j]
    return om * (kappa[j][0] * math.exp(d * om) - kappa[j][1] * math.exp(-d * om))


class _ZeroWing:
    """V^0 == 0.  Makes the first maturity fall through every new branch."""
    def __call__(self, x):
        return 0.0
    def deriv(self, x):
        return 0.0


class _PrevWing:
    def __init__(self, wing):
        self.wing = wing
    def __call__(self, x):
        return eval_wing(self.wing, x)
    def deriv(self, x):
        return eval_wing_deriv(self.wing, x)


def _Atilde(sigma, A_, B_, w, K_):
    return 0.5 * (A_ - sigma * B_) * np.exp(-(w - K_) / sigma) + \
           0.5 * (A_ + sigma * B_) * np.exp((w - K_) / sigma)


def _Btilde(sigma, A_, B_, w, K_):
    return -(0.5 / sigma) * (A_ - sigma * B_) * np.exp(-(w - K_) / sigma) + \
            (0.5 / sigma) * (A_ + sigma * B_) * np.exp((w - K_) / sigma)


def _findSigmaHat(A_, B_, w, K1, K2, V):
    """Smallest sigma with Atilde + Btilde*(K2-w) = V.  Atilde, Btilde are
    strictly decreasing in sigma (Lemma 20), so bisection is safe.
    Also used, with y in place of w and V^{i-1}(K2) in place of V, as the
    Step 1.3.b root search (Lemma 22)."""
    eps = 1.0e-8
    def f(sigma):
        return _Atilde(sigma, A_, B_, w, K1) + \
               _Btilde(sigma, A_, B_, w, K1) * (K2 - w) - V
    low, high = eps, 1.0
    n = 0
    while f(high) > 0 and n < 60:
        high *= 2.0; n += 1
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
        return 0.5 * (At + st * Bt) * np.exp((K2 - w) / st) + \
               0.5 * (At - st * Bt) * np.exp(-(K2 - w) / st) - V
    low, high = eps, 1.0
    n = 0
    while f(high) > 0 and n < 60:
        high *= 2.0; n += 1
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
        return B1 - (0.5 / st) * (At + st * Bt) * np.exp((K2 - w) / st) \
                  + (0.5 / st) * (At - st * Bt) * np.exp(-(K2 - w) / st)
    low, high = sigma_h, sigma_h + 1
    n = 0
    while f(high) > 0 and n < 60:
        high *= 2.0; n += 1
    while high - low > eps:
        mid = 0.5 * (low + high)
        if f(mid) >= 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _find_y(A_, B_, K1t, K2t, V_prev, tol=1e-10):
    """Step 1.3: abscissa where the line A + (K-K1t)*B meets V^{i-1}.

    Returns (y, y_binds).  y_binds is False when the line already dominates
    V^{i-1} at K2t -- always the case when V^{i-1} == 0, which recovers the
    single-maturity path exactly.

    Boundary case.  At the wing origin the LVG boundary condition (23) forces
    V(L) = 0 for EVERY maturity, so A_ == V_prev(K1t) == 0 there.  That is
    required, not a violation: what makes V^i dominate V^{i-1} just to the
    right of L is the derivative, and Step 1's j=0 initialization
        B = d2 * Vbar(T_i,K_1)/(K_1-L) + (1-d2) * V^{i-1}_+(L)
    guarantees B > V^{i-1}_+(L) whenever the market data is strictly
    admissible (convexity of V^{i-1} with V^{i-1}(L)=0 gives
    V^{i-1}_+(L) <= V^{i-1}(K_1)/(K_1-L) < Vbar(T_i,K_1)/(K_1-L)).
    So when the values tie we test the slope instead, then seed the bisection
    strictly to the right where g > 0.
    """
    if A_ + (K2t - K1t) * B_ >= V_prev(K2t):
        return K2t, False

    def g(y):
        return A_ + (y - K1t) * B_ - V_prev(y)

    lo = K1t
    if g(lo) <= 0:
        slope_gap = B_ - V_prev.deriv(K1t)
        if slope_gap <= 0:
            raise RuntimeError(
                f"calendar ordering genuinely violated at K={K1t:.4f}: "
                f"values tie (A={A_:.6e}, V_prev={V_prev(K1t):.6e}) and "
                f"slope does not dominate (B={B_:.6e}, "
                f"V_prev'={V_prev.deriv(K1t):.6e})")
        # g is concave (linear minus a convex function) with g(K1t)=0 and
        # g'(K1t) > 0, so it is strictly positive immediately to the right.
        frac = 1e-9
        seeded = False
        while frac < 1.0:
            cand = K1t + frac * (K2t - K1t)
            if g(cand) > 0:
                lo = cand
                seeded = True
                break
            frac *= 4.0
        if not seeded:
            # never became positive inside the interval; nothing to cross
            return K2t, False

    hi = K2t
    while hi - lo > tol * max(1.0, abs(hi)):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


# How often each branch of Step 1.3 is taken.  If case (b) never fires across a
# whole run, the calendar constraint is slack everywhere and the Figure 6
# machinery is insurance rather than load-bearing.
BRANCH_COUNTS = {"a": 0, "b": 0}


def _extend_interval(A_, B_, K1t, K2t, V2t, B1, V_prev, _depth=0):
    """Extend the time value across one strike interval [K1t, K2t].

    Returns (pieces, A_next, B_next) where pieces is a list of
    (nu, omega, kappa_pair) in left-to-right order.

    Case (a)  (w <= y, always true for the first maturity):
        two pieces at nu = K1t and nu = w.  This is the original code.
    Case (b)  (w > y, Carr-Nadtochiy Step 1.3.b, Figure 6 right panel):
        one piece at nu = K1t with sigma_bar pinning V^i to V^{i-1} at K2t,
        then recurse from y -- which Lemma 23 guarantees lands in case (a).
    """
    y, y_binds = _find_y(A_, B_, K1t, K2t, V_prev)
    w = (V2t + B_ * K1t - A_ - B1 * K2t) / (B_ - B1)

    if (not y_binds) or (w <= y):
        # ---- Step 1.3.a : unchanged from process_data.py -------------------
        BRANCH_COUNTS["a"] += 1
        sigma_h = _findSigmaHat(A_, B_, w, K1t, K2t, V2t)
        sigma = _findSigma(A_, B_, w, K1t, K2t, V2t, B1, sigma_h)
        sigma_t = _sigmaTilde(sigma, A_, B_, w, K1t, K2t, V2t)
        At = _Atilde(sigma, A_, B_, w, K1t)
        Bt = _Btilde(sigma, A_, B_, w, K1t)
        pieces = [
            (K1t, 1.0 / sigma,   [0.5 * (A_ + sigma * B_), 0.5 * (A_ - sigma * B_)]),
            (w,   1.0 / sigma_t, [0.5 * (At + sigma_t * Bt), 0.5 * (At - sigma_t * Bt)]),
        ]
        return pieces, V2t, B1

    # ---- Step 1.3.b -------------------------------------------------------
    BRANCH_COUNTS["b"] += 1
    # sigma_bar solves  Atilde(s) + Btilde(s)*(K2t - y) = V^{i-1}(K2t)
    # (same functional form as _findSigmaHat, with y for w and V_prev for V).
    target = V_prev(K2t)
    sigma_bar = _findSigmaHat(A_, B_, y, K1t, K2t, target)
    At = _Atilde(sigma_bar, A_, B_, y, K1t)
    Bt = _Btilde(sigma_bar, A_, B_, y, K1t)
    first = (K1t, 1.0 / sigma_bar,
             [0.5 * (A_ + sigma_bar * B_), 0.5 * (A_ - sigma_bar * B_)])

    # restart the step from (y, At, Bt); Lemma 23 says this hits case (a)
    if _depth >= 4:
        # Lemma 23 says the restart lands in case (a), so depth should never
        # exceed 1.  If it does, the induction hypothesis has broken down.
        raise RuntimeError(
            f"Step 1.3.b recursed {_depth} deep on [{K1t:.4f}, {K2t:.4f}]")
    rest, A_next, B_next = _extend_interval(At, Bt, y, K2t, V2t, B1, V_prev,
                                            _depth + 1)
    return [first] + rest, A_next, B_next


def _interpolate(K_wing, V_wing, V_prev, delta: float = 0.5):
    """Inner loop of the LVG interpolation on one wing.

    K_wing[0] == 0 and V_wing[0] == 0 (the wing origin: L on the left wing,
    the reflection of Kbar on the right wing).  V_prev is the PREVIOUS
    maturity's time value in the SAME wing coordinate.

    Returns (nu, omega, kappa, B_last).
    """
    N = len(K_wing)
    nu, omega, kappa = [], [], []
    A_ = V_wing[0]

    # Step 1, j = 0 initialization.  The (1-delta)*V_prev'(L) term is new; it
    # is zero for the first maturity, recovering the original expression.
    B_ = delta * (V_wing[1] - V_wing[0]) / (K_wing[1] - K_wing[0]) \
         + (1.0 - delta) * V_prev.deriv(K_wing[0])

    for i in range(N - 2):
        K1t, K2t = K_wing[i], K_wing[i + 1]
        V1t, V2t = V_wing[i], V_wing[i + 1]
        B1 = delta * (V_wing[i + 2] - V2t) / (K_wing[i + 2] - K2t) + \
             (1.0 - delta) * (V2t - V1t) / (K2t - K1t)
        pieces, A_, B_ = _extend_interval(A_, B_, K1t, K2t, V2t, B1, V_prev)
        for (n_, o_, k_) in pieces:
            nu.append(n_); omega.append(o_); kappa.append(k_)

    return nu, omega, kappa, B_


# =============================================================================
# prob5b, multi-maturity
# =============================================================================

def run_prob5b_multi(call: np.ndarray, grid: np.ndarray, S: float, Kbar: float,
                     prev_wings: Optional[tuple] = None, delta: float = 0.5):
    """LVG interpolation for ONE maturity, dominating the previous maturity.

    call      arb-free adjusted prices on the union grid
    prev_wings  (left_wing, right_wing) of maturity i-1, or None for i = 0

    Returns (theta_dict, (left_wing, right_wing)).
    """
    C = list(call)
    K = list(grid)
    N = len(C)

    if prev_wings is None:
        Vp1 = _ZeroWing()
        Vp2 = _ZeroWing()
    else:
        Vp1 = _PrevWing(prev_wings[0])
        Vp2 = _PrevWing(prev_wings[1])

    # locate the interval straddling S
    i0 = 0
    for i in range(N - 1):
        if (K[i] < S) and (S < K[i + 1]):
            i0 = i

    # value at S: bracketed below by the two local linear extrapolations and
    # above by the chord.  NEW: the lower bracket is also floored at the
    # previous maturity's time value, so V^i(S) > V^{i-1}(S) by construction
    # (Carr-Nadtochiy Step 0, where x is inserted as an extra strike).
    Lt1 = C[i0] + (S - K[i0]) * (C[i0] - C[i0 - 1]) / (K[i0] - K[i0 - 1])
    Lt2 = C[i0 + 1] + (S - K[i0 + 1]) * (C[i0 + 2] - C[i0 + 1]) / (K[i0 + 2] - K[i0 + 1])
    Lt = max(Lt1, Lt2)
    Ut = C[i0] * (K[i0 + 1] - S) / (K[i0 + 1] - K[i0]) + \
         C[i0 + 1] * (S - K[i0]) / (K[i0 + 1] - K[i0])
    V_prev_at_S = Vp1(S)
    if V_prev_at_S >= Ut:
        raise RuntimeError(
            f"no room at S0: V^(i-1)(S0)={V_prev_at_S:.6f} >= chord {Ut:.6f}. "
            "The joint LP produced prices too tight against the previous "
            "maturity near the money.")
    Lt = max(Lt, V_prev_at_S)
    Vtemp = delta * Lt + (1 - delta) * Ut

    # left wing: absolute strikes, V = C - (S-K)
    K1 = [0.0]; V1 = [0.0]
    for i in range(N):
        if K[i] < S:
            K1.append(K[i]); V1.append(C[i] - S + K[i])
    K1.append(S); V1.append(Vtemp)

    # right wing: reflected coordinate u = Kbar - K, V = C
    K2 = [0.0]; V2 = [0.0]
    for i in range(N):
        if K[N - 1 - i] > S:
            K2.append(Kbar - K[N - 1 - i]); V2.append(C[N - 1 - i])
    K2.append(Kbar - S); V2.append(Vtemp)

    nu1, omega1, kappa1, B1_last = _interpolate(K1, V1, Vp1, delta)
    nu2, omega2, kappa2, B2_last = _interpolate(K2, V2, Vp2, delta)

    # split the size-1 kink at S between the two wings (Step 3)
    rho = 0.5 + 0.5 * (V2[-2] - V2[-1]) / (K2[-1] - K2[-2]) + \
                0.5 * (V1[-1] - V1[-2]) / (K1[-1] - K1[-2])

    def _append_final(nu_l, omega_l, kappa_l, K_wing, V_wing, B_last, B1, V_prev):
        K1t, K2t = K_wing[-2], K_wing[-1]
        A_, B_ = V_wing[-2], B_last
        pieces, _, _ = _extend_interval(A_, B_, K1t, K2t, V_wing[-1], B1, V_prev)
        for (n_, o_, k_) in pieces:
            nu_l.append(n_); omega_l.append(o_); kappa_l.append(k_)

    _append_final(nu1, omega1, kappa1, K1, V1, B1_last, rho, Vp1)
    _append_final(nu2, omega2, kappa2, K2, V2, B2_last, 1 - rho, Vp2)

    left_wing = (nu1, omega1, kappa1)
    right_wing = (nu2, omega2, kappa2)

    initial_nus1 = list(nu1)
    initial_sigs1 = [1.0 / w for w in omega1]
    initial_nus2 = [Kbar - x for x in reversed(nu2[1:])] + [Kbar]
    initial_sigs2 = [1.0 / w for w in reversed(omega2)]

    def _dump(wing):
        nu, omega, kappa = wing
        return {"nu": [float(x) for x in nu],
                "omega": [float(x) for x in omega],
                "kappa": [[float(k[0]), float(k[1])] for k in kappa]}

    theta = {
        "S0": float(S), "Kbar": float(Kbar),
        "initial_nus1": [float(x) for x in initial_nus1],
        "initial_sigs1": [float(x) for x in initial_sigs1],
        "initial_nus2": [float(x) for x in initial_nus2],
        "initial_sigs2": [float(x) for x in initial_sigs2],
        # Piecewise-exponential coefficients for both wings, in WING
        # coordinates (left wing: absolute strike; right wing: Kbar - K).
        # theta_init_data.json in the single-maturity pipeline threw these
        # away; we keep them so V(K) and C(K) can be evaluated downstream
        # without re-running the recursion.
        "wing_left": _dump(left_wing),
        "wing_right": _dump(right_wing),
    }
    return theta, (left_wing, right_wing)


def time_value(wings, S: float, Kbar: float, x: float) -> float:
    """V(x) in absolute strike coordinates, for checking the ordering."""
    left, right = wings
    if x <= S:
        return eval_wing(left, x)
    return eval_wing(right, Kbar - x)


def check_ordering(all_wings: list, S: float, Kbar: float, n_pts: int = 400):
    """Max violation of V^i >= V^{i-1} on a dense grid.  Should be <= 0."""
    xs = np.linspace(1e-6, Kbar - 1e-6, n_pts)
    worst = -np.inf
    at = None
    for i in range(1, len(all_wings)):
        for x in xs:
            gap = time_value(all_wings[i - 1], S, Kbar, x) - \
                  time_value(all_wings[i], S, Kbar, x)
            if gap > worst:
                worst, at = gap, (i, float(x))
    return float(worst), at


# =============================================================================
# Per-day orchestration
# =============================================================================

def process_day_multi(df_day: pd.DataFrame, trade_date: int, cfg: Config,
                      out_dir: str) -> dict:
    exps = pick_expiries(df_day, trade_date, cfg.maturity_targets_bdays)
    if len(exps) < cfg.min_maturities:
        return {"status": "skip", "reason": f"only {len(exps)} expiries"}

    # per-maturity parity fit
    mats = []
    for exdate, T_bd in exps:
        pairs = filter_strikes(df_day[df_day["exdate"] == exdate])
        if len(pairs) < cfg.min_pairs:
            continue
        try:
            D, F, resid = estimate_DF(pairs)
        except ValueError:
            continue
        mats.append({"exdate": exdate, "T_bdays": T_bd, "pairs": pairs,
                     "D": D, "F": F, "parity_resid": resid})
    if len(mats) < cfg.min_maturities:
        return {"status": "skip", "reason": f"only {len(mats)} usable maturities"}

    # Common S0.  Any positive constant works -- it is a pure scale on the
    # normalized martingale -- but taking the nearest maturity's forward keeps
    # the shortest smile on its own quoted strikes, i.e. exactly the grid the
    # single-maturity pipeline would have used.
    S0 = mats[0]["F"]

    per_mat_adj = []
    for m in mats:
        adj = build_adjusted_calls_common(m["pairs"], m["D"], m["F"], S0)
        if len(adj) < cfg.min_pairs:
            return {"status": "skip",
                    "reason": f"maturity {m['exdate']} lost too many envelopes"}
        per_mat_adj.append(adj)

    grid = build_union_grid([a["kappa"].to_numpy() for a in per_mat_adj],
                            cfg.union_tol)
    if len(grid) < 5:
        return {"status": "skip", "reason": "union grid too small"}

    lbs, ubs, has_quote, box_diag = assemble_boxes(
        grid, per_mat_adj, S0, cfg.union_tol,
        cfg.synthetic_band_trivial, cfg.synthetic_widen)

    # The joint polytope can be genuinely empty: quotes for different
    # maturities are not recorded simultaneously, so a later maturity's ask can
    # sit below an earlier maturity's bid at the same adjusted strike.  The
    # paper hits exactly this (Remark 19) and responds by using fewer
    # maturities.  Drop from the LONG end and retry rather than failing the day
    # or loosening the calendar constraint.
    n_dropped = 0
    while True:
        try:
            prices, diag = run_prob5a_joint(grid, lbs, ubs, S0, cfg.Kbar, cfg)
            break
        except Exception as e:
            if len(mats) - n_dropped - 1 < cfg.min_maturities:
                return {"status": "fail_5a",
                        "reason": f"{e} (after dropping {n_dropped})"}
            n_dropped += 1
            mats = mats[:-1]
            per_mat_adj = per_mat_adj[:-1]
            lbs = lbs[:-1]
            ubs = ubs[:-1]
            has_quote = has_quote[:-1]

    Kbar_eff = max(cfg.Kbar, float(grid[-1]) + 100.0)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "union_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for k in grid:
            w.writerow([f"{k:.6f}"])

    all_wings = []
    prev = None
    per_mat_summary = []
    BRANCH_COUNTS["a"] = 0
    BRANCH_COUNTS["b"] = 0
    for i, m in enumerate(mats):
        sub = os.path.join(out_dir, f"maturity_{i}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "Quotes.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for j, k in enumerate(grid):
                a = "" if not has_quote[i][j] else f"{ubs[i][j]:.4f}"
                b = "" if not has_quote[i][j] else f"{lbs[i][j]:.4f}"
                w.writerow([f"{k:.4f}", a, b])
        with open(os.path.join(sub, "ArbFree.csv"), "w", newline="") as f:
            w = csv.writer(f)
            for c_val, k_val in zip(prices[i], grid):
                w.writerow([f"{c_val:.6f}", f"{k_val:.6f}"])

        try:
            theta, wings = run_prob5b_multi(prices[i], grid, S0, Kbar_eff, prev)
        except Exception as e:
            return {"status": "fail_5b", "reason": f"maturity {i}: {e}"}

        # t_star = T_i - T_{i-1}, in years, for the non-homogeneous LVG step
        T_yr = m["T_bdays"] / 252.0
        T_prev_yr = mats[i - 1]["T_bdays"] / 252.0 if i > 0 else 0.0
        theta["T_years"] = float(T_yr)
        theta["t_star"] = float(T_yr - T_prev_yr)
        theta["exdate"] = int(m["exdate"])
        with open(os.path.join(sub, "theta_init.json"), "w") as f:
            json.dump(theta, f, indent=2)

        all_wings.append(wings)
        prev = wings
        per_mat_summary.append({
            "exdate": int(m["exdate"]), "T_bdays": int(m["T_bdays"]),
            "D": m["D"], "F": m["F"], "parity_resid_max": m["parity_resid"],
            "n_quoted": int(has_quote[i].sum()),
            "n_left_nodes": len(theta["initial_nus1"]),
            "n_right_nodes": len(theta["initial_nus2"]),
        })

    worst_gap, worst_at = check_ordering(all_wings, S0, Kbar_eff)

    summary = {
        "trade_date": int(trade_date),
        "S0": float(S0), "Kbar": float(Kbar_eff),
        "n_union_strikes": int(len(grid)),
        "n_maturities": len(mats),
        "maturities": per_mat_summary,
        "lp": diag,
        "boxes": box_diag,
        "n_maturities_dropped": n_dropped,
        "max_ordering_violation": worst_gap,
        "max_ordering_violation_at": worst_at,
        "step13a_count": BRANCH_COUNTS["a"],
        "step13b_count": BRANCH_COUNTS["b"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return {"status": "ok", **summary}


def main(cfg: Optional[Config] = None):
    if cfg is None:
        cfg = Config()
    print(f"Loading {cfg.spx_csv} ...")
    df = load_spx_csv(cfg.spx_csv)
    print(f"  loaded {len(df):,} rows")

    days = get_trading_days(cfg.start_date, cfg.n_days)
    by_date = {d: g for d, g in df.groupby("date")}

    statuses = []
    for k, d in enumerate(days, start=1):
        if d not in by_date:
            statuses.append({"date": d, "status": "skip", "reason": "no rows"})
            print(f"[{k:>3}/{len(days)}] {d}  SKIP (no rows)")
            continue
        out_dir = os.path.join(cfg.out_root, str(d))
        try:
            res = process_day_multi(by_date[d], d, cfg, out_dir)
        except Exception as e:
            res = {"status": "fail", "reason": f"unhandled: {e}"}
        if res.get("status") == "ok":
            print(f"[{k:>3}/{len(days)}] {d}  M={res['n_maturities']} "
                  f"N_union={res['n_union_strikes']} S0={res['S0']:.2f} "
                  f"cal_slack={res['lp'].get('min_calendar_slack', float('nan')):.2e} "
                  f"order_viol={res['max_ordering_violation']:.2e}")
        else:
            print(f"[{k:>3}/{len(days)}] {d}  {res.get('status')}: {res.get('reason','')}")
        statuses.append({"date": d, **res})

    os.makedirs(cfg.out_root, exist_ok=True)
    with open(os.path.join(cfg.out_root, "_index.json"), "w") as f:
        json.dump(statuses, f, indent=2)
    n_ok = sum(1 for s in statuses if s.get("status") == "ok")
    print(f"\nDONE: {n_ok}/{len(statuses)} days processed")


if __name__ == "__main__":
    main()
