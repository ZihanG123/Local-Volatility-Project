"""
lvg_optim.py
============

LVG/CMA-ES optimization module extracted from cma-es_v5_sigma_VCpp.ipynb.
The notebook and batch driver both import from this so there is one source
of truth for the optimization code.
"""


# ========================================================================
# (from notebook cell 2)
# ========================================================================

import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt

# Infeasible candidates can push sigma toward 0 inside the coefficient
# propagation, producing harmless overflow/invalid warnings (they are caught
# and penalized). Silence them so the optimization log stays readable.
np.seterr(over='ignore', invalid='ignore', divide='ignore')


# ========================================================================
# (from notebook cell 4)
# ========================================================================

def softplus(x):
    """
    Numerically stable softplus: log(1 + exp(x)).
    Clips input to [-500, 20] before computing to avoid overflow,
    and uses the identity softplus(x) ≈ x for x > 20.
    """
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def inv_softplus(x):
    """
    Inverse softplus: log(exp(x) - 1).
    Converts an actual sigma value back to raw form so that
    softplus(raw) + 1e-6 recovers the original sigma.
    """
    return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-8, 500))))


# ========================================================================
# (from notebook cell 6)
# ========================================================================

def _compute_wing_coefficients(theta, R1, R2, S0):
    """
    Compute all LVG model coefficients from a parameter vector theta.

    This is the shared computation used by calculate_J, compute_model_calls,
    and the plotting functions. Factored out to avoid code duplication.

    Parameters
    ----------
    theta : ndarray (2*R1 + 2*R2,)
    R1, R2 : int   number of left/right partition points
    S0 : float     spot price (junction between left and right wings)

    Returns
    -------
    coeff_left  : ndarray (R1, 2)  distance-scaled coefficients, left wing
    coeff_right : ndarray (R2, 2)  distance-scaled coefficients, right wing
    lambda1     : float  left wing amplitude (from C1-matching at S0)
    lambda2     : float  right wing amplitude (from C1-matching at S0)
    left_nodes  : ndarray (R1+1,)  partition points with S0 appended
    right_nodes : ndarray (R2+1,)  partition points with S0 prepended
    local_vols_left  : ndarray (R1,)  actual sigma values, left wing
    local_vols_right : ndarray (R2,)  actual sigma values, right wing
    """

    # ── Unpack theta ──────────────────────────────────────────────────────
    partition_pts_left   = theta[:R1]
    local_vols_left_raw  = theta[R1 : 2*R1]
    partition_pts_right  = theta[2*R1 : 2*R1 + R2]
    local_vols_right_raw = theta[2*R1 + R2 :]

    # S0 serves as the shared boundary between left and right wings:
    #   left wing  covers [0, S0]:   nodes = partition_pts_left  + [S0]
    #   right wing covers [S0, Kbar]: nodes = [S0] + partition_pts_right
    left_nodes  = np.append(partition_pts_left,   S0)   # length R1+1
    right_nodes = np.insert(partition_pts_right, 0, S0)  # length R2+1

    # Convert raw values to actual positive local volatilities
    local_vols_left  = softplus(local_vols_left_raw)  + 1e-6  # shape (R1,)
    local_vols_right = softplus(local_vols_right_raw) + 1e-6  # shape (R2,)

    # ── Left wing: propagate L -> R with unit scale (lambda1 = 1) ────────
    #
    # Boundary condition at K = 0 (left edge):
    #   V(0) = 0  (time value is zero at zero strike)
    #   V'(0) > 0 (time value increases as strike increases from 0)
    #
    # These give the unit starting coefficients for interval 0:
    #   coeff_left[0,0] = -exp(-dist_0 / sigma_0)
    #   coeff_left[0,1] = +exp(+dist_0 / sigma_0)
    # (One can verify: V(0) = coeff[0]*exp(0) + coeff[1]*exp(0) = 0 + ... wait)
    # Actually V at LEFT endpoint of interval 0 means K=left_nodes[0]=0:
    #   V(left_nodes[0]) = cd1*exp(0) + cd2*exp(0) = cd1 + cd2
    # With cd1 = coeff[0,0]*exp(dist/sig) = -1 and cd2 = coeff[0,1]*exp(-dist/sig) = 1
    # So V(0) = -1 + 1 = 0 ✓

    coeff_left    = np.zeros((R1, 2))
    dist_first    = left_nodes[1] - left_nodes[0]
    sigma_first   = local_vols_left[0]
    coeff_left[0, 0] = -np.exp(-dist_first / sigma_first)
    coeff_left[0, 1] =  np.exp( dist_first / sigma_first)

    # Propagate left to right: at each partition point nu_{j+1},
    # enforce C1 continuity (V and V' match across intervals)
    for j in range(R1 - 1):
        # V and V' at the RIGHT endpoint of interval j
        # (= left endpoint of interval j+1 = nu_{j+1})
        V_right_j  = coeff_left[j, 0] + coeff_left[j, 1]
        Vp_right_j = (1.0 / local_vols_left[j]) * (
                         -coeff_left[j, 0] + coeff_left[j, 1])

        # Set distance-scaled coefficients for interval j+1
        # using the C1 matching conditions
        dist_next  = left_nodes[j+2] - left_nodes[j+1]
        sigma_next = local_vols_left[j+1]

        coeff_left[j+1, 0] = 0.5 * (V_right_j - sigma_next * Vp_right_j) \
                              * np.exp(-dist_next / sigma_next)
        coeff_left[j+1, 1] = 0.5 * (V_right_j + sigma_next * Vp_right_j) \
                              * np.exp( dist_next / sigma_next)

    # ── Right wing: propagate R -> L with unit scale (lambda2 = 1) ───────
    #
    # Boundary condition at K = Kbar (right edge):
    #   V(Kbar) = 0  (call price = intrinsic = 0 at Kbar, far OTM)
    #   V'(Kbar) < 0 (time value decreases as strike approaches Kbar)

    coeff_right    = np.zeros((R2, 2))
    dist_last      = right_nodes[-1] - right_nodes[-2]
    sigma_last     = local_vols_right[-1]
    coeff_right[-1, 0] =  np.exp( dist_last / sigma_last)
    coeff_right[-1, 1] = -np.exp(-dist_last / sigma_last)

    # Propagate right to left: enforce C1 continuity at each partition point
    for j in range(R2 - 1, 0, -1):
        V_left_j  = coeff_right[j, 0] + coeff_right[j, 1]
        Vp_left_j = (1.0 / local_vols_right[j]) * (
                        -coeff_right[j, 0] + coeff_right[j, 1])

        dist_prev  = right_nodes[j] - right_nodes[j-1]
        sigma_prev = local_vols_right[j-1]

        coeff_right[j-1, 0] = 0.5 * (V_left_j - sigma_prev * Vp_left_j) \
                               * np.exp( dist_prev / sigma_prev)
        coeff_right[j-1, 1] = 0.5 * (V_left_j + sigma_prev * Vp_left_j) \
                               * np.exp(-dist_prev / sigma_prev)

    # ── Solve for lambda1 and lambda2 at S0 ──────────────────────────────
    #
    # The full call price is: C(K) = lambda1*V_left(K) + (S0-K)+   for K < S0
    #                                C(K) = lambda2*V_right(K)       for K > S0
    #
    # At K = S0, we need C1 continuity of C(K):
    #   Value:      lambda1 * V_left(S0) = lambda2 * V_right(S0)
    #   Derivative: lambda1 * V'_left(S0) - lambda2 * V'_right(S0) = 1
    #               (the +1 comes from the -1 kink in the derivative of (S0-K)+)
    #
    # Solving this 2x2 linear system gives lambda1 and lambda2.

    V_left_S0   = coeff_left[-1, 0]  + coeff_left[-1, 1]
    Vp_left_S0  = (1.0 / local_vols_left[-1]) * (
                      -coeff_left[-1, 0] + coeff_left[-1, 1])

    V_right_S0  = coeff_right[0, 0]  + coeff_right[0, 1]
    Vp_right_S0 = (1.0 / local_vols_right[0]) * (
                      -coeff_right[0, 0] + coeff_right[0, 1])

    denom   = Vp_left_S0 * V_right_S0 - V_left_S0 * Vp_right_S0
    lambda1 = V_right_S0 / denom
    lambda2 = V_left_S0  / denom

    # Scale all coefficients by their respective lambdas
    coeff_left  = coeff_left  * lambda1
    coeff_right = coeff_right * lambda2

    return (coeff_left, coeff_right,
            float(lambda1), float(lambda2),
            left_nodes, right_nodes,
            local_vols_left, local_vols_right)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Evaluate Model Call Prices at Market Strikes
#
# Given a theta vector, reconstruct C(K) at each observed market strike.
# This is used both for the bid-ask feasibility check and for plotting.
# ══════════════════════════════════════════════════════════════════════════════


# ========================================================================
# (from notebook cell 8)
# ========================================================================

def compute_model_calls(theta, market_strikes, R1, R2, S0):
    """
    Compute LVG model call prices C(K) at a set of market strikes.

    C(K) = lambda1 * V_left(K) + (S0 - K)   for K < S0  (left wing)
    C(K) = lambda2 * V_right(K)              for K >= S0 (right wing)

    Parameters
    ----------
    theta          : ndarray (2*R1 + 2*R2,)
    market_strikes : array-like  strike prices to evaluate at
    R1, R2         : int
    S0             : float

    Returns
    -------
    model_calls : ndarray  C(K) at each market strike
    """
    (coeff_left, coeff_right, lambda1, lambda2,
     left_nodes, right_nodes,
     local_vols_left, local_vols_right) = _compute_wing_coefficients(
                                               theta, R1, R2, S0)

    model_calls = []

    for K in market_strikes:
        if K < S0:
            # ── Left wing ──────────────────────────────────────────────
            # Find which interval K falls in
            j = int(np.searchsorted(left_nodes, K, side='right')) - 1
            j = np.clip(j, 0, R1 - 1)

            sig        = local_vols_left[j]
            dist_total = left_nodes[j+1] - left_nodes[j]

            # Recover true coefficients at the LEFT endpoint of interval j
            # (distance-scaled coefficients are relative to the right endpoint)
            cd1 = coeff_left[j, 0] * np.exp( dist_total / sig)
            cd2 = coeff_left[j, 1] * np.exp(-dist_total / sig)

            # Evaluate V(K) as distance from left endpoint of interval
            dist_k = K - left_nodes[j]
            V_K    = cd1 * np.exp(-dist_k / sig) + cd2 * np.exp(dist_k / sig)

            # Call price = time value + intrinsic
            C_K = V_K + (S0 - K)
            model_calls.append(C_K)

        else:
            # ── Right wing ─────────────────────────────────────────────
            j = int(np.searchsorted(right_nodes, K, side='right')) - 1
            j = np.clip(j, 0, R2 - 1)

            sig        = local_vols_right[j]
            dist_total = right_nodes[j+1] - right_nodes[j]

            cd1 = coeff_right[j, 0] * np.exp(-dist_total / sig)
            cd2 = coeff_right[j, 1] * np.exp( dist_total / sig)

            dist_k = K - right_nodes[j]
            V_K    = cd1 * np.exp(-dist_k / sig) + cd2 * np.exp(dist_k / sig)

            # For right wing (OTM calls), intrinsic = 0
            model_calls.append(V_K)

    return np.array(model_calls)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Objective Function J(theta)
#
# J measures the smoothness of C''(K), which is proportional to the
# risk-neutral density. A smooth C'' means no arbitrage opportunities
# and a well-behaved implied volatility surface.
#
# At each partition point nu_j:
#   C''(nu_j from left)  = (1/sigma_j^2) * V(nu_j)
#   C''(nu_j from right) = (1/sigma_{j+1}^2) * V(nu_j)
#   Jump = C''(right) - C''(left)
#
# J = sum of (Jump)^2 over all partition points (including the S0 junction)
# ══════════════════════════════════════════════════════════════════════════════


# ========================================================================
# (from notebook cell 10)
# ========================================================================

def calculate_J(theta, R1, R2, S0):
    """
    Compute J(theta) = sum of squared C'' jumps at all partition points.

    Matches teammates' PyTorch calculate_J exactly.

    Parameters
    ----------
    theta : ndarray (2*R1 + 2*R2,)
    R1, R2 : int
    S0 : float

    Returns
    -------
    J_value : float  the smoothness penalty (lower is better)
    lambda1 : float  left wing scaling constant
    lambda2 : float  right wing scaling constant
    """
    (coeff_left, coeff_right, lambda1, lambda2,
     left_nodes, right_nodes,
     local_vols_left, local_vols_right) = _compute_wing_coefficients(
                                               theta, R1, R2, S0)

    all_jumps = []

    # ── Left wing internal jumps ──────────────────────────────────────────
    # At each partition point left_nodes[j+1] (for j = 0,...,R1-2):
    #   C''(from left)  = (1/sigma_j^2)     * V(left_nodes[j+1])  [end of interval j]
    #   C''(from right) = (1/sigma_{j+1}^2) * V(left_nodes[j+1])  [start of interval j+1]
    #
    # V at end of interval j   = coeff_left[j,0] + coeff_left[j,1]    (distance-scaled)
    # V at start of interval j+1 requires un-scaling by exp(dist/sig)

    for j in range(R1 - 1):
        # C'' at end of interval j (approaching from the left)
        Cpp_from_left = (1.0 / local_vols_left[j]**2) * (
                            coeff_left[j, 0] + coeff_left[j, 1])

        # V at the START of interval j+1 (= left endpoint of interval j+1)
        # coeff_left[j+1] are distance-scaled relative to the right endpoint,
        # so we multiply by exp(dist/sig) to get values at the left endpoint
        dist_j1  = left_nodes[j+2] - left_nodes[j+1]
        sigma_j1 = local_vols_left[j+1]
        V_start_j1 = (coeff_left[j+1, 0] * np.exp( dist_j1 / sigma_j1) +
                      coeff_left[j+1, 1] * np.exp(-dist_j1 / sigma_j1))
        Cpp_from_right = (1.0 / sigma_j1**2) * V_start_j1

        all_jumps.append(Cpp_from_right - Cpp_from_left)

    # ── Junction jump at S0 ───────────────────────────────────────────────
    # The left and right wings meet at S0. Even though V is continuous,
    # C'' can still jump if the local volatility changes across S0.
    Cpp_left_S0  = (1.0 / local_vols_left[-1]**2) * (
                       coeff_left[-1, 0] + coeff_left[-1, 1])
    Cpp_right_S0 = (1.0 / local_vols_right[0]**2) * (
                       coeff_right[0, 0] + coeff_right[0, 1])
    all_jumps.append(Cpp_right_S0 - Cpp_left_S0)

    # ── Right wing internal jumps ─────────────────────────────────────────
    # Same logic as left wing, but propagation direction is reversed.
    # coeff_right[j] are distance-scaled relative to the LEFT endpoint,
    # so evaluating at the RIGHT endpoint = coeff[j,0] + coeff[j,1] directly.

    for j in range(R2 - 1):
        # C'' at start of interval j+1 (approaching from the right)
        Cpp_from_right = (1.0 / local_vols_right[j+1]**2) * (
                             coeff_right[j+1, 0] + coeff_right[j+1, 1])

        # V at the END of interval j (= right endpoint of interval j)
        dist_j  = right_nodes[j+1] - right_nodes[j]
        sigma_j = local_vols_right[j]
        V_end_j = (coeff_right[j, 0] * np.exp(-dist_j / sigma_j) +
                   coeff_right[j, 1] * np.exp( dist_j / sigma_j))
        Cpp_from_left = (1.0 / sigma_j**2) * V_end_j

        all_jumps.append(Cpp_from_right - Cpp_from_left)

    J_value = float(np.sum(np.array(all_jumps)**2))
    return J_value, float(lambda1), float(lambda2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Feasibility Check
#
# CMA-ES works in unconstrained space (it can propose any theta).
# Rather than transforming the parameter space to enforce constraints,
# we check feasibility after sampling and assign a large penalty to
# infeasible candidates — they get killed in the selection step.
#
# Three types of constraints:
#   1. Ordering: partition points must be strictly increasing
#   2. Boundary: left points < S0, right points end at Kbar
#   3. Bid-ask: model prices must lie within market bid-ask spread
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b: New objective — V/C'' (local-variance) smoothness, sigmas only
#
# On each interval  C''(K) = V(K)/sigma_j^2  =>  V(K)/C''(K) = sigma_j^2.
# The jump of V/C'' across node nu_j is (sigma_{j+1}^2 - sigma_j^2): depends on
# SIGMAS ONLY, not on V(nu_j) or the node location.
#
#     J_smooth(theta) = sum_j (sigma_{j+1}^2 - sigma_j^2)^2
#
# OPTIONAL liquid-range mask: the deep tails carry enormous sigma (sigma~600
# near K=0), so their sigma^2 jumps swamp J even though they sit far outside the
# traded strikes. Passing strike_lo/strike_hi zeroes the out-of-range nodes so
# the objective targets the strikes that actually matter for the smile.
# ══════════════════════════════════════════════════════════════════════════════

def _node_positions(theta, R1, R2, S0):
    """Strike location of each of the R1+R2-1 jump nodes (same order as the
    jump vector): left internal nodes, the S0 junction, right internal nodes."""
    nus1 = theta[:R1]
    nus2 = theta[2*R1 : 2*R1 + R2]
    return np.concatenate([nus1[1:R1], [S0], nus2[0:R2-1]])

def _mask_weights(positions, strike_lo, strike_hi, mask_mode='hard',
                  taper_width=30.0, out_floor=0.2):
    """Per-node weights w_j in [0, 1] for the smoothness penalty.

    mask_mode='hard' (default, original behavior):
        1.0 inside [lo, hi], 0.0 outside.
    mask_mode='soft':
        1.0 inside [lo, hi]
        cosine taper from 1.0 down to `out_floor` over `taper_width` strike
        units on each side
        `out_floor` for nodes more than taper_width past the boundary

    The soft variant lets the optimizer trade an in-range cliff (weight 1) for
    a slightly-outside jump (weight ~out_floor), so a single dominant cliff
    can DISPERSE outward instead of being a 0-or-1 indicator.
    """
    pos = np.asarray(positions, dtype=float)
    if mask_mode == 'hard':
        return ((pos >= strike_lo) & (pos <= strike_hi)).astype(float)
    if mask_mode != 'soft':
        raise ValueError(f"unknown mask_mode {mask_mode!r}; use 'hard' or 'soft'")
    w = np.ones_like(pos)
    # left taper: from (strike_lo - taper_width) up to strike_lo
    in_left  = pos < strike_lo
    in_band_left = in_left & (pos >= strike_lo - taper_width)
    in_far_left  = in_left & (pos <  strike_lo - taper_width)
    t = (strike_lo - pos[in_band_left]) / taper_width    # 0 at boundary, 1 at far edge
    w[in_band_left] = out_floor + (1.0 - out_floor) * 0.5*(1 + np.cos(np.pi*t))
    w[in_far_left]  = out_floor
    # right taper: mirror
    in_right = pos > strike_hi
    in_band_right = in_right & (pos <= strike_hi + taper_width)
    in_far_right  = in_right & (pos >  strike_hi + taper_width)
    t = (pos[in_band_right] - strike_hi) / taper_width
    w[in_band_right] = out_floor + (1.0 - out_floor) * 0.5*(1 + np.cos(np.pi*t))
    w[in_far_right]  = out_floor
    return w


def _sigma2_jump_vector(theta, R1, R2, S0=None, strike_lo=None, strike_hi=None,
                        mask_mode='hard', taper_width=30.0, out_floor=0.2):
    """Vector of (sigma^2) jumps across every partition node (length R1+R2-1).

    If strike_lo/strike_hi/S0 are given, jumps are multiplied by per-node mask
    weights -- a hard {0,1} indicator by default, or a soft cosine taper with
    a floor when mask_mode='soft'. The soft mask lets the optimizer disperse
    a dominant cliff at the strike boundary outward.
    """
    sig1 = softplus(theta[R1 : 2*R1])             + 1e-6
    sig2 = softplus(theta[2*R1 + R2 : 2*R1+2*R2]) + 1e-6
    v1, v2 = sig1**2, sig2**2
    jumps = np.concatenate([np.diff(v1), [v2[0] - v1[-1]], np.diff(v2)])
    if strike_lo is not None and strike_hi is not None and S0 is not None:
        pos = _node_positions(theta, R1, R2, S0)
        w   = _mask_weights(pos, strike_lo, strike_hi,
                            mask_mode=mask_mode,
                            taper_width=taper_width, out_floor=out_floor)
        jumps = jumps * w
    return jumps

def calculate_J_smooth(theta, R1, R2, S0, strike_lo=None, strike_hi=None,
                       mode='L2', l1_weight=1.0,
                       mask_mode='hard', taper_width=30.0, out_floor=0.2):
    """V/C'' (local-variance) smoothness penalty over sigma^2 jumps, optionally
    restricted to the liquid strike band [strike_lo, strike_hi].

    mode
    ----
    'L2'      : sum of SQUARED jumps      sum_j (Δσ²_j)²              (default)
                   - punishes big jumps disproportionately
                   - one cliff dominates everything
    'L1'      : sum of ABSOLUTE jumps     sum_j |Δσ²_j|
                   - treats jumps democratically
                   - drives many jumps to EXACTLY zero (sparsity)
                   - tends to produce piecewise-constant sigma² "regimes"
                   - non-differentiable at zero -- fine for CMA-ES, not for
                     gradient methods
    'elastic' : L1 + l1_weight * L2 hybrid
                   - L1 picks the sparsity pattern (which jumps survive)
                   - L2 smooths within it
                   - l1_weight relative scaling (1.0 default)

    Note on scales: L1 has units of σ² (~thousands here) while L2 has units
    of σ⁴ (~millions). They are NOT directly comparable across modes -- the
    'masked J' number means different things under different modes. Always
    compare runs within a single mode.
    """
    j = _sigma2_jump_vector(theta, R1, R2, S0, strike_lo, strike_hi,
                            mask_mode=mask_mode,
                            taper_width=taper_width, out_floor=out_floor)
    if mode == 'L2':
        return float(np.sum(j**2))
    elif mode == 'L1':
        return float(np.sum(np.abs(j)))
    elif mode == 'elastic':
        return float(np.sum(np.abs(j)) + l1_weight * np.sum(j**2))
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'L2', 'L1', or 'elastic'")


# ========================================================================
# (from notebook cell 12)
# ========================================================================

INFEASIBLE_PENALTY = 1e6

def is_feasible(theta, R1, R2, S0, Kbar,
                market_strikes=None, bid_prices=None, ask_prices=None):
    """
    Check whether a candidate theta satisfies all model constraints.

    Parameters
    ----------
    theta          : ndarray (2*R1 + 2*R2,)
    R1, R2         : int
    S0, Kbar       : float
    market_strikes : array-like or None   observed market strikes
    bid_prices     : array-like or None   bid prices at each strike
    ask_prices     : array-like or None   ask prices at each strike

    Returns
    -------
    bool  True if all constraints are satisfied, False otherwise
    """
    partition_pts_left  = theta[:R1]
    partition_pts_right = theta[2*R1 : 2*R1 + R2]

    # ── Constraint 1: Left partition points strictly increasing ───────────
    if np.any(np.diff(partition_pts_left) <= 0):
        return False

    # ── Constraint 2: All left partition points must be below S0 ──────────
    if np.any(partition_pts_left >= S0):
        return False

    # ── Constraint 3: Right partition points strictly increasing ──────────
    if np.any(np.diff(partition_pts_right) <= 0):
        return False

    # ── Constraint 4: Last right partition point must equal Kbar ──────────
    # Allow a small tolerance of 1.0 since Kbar=2000 is fixed and we
    # do not want to penalize floating-point rounding errors near the boundary.
    if abs(partition_pts_right[-1] - Kbar) > 1.0:
        return False

    # ── Constraint 5: Bid-ask feasibility (if market data provided) ───────
    # The model call prices C(K) must lie within the observed bid-ask spread
    # at every market strike. This ensures the optimized model is consistent
    # with market prices and no-arbitrage conditions from the 5a step.
    if market_strikes is not None and bid_prices is not None and ask_prices is not None:
        try:
            model_calls = compute_model_calls(theta, market_strikes, R1, R2, S0)

            # Check each strike: bid <= C(K) <= ask
            tol = 1.5
            if np.any(model_calls < bid_prices - tol) or np.any(model_calls > ask_prices + tol):
                return False
        except Exception:
            # If computation fails (e.g., numerical overflow), reject candidate
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Candidate Evaluation
#
# Wraps the feasibility check and J computation into a single function.
# This is the function passed to the CMA-ES core as the objective.
# ══════════════════════════════════════════════════════════════════════════════

# Large constant assigned to infeasible or numerically invalid candidates.
# Must be much larger than any realistic J value so infeasible candidates
# are always ranked last and never selected as parents.
INFEASIBLE_PENALTY = 1e6

def evaluate_candidate(theta, R1, R2, S0, Kbar,
                        market_strikes=None, bid_prices=None, ask_prices=None):
    """
    Evaluate J(theta) for one CMA-ES candidate.

    Returns INFEASIBLE_PENALTY if constraints are violated so that
    CMA-ES naturally kills the candidate during selection — it will
    never be in the top mu parents, so it won't influence the mean
    or covariance update.

    Parameters
    ----------
    theta          : ndarray
    R1, R2         : int
    S0, Kbar       : float
    market_strikes : array-like or None
    bid_prices     : array-like or None
    ask_prices     : array-like or None

    Returns
    -------
    float  J value or INFEASIBLE_PENALTY
    """
    if not is_feasible(theta, R1, R2, S0, Kbar,
                       market_strikes, bid_prices, ask_prices):
        return INFEASIBLE_PENALTY

    try:
        J_value, _, _ = calculate_J(theta, R1, R2, S0)
        return J_value if np.isfinite(J_value) else INFEASIBLE_PENALTY
    except Exception:
        return INFEASIBLE_PENALTY


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CMA-ES Core Algorithm
#
# Implements the (mu/w, lambda)-CMA-ES with:
#   - Cumulative Step-size Adaptation (CSA) for sigma
#   - Rank-one + Rank-mu Covariance Matrix Adaptation
#   - Periodic eigendecomposition for efficient sampling
#
# Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial."
#            arXiv:1604.00772
#
# The algorithm maintains a multivariate Gaussian search distribution:
#   N(mean, step_size^2 * covariance)
#
# Each generation:
#   1. Sample lambda candidates from the distribution
#   2. Evaluate J for each candidate (infeasible ones get penalty)
#   3. Keep best mu candidates (the "parents")
#   4. Update mean, covariance, step_size using the parents
#   5. Repeat until step_size < tolerance (convergence)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5b: Soft bid-ask penalty for the sigma-only V/C'' objective
#
# With nus FIXED, ordering/boundary constraints can never be violated, so the
# ONLY constraint that bites is bid-ask. We make it SOFT:
#
#     objective(theta) = J_smooth + penalty_weight * sum_i [violation_i]^2
#
# Why soft and not the hard INFEASIBLE_PENALTY: if theta_init sits even slightly
# outside the spread (e.g. the prob5b JSON is out of sync with Quotes.csv), a
# hard penalty makes EVERY nearby candidate infeasible -> 0 feasible forever and
# J frozen at 1e6. A soft penalty gives a continuous gradient back into the
# feasible region, so the optimizer can repair a bad start instead of stalling.
# ══════════════════════════════════════════════════════════════════════════════

def bidask_violation(theta, market_strikes, bid_prices, ask_prices, R1, R2, S0):
    """Sum of squared bid-ask exceedances at the market strikes (0 if inside)."""
    if market_strikes is None or bid_prices is None or ask_prices is None:
        return 0.0
    m     = compute_model_calls(theta, market_strikes, R1, R2, S0)
    over  = np.maximum(0.0, m - ask_prices)
    under = np.maximum(0.0, bid_prices - m)
    return float(np.sum(over**2) + np.sum(under**2))

def evaluate_candidate_smooth(theta, R1, R2, S0, Kbar,
                              market_strikes=None, bid_prices=None, ask_prices=None,
                              penalty_weight=1e8, strike_lo=None, strike_hi=None,
                              objective_mode='L2', l1_weight=1.0,
                              mask_mode='hard', taper_width=30.0, out_floor=0.2,
                              return_feasible=False):
    """Soft-constrained V/C'' objective for the sigma-only search.

    Hard part: structural feasibility (ordering, Kbar) -- trivially satisfied
    while nus are fixed, but still guarded. Soft part: bid-ask violation.

    If return_feasible=True, returns (value, is_truly_feasible) where the flag
    is True iff structural checks pass AND bid-ask violation is (essentially)
    zero. This is what the per-generation log should count, not value<1e6.
    """
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if not is_feasible(theta, R1, R2, S0, Kbar):      # ordering / boundary only
                return (INFEASIBLE_PENALTY, False) if return_feasible else INFEASIBLE_PENALTY
            J    = calculate_J_smooth(theta, R1, R2, S0, strike_lo, strike_hi,
                                       mode=objective_mode, l1_weight=l1_weight,
                                       mask_mode=mask_mode,
                                       taper_width=taper_width, out_floor=out_floor)
            viol = bidask_violation(theta, market_strikes, bid_prices, ask_prices, R1, R2, S0)
            val  = J + penalty_weight * viol
            ok   = np.isfinite(val)
            if not ok:
                return (INFEASIBLE_PENALTY, False) if return_feasible else INFEASIBLE_PENALTY
            if return_feasible:
                truly_feasible = (viol <= 1e-9)               # bid-ask satisfied
                return val, truly_feasible
            return val
    except Exception:
        return (INFEASIBLE_PENALTY, False) if return_feasible else INFEASIBLE_PENALTY


# ========================================================================
# (from notebook cell 14)
# ========================================================================

def _run_cmaes(evaluate_fn, theta_init, initial_step_size=1.0,
               feasibility_fn=None,
               max_generations=500, convergence_tol=1e-10,
               random_seed=42, verbose=True):
    """
    Core CMA-ES optimization loop.

    Parameters
    ----------
    evaluate_fn       : callable theta -> float
                        The objective function (evaluate_candidate)
    theta_init        : ndarray (n,)
                        Starting point for the search distribution mean
    initial_step_size : float
                        Initial value of sigma (global step size).
                        Should be ~10% of the typical parameter gap size.
    max_generations   : int
                        Maximum number of generations before stopping
    convergence_tol   : float
                        Stop when step_size < this value (distribution collapsed)
    random_seed       : int
                        For reproducibility
    verbose           : bool
                        Print progress every 20 generations

    Returns
    -------
    best_theta : ndarray   best parameter vector found across all generations
    best_J     : float     J value at best_theta
    history    : dict      per-generation tracking data
    """
    np.random.seed(random_seed)
    n = len(theta_init)

    # ── CMA-ES Hyperparameters (Hansen 2016, Table 1) ─────────────────────
    #
    # These are the "standard" settings derived from theory. They generally
    # work well without tuning for problems in the 50-300 dimensional range.

    # Population size lambda: roughly 4 + 3*ln(n)
    # More candidates = more exploration but slower per-generation
    population_size = 4 + int(np.floor(3 * np.log(n)))

    # Number of parents mu: best half of the population
    num_parents = population_size // 2

    # Recombination weights: log-spaced so best candidate has highest weight.
    # The 1st candidate gets weight ~log(mu+0.5)-log(1), the mu-th gets ~0.
    raw_weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
    weights     = raw_weights / raw_weights.sum()   # normalize to sum=1

    # Variance-effective selection mass: how many "effective" samples are used
    # per generation. mueff ≈ mu/2 for equal weights, mueff ≈ 1 for rank-1.
    mueff = 1.0 / np.sum(weights**2)

    # Step-size control: how fast sigma adapts
    step_size_decay   = (mueff + 2) / (n + mueff + 5)
    step_size_damping = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + step_size_decay

    # Expected length of ||N(0,I)|| in n dimensions (for step-size reference)
    expected_norm_sphere = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

    # Covariance matrix adaptation rates
    cov_path_decay = (4 + mueff/n) / (n + 4 + 2*mueff/n)
    cov_rank_one   = 2 / ((n + 1.3)**2 + mueff)          # rank-1 learning rate
    cov_rank_mu    = min(1 - cov_rank_one,
                        2*(mueff-2+1/mueff) / ((n+2)**2 + mueff))  # rank-mu rate

    # Threshold for the h_sigma indicator (suppresses rank-1 update if needed)
    h_sigma_thresh = (1.4 + 2/(n+1)) * expected_norm_sphere

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CMA-ES Optimizer  (v4 — direct space + bid-ask constraints)")
        print(f"{'='*65}")
        print(f"  Dimension n          = {n}")
        print(f"  Population size λ    = {population_size}")
        print(f"  Number of parents μ  = {num_parents}")
        print(f"  Effective mass μeff  = {mueff:.2f}")
        print(f"  Initial step size σ₀ = {initial_step_size}")
        print(f"  Max generations      = {max_generations}")
        print(f"  Convergence tol      = {convergence_tol:.1e}")
        print(f"{'='*65}")
        print(f"  {'Gen':>5} | {'Best J':>14} | {'Feasible':>12} | {'σ':>10}")
        print(f"  {'-'*50}")

    # ── State Variables ───────────────────────────────────────────────────

    mean              = theta_init.copy()   # current center of search distribution
    step_size         = initial_step_size   # global step size sigma

    # Evolution paths accumulate the history of mean movements
    evolution_path_C  = np.zeros(n)   # used for rank-1 covariance update
    evolution_path_s  = np.zeros(n)   # used for step-size adaptation

    # Covariance matrix C and its eigendecomposition
    # C = eigenvectors @ diag(axis_lengths^2) @ eigenvectors.T
    eigenvectors      = np.eye(n)     # B: columns are eigenvectors of C
    axis_lengths      = np.ones(n)    # D: sqrt of eigenvalues (axis lengths)
    covariance        = np.eye(n)     # C: the full covariance matrix
    inv_sqrt_cov      = np.eye(n)     # C^{-1/2}: for step-size path normalization

    last_eigen_update = 0             # track when we last decomposed C

    best_J            = np.inf
    best_theta        = mean.copy()

    # History for plotting and diagnostics
    history = {
        'best_J_per_gen':  [],   # best J found so far at each generation
        'mean_J_per_gen':  [],   # mean J across the population (includes penalties)
        'step_size':       [],   # sigma at each generation
        'feasible_count':  [],   # number of feasible candidates per generation
        'generation':      []
    }

    # ── Generation Loop ───────────────────────────────────────────────────
    for generation in range(max_generations):

        # ── Step 1: Sample lambda candidates from N(mean, sigma^2 * C) ───
        #
        # Efficient sampling using eigendecomposition:
        #   x_k = mean + sigma * B * (D * z_k),  z_k ~ N(0, I)
        #
        # This is equivalent to sampling from N(mean, sigma^2 * C) because:
        #   Cov[B*(D*z)] = B * diag(D^2) * B^T = C
        #
        # B (eigenvectors) rotates the standard normal into the ellipse axes
        # D (axis_lengths)  stretches each axis by the corresponding eigenvalue

        standard_samples = np.random.randn(population_size, n)
        candidates = np.array([
            mean + step_size * (eigenvectors @ (axis_lengths * standard_samples[k]))
            for k in range(population_size)
        ])

        # ── Step 2: Evaluate objective for each candidate ─────────────────
        # Infeasible candidates receive INFEASIBLE_PENALTY (1e6) and will
        # be ranked last — they never influence the parameter updates.
        fitness_values = np.array([
            evaluate_fn(candidates[k]) for k in range(population_size)
        ])

        # Per-generation feasibility count. If the caller passed a
        # feasibility_fn (preferred), we count candidates that pass the TRUE
        # bid-ask check; otherwise we fall back to 'objective < penalty wall'
        # (a proxy that's misleading under the soft-penalty regime since
        # J_smooth alone can exceed 1e6).
        if feasibility_fn is not None:
            num_feasible = int(sum(1 for cand in candidates
                                   if feasibility_fn(cand)))
        else:
            num_feasible = int(np.sum(fitness_values < INFEASIBLE_PENALTY))

        # ── Step 3: Sort by fitness (ascending = minimization) ────────────
        sorted_indices = np.argsort(fitness_values)

        # ── Step 4: Update best solution found so far ─────────────────────
        if fitness_values[sorted_indices[0]] < best_J:
            best_J     = fitness_values[sorted_indices[0]]
            best_theta = candidates[sorted_indices[0]].copy()

        # ── Step 5: Update mean (weighted average of best mu parents) ─────
        #
        # New mean = weighted sum of the best num_parents candidates.
        # Better candidates (lower J) get higher weights.
        # This is the "gradient-free gradient step" — we move toward
        # where the best samples were found.
        old_mean = mean.copy()
        mean = np.sum(
            weights[:, None] * candidates[sorted_indices[:num_parents]],
            axis=0
        )

        # ── Step 6: Update step-size evolution path p_s ───────────────────
        #
        # p_s accumulates normalized mean movements.
        # If steps are consistently in the same direction, ||p_s|| grows.
        # If steps oscillate back and forth, ||p_s|| stays small.
        # C^{-1/2} normalizes the step so its expected length is sqrt(n).
        evolution_path_s = (
            (1 - step_size_decay) * evolution_path_s
            + np.sqrt(step_size_decay * (2 - step_size_decay) * mueff)
            * inv_sqrt_cov @ (mean - old_mean) / step_size
        )

        # ── Step 7: Compute h_sigma (Heaviside indicator) ─────────────────
        #
        # h_sigma = 1 if the evolution path is "short enough" to be reliable.
        # h_sigma = 0 suppresses the rank-1 covariance update in the early
        # generations when the path hasn't accumulated enough history.
        path_length_norm = (
            np.linalg.norm(evolution_path_s)
            / np.sqrt(1 - (1 - step_size_decay)**(2*(generation+1)))
        )
        h_sigma = (path_length_norm < h_sigma_thresh)

        # ── Step 8: Update covariance evolution path p_c ──────────────────
        #
        # p_c accumulates the history of mean movements (without C^{-1/2}).
        # It is used for the rank-1 covariance update below.
        # The outer product p_c * p_c^T encodes the direction of consistent
        # improvement across generations.
        evolution_path_C = (
            (1 - cov_path_decay) * evolution_path_C
            + h_sigma * np.sqrt(cov_path_decay * (2 - cov_path_decay) * mueff)
            * (mean - old_mean) / step_size
        )

        # ── Step 9: Update covariance matrix C ────────────────────────────
        #
        # C is updated by two mechanisms:
        #
        # Rank-1 update (cov_rank_one):
        #   Uses the single evolution path p_c. This captures the direction
        #   of consistent improvement accumulated across many generations.
        #   Weight = cov_rank_one * p_c * p_c^T
        #
        # Rank-mu update (cov_rank_mu):
        #   Uses all num_parents successful steps from this generation.
        #   The weighted sum of outer products captures the local curvature.
        #   Weight = cov_rank_mu * sum_i w_i * y_i * y_i^T
        #
        # The old covariance decays at rate (1 - cov_rank_one - cov_rank_mu).

        # Normalized steps of the best parents (relative to old mean, scaled by sigma)
        normalized_steps = (
            (1.0 / step_size)
            * (candidates[sorted_indices[:num_parents]] - old_mean)
        )

        covariance = (
            (1 - cov_rank_one - cov_rank_mu) * covariance
            # Rank-1 update: from evolution path history
            + cov_rank_one * (
                np.outer(evolution_path_C, evolution_path_C)
                + (1 - h_sigma) * cov_path_decay * (2 - cov_path_decay) * covariance
            )
            # Rank-mu update: from current generation's best candidates
            + cov_rank_mu * np.sum(
                weights[:, None, None]
                * (normalized_steps[:, :, None] * normalized_steps[:, None, :]),
                axis=0
            )
        )

        # ── Step 10: Update step size sigma (CSA) ─────────────────────────
        #
        # sigma grows when ||p_s|| > expected_norm_sphere (correlated steps = good)
        # sigma shrinks when ||p_s|| < expected_norm_sphere (oscillating steps = near min)
        step_size = step_size * np.exp(
            (step_size_decay / step_size_damping)
            * (np.linalg.norm(evolution_path_s) / expected_norm_sphere - 1)
        )

        # ── Step 11: Periodic eigendecomposition of C ─────────────────────
        #
        # Computing B and D from C is O(n^3) — expensive for n=209.
        # We only do it every ~n/(10*lambda) generations.
        # Between updates, we use the stored B and D for fast sampling.
        eigendecomp_interval = n / (10 * population_size * (cov_rank_one + cov_rank_mu))
        if generation - last_eigen_update > eigendecomp_interval:
            last_eigen_update = generation

            # Enforce symmetry (numerical drift can make C slightly asymmetric)
            covariance = np.triu(covariance) + np.triu(covariance, 1).T

            # Eigendecomposition: C = B * diag(D^2) * B^T
            eigenvalues_sq, eigenvectors = np.linalg.eigh(covariance)

            # Floor eigenvalues at 1e-20 to prevent numerical issues
            eigenvalues_sq = np.maximum(eigenvalues_sq, 1e-20)
            axis_lengths   = np.sqrt(eigenvalues_sq)        # D = sqrt(eigenvalues)

            # C^{-1/2} = B * diag(1/D) * B^T (for step-size path normalization)
            inv_sqrt_cov = eigenvectors @ np.diag(1.0 / axis_lengths) @ eigenvectors.T

        # ── Step 12: Record history for diagnostics and plotting ───────────
        history['best_J_per_gen'].append(best_J)
        history['mean_J_per_gen'].append(float(np.mean(fitness_values)))
        history['step_size'].append(float(step_size))
        history['feasible_count'].append(num_feasible)
        history['generation'].append(generation)

        # ── Step 13: Print progress ────────────────────────────────────────
        if verbose and generation % 20 == 0:
            print(f"  {generation:>5} | {best_J:>14.8f} | "
                  f"{num_feasible:>5}/{population_size:<5} feasible | "
                  f"σ={step_size:>10.6f}")

        # ── Step 14: Check convergence ─────────────────────────────────────
        # Primary stopping criterion: step size has collapsed
        # This means the search distribution has shrunk to a point —
        # CMA-ES has converged and further generations won't help.
        if step_size < convergence_tol:
            if verbose:
                print(f"\n  Converged at gen {generation}: "
                      f"σ = {step_size:.2e} < tol = {convergence_tol:.2e}")
            break

        # Secondary: J is essentially zero (perfect smoothness)
        if best_J < 1e-12:
            if verbose:
                print(f"\n  Converged at gen {generation}: J = {best_J:.2e} ≈ 0")
            break

    if verbose:
        print(f"  {'-'*50}")

    return best_theta, best_J, history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Public Optimization API
# ══════════════════════════════════════════════════════════════════════════════


# ========================================================================
# (from notebook cell 16)
# ========================================================================

def cmaes_optimize(theta_init, R1, R2, S0, Kbar,
                   theta_baseline=None,
                   market_strikes=None,
                   bid_prices=None,
                   ask_prices=None,
                   initial_step_size=1.0,
                   max_generations=500,
                   convergence_tol=1e-10,
                   random_seed=42,
                   verbose=True,
                   save_plot=True,
                   plot_path='cmaes_result.png'):
    """
    Minimize J(theta) using CMA-ES in direct parameter space.

    Infeasible candidates (ordering violations, boundary violations,
    or bid-ask violations) are killed by the penalty function rather
    than excluded by a coordinate transform.

    Parameters
    ----------
    theta_init        : ndarray (2*R1 + 2*R2,)
                        Starting theta for THIS optimization call.
                        In a sigma schedule, this changes each stage.

    R1, R2            : int   number of left/right partition points
    S0                : float spot price (S0 = 1271.87 for SP500 data)
    Kbar              : float right boundary (fixed = 2000.0)

    theta_baseline    : ndarray or None
                        The ORIGINAL pre-optimization LVG theta.
                        Used as the "before" curve in all plots.
                        If None, uses theta_init (backward compatible).

    market_strikes    : array-like or None
                        Observed market strikes from Quotes.csv.
                        If provided, bid-ask constraint is enforced.
    bid_prices        : array-like or None   bid prices at each strike
    ask_prices        : array-like or None   ask prices at each strike

    initial_step_size : float   CMA-ES initial sigma (default 1.0)
                        Tip: use ~10% of typical partition gap (~0.3-0.5)
    max_generations   : int     maximum generations (default 500)
    convergence_tol   : float   stop when sigma < this (default 1e-10)
    random_seed       : int     for reproducibility (default 42)
    verbose           : bool    print per-generation progress (default True)
    save_plot         : bool    save comparison plot (default True)
    plot_path         : str     file path for the output plot

    Returns
    -------
    dict with keys:
        'best_theta'   ndarray  optimized parameter vector
        'best_J'       float    optimized J value at best_theta
        'J_initial'    float    J value at theta_init (start of this call)
        'J_baseline'   float    J value at theta_baseline (original LVG)
        'improvement'  float    % improvement vs theta_baseline
        'lambda1_opt'  float    lambda1 at best_theta
        'lambda2_opt'  float    lambda2 at best_theta
        'history'      dict     per-generation convergence data
    """

    # Use theta_init as baseline if no separate baseline provided
    if theta_baseline is None:
        theta_baseline = theta_init

    # ── Compute J values ──────────────────────────────────────────────────
    J_baseline, _, _           = calculate_J(theta_baseline, R1, R2, S0)
    J_initial, lambda1_init, lambda2_init = calculate_J(theta_init, R1, R2, S0)

    if verbose:
        print(f"\n  Original LVG baseline J  = {J_baseline:.8f}")
        print(f"  Starting J (this stage)  = {J_initial:.8f}")
        print(f"  Lambda1 (init)           = {lambda1_init:.6f}")
        print(f"  Lambda2 (init)           = {lambda2_init:.6f}")
        if market_strikes is not None:
            print(f"  Bid-ask constraint       = ACTIVE ({len(market_strikes)} strikes)")
        else:
            print(f"  Bid-ask constraint       = OFF (no market data provided)")

    # ── Wrap evaluate_candidate with fixed parameters ─────────────────────
    # Convert market data to numpy arrays once, outside the inner loop
    ms  = np.array(market_strikes)  if market_strikes is not None else None
    bid = np.array(bid_prices)      if bid_prices     is not None else None
    ask = np.array(ask_prices)      if ask_prices     is not None else None

    def objective(theta):
        return evaluate_candidate(theta, R1, R2, S0, Kbar, ms, bid, ask)

    # ── Run CMA-ES ────────────────────────────────────────────────────────
    best_theta, best_J, history = _run_cmaes(
        evaluate_fn       = objective,
        theta_init        = theta_init,
        initial_step_size = initial_step_size,
        max_generations   = max_generations,
        convergence_tol   = convergence_tol,
        random_seed       = random_seed,
        verbose           = verbose
    )

    # ── Compute final lambdas ─────────────────────────────────────────────
    _, lambda1_opt, lambda2_opt = calculate_J(best_theta, R1, R2, S0)

    # Improvement is always measured vs the ORIGINAL LVG baseline
    improvement = 100.0 * (J_baseline - best_J) / J_baseline if J_baseline > 0 else 0.0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Stage Complete")
        print(f"{'='*65}")
        print(f"  Original baseline J  = {J_baseline:.8f}")
        print(f"  This stage start J   = {J_initial:.8f}")
        print(f"  This stage best J    = {best_J:.8f}")
        print(f"  Improvement vs LVG   = {improvement:.4f}%")
        print(f"  Generations run      = {len(history['generation'])}")
        print(f"  Lambda1: {lambda1_init:.4f} -> {lambda1_opt:.4f}")
        print(f"  Lambda2: {lambda2_init:.4f} -> {lambda2_opt:.4f}")
        print(f"{'='*65}\n")

    # ── Save comparison plot ──────────────────────────────────────────────
    if save_plot:
        _generate_plots(
            history        = history,
            J_baseline     = J_baseline,
            best_J         = best_J,
            improvement    = improvement,
            theta_baseline = theta_baseline,    # always the original LVG output
            best_theta     = best_theta,
            R1=R1, R2=R2, S0=S0,
            market_strikes = ms,
            bid_prices     = bid,
            ask_prices     = ask,
            save_path      = plot_path
        )
        if verbose:
            print(f"  Plot saved to: {plot_path}")
    return {
        'best_theta':  best_theta,
        'best_J':      best_J,
        'J_initial':   J_initial,
        'J_baseline':  J_baseline,
        'improvement': improvement,
        'lambda1_opt': lambda1_opt,
        'lambda2_opt': lambda2_opt,
        'history':     history
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Plotting Utilities
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7b: Sigma-only optimization driver (nus FIXED, soft bid-ask, masked J)
# ══════════════════════════════════════════════════════════════════════════════

def cmaes_optimize_sigmas(theta_init, R1, R2, S0, Kbar,
                          theta_baseline=None,
                          market_strikes=None, bid_prices=None, ask_prices=None,
                          penalty_weight=1e8, strike_lo=None, strike_hi=None,
                          objective_mode='L2', l1_weight=1.0,
                          mask_mode='hard', taper_width=30.0, out_floor=0.2,
                          initial_step_size=0.3, max_generations=500,
                          convergence_tol=1e-10, random_seed=42,
                          verbose=True, save_plot=True,
                          plot_path='cmaes_sigma_result.png'):
    if theta_baseline is None:
        theta_baseline = theta_init

    nus1 = theta_init[:R1].copy()
    nus2 = theta_init[2*R1 : 2*R1 + R2].copy()
    x0   = np.concatenate([theta_init[R1 : 2*R1],
                           theta_init[2*R1 + R2 : 2*R1 + 2*R2]])     # raw sigmas

    def assemble(x):
        return np.concatenate([nus1, x[:R1], nus2, x[R1:R1+R2]])

    ms  = np.array(market_strikes) if market_strikes is not None else None
    bid = np.array(bid_prices)     if bid_prices     is not None else None
    ask = np.array(ask_prices)     if ask_prices     is not None else None

    def objective(x):
        return evaluate_candidate_smooth(assemble(x), R1, R2, S0, Kbar,
                                         ms, bid, ask, penalty_weight,
                                         strike_lo, strike_hi,
                                         objective_mode=objective_mode,
                                         l1_weight=l1_weight,
                                         mask_mode=mask_mode,
                                         taper_width=taper_width, out_floor=out_floor)

    Jb       = calculate_J_smooth(theta_baseline, R1, R2, S0, strike_lo, strike_hi,
                                   mode=objective_mode, l1_weight=l1_weight,
                                   mask_mode=mask_mode,
                                   taper_width=taper_width, out_floor=out_floor)
    viol_b   = bidask_violation(theta_baseline, ms, bid, ask, R1, R2, S0)
    feas_b   = is_feasible(theta_baseline, R1, R2, S0, Kbar, ms, bid, ask)
    if verbose:
        print(f"\n  Baseline masked V/C'' J = {Jb:.6f}")
        print(f"  Baseline bid-ask viol   = {viol_b:.6f}   (feasible: {feas_b})")
        print(f"  Free dimension          = {len(x0)}  (sigmas only, nus fixed)")
        print(f"  penalty_weight          = {penalty_weight:.1e}")
        print(f"  objective_mode          = {objective_mode!r}" +
              (f"  (l1_weight={l1_weight})" if objective_mode=='elastic' else ""))
        print(f"  mask_mode               = {mask_mode!r}" +
              (f"  (taper={taper_width}, floor={out_floor})" if mask_mode=='soft' else ""))

    # per-candidate feasibility check used by _run_cmaes for the log column
    def _feas(x):
        th = assemble(x)
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                if not is_feasible(th, R1, R2, S0, Kbar):
                    return False
                v = bidask_violation(th, ms, bid, ask, R1, R2, S0)
                return v <= 1e-9
        except Exception:
            return False

    best_x, best_soft, history = _run_cmaes(
        evaluate_fn=objective, theta_init=x0,
        feasibility_fn=_feas,
        initial_step_size=initial_step_size, max_generations=max_generations,
        convergence_tol=convergence_tol, random_seed=random_seed, verbose=verbose)

    best_theta = assemble(best_x)
    Jf       = calculate_J_smooth(best_theta, R1, R2, S0, strike_lo, strike_hi,
                                   mode=objective_mode, l1_weight=l1_weight,
                                   mask_mode=mask_mode,
                                   taper_width=taper_width, out_floor=out_floor)
    viol_f   = bidask_violation(best_theta, ms, bid, ask, R1, R2, S0)
    feas_f   = is_feasible(best_theta, R1, R2, S0, Kbar, ms, bid, ask)
    Jf_cpp, lam1_opt, lam2_opt = calculate_J(best_theta, R1, R2, S0)
    improvement = 100.0*(Jb - Jf)/Jb if Jb > 0 else 0.0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Sigma-only stage complete")
        print(f"{'='*65}")
        print(f"  masked V/C'' J : {Jb:.6f} -> {Jf:.6f}  ({improvement:.2f}%)")
        print(f"  bid-ask viol   : {viol_b:.6f} -> {viol_f:.6f}   (feasible: {feas_f})")
        print(f"  C'' jump J (side): -> {Jf_cpp:.6f}")
        print(f"{'='*65}\n")

    if save_plot:
        _generate_plots(history=history, J_baseline=Jb, best_J=Jf,
                        improvement=improvement, theta_baseline=theta_baseline,
                        best_theta=best_theta, R1=R1, R2=R2, S0=S0,
                        market_strikes=ms, bid_prices=bid, ask_prices=ask,
                        save_path=plot_path)
    return {'best_theta': best_theta, 'best_J': Jf, 'best_soft': best_soft,
            'J_baseline': Jb, 'improvement': improvement, 'feasible': feas_f,
            'violation': viol_f, 'history': history}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7c: Partition-point densification
#
# Insert new nu_j between selected adjacent partition pairs without breaking
# feasibility. The key trick (equal-left-sigma init): on a sub-interval, setting
# the inserted node's sigma EQUAL to its left neighbor reproduces the SAME ODE
# solution as before splitting, so call prices at every market strike are
# preserved exactly => theta stays bid-ask feasible by construction. Each new
# node starts at sigma^2 jump = 0, giving CMA-ES headroom to smooth gradually
# instead of facing a single cliff.
#
# We expose two modes:
#   - 'uniform'   : insert `factor-1` evenly spaced nodes in every interval of
#                   the chosen wing (mirrors the old DENSE_FACTOR).
#   - 'targeted'  : insert nodes only between the top-k interior nodes whose
#                   current sigma^2 jump is largest within [strike_lo, strike_hi]
#                   -- the cliff fix that motivated this.
# ══════════════════════════════════════════════════════════════════════════════

def densify_partitions(theta, R1, R2, S0,
                       mode='targeted', wing='both',
                       factor=2, top_k=5, n_insert=2,
                       strike_lo=None, strike_hi=None):
    """Return (theta_new, R1_new, R2_new) after inserting partition points.

    Parameters
    ----------
    mode      : 'targeted' inserts around the top_k largest in-range sigma^2
                jumps; 'uniform' inserts factor-1 nodes in every interval.
    wing      : 'left', 'right', or 'both'. Determines which wing(s) get
                considered. 'targeted' uses S0 to assign each large-jump node
                to its wing automatically.
    factor    : (uniform mode) number of sub-intervals to split each interval
                into. factor=2 inserts 1 new node per interval.
    top_k     : (targeted mode) number of largest in-range jumps to address.
    n_insert  : (targeted mode) number of new nodes inserted per selected pair.
    strike_lo, strike_hi : restrict targeted insertions to in-range nodes.

    Returns
    -------
    theta_new : reassembled parameter vector
    R1_new, R2_new : new counts
    """
    sig1 = softplus(theta[R1 : 2*R1])             + 1e-6
    sig2 = softplus(theta[2*R1 + R2 : 2*R1+2*R2]) + 1e-6
    nus1 = theta[:R1].copy()
    nus2 = theta[2*R1 : 2*R1 + R2].copy()

    def _split_interval(nus, sigs, j_left, n_new):
        """Insert n_new evenly spaced nodes between nus[j_left] and nus[j_left+1].
        Each new sigma equals sigs[j_left] (equal-left-sigma init)."""
        a, b = nus[j_left], nus[j_left+1]
        new_nodes = np.linspace(a, b, n_new+2)[1:-1]
        new_sigs  = np.full(n_new, sigs[j_left])    # critical: preserves prices
        new_nus_arr   = np.concatenate([nus[:j_left+1],  new_nodes, nus[j_left+1:]])
        new_sigs_arr  = np.concatenate([sigs[:j_left+1], new_sigs,  sigs[j_left+1:]])
        return new_nus_arr, new_sigs_arr

    if mode == 'uniform':
        n_new = factor - 1
        if n_new <= 0:
            return theta.copy(), R1, R2
        if wing in ('left', 'both'):
            for j in range(R1-2, -1, -1):           # iterate right-to-left so indices stay valid
                nus1, sig1 = _split_interval(nus1, sig1, j, n_new)
        if wing in ('right', 'both'):
            for j in range(R2-2, -1, -1):
                nus2, sig2 = _split_interval(nus2, sig2, j, n_new)

    elif mode == 'targeted':
        # rank intervals by the |sigma^2 jump| AT THEIR LEFT NODE (in-range only)
        jumps = _sigma2_jump_vector(theta, R1, R2)   # length R1+R2-1
        pos   = _node_positions(theta, R1, R2, S0)
        lo = -np.inf if strike_lo is None else strike_lo
        hi =  np.inf if strike_hi is None else strike_hi
        score = np.where((pos >= lo) & (pos <= hi), np.abs(jumps), -1.0)
        order = np.argsort(-score)
        chosen = [k for k in order if score[k] > 0][:top_k]

        # split into (wing, local_interval_index) and de-dup
        left_targets, right_targets = set(), set()
        for node_idx in chosen:
            # mapping: node_idx in [0, R1-2] -> left wing, interval node_idx
            #         node_idx == R1-1       -> junction (the two intervals straddling S0)
            #         node_idx in [R1, R1+R2-2] -> right wing, interval node_idx-R1
            if node_idx <= R1 - 2:
                left_targets.add(node_idx)
            elif node_idx == R1 - 1:
                left_targets.add(R1 - 2); right_targets.add(0)
            else:
                right_targets.add(node_idx - R1)

        if wing in ('left', 'both'):
            for j in sorted(left_targets, reverse=True):
                nus1, sig1 = _split_interval(nus1, sig1, j, n_insert)
        if wing in ('right', 'both'):
            for j in sorted(right_targets, reverse=True):
                nus2, sig2 = _split_interval(nus2, sig2, j, n_insert)
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'uniform' or 'targeted'")

    R1_new, R2_new = len(nus1), len(nus2)
    theta_new = np.concatenate([nus1, inv_softplus(sig1),
                                nus2, inv_softplus(sig2)])
    return theta_new, R1_new, R2_new


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7d: Automatic iterative densification — divide-and-conquer on jumps
#
# Loop: find the largest in-range sigma^2 jumps, split those intervals with new
# nu_j (equal-left-sigma init -> still feasible), re-optimize with a SHORT
# CMA-ES warm-start, measure improvement, and either continue or revert.
#
# Each iteration is gated on actually reducing the masked V/C'' objective. If
# splitting doesn't help (typical for boundary cliffs forced by bid-ask), we
# revert that iteration's split and stop -- so the loop can't make things
# worse, only better-or-equal.
# ══════════════════════════════════════════════════════════════════════════════

def auto_densify_loop(theta, R1, R2, S0, Kbar,
                      market_strikes, bid_prices, ask_prices,
                      strike_lo, strike_hi,
                      max_iterations=8,
                      top_k_per_iter=3,
                      n_insert_per_split=2,
                      max_total_new_nodes=200,
                      tol_relative_improvement=0.05,
                      inner_schedule=((0.10, 200), (0.03, 300), (0.01, 400)),
                      penalty_weight=1e8,
                      verbose=True):
    """Iteratively split the worst in-range sigma^2 jumps and re-optimize.

    Parameters
    ----------
    theta, R1, R2 : starting calibration (already optimized once is best).
    max_iterations : hard cap on iterations.
    top_k_per_iter : number of worst jumps to split per iteration.
    n_insert_per_split : new nodes inserted per chosen interval.
    max_total_new_nodes : stop once we've added this many nodes total.
    tol_relative_improvement : require (J_old - J_new)/J_old > this to keep
                               the iteration; otherwise revert and stop.
    inner_schedule : list of (step_size, max_generations) for the CMA-ES run
                     after each split. Short by design -- this loop is meant
                     to do MANY rounds, not one heroic run.

    Returns
    -------
    dict with best_theta, R1, R2, J_history, R_history (total nodes per iter).
    """
    cur_theta, cur_R1, cur_R2 = theta.copy(), R1, R2
    cur_J = calculate_J_smooth(cur_theta, cur_R1, cur_R2, S0, strike_lo, strike_hi)
    J_hist = [cur_J]
    R_hist = [(cur_R1, cur_R2)]
    total_nodes_added = 0

    if verbose:
        print(f"\n{'='*68}\n  AUTO-DENSIFY LOOP\n{'='*68}")
        print(f"  starting J = {cur_J:.4e}  (R1={cur_R1}, R2={cur_R2})")
        print(f"  top_k={top_k_per_iter}, n_insert={n_insert_per_split}, "
              f"max_iters={max_iterations}, node_budget={max_total_new_nodes}")
        print(f"  inner schedule: {inner_schedule}")

    for it in range(1, max_iterations + 1):
        # snapshot for revert
        snap_theta, snap_R1, snap_R2, snap_J = cur_theta.copy(), cur_R1, cur_R2, cur_J

        # diagnose the cliff *before* splitting -- we'll check after to see if
        # max|Δσ²| dropped, which is the actual measure of dispersion
        max_jump_before = float(np.max(np.abs(
            _sigma2_jump_vector(cur_theta, cur_R1, cur_R2))))

        # split
        new_theta, new_R1, new_R2 = densify_partitions(
            cur_theta, cur_R1, cur_R2, S0,
            mode='targeted', wing='both',
            top_k=top_k_per_iter, n_insert=n_insert_per_split,
            strike_lo=strike_lo, strike_hi=strike_hi)
        nodes_added = (new_R1 + new_R2) - (cur_R1 + cur_R2)

        if verbose:
            print(f"\n  --- iter {it} ---")
            print(f"  split: (R1,R2) {(cur_R1,cur_R2)} -> {(new_R1,new_R2)}   "
                  f"(+{nodes_added} nodes; total added so far {total_nodes_added+nodes_added})")

        # verify feasibility post-split (should always be true under equal-left-sigma)
        feas_post_split = is_feasible(new_theta, new_R1, new_R2, S0, Kbar,
                                       market_strikes, bid_prices, ask_prices)
        if not feas_post_split:
            if verbose: print(f"  post-split infeasible (shouldn't happen) -- reverting and stopping.")
            break

        # short warm-start re-optimization
        warm = new_theta
        for s_idx,(step, ng) in enumerate(inner_schedule, start=1):
            res = cmaes_optimize_sigmas(
                theta_init=warm, R1=new_R1, R2=new_R2, S0=S0, Kbar=Kbar,
                theta_baseline=warm, market_strikes=market_strikes,
                bid_prices=bid_prices, ask_prices=ask_prices,
                penalty_weight=penalty_weight,
                strike_lo=strike_lo, strike_hi=strike_hi,
                initial_step_size=step, max_generations=ng,
                convergence_tol=1e-10, random_seed=7000+it*10+s_idx,
                verbose=False, save_plot=False)
            warm = res['best_theta']
        new_J  = res['best_J']
        max_jump_after = float(np.max(np.abs(
            _sigma2_jump_vector(warm, new_R1, new_R2))))

        rel_improvement = (snap_J - new_J) / snap_J if snap_J > 0 else 0.0
        if verbose:
            print(f"  re-optim J: {snap_J:.4e} -> {new_J:.4e}  ({100*rel_improvement:+.2f}%)")
            print(f"  max|Δσ²|  : {max_jump_before:.2f}  -> {max_jump_after:.2f}   "
                  f"({'DISPERSED' if max_jump_after < 0.9*max_jump_before else 'UNCHANGED'})")
            print(f"  feasible (bid-ask): {res['feasible']}   viol={res['violation']:.4f}")

        if rel_improvement < tol_relative_improvement:
            if verbose:
                print(f"  -> improvement {100*rel_improvement:.2f}% < threshold "
                      f"{100*tol_relative_improvement:.2f}%. REVERTING this iter and stopping.")
            cur_theta, cur_R1, cur_R2, cur_J = snap_theta, snap_R1, snap_R2, snap_J
            break

        # accept this iteration
        cur_theta, cur_R1, cur_R2, cur_J = warm, new_R1, new_R2, new_J
        J_hist.append(cur_J); R_hist.append((cur_R1, cur_R2))
        total_nodes_added += nodes_added

        if total_nodes_added >= max_total_new_nodes:
            if verbose:
                print(f"\n  node budget {max_total_new_nodes} hit "
                      f"(added {total_nodes_added}). Stopping.")
            break

    if verbose:
        print(f"\n{'='*68}\n  DONE: J {J_hist[0]:.4e} -> {cur_J:.4e}  "
              f"({100*(J_hist[0]-cur_J)/J_hist[0]:.2f}% total)")
        print(f"        final (R1,R2) = ({cur_R1}, {cur_R2})   "
              f"nodes added: {total_nodes_added}")
        print(f"{'='*68}\n")
    return {'best_theta': cur_theta, 'R1': cur_R1, 'R2': cur_R2,
            'J_history': J_hist, 'R_history': R_hist,
            'total_nodes_added': total_nodes_added}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7e: Nu-only optimization driver (slack maximization)
#
# WHY a nu stage exists: moving nu_j does NOT change the sigma^2 values, so it
# does NOT directly affect the smoothness objective J = sum(Δσ²)². If we ran
# CMA-ES on the nus with the same J, the landscape would be FLAT.
#
# What nu moves CAN do is release bid-ask binding: by shifting which strikes
# the spread constraint binds at, the next sigma stage can reach values it
# couldn't before. So we optimize nus to MAXIMIZE bid-ask slack, then re-run
# the sigma stage to exploit the freed-up room.
#
# Objective for this stage:  minimize  -min_i slack_i
#   where slack_i = min(model_call_i - bid_i, ask_i - model_call_i)
# We use min-slack (worst strike) rather than sum to focus on the tightest
# constraint -- that's the one that actually limits the next sigma stage.
#
# Hard constraints (structural ordering, Kbar) are enforced by is_feasible
# returning the INFEASIBLE_PENALTY -- the same machinery the original v4
# driver used. The nu vectors must stay strictly increasing.
# ══════════════════════════════════════════════════════════════════════════════

def _bidask_min_slack(theta, R1, R2, S0, market_strikes, bid_prices, ask_prices):
    """Minimum bid-ask slack across all market strikes.

    slack_i = min(model_i - bid_i, ask_i - model_i)
    Positive => price inside spread with this much room to either side.
    Negative => price outside spread (bid-ask violation).
    Returns +inf if no market data passed.
    """
    if market_strikes is None or bid_prices is None or ask_prices is None:
        return np.inf
    m = compute_model_calls(theta, market_strikes, R1, R2, S0)
    return float(np.min(np.minimum(m - bid_prices, ask_prices - m)))


def cmaes_optimize_nus(theta_init, R1, R2, S0, Kbar,
                       market_strikes, bid_prices, ask_prices,
                       initial_step_size=2.0, max_generations=300,
                       convergence_tol=1e-10, random_seed=42,
                       verbose=True):
    """Optimize INTERIOR nu locations to maximize worst-strike bid-ask slack.

    Boundary nus (nus1[0], nus1[-1], nus2[0], nus2[-1]) are HELD FIXED to
    preserve structural feasibility: nus2[-1] = Kbar (constraint 4) and the
    boundary endpoints pinning the wing structure. Only the interior R1-2 +
    R2-2 nus move.

    Adjacent ordering is enforced by projecting each candidate: a tiny epsilon
    gap to the left and right neighbor inside the interior region. This way
    the structural feasibility check (is_feasible) almost never rejects a
    candidate, and the optimizer can focus on the slack signal.

    Sigmas are held fixed. Returns the new theta with updated nus.
    """
    sig_raw_left  = theta_init[R1 : 2*R1].copy()
    sig_raw_right = theta_init[2*R1 + R2 : 2*R1 + 2*R2].copy()
    nus1_0        = theta_init[:R1].copy()
    nus2_0        = theta_init[2*R1 : 2*R1 + R2].copy()

    # Boundary nus are pinned: nus1[0], nus1[-1], nus2[0], nus2[-1]
    # Interior optimizable indices:
    #   left wing : 1 .. R1-2  (count = R1 - 2)
    #   right wing: 1 .. R2-2  (count = R2 - 2)
    n_left_int  = R1 - 2
    n_right_int = R2 - 2
    if n_left_int <= 0 or n_right_int <= 0:
        if verbose:
            print(f"  Nu stage skipped: too few interior nus "
                  f"(left={n_left_int}, right={n_right_int}).")
        return {'best_theta': theta_init.copy(),
                'slack_baseline': _bidask_min_slack(theta_init, R1, R2, S0,
                                                    market_strikes, bid_prices, ask_prices),
                'slack_final': _bidask_min_slack(theta_init, R1, R2, S0,
                                                  market_strikes, bid_prices, ask_prices),
                'history': None}

    x0 = np.concatenate([nus1_0[1:R1-1], nus2_0[1:R2-1]])
    eps = 1e-3       # minimum gap between adjacent nus

    def assemble(x):
        """Reconstruct full theta with projected interior nus.

        Sort the interior block of each wing and clamp so neighbors stay
        eps-separated from the pinned endpoints and from each other.
        """
        x_left  = x[:n_left_int]
        x_right = x[n_left_int : n_left_int + n_right_int]

        # left wing: sort, clamp inside (nus1_0[0]+eps, nus1_0[-1]-eps),
        # and ensure eps gaps between consecutive interior values
        L_lo, L_hi = nus1_0[0] + eps, nus1_0[-1] - eps
        l_sorted = np.sort(np.clip(x_left, L_lo, L_hi))
        for k in range(1, len(l_sorted)):
            if l_sorted[k] - l_sorted[k-1] < eps:
                l_sorted[k] = l_sorted[k-1] + eps
        # final clamp in case the eps-spreading pushed past L_hi
        l_sorted = np.minimum(l_sorted, L_hi)
        # if clamping collapsed the upper tail, walk it back leftward
        for k in range(len(l_sorted)-2, -1, -1):
            if l_sorted[k+1] - l_sorted[k] < eps:
                l_sorted[k] = l_sorted[k+1] - eps

        # right wing: same procedure
        R_lo, R_hi = nus2_0[0] + eps, nus2_0[-1] - eps
        r_sorted = np.sort(np.clip(x_right, R_lo, R_hi))
        for k in range(1, len(r_sorted)):
            if r_sorted[k] - r_sorted[k-1] < eps:
                r_sorted[k] = r_sorted[k-1] + eps
        r_sorted = np.minimum(r_sorted, R_hi)
        for k in range(len(r_sorted)-2, -1, -1):
            if r_sorted[k+1] - r_sorted[k] < eps:
                r_sorted[k] = r_sorted[k+1] - eps

        nus1_full = np.concatenate([[nus1_0[0]], l_sorted, [nus1_0[-1]]])
        nus2_full = np.concatenate([[nus2_0[0]], r_sorted, [nus2_0[-1]]])
        return np.concatenate([nus1_full, sig_raw_left, nus2_full, sig_raw_right])

    ms  = np.asarray(market_strikes); bid = np.asarray(bid_prices); ask = np.asarray(ask_prices)

    # track best feasible candidate separately from the CMA-ES internal best,
    # so we never return a structurally-infeasible theta
    best_seen = {'slack': _bidask_min_slack(theta_init, R1, R2, S0, ms, bid, ask),
                 'theta': theta_init.copy()}

    def objective(x):
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                th = assemble(x)
                if not is_feasible(th, R1, R2, S0, Kbar):     # belt-and-braces
                    return INFEASIBLE_PENALTY
                slack = _bidask_min_slack(th, R1, R2, S0, ms, bid, ask)
                if not np.isfinite(slack):
                    return INFEASIBLE_PENALTY
                if slack > best_seen['slack']:
                    best_seen['slack'] = slack
                    best_seen['theta'] = th.copy()
                return -slack
        except Exception:
            return INFEASIBLE_PENALTY

    def _feas(x):
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                th = assemble(x)
                if not is_feasible(th, R1, R2, S0, Kbar):
                    return False
                return _bidask_min_slack(th, R1, R2, S0, ms, bid, ask) >= 0.0
        except Exception:
            return False

    slack0 = best_seen['slack']
    if verbose:
        print(f"\n  Nu-stage starting: worst-strike slack = {slack0:.4f}")
        print(f"  Free dimension = {len(x0)} (interior nus only; boundary nus pinned)")
        print(f"  Optimizer will try to INCREASE slack -> release bid-ask binding")

    _, _, history = _run_cmaes(
        evaluate_fn=objective, theta_init=x0,
        feasibility_fn=_feas,
        initial_step_size=initial_step_size, max_generations=max_generations,
        convergence_tol=convergence_tol, random_seed=random_seed, verbose=verbose)

    # Use the BEST FEASIBLE candidate we tracked, not _run_cmaes's internal best
    # (which can be the +INFEASIBLE_PENALTY ceiling if everything was bad).
    best_theta  = best_seen['theta']
    final_slack = best_seen['slack']
    if verbose:
        print(f"\n  Nu-stage done: worst-strike slack {slack0:.4f} -> {final_slack:.4f}  "
              f"({'INCREASED' if final_slack>slack0 else 'NO IMPROVEMENT'})")
        nu_shift = (best_theta[:R1] - nus1_0).tolist() + \
                   (best_theta[2*R1:2*R1+R2] - nus2_0).tolist()
        nu_shift = np.array(nu_shift)
        print(f"  nu movement: max|Δν| = {np.max(np.abs(nu_shift)):.3f}  "
              f"mean|Δν| = {np.mean(np.abs(nu_shift)):.3f}")
    return {'best_theta': best_theta,
            'slack_baseline': slack0, 'slack_final': final_slack,
            'history': history}


# ========================================================================
# (from notebook cell 18)
# ========================================================================

def _get_jumps(theta, R1, R2, S0):
    """All C'' jump values from a theta vector. Length = R1+R2-1."""
    (coeff_left, coeff_right, lambda1, lambda2,
     left_nodes, right_nodes,
     local_vols_left, local_vols_right) = _compute_wing_coefficients(theta, R1, R2, S0)
    jumps = []
    for j in range(R1-1):
        dist = left_nodes[j+2]-left_nodes[j+1]; s = local_vols_left[j+1]
        vr   = (1/s**2)*(coeff_left[j+1,0]*np.exp(dist/s)+coeff_left[j+1,1]*np.exp(-dist/s))
        vl   = (1/local_vols_left[j]**2)*(coeff_left[j,0]+coeff_left[j,1])
        jumps.append(vr-vl)
    jumps.append((1/local_vols_right[0]**2)*(coeff_right[0,0]+coeff_right[0,1])
                 - (1/local_vols_left[-1]**2)*(coeff_left[-1,0]+coeff_left[-1,1]))
    for j in range(R2-1):
        dist = right_nodes[j+1]-right_nodes[j]; s = local_vols_right[j]
        vl   = (1/s**2)*(coeff_right[j,0]*np.exp(-dist/s)+coeff_right[j,1]*np.exp(dist/s))
        vr   = (1/local_vols_right[j+1]**2)*(coeff_right[j+1,0]+coeff_right[j+1,1])
        jumps.append(vr-vl)
    return np.array(jumps)

def _get_sigma2_jumps(theta, R1, R2, S0):
    """The V/C'' = sigma^2 jumps across every partition node (unmasked)."""
    return _sigma2_jump_vector(theta, R1, R2)

def _inrange_stats(theta, R1, R2, S0, K_lo, K_hi):
    """(max, mean) |sigma^2 jump| over nodes inside [K_lo, K_hi]."""
    j   = _sigma2_jump_vector(theta, R1, R2)
    pos = _node_positions(theta, R1, R2, S0)
    m   = (pos >= K_lo) & (pos <= K_hi)
    if not np.any(m):
        return 0.0, 0.0
    a = np.abs(j[m])
    return float(np.max(a)), float(np.mean(a))

def _get_curves(theta, R1, R2, S0, n_pts=50):
    """C(K), C''(K), and V/C''=sigma^2(K) on a fine grid within each interval."""
    (coeff_left, coeff_right, lambda1, lambda2,
     left_nodes, right_nodes,
     local_vols_left, local_vols_right) = _compute_wing_coefficients(theta, R1, R2, S0)
    K_all=[]; C_all=[]; Cpp_all=[]; LV_all=[]
    for j in range(R1):
        nu_L=left_nodes[j]; nu_R=left_nodes[j+1]; sig=local_vols_left[j]; dt=nu_R-nu_L
        cd1=coeff_left[j,0]*np.exp(dt/sig); cd2=coeff_left[j,1]*np.exp(-dt/sig)
        for k in np.linspace(nu_L, nu_R, n_pts):
            dk=k-nu_L; V_k=cd1*np.exp(-dk/sig)+cd2*np.exp(dk/sig)
            K_all.append(k); C_all.append(V_k+max(S0-k,0)); Cpp_all.append(V_k/sig**2); LV_all.append(sig**2)
    for j in range(R2):
        nu_L=right_nodes[j]; nu_R=right_nodes[j+1]; sig=local_vols_right[j]; dt=nu_R-nu_L
        cd1=coeff_right[j,0]*np.exp(-dt/sig); cd2=coeff_right[j,1]*np.exp(dt/sig)
        for k in np.linspace(nu_L, nu_R, n_pts):
            dk=k-nu_L; V_k=cd1*np.exp(-dk/sig)+cd2*np.exp(dk/sig)
            K_all.append(k); C_all.append(V_k); Cpp_all.append(V_k/sig**2); LV_all.append(sig**2)
    K=np.array(K_all); idx=np.argsort(K)
    return K[idx], np.array(C_all)[idx], np.array(Cpp_all)[idx], np.array(LV_all)[idx]

def _generate_plots(history, J_baseline, best_J, improvement,
                    theta_baseline, best_theta, R1, R2, S0,
                    market_strikes=None, bid_prices=None, ask_prices=None,
                    save_path='cmaes_sigma_result.png'):
    """Four panels: soft-objective convergence, C(K), C''(K), V/C''=sigma^2(K)."""
    generations = history['generation']
    K_base, C_base, Cpp_base, LV_base = _get_curves(theta_baseline, R1, R2, S0)
    K_opt,  C_opt,  Cpp_opt,  LV_opt  = _get_curves(best_theta,     R1, R2, S0)
    jumps_base = _get_jumps(theta_baseline, R1, R2, S0)
    jumps_opt  = _get_jumps(best_theta,     R1, R2, S0)

    if market_strikes is not None and len(market_strikes) > 0:
        K_lo = float(np.min(market_strikes)) - 30; K_hi = float(np.max(market_strikes)) + 30
    else:
        K_lo, K_hi = S0 - 150, S0 + 200
    s2max_b, s2mean_b = _inrange_stats(theta_baseline, R1, R2, S0, K_lo, K_hi)
    s2max_o, s2mean_o = _inrange_stats(best_theta,     R1, R2, S0, K_lo, K_hi)

    fig, axes = plt.subplots(4, 1, figsize=(13, 21))
    fig.suptitle(f"Sigma-only CMA-ES  —  V/C'' (local-variance) smoothness\n"
                 f"masked J: {J_baseline:.2f} -> {best_J:.2f}   [{improvement:.2f}% improvement]",
                 fontsize=13, fontweight='bold', y=1.005)

    ax=axes[0]
    ax.semilogy(generations, history['best_J_per_gen'], 'b-', lw=2, label='Best objective (J + penalty)')
    ax.set_xlabel('Generation'); ax.set_ylabel('objective [log]')
    ax.set_title("Convergence (soft objective = masked J + bid-ask penalty)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax=axes[1]
    mb=(K_base>=K_lo)&(K_base<=K_hi); mo=(K_opt>=K_lo)&(K_opt<=K_hi)
    ax.plot(K_base[mb], C_base[mb], 'r-', lw=1.8, alpha=0.8, label='C(K) — baseline')
    ax.plot(K_opt[mo],  C_opt[mo],  'b-', lw=1.8, alpha=0.8, label='C(K) — optimized')
    if market_strikes is not None and bid_prices is not None and ask_prices is not None:
        ms=np.array(market_strikes); bd=np.array(bid_prices); ak=np.array(ask_prices); mm=(ms>=K_lo)&(ms<=K_hi)
        ax.scatter(ms[mm], ak[mm], color='gray', s=25, marker='^', zorder=5, label='Ask')
        ax.scatter(ms[mm], bd[mm], color='gray', s=25, marker='v', zorder=5, label='Bid')
    ax.axvline(S0, color='gray', ls='--', lw=1.5, label=f'S0 = {S0:.0f}')
    ax.set_xlim(K_lo, K_hi); ax.set_xlabel('Strike K'); ax.set_ylabel('C(K)')
    ax.set_title('Call Price C(K)  —  Baseline vs Optimized'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax=axes[2]
    Cc=np.concatenate([Cpp_base[mb], Cpp_opt[mo]]); clo,chi=np.percentile(Cc,2),np.percentile(Cc,98)
    ax.plot(K_base[mb], np.clip(Cpp_base[mb],clo,chi), 'r-', lw=1.5, alpha=0.85, label="C''(K) — baseline")
    ax.plot(K_opt[mo],  np.clip(Cpp_opt[mo], clo,chi), 'b-', lw=1.5, alpha=0.85, label="C''(K) — optimized")
    ax.axvline(S0, color='gray', ls='--', lw=1.5); ax.axhline(0, color='black', lw=0.5)
    ax.set_xlim(K_lo, K_hi); ax.set_xlabel('Strike K'); ax.set_ylabel("C''(K)")
    ax.set_title(f"C''(K) density  (Max |jump|: {np.max(np.abs(jumps_base)):.5f} -> {np.max(np.abs(jumps_opt)):.5f})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax=axes[3]
    Lc=np.concatenate([LV_base[mb], LV_opt[mo]]); llo,lhi=np.percentile(Lc,1),np.percentile(Lc,99)
    ax.plot(K_base[mb], np.clip(LV_base[mb],llo,lhi), 'r-', lw=1.5, alpha=0.85, drawstyle='steps-post', label="V/C'' = sigma^2 — baseline")
    ax.plot(K_opt[mo],  np.clip(LV_opt[mo], llo,lhi), 'b-', lw=1.5, alpha=0.85, drawstyle='steps-post', label="V/C'' = sigma^2 — optimized")
    ax.axvline(S0, color='gray', ls='--', lw=1.5)
    ax.set_xlim(K_lo, K_hi); ax.set_xlabel('Strike K'); ax.set_ylabel("V/C''(K) = sigma^2(K)")
    ax.set_title(f"Local variance V/C''  (in-range Max |jump|: {s2max_b:.1f} -> {s2max_o:.1f}  |  Mean: {s2mean_b:.1f} -> {s2mean_o:.1f})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight'); print(f"  Plot saved to: {save_path}")
    except Exception as e:
        print(f"  (could not save plot: {e})")
    plt.show(); plt.close(fig)

    print(f"\n  -- Plot Summary --------------------------------------")
    print(f"    masked V/C'' J     : {J_baseline:.4f}  ->  {best_J:.4f}   ({improvement:.2f}%)")
    print(f"    in-range Max|dsig^2|: {s2max_b:.3f}  ->  {s2max_o:.3f}")
    print(f"    in-range Mean|dsig^2|: {s2mean_b:.3f}  ->  {s2mean_o:.3f}")
    print(f"  ------------------------------------------------------")
