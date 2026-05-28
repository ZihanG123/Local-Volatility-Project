"""
cmaes_optimizer_v3.py
=====================
CMA-ES optimizer for the LVG local volatility model.

Key improvements over v2:
  1. No log-gap transform — CMA-ES works directly in the natural
     parameter space (partition_points, local_vols). Infeasible
     candidates are filtered out (assigned a large penalty J value)
     rather than being excluded by a coordinate transform.
  2. Clearer variable names throughout. 
  3. Richer convergence plot: before/after C'' curves, jump bar chart,
     J convergence curve, and a printed summary with percentages.

Parameter vector layout (same as teammates):
    theta = [
        partition_pts_left  (R1 values),   # nus1
        local_vols_left     (R1 values),   # sigs1_raw  (softplus applied inside J)
        partition_pts_right (R2 values),   # nus2
        local_vols_right    (R2 values),   # sigs2_raw  (softplus applied inside J)
    ]

Constraints (checked by is_feasible, infeasible => penalty):
    - partition_pts_left  strictly increasing, all < S0
    - partition_pts_right strictly increasing, last value == Kbar (fixed)
    - local_vols_left  and local_vols_right: no hard constraint needed
      because softplus(x) + 1e-6 > 0 for all real x

Usage:
------
    from cmaes_optimizer_v3 import cmaes_optimize

    result = cmaes_optimize(
        theta_init = theta0,
        R1         = 70,
        R2         = 34,
        S0         = 1271.87,
        Kbar       = 2000.0,
    )
    best_theta = result['best_theta']
    best_J     = result['best_J']
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Softplus helpers  (match torch.nn.functional.softplus exactly)
# ══════════════════════════════════════════════════════════════════════════════

def softplus(x):
    """
    Numerically stable softplus: log(1 + exp(x)).
    Maps any real number to a positive number.
    Used inside calculate_J to enforce sigma > 0.
    """
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def inv_softplus(x):
    """
    Inverse softplus: log(exp(x) - 1).
    Converts an actual positive sigma value back to the raw form
    stored in theta (so that softplus(raw) + 1e-6 ≈ sigma).
    """
    return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-8, 500))))


# ══════════════════════════════════════════════════════════════════════════════
# Feasibility check
# ══════════════════════════════════════════════════════════════════════════════

def is_feasible(theta, R1, R2, S0, Kbar):
    """
    Check whether a candidate theta satisfies the constraints:
      - partition_pts_left  must be strictly increasing and all < S0
      - partition_pts_right must be strictly increasing and last == Kbar

    If infeasible, CMA-ES assigns a large penalty instead of evaluating J,
    effectively killing the candidate without modifying the search space.

    Parameters
    ----------
    theta : ndarray (2*R1 + 2*R2,)
    R1, R2 : int
    S0, Kbar : float

    Returns
    -------
    bool
    """
    partition_pts_left  = theta[:R1]
    partition_pts_right = theta[2*R1 : 2*R1 + R2]

    # Left partition points must be strictly increasing
    if np.any(np.diff(partition_pts_left) <= 0):
        return False

    # All left partition points must be < S0
    if np.any(partition_pts_left >= S0):
        return False

    # Right partition points must be strictly increasing
    if np.any(np.diff(partition_pts_right) <= 0):
        return False

    # Last right partition point must equal Kbar (fixed boundary)
    if abs(partition_pts_right[-1] - Kbar) > 1.0:
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
# J function  (numpy version of teammates' calculate_J)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_J(theta, R1, R2, S0):
    """
    Compute J(theta) = sum of squared jumps in C'' at all partition points.

    Matches teammates' PyTorch calculate_J exactly.

    Parameters
    ----------
    theta : ndarray (2*R1 + 2*R2,)
        [partition_pts_left(R1), local_vols_left_raw(R1),
         partition_pts_right(R2), local_vols_right_raw(R2)]
    R1, R2 : int
    S0 : float  spot price

    Returns
    -------
    J_value : float
    lambda1 : float  left wing scaling constant
    lambda2 : float  right wing scaling constant
    """

    # ── Unpack theta ──────────────────────────────────────────────────────
    partition_pts_left   = theta[:R1]
    local_vols_left_raw  = theta[R1 : 2*R1]
    partition_pts_right  = theta[2*R1 : 2*R1 + R2]
    local_vols_right_raw = theta[2*R1 + R2 :]

    # S0 is the junction between left and right wings
    # Append S0 to the end of left, prepend S0 to start of right
    left_nodes  = np.append(partition_pts_left, S0)   # length R1+1
    right_nodes = np.insert(partition_pts_right, 0, S0) # length R2+1

    # Convert raw sigma values to actual (positive) local vols via softplus
    local_vols_left  = softplus(local_vols_left_raw)  + 1e-6  # length R1
    local_vols_right = softplus(local_vols_right_raw) + 1e-6  # length R2

    # ── Build left wing coefficients (unit scale: lambda1 = 1) ───────────
    # On each interval [left_nodes[j], left_nodes[j+1]], the time value V(K)
    # satisfies V''(K) = (1/sigma^2) * V(K), giving V(K) = A*exp(K/s) + B*exp(-K/s)
    # We store distance-scaled coefficients to avoid exp overflow.
    #
    # coeff_left[j] = [c1_j, c2_j] where
    #   c1_j = true_c1 * exp(-dist_j / sigma_j)   (scaled at left endpoint)
    #   c2_j = true_c2 * exp(+dist_j / sigma_j)   (scaled at left endpoint)
    # and V at the RIGHT endpoint of interval j = c1_j + c2_j

    coeff_left = np.zeros((R1, 2))

    # Boundary condition at left edge (K=0): V(0) = 0, slope > 0
    # This gives the unit starting coefficients
    dist_first = left_nodes[1] - left_nodes[0]
    sigma_0    = local_vols_left[0]
    coeff_left[0, 0] = -np.exp(-dist_first / sigma_0)
    coeff_left[0, 1] =  np.exp( dist_first / sigma_0)

    # Propagate left -> right using C1 continuity at each partition point
    for j in range(R1 - 1):
        # V and V' at the right endpoint of interval j
        V_at_right_j  = coeff_left[j, 0] + coeff_left[j, 1]
        Vp_at_right_j = (1.0 / local_vols_left[j]) * (
                            -coeff_left[j, 0] + coeff_left[j, 1])

        dist_next   = left_nodes[j+2] - left_nodes[j+1]
        sigma_next  = local_vols_left[j+1]

        coeff_left[j+1, 0] = 0.5 * (V_at_right_j - sigma_next * Vp_at_right_j) \
                              * np.exp(-dist_next / sigma_next)
        coeff_left[j+1, 1] = 0.5 * (V_at_right_j + sigma_next * Vp_at_right_j) \
                              * np.exp( dist_next / sigma_next)

    # ── Build right wing coefficients (unit scale: lambda2 = 1) ──────────
    # Same idea but propagate right -> left
    # Boundary condition at right edge (K=Kbar): V(Kbar) = 0, slope < 0

    coeff_right = np.zeros((R2, 2))

    dist_last  = right_nodes[-1] - right_nodes[-2]
    sigma_last = local_vols_right[-1]
    coeff_right[-1, 0] =  np.exp( dist_last / sigma_last)
    coeff_right[-1, 1] = -np.exp(-dist_last / sigma_last)

    # Propagate right -> left
    for j in range(R2 - 1, 0, -1):
        V_at_left_j  = coeff_right[j, 0] + coeff_right[j, 1]
        Vp_at_left_j = (1.0 / local_vols_right[j]) * (
                            -coeff_right[j, 0] + coeff_right[j, 1])

        dist_prev  = right_nodes[j] - right_nodes[j-1]
        sigma_prev = local_vols_right[j-1]

        coeff_right[j-1, 0] = 0.5 * (V_at_left_j - sigma_prev * Vp_at_left_j) \
                               * np.exp( dist_prev / sigma_prev)
        coeff_right[j-1, 1] = 0.5 * (V_at_left_j + sigma_prev * Vp_at_left_j) \
                               * np.exp(-dist_prev / sigma_prev)

    # ── Solve for lambda1 and lambda2 at S0 ──────────────────────────────
    # At S0, the left and right wings must satisfy C1 continuity:
    #   lambda1 * V_left(S0) = lambda2 * V_right(S0)         (value match)
    #   lambda1 * V'_left(S0) - lambda2 * V'_right(S0) = 1   (derivative jump = -1)
    #
    # The -1 comes from the kink in (S0 - K)+: its derivative jumps by -1 at S0

    V_left_at_S0   = coeff_left[-1, 0]  + coeff_left[-1, 1]
    Vp_left_at_S0  = (1.0 / local_vols_left[-1]) * (
                         -coeff_left[-1, 0] + coeff_left[-1, 1])

    V_right_at_S0  = coeff_right[0, 0]  + coeff_right[0, 1]
    Vp_right_at_S0 = (1.0 / local_vols_right[0]) * (
                         -coeff_right[0, 0] + coeff_right[0, 1])

    denom   = Vp_left_at_S0 * V_right_at_S0 - V_left_at_S0 * Vp_right_at_S0
    lambda1 = V_right_at_S0 / denom
    lambda2 = V_left_at_S0  / denom

    # Scale all coefficients by their respective lambdas
    coeff_left  = coeff_left  * lambda1
    coeff_right = coeff_right * lambda2

    # ── Compute C'' jumps at every partition point ────────────────────────
    # C''(K) = (1/sigma^2) * V(K) on each interval.
    # At partition point nu_j, the jump is:
    #   Delta_C'' = C''(nu_j from right) - C''(nu_j from left)
    #
    # J = sum of (Delta_C'')^2

    all_jumps = []

    # Left wing internal jumps: at left_nodes[j+1] for j = 0,...,R1-2
    for j in range(R1 - 1):
        # C'' approaching nu_{j+1} from the LEFT (end of interval j)
        # V at right endpoint of interval j = coeff_left[j,0] + coeff_left[j,1]
        Cpp_from_left = (1.0 / local_vols_left[j]**2) * (
                            coeff_left[j, 0] + coeff_left[j, 1])

        # C'' leaving nu_{j+1} to the RIGHT (start of interval j+1)
        # V at left endpoint of interval j+1: need to un-scale the coefficients
        dist_j1       = left_nodes[j+2] - left_nodes[j+1]
        sigma_j1      = local_vols_left[j+1]
        V_at_left_j1  = (coeff_left[j+1, 0] * np.exp( dist_j1 / sigma_j1) +
                          coeff_left[j+1, 1] * np.exp(-dist_j1 / sigma_j1))
        Cpp_from_right = (1.0 / sigma_j1**2) * V_at_left_j1

        all_jumps.append(Cpp_from_right - Cpp_from_left)

    # Junction jump at S0 (between left and right wings)
    Cpp_left_at_S0  = (1.0 / local_vols_left[-1]**2) * (
                          coeff_left[-1, 0] + coeff_left[-1, 1])
    Cpp_right_at_S0 = (1.0 / local_vols_right[0]**2) * (
                          coeff_right[0, 0] + coeff_right[0, 1])
    all_jumps.append(Cpp_right_at_S0 - Cpp_left_at_S0)

    # Right wing internal jumps: at right_nodes[j+1] for j = 0,...,R2-2
    for j in range(R2 - 1):
        # C'' leaving nu_{j+1} to the RIGHT (start of interval j+1)
        Cpp_from_right = (1.0 / local_vols_right[j+1]**2) * (
                             coeff_right[j+1, 0] + coeff_right[j+1, 1])

        # C'' approaching nu_{j+1} from the LEFT (end of interval j)
        dist_j        = right_nodes[j+1] - right_nodes[j]
        sigma_j       = local_vols_right[j]
        V_at_right_j  = (coeff_right[j, 0] * np.exp(-dist_j / sigma_j) +
                          coeff_right[j, 1] * np.exp( dist_j / sigma_j))
        Cpp_from_left = (1.0 / sigma_j**2) * V_at_right_j

        all_jumps.append(Cpp_from_right - Cpp_from_left)

    J_value = float(np.sum(np.array(all_jumps)**2))
    return J_value, float(lambda1), float(lambda2)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate a candidate: return J if feasible, large penalty if not
# ══════════════════════════════════════════════════════════════════════════════

INFEASIBLE_PENALTY = 1e6   # assigned to candidates that violate constraints

def evaluate_candidate(theta, R1, R2, S0, Kbar):
    """
    Evaluate J for one candidate theta.
    Returns INFEASIBLE_PENALTY if the candidate violates any constraint,
    so CMA-ES naturally kills it in the selection step.
    """
    if not is_feasible(theta, R1, R2, S0, Kbar):
        return INFEASIBLE_PENALTY

    try:
        J_value, _, _ = calculate_J(theta, R1, R2, S0)
        return J_value if np.isfinite(J_value) else INFEASIBLE_PENALTY
    except Exception:
        return INFEASIBLE_PENALTY


# ══════════════════════════════════════════════════════════════════════════════
# CMA-ES core
# ══════════════════════════════════════════════════════════════════════════════

def _run_cmaes(evaluate_fn, theta_init, initial_step_size=1.0,
               max_generations=500, convergence_tol=1e-10,
               random_seed=42, verbose=True):
    """
    Core CMA-ES algorithm.

    Notation follows Hansen (2016) arXiv:1604.00772 exactly.

    Parameters
    ----------
    evaluate_fn       : callable  theta -> float  (the objective)
    theta_init        : ndarray   initial mean (n,)
    initial_step_size : float     sigma_0
    max_generations   : int       maximum number of generations
    convergence_tol   : float     stop when sigma < this value
    random_seed       : int
    verbose           : bool

    Returns
    -------
    best_theta  : ndarray   best parameter vector found
    best_J      : float     best J value found
    history     : dict      convergence data
    """
    np.random.seed(random_seed)
    n = len(theta_init)

    # ── CMA-ES hyperparameters (Hansen 2016, Table 1) ─────────────────────
    # Population size lambda and number of parents mu
    population_size = 4 + int(np.floor(3 * np.log(n)))
    num_parents     = population_size // 2

    # Recombination weights: log-spaced, normalized to sum to 1
    raw_weights = np.log(num_parents + 0.5) - np.log(np.arange(1, num_parents + 1))
    weights     = raw_weights / raw_weights.sum()

    # Variance-effective selection mass (how much information we use per step)
    mueff = 1.0 / np.sum(weights**2)

    # Step-size control parameters
    step_size_decay      = (mueff + 2) / (n + mueff + 5)
    step_size_damping    = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + step_size_decay
    expected_norm_sphere = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))

    # Covariance matrix adaptation parameters
    cov_path_decay  = (4 + mueff/n) / (n + 4 + 2*mueff/n)
    cov_rank_one    = 2 / ((n + 1.3)**2 + mueff)
    cov_rank_mu     = min(1 - cov_rank_one,
                          2*(mueff-2+1/mueff) / ((n+2)**2 + mueff))
    h_sigma_thresh  = (1.4 + 2/(n+1)) * expected_norm_sphere

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CMA-ES  (v3 — direct parameter space, no log-transform)")
        print(f"{'='*65}")
        print(f"  Dimension           n  = {n}")
        print(f"  Population size     λ  = {population_size}")
        print(f"  Number of parents   μ  = {num_parents}")
        print(f"  Effective mass    μeff = {mueff:.2f}")
        print(f"  Initial step size  σ₀  = {initial_step_size}")
        print(f"  Max generations        = {max_generations}")
        print(f"  Convergence tol        = {convergence_tol:.1e}")
        print(f"{'='*65}")
        print(f"  {'Gen':>5} | {'Best J':>14} | {'Mean J':>14} | "
              f"{'Feasible':>10} | {'σ':>10}")
        print(f"  {'-'*60}")

    # ── State variables ───────────────────────────────────────────────────
    mean             = theta_init.copy()       # current distribution mean
    step_size        = initial_step_size       # global step size sigma
    evolution_path_C = np.zeros(n)             # path for covariance update
    evolution_path_s = np.zeros(n)             # path for step-size update
    eigenvectors     = np.eye(n)               # B: columns are eigenvectors of C
    axis_lengths     = np.ones(n)              # D: sqrt of eigenvalues of C
    covariance       = np.eye(n)               # C: covariance matrix
    inv_sqrt_cov     = np.eye(n)               # C^{-1/2}: for step-size path
    last_eigen_update = 0                      # generation of last eigendecomp

    best_J     = np.inf
    best_theta = mean.copy()

    history = {
        'best_J_per_gen':  [],   # best J found so far at each generation
        'mean_J_per_gen':  [],   # mean J of the population
        'step_size':       [],   # sigma at each generation
        'feasible_count':  [],   # number of feasible candidates per gen
        'generation':      []
    }

    # ── Generation loop ───────────────────────────────────────────────────
    for generation in range(max_generations):

        # 1. Sample lambda candidate solutions from N(mean, sigma^2 * C)
        standard_samples = np.random.randn(population_size, n)
        candidates = np.array([
            mean + step_size * (eigenvectors @ (axis_lengths * standard_samples[k]))
            for k in range(population_size)
        ])

        # 2. Evaluate each candidate
        #    Infeasible candidates receive INFEASIBLE_PENALTY automatically
        fitness_values = np.array([evaluate_fn(candidates[k]) for k in range(population_size)])

        # Count feasible candidates this generation
        num_feasible = int(np.sum(fitness_values < INFEASIBLE_PENALTY))

        # 3. Sort by fitness (ascending: lower J is better)
        sorted_indices = np.argsort(fitness_values)

        # 4. Update best solution found so far
        if fitness_values[sorted_indices[0]] < best_J:
            best_J     = fitness_values[sorted_indices[0]]
            best_theta = candidates[sorted_indices[0]].copy()

        # 5. Update mean: weighted average of the best mu_parents candidates
        old_mean = mean.copy()
        mean = np.sum(
            weights[:, None] * candidates[sorted_indices[:num_parents]],
            axis=0
        )

        # 6. Update step-size evolution path p_s
        #    This accumulates normalized mean movements for step-size control
        evolution_path_s = (
            (1 - step_size_decay) * evolution_path_s
            + np.sqrt(step_size_decay * (2 - step_size_decay) * mueff)
            * inv_sqrt_cov @ (mean - old_mean) / step_size
        )

        # 7. Heaviside indicator h_sigma
        #    Suppresses rank-one update if evolution path is too long
        #    (which would indicate the path is unreliable)
        path_length_normalized = (
            np.linalg.norm(evolution_path_s)
            / np.sqrt(1 - (1 - step_size_decay)**(2*(generation+1)))
        )
        h_sigma = (path_length_normalized < h_sigma_thresh)

        # 8. Update covariance evolution path p_c
        #    This accumulates the history of mean steps for rank-one update
        evolution_path_C = (
            (1 - cov_path_decay) * evolution_path_C
            + h_sigma * np.sqrt(cov_path_decay * (2 - cov_path_decay) * mueff)
            * (mean - old_mean) / step_size
        )

        # 9. Update covariance matrix C
        #    Rank-one update: uses the single evolution path direction
        #    Rank-mu update: uses all num_parents best steps this generation
        normalized_steps = (
            (1.0 / step_size)
            * (candidates[sorted_indices[:num_parents]] - old_mean)
        )
        covariance = (
            (1 - cov_rank_one - cov_rank_mu) * covariance
            + cov_rank_one * (
                np.outer(evolution_path_C, evolution_path_C)
                + (1 - h_sigma) * cov_path_decay * (2 - cov_path_decay) * covariance
            )
            + cov_rank_mu * np.sum(
                weights[:, None, None]
                * (normalized_steps[:, :, None] * normalized_steps[:, None, :]),
                axis=0
            )
        )

        # 10. Update step size sigma via cumulative step-size adaptation (CSA)
        #     sigma grows when steps are correlated (good progress)
        #     sigma shrinks when steps oscillate (near minimum)
        step_size = step_size * np.exp(
            (step_size_decay / step_size_damping)
            * (np.linalg.norm(evolution_path_s) / expected_norm_sphere - 1)
        )

        # 11. Periodically update eigendecomposition of C
        #     (expensive O(n^3) operation, done every ~n/10 generations)
        eigendecomp_interval = n / (10 * population_size * (cov_rank_one + cov_rank_mu))
        if generation - last_eigen_update > eigendecomp_interval:
            last_eigen_update = generation
            covariance        = np.triu(covariance) + np.triu(covariance, 1).T
            eigenvalues_sq, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues_sq    = np.maximum(eigenvalues_sq, 1e-20)
            axis_lengths      = np.sqrt(eigenvalues_sq)
            inv_sqrt_cov      = eigenvectors @ np.diag(1.0 / axis_lengths) @ eigenvectors.T

        # 12. Record history
        history['best_J_per_gen'].append(best_J)
        history['mean_J_per_gen'].append(float(np.mean(fitness_values)))
        history['step_size'].append(float(step_size))
        history['feasible_count'].append(num_feasible)
        history['generation'].append(generation)

        # 13. Print progress every 20 generations
        if verbose and generation % 20 == 0:
            print(f"  {generation:>5} | {best_J:>14.8f} | "
                  f"{np.mean(fitness_values):>14.4f} | "
                  f"{num_feasible:>10}/{population_size} | "
                  f"{step_size:>10.6f}")

        # 14. Convergence check: stop when step size collapses
        if step_size < convergence_tol:
            if verbose:
                print(f"\n  Converged at generation {generation}: "
                      f"σ = {step_size:.2e} < tol = {convergence_tol:.2e}")
            break
        if best_J < 1e-12:
            if verbose:
                print(f"\n  Converged at generation {generation}: "
                      f"J = {best_J:.2e} < 1e-12")
            break

    if verbose:
        print(f"  {'-'*60}")

    return best_theta, best_J, history


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def cmaes_optimize(theta_init, R1, R2, S0, Kbar,
                   initial_step_size=1.0,
                   max_generations=500,
                   convergence_tol=1e-10,
                   random_seed=42,
                   verbose=True,
                   save_plot=True,
                   plot_path='/mnt/user-data/outputs/cmaes_v3_result.png'):
    """
    Minimize J(theta) using CMA-ES, working directly in natural parameter space.

    Infeasible candidates (violated ordering or boundary constraints) are
    automatically filtered by assigning them a large penalty value, so
    they are killed in the selection step without needing a coordinate
    transform.

    Parameters
    ----------
    theta_init        : ndarray (2*R1 + 2*R2,)
                        Initial parameter vector from the LVG calibration.
                        Layout: [partition_pts_left(R1), local_vols_left_raw(R1),
                                 partition_pts_right(R2), local_vols_right_raw(R2)]

    R1, R2            : int   number of left/right partition points
    S0                : float spot price
    Kbar              : float right boundary (fixed = last element of partition_pts_right)
    initial_step_size : float CMA-ES initial sigma (default 1.0)
    max_generations   : int   maximum generations (default 500)
    convergence_tol   : float stop when sigma < this (default 1e-10)
    random_seed       : int   for reproducibility (default 42)
    verbose           : bool  print progress (default True)
    save_plot         : bool  save convergence + comparison plot (default True)
    plot_path         : str   path for the output plot

    Returns
    -------
    dict with keys:
        'best_theta'   ndarray  optimized parameter vector
        'best_J'       float    optimized J value
        'J_initial'    float    baseline J value
        'improvement'  float    percentage improvement
        'lambda1_opt'  float    optimized lambda1
        'lambda2_opt'  float    optimized lambda2
        'history'      dict     convergence data per generation
    """

    # ── Compute baseline ──────────────────────────────────────────────────
    J_initial, lambda1_init, lambda2_init = calculate_J(theta_init, R1, R2, S0)
    if verbose:
        print(f"\nBaseline J(theta_init)  = {J_initial:.8f}")
        print(f"Baseline lambda1        = {lambda1_init:.6f}")
        print(f"Baseline lambda2        = {lambda2_init:.6f}")

    # ── Wrap evaluate_candidate with fixed R1, R2, S0, Kbar ───────────────
    def objective(theta):
        return evaluate_candidate(theta, R1, R2, S0, Kbar)

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

    improvement = 100.0 * (J_initial - best_J) / J_initial if J_initial > 0 else 0.0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Optimization Complete")
        print(f"{'='*65}")
        print(f"  J(theta_init)  = {J_initial:.8f}")
        print(f"  J(theta_opt)   = {best_J:.8f}")
        print(f"  Improvement    = {improvement:.4f}%")
        print(f"  Generations    = {len(history['generation'])}")
        print(f"  lambda1        = {lambda1_init:.4f}  ->  {lambda1_opt:.4f}")
        print(f"  lambda2        = {lambda2_init:.4f}  ->  {lambda2_opt:.4f}")
        print(f"{'='*65}\n")

    # ── Generate and save plots ───────────────────────────────────────────
    if save_plot:
        _generate_plots(
            history     = history,
            J_initial   = J_initial,
            best_J      = best_J,
            improvement = improvement,
            theta_init  = theta_init,
            best_theta  = best_theta,
            R1=R1, R2=R2, S0=S0,
            save_path   = plot_path
        )
        if verbose:
            print(f"  Plot saved to: {plot_path}")

    return {
        'best_theta':  best_theta,
        'best_J':      best_J,
        'J_initial':   J_initial,
        'improvement': improvement,
        'lambda1_opt': lambda1_opt,
        'lambda2_opt': lambda2_opt,
        'history':     history
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _get_jumps(theta, R1, R2, S0):
    """Extract all C'' jumps from a theta vector."""
    _, _, _ = calculate_J(theta, R1, R2, S0)   # verify it runs
    partition_pts_left   = theta[:R1]
    local_vols_left_raw  = theta[R1:2*R1]
    partition_pts_right  = theta[2*R1:2*R1+R2]
    local_vols_right_raw = theta[2*R1+R2:]
    left_nodes  = np.append(partition_pts_left, S0)
    right_nodes = np.insert(partition_pts_right, 0, S0)
    local_vols_left  = softplus(local_vols_left_raw)  + 1e-6
    local_vols_right = softplus(local_vols_right_raw) + 1e-6

    coeff_left = np.zeros((R1, 2))
    d0 = left_nodes[1]-left_nodes[0]; s0v = local_vols_left[0]
    coeff_left[0] = [-np.exp(-d0/s0v), np.exp(d0/s0v)]
    for j in range(R1-1):
        P = coeff_left[j,0]+coeff_left[j,1]
        Sv = (1/local_vols_left[j])*(-coeff_left[j,0]+coeff_left[j,1])
        d = left_nodes[j+2]-left_nodes[j+1]; s = local_vols_left[j+1]
        coeff_left[j+1] = [0.5*(P-s*Sv)*np.exp(-d/s), 0.5*(P+s*Sv)*np.exp(d/s)]

    coeff_right = np.zeros((R2, 2))
    du = right_nodes[-1]-right_nodes[-2]; sl = local_vols_right[-1]
    coeff_right[-1] = [np.exp(du/sl), -np.exp(-du/sl)]
    for j in range(R2-1,0,-1):
        P = coeff_right[j,0]+coeff_right[j,1]
        Sv = (1/local_vols_right[j])*(-coeff_right[j,0]+coeff_right[j,1])
        d = right_nodes[j]-right_nodes[j-1]; s = local_vols_right[j-1]
        coeff_right[j-1] = [0.5*(P-s*Sv)*np.exp(d/s), 0.5*(P+s*Sv)*np.exp(-d/s)]

    vL = coeff_left[-1,0]+coeff_left[-1,1]
    DkL = (1/local_vols_left[-1])*(-coeff_left[-1,0]+coeff_left[-1,1])
    vR = coeff_right[0,0]+coeff_right[0,1]
    DkR = (1/local_vols_right[0])*(-coeff_right[0,0]+coeff_right[0,1])
    denom = DkL*vR - vL*DkR
    coeff_left  *= vR/denom
    coeff_right *= vL/denom

    jumps = []
    for j in range(R1-1):
        d = left_nodes[j+2]-left_nodes[j+1]; s = local_vols_left[j+1]
        vr = (1/s**2)*(coeff_left[j+1,0]*np.exp(d/s)+coeff_left[j+1,1]*np.exp(-d/s))
        vl = (1/local_vols_left[j]**2)*(coeff_left[j,0]+coeff_left[j,1])
        jumps.append(vr-vl)
    jumps.append(
        (1/local_vols_right[0]**2)*(coeff_right[0,0]+coeff_right[0,1])
        - (1/local_vols_left[-1]**2)*(coeff_left[-1,0]+coeff_left[-1,1])
    )
    for j in range(R2-1):
        d = right_nodes[j+1]-right_nodes[j]; s = local_vols_right[j]
        vl = (1/s**2)*(coeff_right[j,0]*np.exp(-d/s)+coeff_right[j,1]*np.exp(d/s))
        vr = (1/local_vols_right[j+1]**2)*(coeff_right[j+1,0]+coeff_right[j+1,1])
        jumps.append(vr-vl)
    return np.array(jumps)


def _generate_plots(history, J_initial, best_J, improvement,
                    theta_init, best_theta, R1, R2, S0, save_path):
    """
    Three-panel figure:
      Panel 1 — J convergence curve (log scale)
      Panel 2 — C(K) call price before vs after
      Panel 3 — C''(K) second derivative before vs after
    """
    generations  = history['generation']
    jumps_before = _get_jumps(theta_init, R1, R2, S0)
    jumps_after  = _get_jumps(best_theta,  R1, R2, S0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # ── Panel 1: J convergence ────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(generations, history['best_J_per_gen'],
                'b-', lw=2, label='Best J per generation')
    ax.axhline(J_initial, color='gray', ls=':', lw=2,
               label=f'Baseline J₀ = {J_initial:.6f}')
    ax.axhline(best_J, color='green', ls='--', lw=1.5,
               label=f'Optimized J* = {best_J:.2e}')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('J(θ)  [log scale]', fontsize=12)
    ax.set_title(
        f'J Convergence — {J_initial:.6f} → {best_J:.2e} '
        f'({improvement:.2f}% improvement)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: C(K) before vs after ────────────────────────────────────
    # Reconstruct C(K) = V(K) + (S0 - K)+ on a fine grid
    def get_C_curve(theta, n_pts=60):
        partition_pts_left   = theta[:R1]
        local_vols_left_raw  = theta[R1:2*R1]
        partition_pts_right  = theta[2*R1:2*R1+R2]
        local_vols_right_raw = theta[2*R1+R2:]
        left_nodes  = np.append(partition_pts_left, S0)
        right_nodes = np.insert(partition_pts_right, 0, S0)
        local_vols_left  = softplus(local_vols_left_raw)  + 1e-6
        local_vols_right = softplus(local_vols_right_raw) + 1e-6

        # Rebuild coefficients
        c1 = np.zeros((R1, 2))
        d0 = left_nodes[1]-left_nodes[0]; s0v = local_vols_left[0]
        c1[0] = [-np.exp(-d0/s0v), np.exp(d0/s0v)]
        for j in range(R1-1):
            P  = c1[j,0]+c1[j,1]
            Sv = (1/local_vols_left[j])*(-c1[j,0]+c1[j,1])
            d  = left_nodes[j+2]-left_nodes[j+1]; s = local_vols_left[j+1]
            c1[j+1] = [0.5*(P-s*Sv)*np.exp(-d/s), 0.5*(P+s*Sv)*np.exp(d/s)]

        c2 = np.zeros((R2, 2))
        du = right_nodes[-1]-right_nodes[-2]; sl = local_vols_right[-1]
        c2[-1] = [np.exp(du/sl), -np.exp(-du/sl)]
        for j in range(R2-1,0,-1):
            P  = c2[j,0]+c2[j,1]
            Sv = (1/local_vols_right[j])*(-c2[j,0]+c2[j,1])
            d  = right_nodes[j]-right_nodes[j-1]; s = local_vols_right[j-1]
            c2[j-1] = [0.5*(P-s*Sv)*np.exp(d/s), 0.5*(P+s*Sv)*np.exp(-d/s)]

        vL  = c1[-1,0]+c1[-1,1]
        DkL = (1/local_vols_left[-1])*(-c1[-1,0]+c1[-1,1])
        vR  = c2[0,0]+c2[0,1]
        DkR = (1/local_vols_right[0])*(-c2[0,0]+c2[0,1])
        denom = DkL*vR - vL*DkR
        c1 *= vR/denom;  c2 *= vL/denom

        K_all = []; C_all = []

        # Left side
        for j in range(R1):
            nu_L = left_nodes[j]; nu_R = left_nodes[j+1]; sig = local_vols_left[j]
            dist_total = nu_R - nu_L
            cd1 = c1[j,0]*np.exp( dist_total/sig)
            cd2 = c1[j,1]*np.exp(-dist_total/sig)
            for k in np.linspace(nu_L, nu_R, n_pts):
                dist_k = k - nu_L
                V_k = cd1*np.exp(-dist_k/sig) + cd2*np.exp(dist_k/sig)
                C_k = V_k + max(S0 - k, 0)
                K_all.append(k); C_all.append(C_k)

        # Right side
        for j in range(R2):
            nu_L = right_nodes[j]; nu_R = right_nodes[j+1]; sig = local_vols_right[j]
            dist_total = nu_R - nu_L
            cd1 = c2[j,0]*np.exp(-dist_total/sig)
            cd2 = c2[j,1]*np.exp( dist_total/sig)
            for k in np.linspace(nu_L, nu_R, n_pts):
                dist_k = k - nu_L
                V_k = cd1*np.exp(-dist_k/sig) + cd2*np.exp(dist_k/sig)
                K_all.append(k); C_all.append(V_k)

        K_arr = np.array(K_all); C_arr = np.array(C_all)
        idx   = np.argsort(K_arr)
        return K_arr[idx], C_arr[idx]

    K_b, C_b = get_C_curve(theta_init)
    K_a, C_a = get_C_curve(best_theta)

    ax = axes[1]
    ax.plot(K_b, C_b, 'r-', lw=1.5, alpha=0.8, label='C(K) before')
    ax.plot(K_a, C_a, 'b-', lw=1.5, alpha=0.8, label='C(K) after')
    ax.axvline(S0, color='gray', ls='--', lw=1.5, label=f'S₀={S0:.0f}')
    ax.set_xlabel('Strike K', fontsize=12)
    ax.set_ylabel('C(K)', fontsize=12)
    ax.set_title('Call Price C(K) — Before vs After', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: C''(K) before vs after ──────────────────────────────────
    def get_Cpp_curve(theta, n_pts=60):
        partition_pts_left   = theta[:R1]
        local_vols_left_raw  = theta[R1:2*R1]
        partition_pts_right  = theta[2*R1:2*R1+R2]
        local_vols_right_raw = theta[2*R1+R2:]
        left_nodes  = np.append(partition_pts_left, S0)
        right_nodes = np.insert(partition_pts_right, 0, S0)
        local_vols_left  = softplus(local_vols_left_raw)  + 1e-6
        local_vols_right = softplus(local_vols_right_raw) + 1e-6

        c1 = np.zeros((R1, 2))
        d0 = left_nodes[1]-left_nodes[0]; s0v = local_vols_left[0]
        c1[0] = [-np.exp(-d0/s0v), np.exp(d0/s0v)]
        for j in range(R1-1):
            P  = c1[j,0]+c1[j,1]
            Sv = (1/local_vols_left[j])*(-c1[j,0]+c1[j,1])
            d  = left_nodes[j+2]-left_nodes[j+1]; s = local_vols_left[j+1]
            c1[j+1] = [0.5*(P-s*Sv)*np.exp(-d/s), 0.5*(P+s*Sv)*np.exp(d/s)]

        c2 = np.zeros((R2, 2))
        du = right_nodes[-1]-right_nodes[-2]; sl = local_vols_right[-1]
        c2[-1] = [np.exp(du/sl), -np.exp(-du/sl)]
        for j in range(R2-1,0,-1):
            P  = c2[j,0]+c2[j,1]
            Sv = (1/local_vols_right[j])*(-c2[j,0]+c2[j,1])
            d  = right_nodes[j]-right_nodes[j-1]; s = local_vols_right[j-1]
            c2[j-1] = [0.5*(P-s*Sv)*np.exp(d/s), 0.5*(P+s*Sv)*np.exp(-d/s)]

        vL  = c1[-1,0]+c1[-1,1]
        DkL = (1/local_vols_left[-1])*(-c1[-1,0]+c1[-1,1])
        vR  = c2[0,0]+c2[0,1]
        DkR = (1/local_vols_right[0])*(-c2[0,0]+c2[0,1])
        denom = DkL*vR - vL*DkR
        c1 *= vR/denom;  c2 *= vL/denom

        K_all = []; Cpp_all = []

        for j in range(R1):
            nu_L = left_nodes[j]; nu_R = left_nodes[j+1]; sig = local_vols_left[j]
            dist_total = nu_R - nu_L
            cd1 = c1[j,0]*np.exp( dist_total/sig)
            cd2 = c1[j,1]*np.exp(-dist_total/sig)
            for k in np.linspace(nu_L, nu_R, n_pts):
                dist_k = k - nu_L
                V_k    = cd1*np.exp(-dist_k/sig) + cd2*np.exp(dist_k/sig)
                K_all.append(k); Cpp_all.append(V_k / sig**2)

        for j in range(R2):
            nu_L = right_nodes[j]; nu_R = right_nodes[j+1]; sig = local_vols_right[j]
            dist_total = nu_R - nu_L
            cd1 = c2[j,0]*np.exp(-dist_total/sig)
            cd2 = c2[j,1]*np.exp( dist_total/sig)
            for k in np.linspace(nu_L, nu_R, n_pts):
                dist_k = k - nu_L
                V_k    = cd1*np.exp(-dist_k/sig) + cd2*np.exp(dist_k/sig)
                K_all.append(k); Cpp_all.append(V_k / sig**2)

        K_arr   = np.array(K_all); Cpp_arr = np.array(Cpp_all)
        idx     = np.argsort(K_arr)
        return K_arr[idx], Cpp_arr[idx]

    K_b2, Cpp_b = get_Cpp_curve(theta_init)
    K_a2, Cpp_a = get_Cpp_curve(best_theta)

    # Clip extreme outliers for clean display
    clip_hi = np.percentile(np.concatenate([Cpp_b, Cpp_a]), 99)
    clip_lo = np.percentile(np.concatenate([Cpp_b, Cpp_a]), 1)

    ax = axes[2]
    ax.plot(K_b2, np.clip(Cpp_b, clip_lo, clip_hi),
            'r-', lw=1.5, alpha=0.8, label='C\'\'(K) before')
    ax.plot(K_a2, np.clip(Cpp_a, clip_lo, clip_hi),
            'b-', lw=1.5, alpha=0.8, label='C\'\'(K) after')
    ax.axvline(S0, color='gray', ls='--', lw=1.5, label=f'S₀={S0:.0f}')
    ax.axhline(0,  color='black', lw=0.5)
    ax.set_xlabel('Strike K', fontsize=12)
    ax.set_ylabel("C''(K)", fontsize=12)
    ax.set_title(
        f"C''(K) — Before vs After\n"
        f"Max |jump|: {np.max(np.abs(jumps_before)):.5f} → "
        f"{np.max(np.abs(jumps_after)):.5f}  |  "
        f"Mean |jump|: {np.mean(np.abs(jumps_before)):.5f} → "
        f"{np.mean(np.abs(jumps_after)):.5f}",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Summary:")
    print(f"    J before    = {J_initial:.8f}")
    print(f"    J after     = {best_J:.8f}")
    print(f"    Improvement = {improvement:.4f}%")
    print(f"    Max |jump|  : {np.max(np.abs(jumps_before)):.6f}  ->  "
          f"{np.max(np.abs(jumps_after)):.6f}")
    print(f"    Mean|jump|  : {np.mean(np.abs(jumps_before)):.6f}  ->  "
          f"{np.mean(np.abs(jumps_after)):.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import numpy as np

    R1, R2 = 70, 34
    S0     = 1271.87
    Kbar   = 2000.0

    initial_sigs1 = np.array([610.43, 164.05, 21.41, 41.04, 32.59, 48.78, 42.22, 55.82, 51.21, 60.69, 59.95, 58.90, 61.25, 58.78, 61.94, 57.79, 62.28, 54.62, 60.39, 50.96, 57.50, 46.58, 54.08, 40.60, 46.80, 40.23, 44.18, 40.07, 42.59, 41.97, 43.55, 43.64, 46.68, 40.67, 45.75, 37.59, 43.37, 34.28, 38.64, 34.12, 38.34, 33.12, 37.54, 30.90, 34.63, 31.93, 34.99, 31.44, 34.80, 31.01, 33.67, 32.96, 33.88, 37.49, 36.95, 42.71, 46.47, 33.85, 43.68, 24.47, 29.27, 33.25, 32.86, 36.88, 38.57, 34.89, 32.34, 54.86, 42.54, 31.82])
    initial_sigs2 = np.array([24.03, 38.31, 64.22, 24.95, 31.48, 31.17, 29.04, 30.23, 28.25, 27.35, 26.08, 24.72, 23.36, 22.45, 22.00, 19.57, 19.32, 17.76, 19.39, 39.62, 26.39, 49.90, 28.53, 31.87, 41.72, 83.05, 40.34, 37.01, 39.35, 30.27, 73.53, 57.33, 169.44, 313.04])
    initial_nus1  = np.array([0, 985.90, 1103.9, 1105.63, 1108.9, 1110.93, 1113.9, 1116.08, 1118.9, 1121.22, 1123.9, 1126.45, 1128.9, 1131.48, 1133.9, 1136.52, 1138.9, 1141.60, 1143.9, 1146.64, 1148.9, 1151.70, 1153.9, 1156.79, 1158.9, 1161.63, 1163.9, 1166.56, 1168.9, 1171.46, 1173.9, 1176.44, 1178.9, 1181.62, 1183.9, 1186.69, 1188.9, 1191.74, 1193.9, 1196.55, 1198.8, 1201.53, 1203.8, 1206.59, 1208.8, 1211.46, 1213.8, 1216.49, 1218.8, 1221.50, 1223.8, 1226.39, 1228.8, 1231.23, 1233.8, 1236.18, 1238.8, 1241.75, 1243.8, 1247.06, 1248.8, 1251.20, 1253.8, 1256.22, 1258.8, 1261.48, 1263.8, 1265.71, 1268.8, 1270.58])
    initial_nus2  = np.array([1272.60, 1273.8, 1277.34, 1278.8, 1281.24, 1283.8, 1286.17, 1288.8, 1291.26, 1293.8, 1296.28, 1298.8, 1301.21, 1303.7, 1306.26, 1308.7, 1311.23, 1313.7, 1316.81, 1323.7, 1327.01, 1333.7, 1336.02, 1338.7, 1343.47, 1353.7, 1356.28, 1358.7, 1361.50, 1363.7, 1377.34, 1388.7, 1503.72, 2000.0])

    sigs1_raw = inv_softplus(initial_sigs1)
    sigs2_raw = inv_softplus(initial_sigs2)

    # ── Baseline ──────────────────────────────────────────────────────────
    theta_init = np.concatenate([initial_nus1, sigs1_raw,
                                  initial_nus2, sigs2_raw])
    J_baseline, _, _ = calculate_J(theta_init, R1, R2, S0)
    print(f"\nBaseline J = {J_baseline:.8f}")
    print("="*55)

    # ── Step 1 + 2: Sigma schedule with generation schedule ───────────────
    sigma_schedule       = [0.3,  0.1,  0.05, 0.01, 0.005, 0.001, 0.0005]
    generations_schedule = [300,  300,  300,  500,  500,   1000,  1000  ]
    current_theta        = theta_init.copy()

    for stage, (sigma, n_gen) in enumerate(zip(sigma_schedule, generations_schedule)):
        print(f"\nStage {stage+1}/{len(sigma_schedule)} — sigma={sigma}, max_gen={n_gen}")
        print("-"*55)

        result = cmaes_optimize(
            theta_init        = current_theta,
            R1                = R1,
            R2                = R2,
            S0                = S0,
            Kbar              = Kbar,
            initial_step_size = sigma,
            max_generations   = n_gen,
            convergence_tol   = 1e-10,
            random_seed       = 42,
            verbose           = True,
            save_plot         = True,
            plot_path         = f'/Users/lucasxia/Desktop/cmaes_stage{stage+1}.png'
        )

        current_theta = result['best_theta']
        improvement_vs_baseline = 100 * (J_baseline - result['best_J']) / J_baseline
        print(f"\n  Stage {stage+1} result:")
        print(f"    J           = {result['best_J']:.8f}")
        print(f"    vs baseline = {improvement_vs_baseline:.2f}% improvement")

    # ── Step 3: Multi-start refinement ────────────────────────────────────
    print("\n" + "="*55)
    print("STEP 3: Multi-start refinement")
    print("="*55)

    best_J_so_far     = calculate_J(current_theta, R1, R2, S0)[0]
    best_theta_so_far = current_theta.copy()

    for trial in range(20):
        np.random.seed(trial * 7)
        noise           = np.random.randn(len(current_theta)) * 0.0001
        theta_perturbed = current_theta + noise

        result_trial = cmaes_optimize(
            theta_init        = theta_perturbed,
            R1                = R1,
            R2                = R2,
            S0                = S0,
            Kbar              = Kbar,
            initial_step_size = 0.0001,
            max_generations   = 1000,
            convergence_tol   = 1e-10,
            random_seed       = trial,
            verbose           = False,
            save_plot         = False,
        )

        improvement_trial = 100 * (J_baseline - result_trial['best_J']) / J_baseline
        print(f"  Trial {trial+1:>2}/20 — J = {result_trial['best_J']:.8f} "
              f"({improvement_trial:.4f}%)")

        if result_trial['best_J'] < best_J_so_far:
            best_J_so_far     = result_trial['best_J']
            best_theta_so_far = result_trial['best_theta'].copy()
            print(f"           ↑ new best!")

    current_theta = best_theta_so_far

    # ── Final plot and summary ─────────────────────────────────────────────
    J_final, lam1, lam2 = calculate_J(current_theta, R1, R2, S0)

    cmaes_optimize(
        theta_init        = current_theta,
        R1                = R1,
        R2                = R2,
        S0                = S0,
        Kbar              = Kbar,
        initial_step_size = 0.00001,
        max_generations   = 1,
        convergence_tol   = 1e-10,
        random_seed       = 42,
        verbose           = False,
        save_plot         = True,
        plot_path         = '/Users/lucasxia/Desktop/QF Research/optimization results/cmaes_final.png'
    )

    print(f"\n{'='*55}")
    print(f"FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  Baseline J        = {J_baseline:.8f}")
    print(f"  Final J           = {J_final:.8f}")
    print(f"  Total improvement = {100*(J_baseline-J_final)/J_baseline:.2f}%")
    print(f"  lambda1           = {lam1:.6f}")
    print(f"  lambda2           = {lam2:.6f}")
    print(f"{'='*55}")