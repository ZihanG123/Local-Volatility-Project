"""
cmaes_optimizer_v2.py
=====================
CMA-ES optimizer updated to match teammates' calculate_J exactly.

Key differences from v1:
- J function reimplemented in numpy matching teammates' PyTorch version
- lam1, lam2 computed analytically (not from V0/Vp0 boundary conditions)
- sigs use softplus transform (not log) for positivity
- nus1[0]=0 and nus2[-1]=Kbar are fixed boundaries
- CMA-ES operates on log-gaps for nus ordering + raw sigs (unconstrained)

Teammates' theta packing (matches exactly):
    theta = [nus1 (R1), sigs1_raw (R1), nus2 (R2), sigs2_raw (R2)]

where:
    - nus1: R1 partition points, nus1[0]=0 (fixed L), rest increasing < S0
    - sigs1_raw: R1 raw sigmas (softplus applied internally for positivity)
    - nus2: R2 partition points, nus2[-1]=Kbar (fixed), rest increasing > S0
    - sigs2_raw: R2 raw sigmas (softplus applied internally for positivity)

Usage:
------
    from cmaes_optimizer_v2 import cmaes_optimize

    result = cmaes_optimize(
        theta0  = theta0,     # teammates' initial theta vector
        R1      = 70,
        R2      = 34,
        S0      = 1271.87,
        Kbar    = 2000.0,
        sigma0  = 0.5,
        maxiter = 500,
        seed    = 42
    )
    best_theta = result['best_theta']
    best_J     = result['best_J']
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# Softplus (matches torch.nn.functional.softplus)
# ══════════════════════════════════════════════════════════════════════════════

def softplus(x):
    """log(1 + exp(x)), numerically stable."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))

def inv_softplus(x):
    """Inverse softplus: log(exp(x) - 1), for x > 0."""
    return np.where(x > 20, x, np.log(np.expm1(np.clip(x, 1e-8, 500))))


# ══════════════════════════════════════════════════════════════════════════════
# J function — numpy reimplementation of teammates' calculate_J
# ══════════════════════════════════════════════════════════════════════════════

def calculate_J(R1, R2, S0, theta):
    """
    Compute J(theta) matching teammates' PyTorch calculate_J exactly.

    Parameters
    ----------
    R1    : int   — number of left partition points (including L=0)
    R2    : int   — number of right partition points (including Kbar)
    S0    : float — spot price
    theta : ndarray (2*R1 + 2*R2,)
            [nus1(R1), sigs1_raw(R1), nus2(R2), sigs2_raw(R2)]

    Returns
    -------
    j_val : float — smoothness penalty J(theta)
    lam1  : float — left wing scaling lambda
    lam2  : float — right wing scaling lambda
    """
    # ── 1. Unpack theta ───────────────────────────────────────────────────
    idx = 0
    nus1      = theta[idx:idx+R1];  idx += R1
    sigs1_raw = theta[idx:idx+R1];  idx += R1
    nus2      = theta[idx:idx+R2];  idx += R2
    sigs2_raw = theta[idx:idx+R2]

    # Append / prepend S0 (matches teammates' code exactly)
    nus1 = np.append(nus1, S0)        # length R1+1
    nus2 = np.insert(nus2, 0, S0)     # length R2+1

    # Enforce strict positivity via softplus (matches torch.nn.functional.softplus)
    sigs1 = softplus(sigs1_raw) + 1e-6  # length R1
    sigs2 = softplus(sigs2_raw) + 1e-6  # length R2

    # ── 2. Left wing unit coefficients (lam1 = 1) ────────────────────────
    # Initial: distance from nus1[0] to nus1[1]
    c_v1 = np.zeros((R1, 2))
    dist0 = nus1[1] - nus1[0]
    c_v1[0, 0] = -np.exp(-dist0 / sigs1[0])   # c1 (distance-scaled)
    c_v1[0, 1] =  np.exp( dist0 / sigs1[0])   # c2 (distance-scaled)

    # Recursive propagation left -> right
    for j in range(R1 - 1):
        P    = c_v1[j, 0] + c_v1[j, 1]
        S    = (1.0 / sigs1[j]) * (-c_v1[j, 0] + c_v1[j, 1])
        dist = nus1[j+2] - nus1[j+1]
        c_v1[j+1, 0] = 0.5 * (P - sigs1[j+1] * S) * np.exp(-dist / sigs1[j+1])
        c_v1[j+1, 1] = 0.5 * (P + sigs1[j+1] * S) * np.exp( dist / sigs1[j+1])

    # ── 3. Right wing unit coefficients (lam2 = 1) ───────────────────────
    # Initial: distance from nus2[-2] to nus2[-1]
    c_v2 = np.zeros((R2, 2))
    dist_u = nus2[-1] - nus2[-2]
    c_v2[-1, 0] =  np.exp( dist_u / sigs2[-1])   # c1
    c_v2[-1, 1] = -np.exp(-dist_u / sigs2[-1])   # c2

    # Recursive propagation right -> left
    for j in range(R2 - 1, 0, -1):
        P    = c_v2[j, 0] + c_v2[j, 1]
        S    = (1.0 / sigs2[j]) * (-c_v2[j, 0] + c_v2[j, 1])
        dist = nus2[j] - nus2[j-1]
        c_v2[j-1, 0] = 0.5 * (P - sigs2[j-1] * S) * np.exp( dist / sigs2[j-1])
        c_v2[j-1, 1] = 0.5 * (P + sigs2[j-1] * S) * np.exp(-dist / sigs2[j-1])

    # ── 4. Solve for lam1, lam2 at S0 ────────────────────────────────────
    # Unit left wing value and derivative at S0 (= nus1[-1])
    v1    = c_v1[-1, 0] + c_v1[-1, 1]
    Dk_v1 = (1.0 / sigs1[-1]) * (-c_v1[-1, 0] + c_v1[-1, 1])

    # Unit right wing value and derivative at S0 (= nus2[0])
    v2    = c_v2[0, 0] + c_v2[0, 1]
    Dk_v2 = (1.0 / sigs2[0]) * (-c_v2[0, 0] + c_v2[0, 1])

    # Solve: lam1*V1 = lam2*V2  and  lam1*DkV1 - lam2*DkV2 = 1
    denom = Dk_v1 * v2 - v1 * Dk_v2
    lam1  = v2 / denom
    lam2  = v1 / denom

    # ── 5. Scale coefficients by lam1, lam2 ──────────────────────────────
    c_v1 *= lam1   # shape (R1, 2)
    c_v2 *= lam2   # shape (R2, 2)

    # ── 6. Compute jumps in C'' at each partition point ───────────────────
    jumps = []

    # Left wing internal jumps: at nus1[j+1] for j = 0,...,R1-2
    for j in range(R1 - 1):
        # C''(nu_j+1, left)  = (1/sig_j^2) * V(nu_j+1 from left)
        #                     = (1/sig_j^2) * (c1_j + c2_j)  [at right endpoint of interval j]
        v_left  = (1.0 / sigs1[j]**2)   * (c_v1[j, 0] + c_v1[j, 1])
        # C''(nu_j+1, right) = (1/sig_{j+1}^2) * V(nu_j+1 from right)
        dist    = nus1[j+2] - nus1[j+1]
        v_right = (1.0 / sigs1[j+1]**2) * (c_v1[j+1, 0] * np.exp( dist / sigs1[j+1]) +
                                            c_v1[j+1, 1] * np.exp(-dist / sigs1[j+1]))
        jumps.append(v_right - v_left)

    # Junction jump at S0
    v1_S0 = (1.0 / sigs1[-1]**2) * (c_v1[-1, 0] + c_v1[-1, 1])
    v2_S0 = (1.0 / sigs2[0]**2)  * (c_v2[0, 0]  + c_v2[0, 1])
    jumps.append(v2_S0 - v1_S0)

    # Right wing internal jumps: at nus2[j] for j = 0,...,R2-2
    for j in range(R2 - 1):
        # C''(nu_{j+1}, right) at left endpoint of interval j+1
        v_right = (1.0 / sigs2[j+1]**2) * (c_v2[j+1, 0] + c_v2[j+1, 1])
        # C''(nu_{j+1}, left)  at right endpoint of interval j
        dist    = nus2[j+1] - nus2[j]
        v_left  = (1.0 / sigs2[j]**2)   * (c_v2[j, 0] * np.exp(-dist / sigs2[j]) +
                                            c_v2[j, 1] * np.exp( dist / sigs2[j]))
        jumps.append(v_right - v_left)

    j_val = float(np.sum(np.array(jumps)**2))
    return j_val, float(lam1), float(lam2)


# ══════════════════════════════════════════════════════════════════════════════
# Unconstrained <-> constrained transforms for CMA-ES
# ══════════════════════════════════════════════════════════════════════════════

def to_unconstrained(theta, R1, R2, S0, Kbar):
    """
    Transform constrained theta into unconstrained z.

    Fixed parameters (not optimized):
        nus1[0]  = 0    (left boundary L)
        nus2[-1] = Kbar (right boundary)

    Transform:
        nus1[1:]   -> log-gaps from nus1[0]=0      [R1-1 values]
        sigs1_raw  -> unchanged (already unconstrained) [R1 values]
        nus2[:-1]  -> log-gaps from S0             [R2-1 values]
        sigs2_raw  -> unchanged (already unconstrained) [R2 values]
    """
    nus1      = theta[:R1]
    sigs1_raw = theta[R1:2*R1]
    nus2      = theta[2*R1:2*R1+R2]
    sigs2_raw = theta[2*R1+R2:]

    # Left: gaps between consecutive nus1 (nus1[0]=0 is fixed)
    gaps1  = np.diff(nus1)                   # R1-1 gaps, all should be > 0
    lgaps1 = np.log(np.maximum(gaps1, 1e-10))

    # Right: gaps between consecutive nus2[:-1] starting from S0
    #        nus2[-1]=Kbar is fixed, so we have R2-1 free interior points
    nus2_free = nus2[:-1]                    # R2-1 free points
    anchors   = np.concatenate([[S0], nus2_free[:-1]])
    gaps2     = nus2_free - anchors          # R2-1 gaps, all should be > 0
    lgaps2    = np.log(np.maximum(gaps2, 1e-10))

    return np.concatenate([lgaps1, sigs1_raw, lgaps2, sigs2_raw])


def to_constrained(z, R1, R2, S0, Kbar):
    """
    Transform unconstrained z back into constrained theta.
    Inverse of to_unconstrained.
    """
    # Unpack z
    lgaps1    = z[:R1-1]
    sigs1_raw = z[R1-1:2*R1-1]
    lgaps2    = z[2*R1-1:2*R1+R2-2]
    sigs2_raw = z[2*R1+R2-2:]

    # Reconstruct nus1: fix nus1[0]=0, build from gaps
    gaps1 = np.exp(lgaps1)
    nus1  = np.concatenate([[0.0], np.cumsum(gaps1)])
    nus1  = np.minimum(nus1, S0 - 1e-6)     # clip to stay below S0

    # Reconstruct nus2: build from S0, fix nus2[-1]=Kbar
    gaps2     = np.exp(lgaps2)
    nus2_free = S0 + np.cumsum(gaps2)        # R2-1 free points
    nus2_free = np.minimum(nus2_free, Kbar - 1e-6)
    nus2      = np.append(nus2_free, Kbar)  # add fixed Kbar at end

    return np.concatenate([nus1, sigs1_raw, nus2, sigs2_raw])


# ══════════════════════════════════════════════════════════════════════════════
# CMA-ES core (same algorithm as v1)
# ══════════════════════════════════════════════════════════════════════════════

def _cmaes_core(func_z, z0, sigma0=0.5, maxiter=500, tol=1e-10,
                seed=42, verbose=True):
    """
    Core CMA-ES in unconstrained space z.
    (mu/w, lambda)-CMA-ES with CSA and rank-mu + rank-one covariance update.
    Reference: Hansen 2016, arXiv:1604.00772
    """
    np.random.seed(seed)
    n = len(z0)

    # Hyperparameters
    lam    = 4 + int(np.floor(3 * np.log(n)))
    mu_cma = lam // 2
    w      = np.log(mu_cma + 0.5) - np.log(np.arange(1, mu_cma + 1))
    w     /= w.sum()
    mueff  = 1.0 / np.sum(w**2)

    cs    = (mueff + 2) / (n + mueff + 5)
    ds    = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1))-1) + cs
    chiN  = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    cc    = (4 + mueff/n) / (n + 4 + 2*mueff/n)
    c1    = 2 / ((n + 1.3)**2 + mueff)
    cmu   = min(1 - c1, 2*(mueff-2+1/mueff) / ((n+2)**2 + mueff))
    hth   = (1.4 + 2/(n+1)) * chiN

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CMA-ES Optimizer  (v2 — matched to teammates' J)")
        print(f"{'='*65}")
        print(f"  Dimension    n    = {n}")
        print(f"  Population   λ    = {lam}")
        print(f"  Parents      μ    = {mu_cma}")
        print(f"  mueff             = {mueff:.2f}")
        print(f"  Initial step σ₀   = {sigma0}")
        print(f"  Max generations   = {maxiter}")
        print(f"  Convergence tol   = {tol:.1e}")
        print(f"{'='*65}")
        print(f"  {'Gen':>5} | {'J_best':>14} | {'J_mean':>14} | {'σ':>10}")
        print(f"  {'-'*50}")

    # State
    m         = z0.copy()
    sigma     = sigma0
    pc        = np.zeros(n)
    ps        = np.zeros(n)
    B         = np.eye(n)
    D         = np.ones(n)
    C         = np.eye(n)
    invsqrtC  = np.eye(n)
    eigeneval = 0
    best_J    = np.inf
    best_z    = m.copy()
    history   = {'J_best': [], 'J_mean': [], 'sigma': [], 'gen': []}

    for gen in range(maxiter):
        # Sample
        arz = np.random.randn(lam, n)
        arx = np.array([m + sigma * (B @ (D * arz[k])) for k in range(lam)])

        # Evaluate
        fitvals = np.array([func_z(arx[k]) for k in range(lam)])
        idx_s   = np.argsort(fitvals)

        # Track best
        if fitvals[idx_s[0]] < best_J:
            best_J = fitvals[idx_s[0]]
            best_z = arx[idx_s[0]].copy()

        # Update mean
        m_old = m.copy()
        m     = np.sum(w[:, None] * arx[idx_s[:mu_cma]], axis=0)

        # Update ps (step-size path)
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (m - m_old) / sigma

        # h_sigma
        h_sigma = (np.linalg.norm(ps) / np.sqrt(1-(1-cs)**(2*(gen+1))) < hth)

        # Update pc (covariance path)
        pc = ((1-cc)*pc
              + h_sigma * np.sqrt(cc*(2-cc)*mueff) * (m - m_old) / sigma)

        # Update C
        artmp = (1/sigma) * (arx[idx_s[:mu_cma]] - m_old)
        C = ((1-c1-cmu)*C
             + c1*(np.outer(pc, pc) + (1-h_sigma)*cc*(2-cc)*C)
             + cmu * np.sum(w[:, None, None] *
                            (artmp[:, :, None] * artmp[:, None, :]), axis=0))

        # Update sigma
        sigma = sigma * np.exp((cs/ds) * (np.linalg.norm(ps)/chiN - 1))

        # Eigendecomposition (periodic)
        if gen - eigeneval > n / (10 * lam * (c1 + cmu)):
            eigeneval = gen
            C         = np.triu(C) + np.triu(C, 1).T
            D2, B     = np.linalg.eigh(C)
            D2        = np.maximum(D2, 1e-20)
            D         = np.sqrt(D2)
            invsqrtC  = B @ np.diag(1/D) @ B.T

        # History
        history['J_best'].append(best_J)
        history['J_mean'].append(float(np.mean(fitvals)))
        history['sigma'].append(float(sigma))
        history['gen'].append(gen)

        if verbose and gen % 20 == 0:
            print(f"  {gen:>5} | {best_J:>14.8f} | "
                  f"{np.mean(fitvals):>14.6f} | {sigma:>10.6f}")

        if sigma < tol:
            if verbose:
                print(f"\n  Converged gen={gen}: σ={sigma:.2e} < tol={tol:.2e}")
            break
        if best_J < 1e-12:
            if verbose:
                print(f"\n  Converged gen={gen}: J={best_J:.2e} < 1e-12")
            break

    if verbose:
        print(f"  {'-'*50}")

    return best_z, best_J, history


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def cmaes_optimize(theta0, R1, R2, S0, Kbar,
                   sigma0=0.5, maxiter=500, tol=1e-10,
                   seed=42, verbose=True, plot=True,
                   plot_path='/mnt/user-data/outputs/cmaes_v2_result.png'):
    """
    Minimize J(theta) using CMA-ES, matched to teammates' calculate_J.

    Parameters
    ----------
    theta0  : ndarray (2*R1 + 2*R2,)
              Initial theta: [nus1(R1), sigs1_raw(R1), nus2(R2), sigs2_raw(R2)]
              sigs_raw should be inverse-softplus of actual sigmas.

    R1      : int   — number of left partition points (including L=0)
    R2      : int   — number of right partition points (including Kbar)
    S0      : float — spot price
    Kbar    : float — right boundary (fixed, = nus2[-1])
    sigma0  : float — initial CMA-ES step size
    maxiter : int   — max generations
    tol     : float — convergence tolerance
    seed    : int   — random seed
    verbose : bool  — print progress
    plot    : bool  — save convergence plot
    plot_path: str  — path for plot

    Returns
    -------
    dict with keys:
        'best_theta'  : ndarray — optimized theta (same format as input)
        'best_J'      : float   — optimized J value
        'J0'          : float   — baseline J value
        'improvement' : float   — % improvement
        'history'     : dict    — convergence history
        'lam1_opt'    : float   — optimized lambda1
        'lam2_opt'    : float   — optimized lambda2
    """
    # ── Baseline ──────────────────────────────────────────────────────────
    J0, lam1_0, lam2_0 = calculate_J(R1, R2, S0, theta0)
    if verbose:
        print(f"Baseline J(theta0) = {J0:.8f}")
        print(f"Baseline lam1      = {lam1_0:.6f}")
        print(f"Baseline lam2      = {lam2_0:.6f}")

    # ── Transform to unconstrained space ──────────────────────────────────
    z0 = to_unconstrained(theta0, R1, R2, S0, Kbar)
    if verbose:
        print(f"Unconstrained dim  = {len(z0)}  "
              f"(fixed: nus1[0]=0, nus2[-1]={Kbar})")

    # ── Wrap J to work in unconstrained space ─────────────────────────────
    def func_z(z):
        theta = to_constrained(z, R1, R2, S0, Kbar)
        try:
            j_val, _, _ = calculate_J(R1, R2, S0, theta)
            return j_val if np.isfinite(j_val) else 1e12
        except Exception:
            return 1e12

    # ── Run CMA-ES ────────────────────────────────────────────────────────
    best_z, best_J, history = _cmaes_core(
        func_z  = func_z,
        z0      = z0,
        sigma0  = sigma0,
        maxiter = maxiter,
        tol     = tol,
        seed    = seed,
        verbose = verbose
    )

    # ── Recover best theta and lambdas ────────────────────────────────────
    best_theta            = to_constrained(best_z, R1, R2, S0, Kbar)
    _, lam1_opt, lam2_opt = calculate_J(R1, R2, S0, best_theta)
    improvement           = 100.0 * (J0 - best_J) / J0 if J0 > 0 else 0.0

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Optimization Complete")
        print(f"{'='*65}")
        print(f"  J(theta0)    = {J0:.8f}")
        print(f"  J(theta_opt) = {best_J:.8f}")
        print(f"  Improvement  = {improvement:.4f}%")
        print(f"  lam1_opt     = {lam1_opt:.6f}")
        print(f"  lam2_opt     = {lam2_opt:.6f}")
        print(f"{'='*65}\n")

    if plot:
        _plot_results(history, J0, best_J, improvement,
                      theta0, best_theta, R1, R2, S0, Kbar, plot_path)
        if verbose:
            print(f"Plot saved to: {plot_path}")

    return {
        'best_theta':  best_theta,
        'best_J':      best_J,
        'J0':          J0,
        'improvement': improvement,
        'history':     history,
        'lam1_opt':    lam1_opt,
        'lam2_opt':    lam2_opt
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(history, J0, best_J, improvement,
                  theta0, best_theta, R1, R2, S0, Kbar, save_path):
    """Plot convergence + jump comparison before/after."""

    # Compute jumps before and after
    def get_jumps(theta):
        nus1      = theta[:R1]
        sigs1_raw = theta[R1:2*R1]
        nus2      = theta[2*R1:2*R1+R2]
        sigs2_raw = theta[2*R1+R2:]
        nus1_ext  = np.append(nus1, S0)
        nus2_ext  = np.insert(nus2, 0, S0)
        sigs1     = softplus(sigs1_raw) + 1e-6
        sigs2     = softplus(sigs2_raw) + 1e-6

        # Recompute c coefficients (abbreviated)
        c_v1 = np.zeros((R1, 2))
        dist0 = nus1_ext[1] - nus1_ext[0]
        c_v1[0] = [-np.exp(-dist0/sigs1[0]), np.exp(dist0/sigs1[0])]
        for j in range(R1-1):
            P = c_v1[j,0]+c_v1[j,1]; S = (1/sigs1[j])*(-c_v1[j,0]+c_v1[j,1])
            d = nus1_ext[j+2]-nus1_ext[j+1]
            c_v1[j+1] = [0.5*(P-sigs1[j+1]*S)*np.exp(-d/sigs1[j+1]),
                         0.5*(P+sigs1[j+1]*S)*np.exp( d/sigs1[j+1])]
        c_v2 = np.zeros((R2, 2))
        du = nus2_ext[-1]-nus2_ext[-2]
        c_v2[-1] = [np.exp(du/sigs2[-1]), -np.exp(-du/sigs2[-1])]
        for j in range(R2-1, 0, -1):
            P = c_v2[j,0]+c_v2[j,1]; S = (1/sigs2[j])*(-c_v2[j,0]+c_v2[j,1])
            d = nus2_ext[j]-nus2_ext[j-1]
            c_v2[j-1] = [0.5*(P-sigs2[j-1]*S)*np.exp( d/sigs2[j-1]),
                         0.5*(P+sigs2[j-1]*S)*np.exp(-d/sigs2[j-1])]
        v1 = c_v1[-1,0]+c_v1[-1,1]; Dk1 = (1/sigs1[-1])*(-c_v1[-1,0]+c_v1[-1,1])
        v2 = c_v2[0,0]+c_v2[0,1];   Dk2 = (1/sigs2[0])*(-c_v2[0,0]+c_v2[0,1])
        denom = Dk1*v2-v1*Dk2; lam1=v2/denom; lam2=v1/denom
        c_v1 *= lam1; c_v2 *= lam2

        jumps = []
        for j in range(R1-1):
            d = nus1_ext[j+2]-nus1_ext[j+1]
            jumps.append((1/sigs1[j+1]**2)*(c_v1[j+1,0]*np.exp(d/sigs1[j+1])+
                                             c_v1[j+1,1]*np.exp(-d/sigs1[j+1]))
                        -(1/sigs1[j]**2)*(c_v1[j,0]+c_v1[j,1]))
        jumps.append((1/sigs2[0]**2)*(c_v2[0,0]+c_v2[0,1])
                    -(1/sigs1[-1]**2)*(c_v1[-1,0]+c_v1[-1,1]))
        for j in range(R2-1):
            d = nus2_ext[j+1]-nus2_ext[j]
            jumps.append((1/sigs2[j+1]**2)*(c_v2[j+1,0]+c_v2[j+1,1])
                        -(1/sigs2[j]**2)*(c_v2[j,0]*np.exp(-d/sigs2[j])+
                                          c_v2[j,1]*np.exp(d/sigs2[j])))
        return np.array(jumps)

    jumps_before = get_jumps(theta0)
    jumps_after  = get_jumps(best_theta)

    gen = history['gen']
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Panel 1: Convergence
    ax = axes[0]
    ax.semilogy(gen, history['J_best'], 'b-',  lw=2,   label='Best J(θ)')
    ax.semilogy(gen, history['J_mean'], 'r--', lw=1.5, label='Mean J(θ)')
    ax.axhline(J0, color='gray', ls=':', lw=2,
               label=f'Baseline J₀ = {J0:.6f}')
    ax.axhline(best_J, color='green', ls='--', lw=1.5,
               label=f'Optimized J* = {best_J:.2e}')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('J(θ) [log scale]', fontsize=12)
    ax.set_title(f'CMA-ES Convergence  —  '
                 f'J: {J0:.6f} → {best_J:.2e}  ({improvement:.2f}% improvement)',
                 fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel 2: Step size
    ax = axes[1]
    ax.semilogy(gen, history['sigma'], 'g-', lw=2)
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Step size σ [log scale]', fontsize=12)
    ax.set_title('CMA-ES Step Size Adaptation', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Panel 3: Jumps before vs after
    ax = axes[2]
    idx_all = np.arange(len(jumps_before))
    ax.bar(idx_all, np.abs(jumps_before), color='lightcoral',
           label='Before', alpha=0.8)
    ax.bar(idx_all, np.abs(jumps_after),  color='steelblue',
           label='After',  alpha=0.8)
    ax.axvline(R1-1, color='gray', ls='--', lw=1.5, label='S₀ junction')
    ax.set_xlabel('Jump index j', fontsize=12)
    ax.set_ylabel("|ΔC''(νⱼ)|", fontsize=12)
    ax.set_title(f"Jumps in C'' Before vs After CMA-ES\n"
                 f"Total jumps: {len(jumps_before)}  "
                 f"(left: {R1-1}, S0: 1, right: {R2-1})",
                 fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Run with teammates' exact initial theta
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import numpy as np

    R1, R2 = 70, 34
    S0     = 1271.87
    Kbar   = 2000.0

    # ── Teammates' exact initial values ──────────────────────────────────
    initial_sigs1 = np.array([610.4320996213446, 164.04874695079099, 21.410699478563217, 41.036799095433764, 32.59140099151007, 48.77559067919313, 42.2176372999541, 55.823566769595985, 51.211167181688566, 60.691607330511374, 59.950820637770775, 58.898463641159154, 61.253284806200256, 58.77606559998955, 61.9431978849179, 57.790240203069374, 62.2779153688338, 54.621537161593736, 60.38808834808558, 50.96497274766389, 57.49812984113923, 46.576659253726845, 54.080819207223875, 40.60311710082493, 46.79656147874567, 40.230053767550686, 44.17878499997829, 40.0738197788728, 42.59425882166232, 41.97429249407287, 43.554019179773334, 43.64066412990515, 46.68459597597928, 40.66643248490499, 45.74533322390213, 37.58570160756966, 43.36876861252296, 34.281419881606425, 38.63837902433531, 34.11699307804324, 38.33534468465065, 33.11788846662374, 37.54304297301797, 30.8995436545175, 34.629132576982585, 31.932382438352974, 34.99302768634262, 31.44000481459117, 34.80339761657143, 31.011124152999194, 33.67039062291056, 32.963580616942636, 33.88070107871873, 37.490458108901564, 36.9523875799828, 42.71457754035583, 46.473166992677335, 33.8545330176666, 43.67858033724246, 24.46553767335429, 29.27267842043444, 33.253868886417095, 32.860201483329625, 36.87919843993161, 38.56770262319378, 34.89031672560033, 32.336062185014455, 54.86203086146442, 42.54094598147429, 31.82466458162184])
    initial_sigs2 = np.array([24.02631471182734, 38.31350500726014, 64.22160217291687, 24.945880842361603, 31.475272346451945, 31.167940240782592, 29.04149431455202, 30.22926871363152, 28.245473181139502, 27.3473961133823, 26.078699124964757, 24.718907886876814, 23.35903470115683, 22.454368334929534, 22.002320937211316, 19.569326547161026, 19.315256975829875, 17.756271425921597, 19.392408982133873, 39.61543966355491, 26.387286176796366, 49.89596975113815, 28.527769674700764, 31.874820986884167, 41.72362039958813, 83.04797290752168, 40.34143759307382, 37.0089135084597, 39.353339426345215, 30.273572809409785, 73.52830072545393, 57.33057445441953, 169.4372302588361, 313.0441713343745])
    initial_nus1  = np.array([0, 985.8952130330157, 1103.9, 1105.6299717509714, 1108.9, 1110.9261114114317, 1113.9, 1116.0803462760916, 1118.9, 1121.217532062242, 1123.9, 1126.4524566940233, 1128.9, 1131.482542383951, 1133.9, 1136.5181574323626, 1138.9, 1141.5955696582002, 1143.9, 1146.6439410206626, 1148.9, 1151.6955052869373, 1153.9, 1156.7899943184998, 1158.9, 1161.6254606809712, 1163.9, 1166.5610218554195, 1168.9, 1171.459730260086, 1173.9, 1176.4403363141735, 1178.9, 1181.615546573849, 1183.9, 1186.6888172217348, 1188.9, 1191.7378244235415, 1193.9, 1196.5483693501587, 1198.8, 1201.5325443234078, 1203.8, 1206.5944516093734, 1208.8, 1211.4560716332721, 1213.8, 1216.4902257056644, 1218.8, 1221.5020603603978, 1223.8, 1226.3861829743955, 1228.8, 1231.2336479134826, 1233.8, 1236.1780621637572, 1238.8, 1241.7477258299014, 1243.8, 1247.0561689330698, 1248.8, 1251.2009300066638, 1253.8, 1256.2167752285077, 1258.8, 1261.4846983461696, 1263.8, 1265.7099860650808, 1268.8, 1270.5770900753105])
    initial_nus2  = np.array([1272.6042131350682, 1273.8, 1277.3405408056378, 1278.8, 1281.2353139855318, 1283.8, 1286.1696012675789, 1288.8, 1291.2560021979807, 1293.8, 1296.2794535546955, 1298.8, 1301.213819395613, 1303.7, 1306.2617186435, 1308.7, 1311.232700889095, 1313.7, 1316.810048833819, 1323.7, 1327.0068556961314, 1333.7, 1336.0240360741054, 1338.7, 1343.4698860472697, 1353.7, 1356.277830214803, 1358.7, 1361.5021167401464, 1363.7, 1377.3378359299822, 1388.7, 1503.7243622969859, 2000.0])

    # Convert actual sigmas -> sigs_raw via inverse softplus (matches teammates)
    sigs1_raw = inv_softplus(initial_sigs1)
    sigs2_raw = inv_softplus(initial_sigs2)

    # Pack theta exactly as teammates do
    theta0 = np.concatenate([initial_nus1, sigs1_raw,
                              initial_nus2, sigs2_raw])

    print(f"theta0 length = {len(theta0)}  (expected {2*R1+2*R2} = {2*70+2*34})")

    # ── Verify our J matches their description ────────────────────────────
    J0, lam1, lam2 = calculate_J(R1, R2, S0, theta0)
    print(f"Our numpy J(theta0) = {J0:.8f}")
    print(f"lam1 = {lam1:.6f},  lam2 = {lam2:.6f}")

    # ── Run CMA-ES ────────────────────────────────────────────────────────
    result = cmaes_optimize(
        theta0    = theta0,
        R1        = R1,
        R2        = R2,
        S0        = S0,
        Kbar      = Kbar,
        sigma0    = 0.5,
        maxiter   = 300,
        tol       = 1e-10,
        seed      = 42,
        verbose   = True,
        plot      = True,
        plot_path = '/mnt/user-data/outputs/cmaes_v2_result.png'
    )

    print(f"\nFinal results:")
    print(f"  J0      = {result['J0']:.8f}")
    print(f"  J_opt   = {result['best_J']:.8f}")
    print(f"  Improve = {result['improvement']:.4f}%")
    print(f"  lam1    = {result['lam1_opt']:.6f}")
    print(f"  lam2    = {result['lam2_opt']:.6f}")
