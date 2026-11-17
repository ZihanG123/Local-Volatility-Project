from calibration import V_K
from calibration import V_K_prime
import numpy as np
from calibration import load_c_values

try:
    from _optimization_cpp import coordinate_descent as _native_coordinate_descent
except ImportError:
    _native_coordinate_descent = None

def objective(sigs, nus):
    return np.sum(np.diff(sigs)**2)


def coordinate_descent(
    S0,
    sigs_nus_compiled,
    bid_ask_compiled,
    num_iterations=1,
    num_indices=1,
    num_strikes=75,
):
    """Run the native constrained coordinate-descent optimizer."""
    if _native_coordinate_descent is None:
        raise ImportError(
            "The native CCD optimizer is unavailable. Build the "
            "_optimization_cpp extension before running coordinate_descent."
        )
    return [
        (np.asarray(sigs, dtype=float), np.asarray(nus, dtype=float))
        for sigs, nus in _native_coordinate_descent(
            S0,
            sigs_nus_compiled,
            bid_ask_compiled,
            num_iterations=num_iterations,
            num_indices=num_indices,
            num_strikes=num_strikes,
        )
    ]


def coordinate_descent_compiled(
    S0,
    sigs_nus_compiled,
    bid_ask_compiled,
    num_iterations=1,
    num_indices=1,
    num_strikes=75,
):
    """Backward-compatible name for the native CCD optimizer."""
    return coordinate_descent(
        S0,
        sigs_nus_compiled,
        bid_ask_compiled,
        num_iterations=num_iterations,
        num_indices=num_indices,
        num_strikes=num_strikes,
    )


def add_nus_uniform(sigs, nus, max_num=1000):
    """Bisect the widest intervals until the knot count reaches ``max_num``.

    Curves use one sigma per knot, with the final sigma representing the
    right-hand tail, so ``sigs`` and ``nus`` have equal lengths.
    """
    sigs = np.asarray(sigs, dtype=float).copy()
    nus = np.asarray(nus, dtype=float).copy()
    if nus.size != sigs.size:
        raise ValueError("sigs and nus must have equal lengths")

    while sigs.size < max_num:
        interval = int(np.argmax(np.diff(nus)))
        midpoint = 0.5 * (nus[interval] + nus[interval + 1])
        nus = np.insert(nus, interval + 1, midpoint)
        sigs = np.insert(sigs, interval + 1, sigs[interval])
    return sigs, nus

def generate_lower_bound(S0, sigs, nus, c_v1, c_v2, num_strikes = 200):
    strikes = np.linspace(nus[0], nus[-1], num_strikes)
    values = V_K(S0, sigs, nus, c_v1, c_v2, strikes, type="C")
    slopes = V_K_prime(S0, sigs, nus, c_v1, c_v2, strikes, type="C")

    slope_differences = slopes[:-1] - slopes[1:]

    intersection_strikes = (
        values[1:]
        - values[:-1]
        + slopes[:-1] * strikes[:-1]
        - slopes[1:] * strikes[1:]
    ) / slope_differences
    intersection_values = values[:-1] + slopes[:-1] * (
        intersection_strikes - strikes[:-1]
    )
    lower_bound = np.column_stack((intersection_strikes, intersection_values))
    return lower_bound

def check_monotonicity(
    S0,
    sigs_nus_compiled,
    c_vals_compiled,
    lower_bounds_compiled,
    test_sigs,
    test_nus,
    test_c_vals,
    index,
    num_strikes=200,
    kbar=None,
):
    if kbar is None:
        kbar = max(float(nus[-1]) for _, nus in sigs_nus_compiled)

    candidate_lower_bound = None
    if len(sigs_nus_compiled) > 1:
        candidate_lower_bound = generate_lower_bound(
            S0,
            test_sigs,
            test_nus,
            test_c_vals[0],
            test_c_vals[1],
            num_strikes=num_strikes,
        )

    if index > 0:
        previous_sigs, previous_nus = sigs_nus_compiled[index - 1]
        previous_kbar_value = V_K(
            S0,
            previous_sigs,
            previous_nus,
            c_vals_compiled[index - 1][0],
            c_vals_compiled[index - 1][1],
            np.asarray([kbar]),
            type="C",
        )[0]
        candidate_kbar_value = V_K(
            S0,
            test_sigs,
            test_nus,
            test_c_vals[0],
            test_c_vals[1],
            np.asarray([kbar]),
            type="C",
        )[0]
        if (
            not np.isfinite(previous_kbar_value)
            or not np.isfinite(candidate_kbar_value)
            or previous_kbar_value > candidate_kbar_value
            or previous_sigs[-1] > test_sigs[-1]
        ):
            return False, candidate_lower_bound
        previous_values = V_K(
            S0,
            previous_sigs,
            previous_nus,
            c_vals_compiled[index - 1][0],
            c_vals_compiled[index - 1][1],
            candidate_lower_bound[:, 0],
            type="C",
        )
        if not np.all(previous_values <= candidate_lower_bound[:, 1]):
            return False, candidate_lower_bound

    if index < len(sigs_nus_compiled) - 1:
        next_sigs, next_nus = sigs_nus_compiled[index + 1]
        next_kbar_value = V_K(
            S0,
            next_sigs,
            next_nus,
            c_vals_compiled[index + 1][0],
            c_vals_compiled[index + 1][1],
            np.asarray([kbar]),
            type="C",
        )[0]
        candidate_kbar_value = V_K(
            S0,
            test_sigs,
            test_nus,
            test_c_vals[0],
            test_c_vals[1],
            np.asarray([kbar]),
            type="C",
        )[0]
        if (
            not np.isfinite(candidate_kbar_value)
            or not np.isfinite(next_kbar_value)
            or candidate_kbar_value > next_kbar_value
            or test_sigs[-1] > next_sigs[-1]
        ):
            return False, candidate_lower_bound
        next_lower_bound = lower_bounds_compiled[index + 1]
        candidate_values = V_K(
            S0,
            test_sigs,
            test_nus,
            test_c_vals[0],
            test_c_vals[1],
            next_lower_bound[:, 0],
            type="C",
        )
        if not np.all(candidate_values <= next_lower_bound[:, 1]):
            return False, candidate_lower_bound

    return True, candidate_lower_bound

def prepare_curve_data(S0, sigs_nus_compiled, num_strikes):
    c_vals_compiled = [
        load_c_values(S0, sigs, nus)
        for sigs, nus in sigs_nus_compiled
    ]
    lower_bounds_compiled = [
        generate_lower_bound(
            S0,
            sigs,
            nus,
            c_vals[0],
            c_vals[1],
            num_strikes=num_strikes,
        )
        for (sigs, nus), c_vals in zip(
            sigs_nus_compiled, c_vals_compiled
        )
    ]
    return c_vals_compiled, lower_bounds_compiled

def run_optimizer(
    S0,
    sigs_nus_compiled,
    bid_ask_compiled,
    num_strikes=200,
):
    initial_sigs_nus_compiled = [
        (np.array(sigs, copy=True), np.array(nus, copy=True))
        for sigs, nus in sigs_nus_compiled
    ]
    initial_objectives = np.array(
        [objective(sigs, nus) for sigs, nus in initial_sigs_nus_compiled]
    )

    # First pass: optimize the original calibrated grid one coordinate at a time.
    ccd_sigs_nus_compiled = coordinate_descent(
        S0,
        sigs_nus_compiled,
        bid_ask_compiled,
        num_iterations=10,
        num_indices=1,
        num_strikes=num_strikes,
    )

    # Add one knot in every interval, then run a wider-block CCD pass.
    refined_sigs_nus_compiled = [
        add_nus_uniform(
            sigs,
            nus,
            max_num=100,
        )
        for sigs, nus in ccd_sigs_nus_compiled
    ]
    sigs_nus_compiled = coordinate_descent(
        S0,
        refined_sigs_nus_compiled,
        bid_ask_compiled,
        num_iterations=10,
        num_indices=max(
            1,
            max(len(nus) for _, nus in refined_sigs_nus_compiled) // 10,
        ),
        num_strikes=num_strikes,
    )

    final_check_num_strikes = 10000
    kbar = max(float(nus[-1]) for _, nus in sigs_nus_compiled)
    c_vals_compiled, final_lower_bounds = prepare_curve_data(
        S0, sigs_nus_compiled, final_check_num_strikes
    )
    monotonicity_results = [
        check_monotonicity(
            S0,
            sigs_nus_compiled,
            c_vals_compiled,
            final_lower_bounds,
            sigs,
            nus,
            c_vals,
            index,
            num_strikes=final_check_num_strikes,
            kbar=kbar,
        )
        for index, ((sigs, nus), c_vals) in enumerate(
            zip(sigs_nus_compiled, c_vals_compiled)
        )
    ]
    failed_maturities = [
        index + 1
        for index, (is_monotonic, _) in enumerate(monotonicity_results)
        if not is_monotonic
    ]
    if failed_maturities:
        print("Final monotonicity diagnostics:")
        for index, ((sigs, nus), c_vals) in enumerate(
            zip(sigs_nus_compiled, c_vals_compiled)
        ):
            candidate_lower_bound = generate_lower_bound(
                S0,
                sigs,
                nus,
                c_vals[0],
                c_vals[1],
                num_strikes=final_check_num_strikes,
            )
            checks = []
            if index > 0:
                previous_sigs, previous_nus = sigs_nus_compiled[index - 1]
                previous_values = V_K(
                    S0,
                    previous_sigs,
                    previous_nus,
                    c_vals_compiled[index - 1][0],
                    c_vals_compiled[index - 1][1],
                    candidate_lower_bound[:, 0],
                    type="C",
                )
                checks.append(
                    (
                        "previous curve vs current lower bound",
                        candidate_lower_bound[:, 0],
                        previous_values - candidate_lower_bound[:, 1],
                    )
                )
            if index < len(sigs_nus_compiled) - 1:
                next_lower_bound = final_lower_bounds[index + 1]
                current_values = V_K(
                    S0,
                    sigs,
                    nus,
                    c_vals[0],
                    c_vals[1],
                    next_lower_bound[:, 0],
                    type="C",
                )
                checks.append(
                    (
                        "current curve vs next lower bound",
                        next_lower_bound[:, 0],
                        current_values - next_lower_bound[:, 1],
                    )
                )
            for description, strikes, violations in checks:
                finite = np.isfinite(violations)
                if not np.any(finite):
                    print(
                        f"  Maturity {index + 1}: {description}: "
                        "no finite comparison values"
                    )
                    continue
                finite_violations = violations[finite]
                finite_strikes = strikes[finite]
                worst_position = int(np.argmax(finite_violations))
                print(
                    f"  Maturity {index + 1}: {description}: "
                    f"max difference={finite_violations[worst_position]:.6e} "
                    f"at strike={finite_strikes[worst_position]:.6f}; "
                    f"nonfinite={np.count_nonzero(~finite)}"
                )
        print(
            "Final monotonicity check failed for maturities: "
            + ", ".join(map(str, failed_maturities))
        )
    final_objectives = np.array([objective(sigs, nus) for sigs, nus in sigs_nus_compiled])
    return (
        initial_objectives,
        final_objectives,
        100 * (initial_objectives - final_objectives) / initial_objectives,
        initial_sigs_nus_compiled,
        sigs_nus_compiled,
    )


