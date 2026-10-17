from calibration import V_K
from calibration import V_K_prime
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import time
from plotting import plot_sigs
from calibration import load_c_values
import matplotlib.pyplot as plt

def objective(sigs, nus):
    return np.sum(np.diff(sigs)**2)

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
):
    candidate_lower_bound = None
    if len(sigs_nus_compiled) > 1:
        candidate_lower_bound = generate_lower_bound(
            S0, test_sigs, test_nus, test_c_vals[0], test_c_vals[1]
        )

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
        if not np.all(previous_values < candidate_lower_bound[:, 1]):
            return False, candidate_lower_bound

    if index < len(sigs_nus_compiled) - 1:
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
        if not np.all(candidate_values < next_lower_bound[:, 1]):
            return False, candidate_lower_bound

    return True, candidate_lower_bound
    

def validate_and_check_market(S0, sigs, nus, bid_ask_arr, c_vals=None):
    if c_vals is None:
        c_vals = load_c_values(S0, sigs, nus)
    K_vec = bid_ask_arr[:, 0]
    results = V_K(S0, sigs, nus, c_vals[0], c_vals[1], K_vec, type="C")

    return np.all((results >= bid_ask_arr[:, 1]) & (results <= bid_ask_arr[:, 2]))

def coordinateDescent(S0, sigs_nus_compiled, bid_ask_compiled, num_iterations = 1, num_indices = 1):
    M = len(sigs_nus_compiled)
    c_vals_compiled = [load_c_values(S0, sigs_nus_compiled[m][0], sigs_nus_compiled[m][1]) for m in range(M)]
    lower_bounds_compiled = [
        generate_lower_bound(S0, sigs, nus, c_vals[0], c_vals[1])
        for (sigs, nus), c_vals in zip(sigs_nus_compiled, c_vals_compiled)
    ] if M > 1 else [None]

    for N in range(num_iterations):
        print("On iteration:", N + 1)
        for m in range(M):
            print(f"Optimizing maturity {m + 1}")
            sigs, nus = sigs_nus_compiled[m]

            current_loss = objective(sigs, nus)
            if not validate_and_check_market(
                S0, sigs, nus, bid_ask_compiled[m], c_vals_compiled[m]
            ):
                raise ValueError("Initial sigs and nus do not satisfy market constraints.")

            length = len(sigs) - num_indices + 1

            for i in range(length):

                grad_sigs = np.zeros_like(sigs)

                s_prev = sigs[:-2]
                s_next = sigs[2:]

                target = (s_prev + s_next) / 2
                grad_sigs[1:-1] = target - sigs[1:-1]

                grad_sigs[0] = (sigs[1] - sigs[0])
                grad_sigs[-1] = (sigs[-2] - sigs[-1])

                start = i % length
                indices = range(start, start+num_indices)

                mask = np.zeros(grad_sigs.shape, dtype=bool)
                mask[indices] = True

                grad_sigs *= mask

                curr_alpha = 1
                for A in range(5):
                    test_sigs = sigs + curr_alpha * grad_sigs
                    test_loss = objective(test_sigs, nus)
                    if test_loss < current_loss:
                        test_c_vals = load_c_values(S0, test_sigs, nus)
                        market_valid = validate_and_check_market(
                            S0,
                            test_sigs,
                            nus,
                            bid_ask_compiled[m],
                            test_c_vals,
                        )
                        if market_valid:
                            monotonic, test_lower_bound = check_monotonicity(
                                S0,
                                sigs_nus_compiled,
                                c_vals_compiled,
                                lower_bounds_compiled,
                                test_sigs,
                                nus,
                                test_c_vals,
                                m,
                            )
                            if monotonic:
                                sigs = test_sigs
                                current_loss = test_loss
                                c_vals_compiled[m] = test_c_vals
                                if test_lower_bound is not None:
                                    lower_bounds_compiled[m] = test_lower_bound
                                break
                    curr_alpha *= 0.5
            sigs_nus_compiled[m] = (sigs, nus)
    return sigs_nus_compiled


def coordinateDescentCpp(
    S0,
    sigs_nus_compiled,
    bid_ask_compiled,
    num_iterations=1,
    num_indices=1,
    num_strikes=200,
):
    try:
        import _optimization_cpp
    except ImportError as error:
        raise ImportError(
            "The native optimizer is not built. Run `python -m pip install -e .` "
            "from the project directory."
        ) from error

    return _optimization_cpp.coordinate_descent(
        S0,
        sigs_nus_compiled,
        bid_ask_compiled,
        num_iterations=num_iterations,
        num_indices=num_indices,
        num_strikes=num_strikes,
    )

def add_nus_uniform(sigs, nus, max_num=1000):
    min_length = np.min(np.diff(nus))
    for _ in range(max_num):
        index = np.argmax(np.diff(nus))
        if (nus[index + 1] - nus[index]) / 2 < min_length:
            break
        nus = np.insert(nus, index + 1, (nus[index + 1] + nus[index]) / 2)
        sigs = np.insert(sigs, index + 1, sigs[index])
    return (sigs, nus)


def run_optimizer(S0, sigs_nus_compiled, bid_ask_compiled, backend="cpp"):
    if backend == "cpp":
        optimizer = coordinateDescentCpp
    elif backend == "python":
        optimizer = coordinateDescent
    else:
        raise ValueError("backend must be 'cpp' or 'python'")

    fig, axs = plt.subplots(2, figsize=(20, 10))
    for m, (init_sigs, init_nus) in enumerate(sigs_nus_compiled):
        label = f"Maturity {m + 1}"
        plot_sigs(S0, init_sigs, init_nus, ax=axs[0], label=label)
    sigs_nus_compiled = optimizer(S0, sigs_nus_compiled, bid_ask_compiled, num_iterations=1, num_indices=1)
    sigs_nus_compiled = [add_nus_uniform(sigs, nus, max_num=100) for (sigs, nus) in sigs_nus_compiled]
    sigs_nus_compiled = optimizer(S0, sigs_nus_compiled, bid_ask_compiled, num_iterations=100, num_indices=len(sigs_nus_compiled[0][0])//10)
    for m, (final_sigs, final_nus) in enumerate(sigs_nus_compiled):
        label = f"Maturity {m + 1}"
        plot_sigs(S0, final_sigs, final_nus, ax=axs[1], label=label)
    plt.show()