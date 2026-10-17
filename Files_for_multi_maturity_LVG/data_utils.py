from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp
from IPython.display import display


_SPX_CACHE_PATH = Path(__file__).with_name("spx_prices.csv")
_spx_cache_df = pd.read_csv(_SPX_CACHE_PATH, index_col=0, parse_dates=True)
_spx_cache = {
    int(pd.to_datetime(date).strftime("%Y%m%d")): float(value)
    for date, value in _spx_cache_df["Close"].to_dict().items()
}


def get_option_data(df, date_val, exdate_val):
    filtered_df = df[
        (df["date"] == int(date_val)) & (df["exdate"] == int(exdate_val))
    ]
    result_df = filtered_df[
        ["strike_price", "cp_flag", "best_bid", "best_offer"]
    ]
    return result_df.to_numpy()


def get_spx_close(date_val):
    date_int = int(date_val)
    if date_int in _spx_cache:
        return _spx_cache[date_int]
    return None


def process_date_exdate(df, date_val, exdate_val, S0):
    test = get_option_data(df, date_val, exdate_val)
    test_calls = test[test[:, 1] == "C"]
    test_puts = test[test[:, 1] == "P"]
    test_calls = np.delete(test_calls, 1, axis=1).astype(float)
    test_puts = np.delete(test_puts, 1, axis=1).astype(float)

    common_strikes = np.intersect1d(test_calls[:, 0], test_puts[:, 0])
    test_calls = test_calls[np.isin(test_calls[:, 0], common_strikes)]
    test_puts = test_puts[np.isin(test_puts[:, 0], common_strikes)]
    test_calls = test_calls[test_calls[:, 0].argsort()]
    test_puts = test_puts[test_puts[:, 0].argsort()]

    mid_calls = test_calls[:, 1:3].mean(axis=1)
    mid_puts = test_puts[:, 1:3].mean(axis=1)
    strikes = test_calls[:, 0]
    diff_prices = mid_calls - mid_puts
    lower_bound = test_calls[:, 1] - test_puts[:, 2]
    upper_bound = test_calls[:, 2] - test_puts[:, 1]

    A = cp.Variable()
    B = cp.Variable()
    margin = cp.Variable(nonneg=True)
    fitted_values = A + B * strikes
    problem = cp.Problem(
        cp.Maximize(margin),
        [
            fitted_values + margin <= upper_bound,
            fitted_values - margin >= lower_bound,
        ],
    )
    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_feas=1e-10,
    )
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(f"Affine quote-fit Chebyshev solve failed: {problem.status}")

    y_pred = np.asarray(fitted_values.value).ravel()
    B = B.value
    A = A.value
    within_bounds = (y_pred >= lower_bound) & (y_pred <= upper_bound)
    if not np.all(within_bounds):
        violations_df = pd.DataFrame(
            {
                "Strike": strikes[~within_bounds],
                "Lower Bound": lower_bound[~within_bounds],
                "Predicted": y_pred[~within_bounds],
                "Upper Bound": upper_bound[~within_bounds],
            }
        )
        display(violations_df)
        raise ValueError("No affine fit satisfies all call-put quote bounds")

    D = -B
    F = A / D
    strike_scale = S0 / F
    price_scale = S0 / (D * F)

    test_calls[:, 0] *= strike_scale
    test_calls[:, 1:3] *= price_scale
    test_puts[:, 0] *= strike_scale
    test_puts[:, 1:3] *= price_scale

    K = test_calls[:, 0]
    C_h = np.minimum(test_calls[:, 2], test_puts[:, 2] + S0 - K)
    C_l = np.maximum(test_calls[:, 1], test_puts[:, 1] + S0 - K)
    return K, C_l, C_h
