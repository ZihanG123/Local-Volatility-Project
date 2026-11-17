import numpy as np
import warnings
import pandas as pd
import json
from pathlib import Path

from optimization import run_optimizer
import matplotlib.pyplot as plt
from calibration import load_theta
from data_utils import get_spx_close
from data_utils import process_date_exdate
from initial_point import find_initial_point, find_single_initial_point
from initial_point import smart_K_union
from initial_point import find_bad_constraints
from plotting import plot_V_K, scatter_initial_points


def test_date_index(index, dataframe):
    df = dataframe.copy()
    date_counts = df.groupby("date")["exdate"].nunique()
    valid_dates = sorted(date_counts[date_counts > 0].index.tolist())
    if index >= len(valid_dates):
        print(f"Index {index} out of range for {len(valid_dates)} valid dates.")
        return None

    target_date = valid_dates[index]
    exdates = sorted(df[df["date"] == target_date]["exdate"].unique())
    bid_ask_compiled = []
    strikes = []
    S0 = get_spx_close(target_date)
    print(f"Processing date: {target_date}")

    feasible_exdates = []
    single_maturity_kbars = []
    for maturity_index, exdate in enumerate(exdates):
        K_curr, C_l, C_h = process_date_exdate(
            dataframe, target_date, exdate, S0
        )
        bid_ask = np.column_stack([K_curr, C_l, C_h])
        margin = 400
        try:
            find_single_initial_point(
                S0,
                K_curr,
                bid_ask,
            )
            feasible_exdates.append(exdate)
            bid_ask_compiled.append(bid_ask)
            strikes.append(K_curr)
            single_maturity_kbars.append(K_curr[-1] + margin)
        except ValueError: continue
    exdates = np.sort(np.asarray(feasible_exdates))
    if exdates.size == 0:
        print(f"No feasible expirations remain for date {target_date}.")
        return None

    removed_constraints = []

    

    eps = 50
    tolerance = 1e-8
    while eps >= 0:
        K_array = smart_K_union(strikes, eps=eps)

        Kbar = np.max(K_array) + 200
        new_col = np.full((K_array.shape[0], 1), Kbar)
        K_array = np.hstack((K_array, new_col))

        try:
            initial_point = find_initial_point(
                S0,
                K_array,
                bid_ask_compiled,
            )
            break
        except ValueError:
            if (eps == 0): eps = -1
            eps = eps // 2
            if eps < 0:
                bid_ask_compiled, removed_strikes, tolerance = find_bad_constraints(S0, K_array, bid_ask_compiled, tolerance)
                removed_constraints.extend(removed_strikes)
                eps = 50
                strikes = [bid_ask[:, 0] for bid_ask in bid_ask_compiled]

    sigs_nus_compiled = []
    prev_sigs, prev_nus = None, None
    target_date_dt = pd.to_datetime(target_date, format="%Y%m%d")
    for i, exdate in enumerate(exdates):
        exdate_dt = pd.to_datetime(exdate, format="%Y%m%d")
        t_to_expiry = len(pd.bdate_range(target_date_dt, exdate_dt)) - 1
        sigs, nus = load_theta(
            S0,
            initial_point[i, :],
            K_array[i, :],
            0,
            prev_sigs=prev_sigs,
            prev_nus=prev_nus,
        )
        prev_sigs, prev_nus = sigs, nus
        sigs_nus_compiled.append((sigs, nus))

    for maturity, (sigs, nus) in enumerate(sigs_nus_compiled, start=1):
        if not np.all(np.isfinite(sigs)) or not np.all(np.isfinite(nus)):
            print(
                f"Inf or NaN detected for date {target_date}, "
                f"maturity {maturity}; skipping optimization."
            )
            return None

    try:
        result = run_optimizer(
            S0,
            sigs_nus_compiled,
            bid_ask_compiled,
        )
        if result is not None:
            return target_date, result, exdates, S0
        print(
            f"Optimization did not produce a valid result for date "
            f"{target_date}; final monotonicity check failed."
        )
        return None
    except ValueError as error:
        print(f"Optimizer error for date {target_date}: {error}")
        return None


def result_to_csv_row(target_date, result, exdates, spot):
    (
        initial_objectives,
        final_objectives,
        percent_reductions,
        initial_sigs_nus_compiled,
        sigs_nus_compiled,
    ) = result
    return {
        "target_date": target_date,
        "exdates": json.dumps(np.asarray(exdates).tolist()),
        "spot": float(spot),
        "initial_objectives": json.dumps(
            np.asarray(initial_objectives).tolist()
        ),
        "final_objectives": json.dumps(
            np.asarray(final_objectives).tolist()
        ),
        "percent_reductions": json.dumps(
            np.asarray(percent_reductions).tolist()
        ),
        "initial_sigs_nus_compiled": json.dumps(
            [
                {
                    "sigs": np.asarray(sigs).tolist(),
                    "nus": np.asarray(nus).tolist(),
                }
                for sigs, nus in initial_sigs_nus_compiled
            ]
        ),
        "sigs_nus_compiled": json.dumps(
            [
                {
                    "sigs": np.asarray(sigs).tolist(),
                    "nus": np.asarray(nus).tolist(),
                }
                for sigs, nus in sigs_nus_compiled
            ]
        ),
    }


def save_results_csv(
    results,
    output_path=None,
):
    new_results = pd.DataFrame(results)
    output_file = (
        Path(output_path)
        if output_path is not None
        else Path(__file__).resolve().parent / "optimization_results.csv"
    )

    if new_results.empty:
        return

    if output_file.exists() and output_file.stat().st_size > 0:
        try:
            existing_results = pd.read_csv(
                output_file,
                dtype={"target_date": str},
            )
        except pd.errors.EmptyDataError:
            existing_results = pd.DataFrame()
        combined_results = pd.concat(
            [existing_results, new_results],
            ignore_index=True,
        )
    else:
        combined_results = new_results

    combined_results["target_date"] = pd.to_datetime(
        combined_results["target_date"].astype(str),
        format="%Y%m%d",
    ).dt.strftime("%Y%m%d")
    combined_results = (
        combined_results
        .drop_duplicates(subset="target_date", keep="last")
        .sort_values("target_date")
        .reset_index(drop=True)
    )
    combined_results.to_csv(output_file, index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    dataframe = pd.read_csv("datasets/SPX_opt_2011_2012.csv")
    results = []
    for i in range(10):
        try:
            result = test_date_index(i, dataframe)
        except ValueError as error:
            print(f"Skipping date index {i}: {error}")
            continue
        if result is not None:
            target_date, optimizer_result, exdates, spot = result
            results.append(
                result_to_csv_row(
                    target_date,
                    optimizer_result,
                    exdates,
                    spot,
                )
            )
            save_results_csv(results)
    # save_results_csv(results)

    
