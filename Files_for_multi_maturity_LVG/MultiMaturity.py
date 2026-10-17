import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from optimization import generate_lower_bound
from optimization import run_optimizer
from calibration import load_c_values
from calibration import load_theta
from calibration import V_K
from data_utils import get_spx_close
from data_utils import process_date_exdate
from initial_point import find_initial_point
from initial_point import smart_K_union
from initial_point import find_bad_constraints
from plotting import plot_sigs
from plotting import plot_V_K
from plotting import scatter_initial_points


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

    for exdate in exdates:
        K_curr, C_l, C_h = process_date_exdate(dataframe, target_date, exdate, S0)
        bid_ask_compiled.append(np.column_stack([K_curr, C_l, C_h]))
        strikes.append(K_curr)

    removed_constraints = []

    eps = 50
    tolerance = 1e-8
    while eps >= 0:
        K_array = smart_K_union(strikes, eps=eps)
        Kbar = np.max(K_array[:, -1]) + 20
        try:
            initial_point = find_initial_point(
                S0,
                K_array,
                bid_ask_compiled,
                Kbar
            )
            break
        except ValueError:
            if (eps == 0): eps = -1
            eps = eps // 2
            if eps < 0:
                bid_ask_compiled, removed_strikes, tolerance = find_bad_constraints(S0, K_array, bid_ask_compiled, Kbar, tolerance)
                removed_constraints.extend(removed_strikes)
                eps = 50
                strikes = [bid_ask[:, 0] for bid_ask in bid_ask_compiled]

    sigs_nus_compiled = []
    prev_sigs, prev_nus = None, None
    fig, ax = plt.subplots(figsize=(20, 10))
    target_date_dt = pd.to_datetime(target_date, format="%Y%m%d")
    for i, exdate in enumerate(exdates):
        exdate_dt = pd.to_datetime(exdate, format="%Y%m%d")
        t_to_expiry = len(pd.bdate_range(target_date_dt, exdate_dt)) - 1
        sigs, nus = load_theta(
            S0,
            initial_point[i, :],
            K_array[i, :],
            0,
            Kbar,
            prev_sigs=prev_sigs,
            prev_nus=prev_nus,
        )
        prev_sigs, prev_nus = sigs, nus
        sigs_nus_compiled.append((sigs, nus))

    print(sigs_nus_compiled)
    for m, (sigs, nus) in enumerate(sigs_nus_compiled):
        label = f"Maturity {m + 1}"
        plot_sigs(S0, sigs, nus, ax=ax, label=label)
    plt.show()


if __name__ == "__main__":
    dataframe = pd.read_csv("datasets/SPX_opt_2011_2012.csv")
    test_date_index(4, dataframe)
