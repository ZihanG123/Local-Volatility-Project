import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from plot_colors import maturity_colors
from calibration import V_K, load_c_values


def plot_objective_histograms(objectives):
    """Plot objective relationships and percent-reduction distributions."""
    log_initial = np.log(objectives[:, 0])
    log_final = np.log(objectives[:, 1])
    percent_reductions = objectives[:, 2]
    average_initial = float(np.mean(objectives[:, 0]))
    average_log_initial = float(np.log(average_initial))

    plt.style.use("default")
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        constrained_layout=True,
    )

    relationships = (
        (axes[0], log_initial, log_final, "ln(Initial Objective)", "ln(Final Objective)",
         "ln(Initial Objective) vs ln(Final Objective)"),
        (axes[1], log_initial, percent_reductions, "ln(Initial Objective)",
         "Percent Reduction", "ln(Initial Objective) vs Percent Reduction"),
    )
    for ax, x_values, y_values, xlabel, ylabel, title in relationships:
        ax.hist2d(
            x_values,
            y_values,
            bins=100,
            cmap="viridis",
            cmin=1,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axvline(
            average_log_initial,
            color="#f0c419",
            linestyle="--",
            linewidth=1,
            label=f"Avg Initial: {average_initial:.2f}",
        )
        if ylabel == "Percent Reduction":
            ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.axhline(
            float(np.mean(y_values)),
            color="#ff5a5f",
            linestyle="--",
            linewidth=1,
            label=f"Avg: {np.mean(y_values):.2f}",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.35)
        ax.grid(False)
        ax.set_box_aspect(1)

    average_reduction = float(np.mean(percent_reductions))
    axes[2].hist(
        percent_reductions,
        bins=100,
        color="#377eb8",
        alpha=0.8,
        edgecolor="#8ecae6",
        linewidth=0.6,
        label="100 bins",
    )
    axes[2].axvline(
        average_reduction,
        color="#ff5a5f",
        linestyle="--",
        linewidth=1,
        label=f"Avg: {average_reduction:.2f}%",
    )
    axes[2].set_title("Percent Reduction Distribution", fontsize=10)
    axes[2].set_xlabel("Percent Reduction")
    axes[2].set_ylabel("Count")
    axes[2].xaxis.set_major_formatter(PercentFormatter(100))
    axes[2].legend(loc="upper right", fontsize=8, framealpha=0.35)
    axes[2].grid(False)
    axes[2].set_box_aspect(1)

    return fig, axes


def _load_optimization_results(optimization_results):
    if isinstance(optimization_results, (str, Path)):
        return pd.read_csv(optimization_results, dtype={"target_date": str})
    if isinstance(optimization_results, pd.DataFrame):
        return optimization_results
    raise TypeError("optimization_results must be a pandas DataFrame or CSV path")


def _parse_json_column(dataframe, column):
    try:
        return dataframe[column].map(json.loads).tolist()
    except KeyError as error:
        raise KeyError(f"optimization_results is missing column {column!r}") from error


def _concatenate_json_arrays(dataframe, column):
    return np.concatenate(
        [np.asarray(values, dtype=float) for values in _parse_json_column(dataframe, column)]
    )


def _plot_sigs_nus(ax, curves, title, colors=None):
    if colors is None:
        colors = maturity_colors(len(curves))
    # The last sigma is the constant value on the infinite right-tail
    # interval. Use the largest finite knot as the visualization cutoff.
    plot_right = max(
        float(np.asarray(curve["nus"], dtype=float)[-1])
        for curve in curves
    )
    for maturity, curve in enumerate(curves, start=1):
        sigs = np.asarray(curve["sigs"], dtype=float)
        nus = np.asarray(curve["nus"], dtype=float)
        if sigs.ndim != 1 or nus.ndim != 1 or nus.size != sigs.size:
            raise ValueError(
                "Each sigs/nus curve must contain one-dimensional arrays with "
                "equal lengths"
            )
        if sigs.size == 0 or np.any(np.diff(nus) < 0):
            raise ValueError("Each curve must contain sorted, non-empty nus")

        # sigma[i] applies on [nus[i], nus[i + 1]) and sigma[-1] applies
        # from nus[-1] through the finite plotting cutoff.
        interval_ends = np.append(nus, max(plot_right, nus[-1]))
        interval_values = np.append(sigs, sigs[-1])
        ax.step(
            interval_ends,
            interval_values,
            where="post",
            linewidth=1.4,
            color=colors[maturity - 1],
            label=f"Maturity {maturity}",
        )

    ax.set_title(title)
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\sigma$")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.margins(x=0.02, y=0.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        fontsize=8, framealpha=0.85, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )


def _curve_values(spot, curves, strikes):
    # Intrinsic value is identical across maturities, so V differences equal C
    # differences without subtracting large intrinsic values in the tails.
    return np.asarray([
        V_K(spot, sigs, nus, *load_c_values(spot, sigs, nus), strikes, type="V")
        for sigs, nus in curves
    ])


def _safe_ratio(numerator, denominator, scale):
    """Leave undefined/roundoff-scale ratios as gaps rather than infinities."""
    tolerance = 64 * np.finfo(float).eps * scale
    valid = np.isfinite(denominator) & (np.abs(denominator) > tolerance)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=valid & np.isfinite(numerator))
    return result


def plot_sampled_maturity_gaps(dataframe, sample_count=20, seed=None, num_strikes=2000,
                               gap_index=None):
    """One three-panel figure per randomly sampled unique date.

    Normalized price: (C_i - C_1)/(C_N - C_1). Vertical spacing between
    adjacent curves is that pair's share of the total maturity spread.
    The third panel uses raw gaps: (C_after[i+1]-C_after[i]) /
    (C_before[i+1]-C_before[i]).
    """
    plt.style.use("default")
    dates = dataframe.drop_duplicates("target_date")
    eligible = dates[dates["exdates"].map(lambda value: len(json.loads(value)) >= 2)]
    if gap_index is None:
        sampled = eligible.sample(n=min(sample_count, len(eligible)), random_state=seed)
    else:
        if not 0 <= gap_index < len(dataframe):
            raise IndexError(f"gap_index must be between 0 and {len(dataframe) - 1}")
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1 when gap_index is supplied")
        chosen = dataframe.iloc[[gap_index]]
        if len(json.loads(chosen.iloc[0]["exdates"])) < 2:
            raise ValueError("The selected gap row needs at least two maturities")
        remaining = eligible[eligible["target_date"] != chosen.iloc[0]["target_date"]]
        random_rows = remaining.sample(n=min(sample_count - 1, len(remaining)), random_state=seed)
        sampled = pd.concat([chosen, random_rows])
    figures = []
    for _, row in sampled.iterrows():
        before, after = [
            [(np.asarray(curve["sigs"], dtype=float), np.asarray(curve["nus"], dtype=float))
             for curve in json.loads(row[column])]
            for column in ("initial_sigs_nus_compiled", "sigs_nus_compiled")
        ]
        exdates = json.loads(row["exdates"])
        if len(before) != len(after) or len(before) != len(exdates):
            raise ValueError(f"Maturity counts do not match for {row['target_date']}")
        spot = float(row["spot"])
        all_curves = before + after
        lower = max(nus[0] for _, nus in all_curves)
        upper = min(nus[-1] for _, nus in all_curves)
        # Exclude the boundary knots, where all maturity gaps can vanish.
        strikes = np.linspace(lower, upper, num_strikes + 2)[1:-1]
        initial = _curve_values(spot, before, strikes)
        final = _curve_values(spot, after, strikes)
        initial_scale = np.max(np.abs(initial), axis=0)
        final_scale = np.max(np.abs(final), axis=0)
        initial_normalized = _safe_ratio(initial - initial[0], initial[-1] - initial[0], initial_scale)
        final_normalized = _safe_ratio(final - final[0], final[-1] - final[0], final_scale)
        ratios = _safe_ratio(np.diff(final, axis=0), np.diff(initial, axis=0), initial_scale)

        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
        date = pd.to_datetime(str(row["target_date"]), format="%Y%m%d")
        fig.suptitle(f"Maturity spacing · {date:%Y-%m-%d}", fontsize=14)
        colors = maturity_colors(len(exdates))
        for ax, values, title in zip(
            axes[:2], (initial_normalized, final_normalized),
            ("Before optimization", "After optimization"),
        ):
            for i, expiry in enumerate(exdates):
                ax.plot(strikes, values[i], color=colors[i], linewidth=1.2,
                        label=f"T{i + 1}: {expiry}")
            ax.set(title=title, ylabel="(C_i − C_1) / (C_N − C_1)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        pair_colors = maturity_colors(len(exdates) - 1)
        for i, ratio in enumerate(ratios):
            axes[2].plot(strikes, ratio, color=pair_colors[i], linewidth=1.2,
                         label=f"T{i + 1}–T{i + 2}")
        axes[2].axhline(1, color="black", linestyle="--", linewidth=0.8, label="Unchanged gap")
        axes[2].set_yscale("log", base=10, nonpositive="mask")
        axes[2].set(title="Adjacent maturity gap ratios", ylabel="Gap after / gap before",
                    xlabel="Strike K")
        axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        for ax in axes:
            ax.axvline(spot, color="red", linestyle=":", linewidth=1)
            ax.grid(alpha=0.2)
        fig.supxlabel("Red dotted line: spot. Undefined or numerically negligible denominators are omitted.",
                      fontsize=9)
        figures.append(fig)
    return figures


def plot_optimization_results(optimization_results, index=0, sample_count=20, seed=None,
                              gap_index=None):
    """Return objective plots, sigma curves, and sampled-date gap figures.

    Gap figures show normalized maturity prices before/after optimization and
    raw adjacent-price gap ratios. The optional seed makes date sampling repeatable.
    """
    dataframe = _load_optimization_results(optimization_results)
    if not 0 <= index < len(dataframe):
        raise IndexError(
            f"index must be between 0 and {len(dataframe) - 1}, got {index}"
        )

    initial_objectives = _concatenate_json_arrays(dataframe, "initial_objectives")
    final_objectives = _concatenate_json_arrays(dataframe, "final_objectives")
    percent_reductions = _concatenate_json_arrays(dataframe, "percent_reductions")
    objective_figures = plot_objective_histograms(
        np.column_stack(
            (
                initial_objectives.reshape(-1),
                final_objectives.reshape(-1),
                percent_reductions.reshape(-1),
            )
        )
    )

    initial_curves = json.loads(dataframe.iloc[index]["initial_sigs_nus_compiled"])
    final_curves = json.loads(dataframe.iloc[index]["sigs_nus_compiled"])
    curves_figure, curves_axes = plt.subplots(
        2, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True,
    )
    curves_figure.suptitle(f"Sigma–nu curves · result {index}", fontsize=14)
    colors = maturity_colors(max(len(initial_curves), len(final_curves)))
    _plot_sigs_nus(
        curves_axes[0],
        initial_curves,
        f"Initial sigmas and nus (index {index})",
        colors=colors,
    )
    _plot_sigs_nus(
        curves_axes[1],
        final_curves,
        f"Final sigmas and nus (index {index})",
        colors=colors,
    )

    # The chosen gap row is shown first and counts toward sample_count.
    gap_figures = plot_sampled_maturity_gaps(
        dataframe, sample_count=sample_count, seed=seed, gap_index=gap_index,
    )
    return objective_figures, curves_figure, gap_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot optimization results.")
    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=0,
        help="Row index used for the initial/final sigma-nu plots (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible date sampling.")
    parser.add_argument("--sample-count", type=int, default=20, help="Total gap dates, including any chosen index (default: 20).")
    parser.add_argument("--gap-index", type=int, default=0,
                        help="Include this zero-based row first; remaining gap dates are random.")
    args = parser.parse_args()

    plot_optimization_results(
        Path(__file__).with_name("optimization_results.csv"),
        index=args.index,
        sample_count=args.sample_count,
        seed=args.seed,
        gap_index=args.gap_index,
    )
    plt.show()
