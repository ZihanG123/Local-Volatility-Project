import json
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from calibration import V_K, V_K_prime, load_c_values

try:
    go = importlib.import_module("plotly.graph_objects")
except ImportError:  # pragma: no cover - exercised only when the optional dependency is absent
    go = None

def find_min_C2K(S0, sigs_nus_compiled):
    minima = []
    for sigs, nus in sigs_nus_compiled:
        sigs = np.asarray(sigs, dtype=float)
        nus = np.asarray(nus, dtype=float)
        c_v1, c_v2 = load_c_values(S0, sigs, nus)
        values = V_K(S0, sigs, nus, c_v1, c_v2, nus, type="V")
        minima.append(min(np.min(values[:-1] / sigs), np.min(values[1:] / sigs)))

    return np.asarray(minima, dtype=float)

def evaluate_curves(S0, curves, strikes, coefficients=None):
    """Evaluate each maturity once; arrays have shape (maturities, strikes)."""
    if coefficients is None:
        coefficients = [
            load_c_values(S0, sigs, nus) for sigs, nus in curves
        ]
    if len(coefficients) != len(curves):
        raise ValueError("coefficients and curves must have equal lengths")

    prices, first_derivatives, second_derivatives = [], [], []
    intrinsic = np.maximum(S0 - strikes, 0)
    for (sigs, nus), curve_coefficients in zip(curves, coefficients):
        price = V_K(
            S0, sigs, nus, *curve_coefficients, strikes, type="C"
        )
        sigma_indices = np.clip(
            np.searchsorted(nus, strikes, side="right") - 1, 0, len(sigs) - 1,
        )
        prices.append(price)
        first_derivatives.append(
            V_K_prime(
                S0, sigs, nus, *curve_coefficients, strikes, type="C"
            )
        )
        second_derivatives.append((price - intrinsic) / np.asarray(sigs)[sigma_indices]**2)
    return tuple(np.asarray(values) for values in
                 (prices, first_derivatives, second_derivatives))


def _interpolation_terms(curve_data, days_to_expiry, *, return_auxiliary=False):
    """Reuse maturity secants for slopes, curvature ratios, and interpolation."""
    prices, first_derivatives, second_derivatives = curve_data
    dT = np.diff(days_to_expiry)
    A, A_1, A_2 = (
        np.diff(values, axis=0) / dT[:, None] for values in curve_data
    )
    a, b = A[:-1], A[1:]
    a_1, b_1 = A_1[:-1], A_1[1:]
    a_2, b_2 = A_2[:-1], A_2[1:]
    total = a + b
    interior_curvature = (
        (2*b*b*a_2 + 2*a*a*b_2) / total**2
        - 4*(a*b_1 - b*a_1)**2 / total**3
    )
    m = np.vstack((A[0] / 2, 2 / (1/a + 1/b), A[-1] / 2))
    m_2K = np.vstack((A_2[0] / 2, interior_curvature, A_2[-1] / 2))
    result = m_2K / second_derivatives, m, A, dT
    if return_auxiliary:
        return (*result, m_2K, A_2)
    return result


def m_2K_over_second_derivative(
    S0, sigs_nus_compiled, days_to_expiry, strike, maturity_index,
    *, coefficients=None,
):
    """Return ``m_2K / C_KK`` at one strike and one maturity.

    ``maturity_index`` is zero-based. Only the requested strike is evaluated;
    the maturity secants are still formed across all calibrated maturities,
    since the selected maturity's ``m_2K`` depends on its neighboring curves.
    The endpoint and interior formulas match ``_interpolation_terms``.
    """
    days_to_expiry = np.asarray(days_to_expiry, dtype=float)
    maturity_count = len(sigs_nus_compiled)
    if maturity_count != len(days_to_expiry):
        raise ValueError(
            "sigs_nus_compiled and days_to_expiry must have equal lengths"
        )
    if maturity_count < 2:
        raise ValueError("At least two maturities are required")
    if not 0 <= maturity_index < maturity_count:
        raise IndexError(
            f"maturity_index must be between 0 and {maturity_count - 1}"
        )

    left = max(0, maturity_index - 1)
    right = min(maturity_count, maturity_index + 2)
    local_curves = sigs_nus_compiled[left:right]
    local_days_to_expiry = days_to_expiry[left:right]
    local_coefficients = (
        None if coefficients is None else coefficients[left:right]
    )
    curve_data = evaluate_curves(
        S0,
        local_curves,
        np.asarray([strike], dtype=float),
        coefficients=local_coefficients,
    )
    prices, first_derivatives, second_derivatives = (
        values[:, 0] for values in curve_data
    )
    dT = np.diff(local_days_to_expiry)
    if np.any(dT <= 0):
        raise ValueError("days_to_expiry must be strictly increasing")

    A = np.diff(prices) / dT
    A_1 = np.diff(first_derivatives) / dT
    A_2 = np.diff(second_derivatives) / dT
    if maturity_index == 0:
        m_2K = A_2[0] / 2
    elif maturity_index == maturity_count - 1:
        m_2K = A_2[-1] / 2
    else:
        a, b = A[0], A[1]
        a_1, b_1 = A_1[0], A_1[1]
        a_2, b_2 = A_2[0], A_2[1]
        total = a + b
        m_2K = (
            (2 * b * b * a_2 + 2 * a * a * b_2) / total**2
            - 4 * (a * b_1 - b * a_1) ** 2 / total**3
        )
    return float(m_2K / second_derivatives[maturity_index - left])


def _maximize_ratio_on_nu_intervals(
    S0, sigs_nus_compiled, union_nus, days_to_expiry, maturity_index, sign,
    coefficients,
):
    """Maximize ``sign * R`` independently on every union-nu interval."""
    interval_maxima = []
    lower_buffer = 0.1
    kbar = min(
        float(np.asarray(nus, dtype=float)[-1])
        for _, nus in sigs_nus_compiled
    )
    upper_buffer = kbar - lower_buffer

    def objective(strike):
        ratio = m_2K_over_second_derivative(
            S0,
            sigs_nus_compiled,
            days_to_expiry,
            strike,
            maturity_index,
            coefficients=coefficients,
        )
        if not np.isfinite(ratio):
            return np.inf
        return -sign * ratio

    for left, right in zip(union_nus[:-1], union_nus[1:]):
        left = max(float(left), lower_buffer)
        right = min(float(right), upper_buffer)
        if right <= left:
            continue
        result = minimize_scalar(
            objective,
            bounds=(left, right),
            method="bounded",
            options={"xatol": max(1e-10, (right - left) * 1e-8)},
        )
        candidates = [left, right]
        if result.success and np.isfinite(result.x):
            candidates.append(result.x)
        finite_values = [
            sign * m_2K_over_second_derivative(
                S0,
                sigs_nus_compiled,
                days_to_expiry,
                candidate,
                maturity_index,
                coefficients=coefficients,
            )
            for candidate in candidates
        ]
        finite_values = [value for value in finite_values if np.isfinite(value)]
        if finite_values:
            interval_maxima.append(max(finite_values))

    final_max = np.nan if not interval_maxima else float(max(interval_maxima))
    quantity = "R" if sign > 0 else "-R"
    print(
        f"maximize {quantity}: maturity={maturity_index}, "
        f"final max={final_max:.6e}"
    )
    return final_max


def _optimized_interpolation_N(
    S0, sigs_nus_compiled, days_to_expiry
):
    """Compute interpolation exponents from locally optimized R extrema.

    The ratio for maturity ``i`` only depends on its own curve and its
    adjacent maturity curves, so its optimization intervals use that local
    union of nus rather than the union across every maturity.
    """
    maturity_count = len(sigs_nus_compiled)
    coefficients = [
        load_c_values(S0, sigs, nus)
        for sigs, nus in sigs_nus_compiled
    ]
    positive_R = np.empty(maturity_count, dtype=float)
    negative_R = np.empty(maturity_count, dtype=float)
    for maturity_index in range(maturity_count):
        left = max(0, maturity_index - 1)
        right = min(maturity_count, maturity_index + 2)
        local_union_nus = np.unique(np.concatenate([
            np.asarray(nus, dtype=float)
            for _, nus in sigs_nus_compiled[left:right]
        ]))
        positive_R[maturity_index] = _maximize_ratio_on_nu_intervals(
            S0,
            sigs_nus_compiled,
            local_union_nus,
            days_to_expiry,
            maturity_index,
            sign=1.0,
            coefficients=coefficients,
        )
        negative_R[maturity_index] = _maximize_ratio_on_nu_intervals(
            S0,
            sigs_nus_compiled,
            local_union_nus,
            days_to_expiry,
            maturity_index,
            sign=-1.0,
            coefficients=coefficients,
        )

    dT = np.diff(np.asarray(days_to_expiry, dtype=float))
    required_N = np.maximum(
        negative_R[:-1] * dT,
        positive_R[1:] * dT,
    )
    return np.ceil(np.maximum(4.0, required_N)).astype(int)


def find_R1_R2(S0, sigs_nus_compiled, days_to_expiry, strikes):
    curve_data = evaluate_curves(S0, sigs_nus_compiled, np.asarray(strikes, dtype=float))
    R, m, _, _ = _interpolation_terms(curve_data, days_to_expiry)
    return R, m


def get_surface(S0, sigs_nus_compiled, days_to_expiry, strikes, times,
                *, return_prices=False, return_local_vol=False,
                chunk_rows=128, use_optimized_N=True):
    """Return (times, strikes) prices, optionally with calibrated maturity prices.

    If requested, local volatility is computed from Dupire's formula using the
    same interpolation terms as the call-price surface. ``use_optimized_N``
    selects between interval-wise scalar optimization and the original
    sampled-strike estimate of the interpolation exponents.
    """
    strikes = np.asarray(strikes, dtype=float)
    times = np.asarray(times, dtype=float)
    days_to_expiry = np.asarray(days_to_expiry, dtype=float)
    calculation_strikes = strikes
    curve_data = evaluate_curves(S0, sigs_nus_compiled, calculation_strikes)
    prices = curve_data[0]
    second_derivatives = curve_data[2]
    R, m, A, dT, m_2K, A_2 = _interpolation_terms(
        curve_data, days_to_expiry, return_auxiliary=True
    )
    if use_optimized_N:
        N = _optimized_interpolation_N(
            S0,
            sigs_nus_compiled,
            days_to_expiry,
        )
    else:
        bounds = np.maximum(-R[:-1].min(axis=1), R[1:].max(axis=1)) * dT
        N = np.maximum(bounds, 4).astype(int) + 1
    print(N)
    I = np.clip(np.searchsorted(days_to_expiry, times, side="left") - 1,
                0, len(days_to_expiry) - 2)
    u = (times - days_to_expiry[I]) / dT[I]
    result = np.empty((len(times), len(calculation_strikes)))
    local_vol = (
        np.empty_like(result) if return_local_vol else None
    )
    for start in range(0, len(times), chunk_rows):
        stop = min(start + chunk_rows, len(times))
        intervals = I[start:stop]
        weight = u[start:stop, None]
        exponent = N[intervals, None] - 1
        A_i = A[intervals, :]
        m_i = m[:-1][intervals, :]
        m_i1 = m[1:][intervals, :]
        result[start:stop] = prices[intervals] + dT[intervals, None] * (
            A_i * weight
            + (m_i - A_i) * weight * (1 - weight)**exponent
            + (A_i - m_i1) * weight**exponent * (1 - weight)
        )

        if return_local_vol:
            time_derivative = (
                A_i
                + (m_i - A_i)
                * (1 - weight)**(exponent - 1)
                * (1 - (exponent + 1) * weight)
                + (A_i - m_i1)
                * weight**(exponent - 1)
                * (exponent - (exponent + 1) * weight)
            )
            strike_curvature = (
                (1 - weight) * second_derivatives[intervals]
                + weight * second_derivatives[intervals + 1]
                + dT[intervals, None]
                * (m_2K[:-1][intervals, :] - A_2[intervals, :])
                * weight * (1 - weight)**exponent
                + dT[intervals, None]
                * (A_2[intervals, :] - m_2K[1:][intervals, :])
                * weight**exponent * (1 - weight)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                local_variance = (
                    2 * time_derivative
                    / (calculation_strikes[None, :]**2 * strike_curvature)
                )
            local_vol[start:stop] = np.where(
                local_variance >= 0,
                np.sqrt(local_variance),
                np.nan,
            )

    if return_local_vol:
        finite_count = np.count_nonzero(np.isfinite(local_vol))
        nan_count = np.count_nonzero(np.isnan(local_vol))
        inf_count = np.count_nonzero(np.isinf(local_vol))
        print(
            "Local volatility diagnostics: "
            f"{finite_count}/{local_vol.size} finite, "
            f"{nan_count} NaN, {inf_count} infinite"
        )

    if return_prices and return_local_vol:
        return result, prices, local_vol
    if return_prices:
        return result, prices
    if return_local_vol:
        return result, local_vol
    return result


def check_surface(surface, strikes, times, chunk_rows=128):
    """Check strict inequalities on the sampled grid, counting equality as failure.

    Checks are chunked to limit temporary memory. Passing these checks does not
    establish the inequalities between sampled points.
    """
    strikes = np.asarray(strikes, dtype=float)
    times = np.asarray(times, dtype=float)
    counts = {"increasing in maturity": 0, "decreasing in strike": 0,
              "convex in strike": 0}
    first = {}
    widths = np.diff(strikes)

    def record(name, valid, start):
        bad = ~valid
        count = int(np.count_nonzero(bad))
        counts[name] += count
        if count and name not in first:
            row, col = np.unravel_index(np.argmax(bad), bad.shape)
            first[name] = (start + int(row), int(col))

    for start in range(0, len(times), chunk_rows):
        stop = min(start + chunk_rows, len(times))
        block = surface[start:stop]
        if not np.all(np.isfinite(block)):
            raise ValueError("The generated surface contains non-finite call prices")
        record("decreasing in strike", block[:, 1:] < block[:, :-1], start)
        slopes = np.diff(block, axis=1) / widths[None, :]
        record("convex in strike", slopes[:, 1:] >= slopes[:, :-1], start)
        # Include the comparison crossing each chunk boundary.
        pair_stop = min(stop, len(times) - 1)
        if start < pair_stop:
            record("increasing in maturity",
                   surface[start + 1:pair_stop + 1] > surface[start:pair_stop], start)

    for name, count in counts.items():
        if not count:
            print(f"PASS: strictly {name} on the sampled grid.")
            continue
        row, col = first[name]
        if name == "increasing in maturity":
            location = (f"K={strikes[col]:.10g}, times={times[row:row + 2].tolist()}, "
                        f"prices={surface[row:row + 2, col].tolist()}")
        else:
            size = 3 if name == "convex in strike" else 2
            location = (f"time={times[row]:.10g}, strikes={strikes[col:col + size].tolist()}, "
                        f"prices={surface[row, col:col + size].tolist()}")
            if size == 3:
                local_slopes = np.diff(surface[row, col:col + 3]) / widths[col:col + 2]
                location += f", secant slopes={local_slopes.tolist()}"
        print(f"VIOLATION: strictly {name}: {count:,} failed checks. First: {location}")
    return counts


def plot_interpolation_diagnostics(S0, curves, days_to_expiry, strikes, times, surface,
                                   prices=None):
    """Compare the interpolated surface with linear interpolation in time."""
    if prices is None:
        prices = np.asarray([
            V_K(S0, sigs, nus, *load_c_values(S0, sigs, nus), strikes, type="C")
            for sigs, nus in curves
        ])
    intervals = np.clip(np.searchsorted(days_to_expiry, times, side="right") - 1,
                        0, len(days_to_expiry) - 2)
    u = ((times - days_to_expiry[intervals]) /
         (days_to_expiry[intervals + 1] - days_to_expiry[intervals]))[:, None]
    baseline = (1 - u) * prices[intervals] + u * prices[intervals + 1]
    residual = surface - baseline
    peak_time, peak_strike = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
    limit = max(float(np.max(np.abs(residual))), np.finfo(float).eps)

    heat_fig, heat_ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    heat = heat_ax.pcolormesh(strikes, times, residual, shading="auto",
                             cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    for maturity in days_to_expiry:
        heat_ax.axhline(maturity, color="black", alpha=0.3, linewidth=0.6)
    heat_ax.plot(strikes[peak_strike], times[peak_time], "kx", label="Largest absolute deviation")
    heat_ax.axvline(S0, color="red", linestyle=":", linewidth=1.5,
                    label=f"Spot = {S0:.2f}")
    heat_ax.set(xlabel="Strike K", ylabel="Days to expiry",
                title="Interpolation minus linear baseline (horizontal lines: calibrated maturities)")
    heat_ax.legend()
    heat_fig.colorbar(heat, ax=heat_ax, label="Call-price difference")

    # Rank multiplicative jumps symmetrically for increases and decreases.
    # Subtract logs to avoid overflow/underflow when forming the ratio.
    jumps = [
        (abs(np.log(float(sigs[i])) - np.log(float(sigs[i - 1]))), float(nus[i]), maturity)
        for maturity, (sigs, nus) in enumerate(curves, start=1)
        for i in range(1, len(sigs))
    ]
    selected = [(int(np.argmin(np.abs(strikes - S0))), "Spot")]
    if jumps:
        jump_size, jump_strike, maturity = max(jumps, key=lambda item: item[0])
        selected.append((
            int(np.argmin(np.abs(strikes - jump_strike))),
            f"Largest multiplicative sigma jump (maturity {maturity}, |log ratio|={jump_size:.2f})",
        ))
    selected.append((int(peak_strike), "Largest absolute deviation"))
    slice_fig, axes = plt.subplots(len(selected), 2, figsize=(14, 3.5 * len(selected)),
                                   squeeze=False, sharex=True, constrained_layout=True)
    for row_index, (k, description) in enumerate(selected):
        price_ax, residual_ax = axes[row_index]
        price_ax.plot(times, surface[:, k], label="Interpolated", linewidth=1.5)
        price_ax.plot(times, baseline[:, k], "--", label="Linear baseline", linewidth=1)
        price_ax.scatter(days_to_expiry, prices[:, k], color="black", s=18,
                         label="Calibrated prices", zorder=3)
        price_ax.set(title=f"{description} · grid strike {strikes[k]:.2f}", ylabel="Call price")
        residual_ax.plot(times, residual[:, k], color="tab:purple")
        residual_ax.axhline(0, color="black", linewidth=0.7)
        residual_ax.set(title="Deviation from linear baseline", ylabel="Price difference")
        for maturity in days_to_expiry:
            residual_ax.axvline(maturity, color="black", alpha=0.2, linewidth=0.6)
        for axis in (price_ax, residual_ax):
            axis.grid(alpha=0.2)
        price_ax.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Days to expiry")
    return heat_fig, slice_fig


def _surface_derivative_grids(S0, curves, days_to_expiry, strikes, times):
    """Return C_KK and C_T on the supplied strike-time grid."""
    curve_data = evaluate_curves(S0, curves, strikes)
    _, _, second_derivatives = curve_data
    R, m, A, dT, m_2K, A_2 = _interpolation_terms(
        curve_data, days_to_expiry, return_auxiliary=True
    )
    bounds = np.maximum(-R[:-1].min(axis=1), R[1:].max(axis=1)) * dT
    N = np.maximum(bounds, 4).astype(int) + 1
    intervals = np.clip(
        np.searchsorted(days_to_expiry, times, side="left") - 1,
        0,
        len(days_to_expiry) - 2,
    )
    weight = (
        (times - days_to_expiry[intervals]) / dT[intervals]
    )[:, None]
    exponent = N[intervals, None] - 1
    A_i = A[intervals]
    m_i = m[:-1][intervals]
    m_i1 = m[1:][intervals]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        time_derivative = (
            A_i
            + (m_i - A_i)
            * (1 - weight) ** (exponent - 1)
            * (1 - (exponent + 1) * weight)
            + (A_i - m_i1)
            * weight ** (exponent - 1)
            * (exponent - (exponent + 1) * weight)
        )
        strike_curvature = (
            (1 - weight) * second_derivatives[intervals]
            + weight * second_derivatives[intervals + 1]
            + dT[intervals, None]
            * (m_2K[:-1][intervals] - A_2[intervals])
            * weight * (1 - weight) ** exponent
            + dT[intervals, None]
            * (A_2[intervals] - m_2K[1:][intervals])
            * weight ** exponent * (1 - weight)
        )
    return strike_curvature, time_derivative


def _export_plotly_surface_html(strikes, times, values, title, zlabel,
                                html_path, log_z=False, mesh_density=1):
    """Export a sampled surface as a self-contained interactive Plotly HTML."""
    if go is None:
        raise ImportError(
            "Plotly is required for HTML surface exports. Install it with "
            "`python -m pip install plotly`."
        )

    strikes = np.asarray(strikes, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if not 0 < mesh_density <= 1:
        raise ValueError("mesh_density must be in the interval (0, 1]")
    z_values = values if not log_z else np.where(
        values > 0, np.log10(values), np.nan
    )
    green_yellow_red = [
        [0.0, "#2ca25f"],
        [0.5, "#ffffbf"],
        [1.0, "#d73027"],
    ]
    surface = go.Surface(
        x=strikes,
        y=times,
        z=z_values,
        colorscale=green_yellow_red,
        opacity=1.0,
        showscale=True,
        colorbar={"title": f"log10({zlabel})" if log_z else zlabel},
    )

    # Match the PyVista wireframe density: mesh_density is an area fraction,
    # so each axis uses sqrt(mesh_density) of the surface samples.
    axis_fraction = np.sqrt(mesh_density)
    mesh_strike_count = min(
        len(strikes), max(2, int(np.ceil(len(strikes) * axis_fraction)))
    )
    mesh_time_count = min(
        len(times), max(2, int(np.ceil(len(times) * axis_fraction)))
    )
    mesh_strike_indices = np.linspace(
        0, len(strikes) - 1, mesh_strike_count, dtype=int
    )
    mesh_time_indices = np.linspace(
        0, len(times) - 1, mesh_time_count, dtype=int
    )

    # Plotly has no direct structured-grid wireframe primitive, so create one
    # line trace for all strike-direction and time-direction mesh lines.
    mesh_x, mesh_y, mesh_z = [], [], []
    for time_index in mesh_time_indices:
        mesh_x.extend(strikes.tolist() if mesh_density == 1 else strikes[mesh_strike_indices].tolist())
        mesh_y.extend([times[time_index]] * mesh_strike_count)
        mesh_z.extend(z_values[time_index, mesh_strike_indices].tolist())
        mesh_x.append(None)
        mesh_y.append(None)
        mesh_z.append(None)
    for strike_index in mesh_strike_indices:
        mesh_x.extend([strikes[strike_index]] * mesh_time_count)
        mesh_y.extend(times[mesh_time_indices].tolist())
        mesh_z.extend(z_values[mesh_time_indices, strike_index].tolist())
        mesh_x.append(None)
        mesh_y.append(None)
        mesh_z.append(None)
    wireframe = go.Scatter3d(
        x=mesh_x,
        y=mesh_y,
        z=mesh_z,
        mode="lines",
        line={"color": "black", "width": 1},
        hoverinfo="skip",
        showlegend=False,
    )
    figure = go.Figure(data=[surface, wireframe])
    axis_style = {
        "backgroundcolor": "#101820",
        "gridcolor": "#d9e2ec",
        "zerolinecolor": "#d9e2ec",
        "color": "white",
    }
    figure.update_layout(
        title=title,
        paper_bgcolor="#101820",
        plot_bgcolor="#101820",
        font={"color": "white"},
        scene={
            "xaxis": {"title": "Strike K", **axis_style},
            "yaxis": {"title": "Days to expiry", **axis_style},
            "zaxis": {
                "title": f"log10({zlabel})" if log_z else zlabel,
                **axis_style,
            },
            # These are visual multipliers only; the axis coordinates remain
            # the original strike, time, and value coordinates.
            "aspectmode": "manual",
            "aspectratio": {"x": 2, "y": 2, "z": 1},
        },
    )
    if html_path is not None:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    return figure


def _plotly_surface(strikes, times, values, title, zlabel, cmap="RdYlGn_r",
                    log_z=False, show=True, show_one_at_a_time=True,
                    mesh_density=1, html_path=None):
    """Create and optionally display/export an interactive Plotly surface."""

    strikes = np.asarray(strikes, dtype=float)
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(times), len(strikes)):
        raise ValueError(
            f"Surface values must have shape {(len(times), len(strikes))}; "
            f"got {values.shape}"
        )
    if not 0 < mesh_density <= 1:
        raise ValueError("mesh_density must be in the interval (0, 1]")
    figure = _export_plotly_surface_html(
        strikes, times, values, title, zlabel,
        html_path,
        log_z=log_z,
        mesh_density=mesh_density,
    )
    if show:
        figure.show()
        if show_one_at_a_time:
            figure = None
    return figure


def plot_surface_results(S0, curves, days_to_expiry, strikes, times, surface,
                         maturity_prices=None, local_vol=None,
                         title_suffix="", show=True, plot_full_grid=False,
                         num_strikes_sampled=300, num_times_sampled=300,
                         show_one_at_a_time=True, mesh_density=1,
                         html_output_dir="."):
    """Create surface diagnostics, derivatives, and local-volatility plots.

    By default, every plot uses the supplied full strike-time grid. Set
    ``plot_full_grid=False`` to use the configured sampled grid instead.
    With ``show_one_at_a_time=True``, each figure is displayed and closed
    before the next figure is created.
    """
    def display_and_close(figure):
        if show and show_one_at_a_time:
            plt.figure(figure.number)
            plt.show(block=True)
            plt.close(figure)

    close_after_display = show and show_one_at_a_time

    if plot_full_grid:
        strike_indices = np.arange(len(strikes))
        time_indices = np.arange(len(times))
    else:
        if num_strikes_sampled <= 0 or num_times_sampled <= 0:
            raise ValueError(
                "num_strikes_sampled and num_times_sampled must be positive"
            )
        strike_indices = np.linspace(
            0,
            len(strikes) - 1,
            min(num_strikes_sampled, len(strikes)),
            dtype=int,
        )
        time_indices = np.linspace(
            0,
            len(times) - 1,
            min(num_times_sampled, len(times)),
            dtype=int,
        )
    plot_strikes = strikes[strike_indices]
    plot_times = times[time_indices]
    plot_surface = surface[np.ix_(time_indices, strike_indices)]
    has_local_vol_figures = local_vol is not None

    fig = _plotly_surface(
        plot_strikes,
        plot_times,
        plot_surface,
        f"Call price C(K, T){title_suffix}",
        "Call price C(K, T)",
        cmap="RdYlGn_r",
        show=show,
        show_one_at_a_time=show_one_at_a_time,
        mesh_density=mesh_density,
    )

    strike_curvature, time_derivative = _surface_derivative_grids(
        S0, curves, days_to_expiry, plot_strikes, plot_times
    )
    derivative_figures = []
    for values, title, zlabel, cmap in (
        (strike_curvature, r"Call-price curvature $C_{KK}$", r"$C_{KK}$", "RdYlGn_r"),
        (time_derivative, r"Call-price time derivative $C_T$", r"$C_T$", "RdYlGn_r"),
    ):
        derivative_figures.append(_plotly_surface(
            plot_strikes,
            plot_times,
            values,
            f"{title}{title_suffix}",
            zlabel,
            cmap=cmap,
            show=show,
            show_one_at_a_time=show_one_at_a_time,
            mesh_density=mesh_density,
        ))
    if close_after_display:
        derivative_figures = [None, None]
    del strike_curvature, time_derivative

    local_vol_fig = None
    local_vol_log_fig = None
    if local_vol is not None:
        html_output_dir = Path(html_output_dir).resolve()
        if plot_full_grid:
            linear_strike_indices = strike_indices
        else:
            eligible_strike_indices = np.flatnonzero(
                (strikes >= S0 - 600) & (strikes <= S0 + 600)
            )
            if eligible_strike_indices.size == 0:
                raise ValueError(
                    "The requested strike grid does not contain any points within "
                    "300 of spot."
                )
            sampled_count = min(num_strikes_sampled, eligible_strike_indices.size)
            sampled_positions = np.linspace(
                0, eligible_strike_indices.size - 1, sampled_count, dtype=int
            )
            linear_strike_indices = eligible_strike_indices[sampled_positions]
        linear_strikes = strikes[linear_strike_indices]
        linear_local_vol = local_vol[
            np.ix_(time_indices, linear_strike_indices)
        ]
        if linear_local_vol.shape != (
            len(plot_times), len(linear_strikes)
        ):
            raise ValueError(
                "local_vol must have shape (time, strike); "
                f"got {linear_local_vol.shape}"
            )
        local_vol_fig = _plotly_surface(
            linear_strikes,
            plot_times,
            linear_local_vol,
            f"Dupire local volatility{title_suffix}",
            "Local volatility",
            cmap="RdYlGn_r",
            show=show,
            show_one_at_a_time=show_one_at_a_time,
            mesh_density=mesh_density,
            html_path=html_output_dir / "local_volatility.html",
        )
        del linear_local_vol

        plot_local_vol = local_vol[np.ix_(time_indices, strike_indices)]
        positive_local_vol = np.where(
            plot_local_vol > 0, plot_local_vol, np.nan
        )
        finite_positive = positive_local_vol[np.isfinite(positive_local_vol)]
        if finite_positive.size == 0:
            raise ValueError(
                "Cannot create a logarithmic local-volatility plot: "
                "there are no finite positive local-volatility values."
            )
        local_vol_log_fig = _plotly_surface(
            plot_strikes,
            plot_times,
            positive_local_vol,
            f"Dupire local volatility (log){title_suffix}",
            "Local volatility",
            cmap="RdYlGn_r",
            log_z=True,
            show=show,
            show_one_at_a_time=show_one_at_a_time,
            mesh_density=mesh_density,
            html_path=html_output_dir / "local_volatility_log.html",
        )
        del plot_local_vol, positive_local_vol, finite_positive
    heat_fig, slice_fig = plot_interpolation_diagnostics(
        S0, curves, days_to_expiry, plot_strikes, plot_times, plot_surface,
        prices=None if maturity_prices is None else maturity_prices[:, strike_indices],
    )
    display_and_close(heat_fig)
    display_and_close(slice_fig)
    if close_after_display:
        heat_fig = None
        slice_fig = None
    if show:
        if not show_one_at_a_time:
            plt.show()
    if has_local_vol_figures:
        return (
            fig,
            heat_fig,
            slice_fig,
            derivative_figures[0],
            derivative_figures[1],
            local_vol_fig,
            local_vol_log_fig,
        )
    return fig, heat_fig, slice_fig, derivative_figures[0], derivative_figures[1]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results = pd.read_csv(base_dir / "optimization_results.csv", dtype={"target_date": str})
    violations_path = base_dir / "interpolation_violations.csv"
    violation_columns = [
        "index", "target_date", "increasing_in_maturity_violations",
        "decreasing_in_strike_violations", "convex_in_strike_violations",
    ]
    # Start a fresh report for this run, including a header if all rows pass.
    pd.DataFrame(columns=violation_columns).to_csv(violations_path, index=False)

    for index in [0]:
        row = results.iloc[index]
        target_date = pd.to_datetime(row["target_date"], format="%Y%m%d")
        print(f"\nIndex {index} · {target_date:%Y-%m-%d}", flush=True)
        # Preserve the saved expiry order to match the calibrated curves.
        exdates = json.loads(row["exdates"])
        days_to_expiry = (
            pd.to_datetime(exdates, format="%Y%m%d") - target_date
        ).days.to_numpy()
        sigs_nus_final = [
            (np.asarray(curve["sigs"], dtype=float), np.asarray(curve["nus"], dtype=float))
            for curve in json.loads(row["sigs_nus_compiled"])
        ]
        S0 = float(row["spot"])
        Kbar = float(sigs_nus_final[0][1][-1])
        strikes = np.linspace(1, Kbar - 1, 10000)
        times = np.linspace(days_to_expiry[0], days_to_expiry[-1], 10000)
        surface, maturity_prices, local_vol = get_surface(
            S0,
            sigs_nus_final,
            days_to_expiry,
            strikes,
            times,
            return_prices=True,
            return_local_vol=True,
            use_optimized_N=False
        )
        counts = check_surface(surface, strikes, times)
        if any(counts.values()):
            pd.DataFrame([{
                "index": index,
                "target_date": row["target_date"],
                "increasing_in_maturity_violations": counts["increasing in maturity"],
                "decreasing_in_strike_violations": counts["decreasing in strike"],
                "convex_in_strike_violations": counts["convex in strike"],
            }], columns=violation_columns).to_csv(
                violations_path, mode="a", header=False, index=False,
            )
        plot_surface_results(
            S0,
            sigs_nus_final,
            days_to_expiry,
            strikes,
            times,
            surface,
            maturity_prices,
            local_vol,
            title_suffix=f" · {target_date:%Y-%m-%d} · row {index}",
        )
        # Release the full grid before allocating the next row's surface.
        del surface, maturity_prices, local_vol
