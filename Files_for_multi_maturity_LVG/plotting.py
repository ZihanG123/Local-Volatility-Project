import random

import matplotlib.pyplot as plt
import numpy as np

from calibration import load_c_values
from calibration import V_K


def scatter_initial_points(K_array, initial_points, ax=None):
    K_array = np.asarray(K_array)
    initial_points = np.asarray(initial_points)

    if K_array.ndim != 2 or initial_points.ndim != 2:
        raise ValueError("K_array and initial_points must be two-dimensional")
    if K_array.shape != initial_points.shape:
        raise ValueError("K_array and initial_points must have matching shapes")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    for maturity in range(K_array.shape[0]):
        ax.scatter(
            K_array[maturity, :],
            initial_points[maturity, :],
            color="blue",
        )

    ax.set_xlabel("K")
    ax.set_ylabel("Initial point")
    ax.set_title("Initial points by maturity")
    ax.legend()
    ax.grid(True)
    return ax


def plot_V_K(S0, sigs, nus, ax=None, label=None):
    c_v1, c_v2 = load_c_values(S0, sigs, nus)
    K_vals = np.linspace(nus[0], nus[-1], 1000)
    V_vals = V_K(S0, sigs, nus, c_v1, c_v2, K_vals) + np.maximum(S0 - K_vals, 0)
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 8))
    ax.plot(K_vals, V_vals, label=str(label))
    ax.set_xlabel("K")
    ax.set_ylabel("V(K)")
    ax.set_title("Plot of $V(K)$ against $K$")
    ax.legend()
    ax.grid(True)
    if ax is None:
        plt.show()


def plot_sigs(S0, sigs, nus, ax=None, label=None):
    random_color = (random.random(), random.random(), random.random())
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 8))
    first = True
    for i in range(len(sigs)):
        if 250 < nus[i]:
            if first:
                ax.plot(
                    [np.log(nus[i] / S0), np.log(nus[i + 1] / S0)],
                    [sigs[i], sigs[i]],
                    lw=0.5,
                    color=random_color,
                    label=str(label),
                )
                first = False
            else:
                ax.plot(
                    [np.log(nus[i] / S0), np.log(nus[i + 1] / S0)],
                    [sigs[i], sigs[i]],
                    lw=0.5,
                    color=random_color,
                )
            if i != len(sigs) - 1:
                ax.plot(
                    [np.log(nus[i + 1] / S0), np.log(nus[i + 1] / S0)],
                    [sigs[i], sigs[i + 1]],
                    lw=0.5,
                    color=random_color,
                )
    ax.set_xlabel("log(K/S0)")
    ax.set_ylabel("sigma")
    ax.set_title("Square root of local vol against log-moneyness")
    ax.legend()
    ax.grid(True)
    if ax is None:
        plt.show()
