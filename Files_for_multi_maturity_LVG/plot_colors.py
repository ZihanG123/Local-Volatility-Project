"""Shared, ordered colors for maturity curves."""

import numpy as np
from matplotlib import pyplot as plt


def maturity_colors(count):
    """Sample viridis from purple to green, avoiding pale yellow on light backgrounds."""
    return plt.get_cmap("viridis")(np.linspace(0.1, 0.85, count))


def next_curve_color(ax):
    """Use a deterministic ten-color gradient for individually added curves."""
    handles, _ = ax.get_legend_handles_labels()
    return maturity_colors(10)[len(handles) % 10]
