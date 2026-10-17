#!/usr/bin/env python3

import functools

import numpy as np
import matplotlib.pyplot as plt

import linear
import optimization
from getObjFVal import getObjFVal, getObjFGrad, reset_counters
from optimization import (conjugate_gradients_Fletcher_Reeves, BFGS,
                          crit_2_norm, MaxIterError)

# the default line search of linear.py is newton-raphson, which has no iteration
# cap and can return a negative alpha on f2 -> pin an explicit one here so that
# the figure measures the restart and not the line search
LINE_SEARCH = "dichotomy"
optimization.linear_search = functools.partial(linear.linear_search, method=LINE_SEARCH)

MaxIter = 300
Epsilon = 1e-5
N = 8  # N x N grid of starting points in the domain [-15, 15]^2
RESTARTS = [1, 2, 3, 4, 5, 6, 8, 10]  # p = 1 -> steepest descent for CG
CRITERION = crit_2_norm

methods = {
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
}

# categorical slots, fixed order, validated for colour-vision deficiency
styles = {
    "getObjFVal": dict(color="blue", marker="o"),
    "getObjFGrad": dict(color="red", marker="s"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.figsize": (14, 11),
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.linewidth": 0.6,
    "savefig.format": "pdf",
})


# median number of calls over the grid of starting points, for one setting
def sweep(method, functionID, restart):
    xi = np.linspace(-14, 14, N)
    fcalls, gcalls, failed = [], [], 0

    for x0 in ((a, b) for a in xi for b in xi):
        reset_counters()
        x = np.zeros((2, MaxIter))
        x[:, 0] = x0
        try:
            method(x, 2, functionID, MaxIter, Epsilon,
                   CRITERION, restart=restart, verbose=False)
        except MaxIterError:  # stopped at MaxIter, not a converged result
            failed += 1
            continue
        fcalls.append(getObjFVal.calls)
        gcalls.append(getObjFGrad.calls)

    # a setting is only reported when EVERY starting point converged: a median
    # taken over the subset that did converge would be measured on an easier
    # problem than the other points, so it is dropped instead of plotted
    if failed or not fcalls:
        return np.nan, np.nan, failed
    return np.median(fcalls), np.median(gcalls), failed


if __name__ == "__main__":

    fig, axes = plt.subplots(2, 2, sharex=True)

    for row, functionID in enumerate((1, 2)):
        for col, name in enumerate(methods):
            ax = axes[row, col]
            method = methods[name]

            # reference: the method as it is, without re-initialization
            base_f, base_g, base_fail = sweep(method, functionID, 0)

            # then every restart period
            data = [sweep(method, functionID, p) for p in RESTARTS]

            for key, values, base in (
                ("getObjFVal", [d[0] for d in data], base_f),
                ("getObjFGrad", [d[1] for d in data], base_g),
            ):
                ax.plot(RESTARTS, values, label=f"{key} calls", **styles[key])
                if np.isfinite(base):  # no baseline when it did not converge
                    ax.axhline(base, color=styles[key]["color"], linestyle="--",
                               linewidth=1.5, alpha=0.7)

            ax.set_yscale("log")
            ax.set_title(f"{name} on f{functionID}")
            ax.grid(True, which="both", alpha=0.35)
            if row == 1:
                ax.set_xlabel("re-initialization period p [iterations]")
            if col == 0:
                ax.set_ylabel("calls, median over starting points [-]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [],linestyle="--", linewidth=1.5))
    labels.append("no re-initialization")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Effect of the periodic re-initialization (line search: {LINE_SEARCH})")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig("img/reinit.pdf", bbox_inches="tight")
