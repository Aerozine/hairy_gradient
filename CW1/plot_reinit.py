#!/usr/bin/env python3

import inspect

import numpy as np
import matplotlib.pyplot as plt

from getObjFVal import getObjFGrad, reset_counters
from linear import linear_search
from optimization import (conjugate_gradients_Fletcher_Reeves, BFGS,
                          crit_2_norm, MaxIterError)

# the methods call linear_search without naming a method, so the figure is made
# with whichever one linear.py has as its default. read it, only to label it.
LINE_SEARCH = inspect.signature(linear_search).parameters["method"].default

MaxIter = 300
Epsilon = 1e-5
lb, up = -15, 15  # the domain of the statement
N = 8  # N x N grid of starting points in [lb, up]^2
RESTARTS = [1, 2, 3, 4, 5, 6, 8, 10]  # p = 1 -> steepest descent for CG
CRITERION = crit_2_norm

methods = {
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
}

# only getObjFGrad is counted: crit_2_norm (61) and the dichotomy (125-127) use
# the gradient only, so the getObjFVal count was entirely produced by the back
# search of linear.py, which is not part of the course
style = dict(color="red", marker="s")

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


# mean, min and max number of gradient calls over the grid of starting points
def sweep(method, functionID, restart):
    xi = np.linspace(lb, up, N)
    gcalls, failed = [], 0

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
        gcalls.append(getObjFGrad.calls)

    # a setting is only reported when EVERY starting point converged: a mean
    # taken over the subset that did converge would be measured on an easier
    # problem than the other points, so it is dropped instead of plotted
    if failed or not gcalls:
        return np.nan, np.nan, np.nan, failed
    g = np.array(gcalls, float)
    return g.mean(), g.min(), g.max(), failed


if __name__ == "__main__":

    fig, axes = plt.subplots(2, 2, sharex=True)

    for row, functionID in enumerate((1, 2)):
        for col, name in enumerate(methods):
            ax = axes[row, col]
            method = methods[name]

            # reference: the method as it is, without re-initialization
            base, base_lo, base_hi, base_fail = sweep(method, functionID, 0)

            # then every restart period
            mean, lo, hi, fails = (np.array(v, float) for v in
                                   zip(*[sweep(method, functionID, p)
                                         for p in RESTARTS]))

            # the bar spans the cheapest and the dearest starting point, the
            # marker is the mean over all of them
            ax.errorbar(RESTARTS, mean, yerr=[mean - lo, hi - mean],
                        capsize=4, label="with re-initialization", **style)
            if np.isfinite(base):  # dropped when some starting point diverged
                ax.axhspan(base_lo, base_hi, color=style["color"], alpha=0.10)
                ax.axhline(base, color=style["color"], linestyle="--",
                           linewidth=1.5, alpha=0.7, label="no re-initialization")
            else:
                print(f"f{functionID} {name}: no restart -> "
                      f"{base_fail}/{N * N} starting points hit MaxIter, point dropped")

            ax.set_title(f"{name} on f{functionID}")
            ax.grid(True, alpha=0.35)
            if row == 1:
                ax.set_xlabel("re-initialization period p [iterations]")
            if col == 0:
                ax.set_ylabel("getObjFGrad calls [-]")

    handles = [plt.Line2D([], [], **style), 
               plt.Line2D([], [], color=style["color"], linestyle="--",
                          linewidth=1.5, alpha=0.7)]
    fig.legend(handles, ["with re-initialization", "no re-initialization"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Effect of the periodic re-initialization "
                 f"(line search: {LINE_SEARCH})\n"
                 f"marker = mean over the {N}x{N} starting points, "
                 f"bar = min to max")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig("img/reinit.pdf", bbox_inches="tight")
