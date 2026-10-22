#!/usr/bin/env python3

import inspect

import numpy as np
import matplotlib.pyplot as plt

from getObjFVal import getObjFGrad, reset_counters
from linear import linear_search
from matplotlib.ticker import ScalarFormatter
from optimization import (conjugate_gradients_Fletcher_Reeves, BFGS,
                          crit_2_norm, MaxIterError)

# the methods call linear_search without naming a method, so the figure is made
# with whichever one linear.py has as its default. read it, only to label it.
LINE_SEARCH = inspect.signature(linear_search).parameters["method"].default

MaxIter = 300
Epsilon = 1e-5
lb, up = -15, 15  # the domain of the statement
N = 8  # N x N grid of starting points in [lb, up]^2
# p = 0 -> no re-initialization. p = 1 is left out: it re-initializes at every
# iteration, which turns CG back into steepest descent, and its cost is so far
# above the others that it flattens the variation between the values that matter
RESTARTS = [0, 2, 3, 4, 5, 6, 8, 10]
CRITERION = crit_2_norm

methods = {
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
}

# only getObjFGrad is counted: crit_2_norm (61) and the dichotomy (125-127) use
# the gradient only, so the getObjFVal count was entirely produced by the back
# search of linear.py, which is not part of the course
style = dict(color="blue", marker="s")

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


# median, min and max number of gradient calls over the grid of starting points
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

    # a setting is only reported when EVERY starting point converged: a median
    # taken over the subset that did converge would be measured on an easier
    # problem than the other points, so it is dropped instead of plotted
    if failed or not gcalls:
        return np.nan, np.nan, np.nan, failed
    g = np.array(gcalls, float)
    return np.median(g), g.min(), g.max(), failed


if __name__ == "__main__":

    print(f"line search: {LINE_SEARCH}, p = 0 is the method without "
          f"re-initialization")

    fig, axes = plt.subplots(2, 2, sharex=True)

    for row, functionID in enumerate((1, 2)):
        for col, name in enumerate(methods):
            ax = axes[row, col]
            method = methods[name]

            # p = 0 is the method without re-initialization, so it is simply
            # the first point of the curve rather than a separate reference
            med, lo, hi, fails = (np.array(v, float) for v in
                                   zip(*[sweep(method, functionID, p)
                                         for p in RESTARTS]))

            for p, f in zip(RESTARTS, fails):
                if f:
                    print(f"f{functionID} {name}: p={p:.0f} -> {f:.0f}/{N * N} "
                          f"starting points hit MaxIter, point dropped")

            # the bar spans the cheapest and the dearest starting point, the
            # marker is the median over all of them. the scale is logarithmic:
            # the spread is wide enough to flatten the variation of the median
            # on a linear axis, which is what the figure is about
            ax.errorbar(RESTARTS, med, yerr=[med - lo, hi - med], capsize=4,
                        label="median, bar = min to max", **style)
            ax.set_title(name)
            ax.set_yscale("log")
            # a count of calls reads as 40, 50, 100, not as 4 x 10^1
            for axis in (ax.yaxis.set_major_formatter,
                         ax.yaxis.set_minor_formatter):
                fmt = ScalarFormatter()
                fmt.set_scientific(False)
                axis(fmt)

            ax.grid(True, alpha=0.35, which="both")
            ax.set_xticks(RESTARTS)
            if row == 1:
                ax.set_xlabel("re-initialization period p [iterations]")
            if col == 0:
                # the function is named here: the legend is shared by the four
                # panels and cannot carry it
                ax.set_ylabel(rf"$\nabla F$ calls on $f_{functionID}$  [-]")

    # the four panels draw the same thing, so they share one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center",
               title=f"{N}x{N} starting points")
    fig.savefig("img/reinit.pdf", bbox_inches="tight")
