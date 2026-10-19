# because paralellism is fun
import inspect
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter

from getObjFVal import *
from linear import linear_search
from optimization import *

# based on plotOptimizationPath
FUNCTIONS = (1, 2)
lb, up = -15, 15
N = 80  # N x N starting points
MaxIter = 300
Epsilon = 1e-5
CRITERION = crit_2_norm

metrics = ["Iterations", "getObjFGrad calls"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "savefig.format": "pdf",
})


# the methods are called without naming restart, so each one runs with its own
# default: say which, the figure compares them side by side
def reinit_label(method):
    p = inspect.signature(method).parameters.get("restart")
    if p is None:
        return "no re-initialization"
    return f"re-init every {p.default} it." if p.default else "no re-initialization"


# one point job: take a position, a function and a name, execute the algo
def run_point(task):
    name, functionID, x0 = task
    reset_counters()  # init counter to 0
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    try:
        nit = methods[name](
            x, 2, functionID, MaxIter, Epsilon, CRITERION, verbose=False
        ).shape[1]
    except MaxIterError:
        return np.nan, np.nan  # did not converge, drawn in red on the map
    return nit, getObjFGrad.calls


if __name__ == "__main__":
    xi = np.linspace(lb, up, N)
    # pts is basically all combination possible from xi X xi
    pts = [(a, b) for b in xi for a in xi]
    # contour of f, as in plotOptimizationPath (0.1 is enough to draw it)
    xc = np.arange(lb, up + 0.1, 0.1)
    X, Y = np.meshgrid(xc, xc)  # F[j,i] = getObjFVal([xc[i], xc[j]])

    cmap = plt.get_cmap("viridis").copy()
    # starting points that never converged. red is absent from viridis, so it
    # stays distinct from every step of the ramp (worst pair dE 11.8 in deutan)
    cmap.set_bad("#d03b3b")

    for functionID in FUNCTIONS:
        F = getObjFVal([X, Y], functionID)

        data = {}
        for name in methods:
            with ProcessPoolExecutor() as ex:
                res = list(ex.map(run_point,
                                  [(name, functionID, p) for p in pts],
                                  chunksize=50))
            data[name] = np.array(res, float).reshape(N, N, len(metrics))
            bad = np.isnan(data[name][:, :, 0]).sum()
            print(f"f{functionID} {name:20s} done, {bad}/{N * N} hit MaxIter",
                  flush=True)

        # one row per metric, one column per method
        fig, axes = plt.subplots(len(metrics), len(methods),
                                 figsize=(5.4 * len(methods), 5.0 * len(metrics)),
                                 sharex=True, sharey=True)

        for row in range(len(metrics)):
            for col, name in enumerate(methods):
                ax = axes[row, col]
                v = data[name][:, :, row]
                vmin, vmax = np.nanmin(v), np.nanmax(v)
                # a scale of its own per panel: a scale shared with steepest
                # descent flattens the other two into a single colour
                norm = LogNorm(vmin=max(vmin, 1), vmax=vmax) if vmax > vmin else None
                ax.imshow(v, origin="lower", extent=[lb, up, lb, up],
                          aspect="equal", cmap=cmap, norm=norm)
                # contour lines of f, in a neutral colour so that they stay
                # readable on the heat map (a viridis contour on viridis is not)
                C = ax.contour(xc, xc, F, 8, colors="white",
                               linewidths=0.6, alpha=0.55)
                ax.clabel(C, inline=True, fontsize=7, fmt="%.0f")

                if row == 0:
                    ax.set_title(f"{name}\n{reinit_label(methods[name])}")
                if row == len(metrics) - 1:
                    ax.set_xlabel("x")
                if col == 0:
                    ax.set_ylabel("y")
                    ax.annotate(metrics[row], xy=(-0.28, 0.5),
                                xycoords="axes fraction", rotation=90,
                                ha="center", va="center", fontsize=15)

                # a few log-spaced integer ticks, otherwise the default log
                # locator writes every minor tick in scientific notation
                ticks = np.unique(np.round(
                    np.geomspace(max(vmin, 1), max(vmax, 2), 5)).astype(int))
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                  ax=ax, shrink=0.82, pad=0.03,
                                  ticks=ticks, format=ScalarFormatter())
                cb.ax.yaxis.set_minor_formatter(NullFormatter())
                cb.ax.minorticks_off()

        fig.suptitle(f"Cost against the starting point, f{functionID} "
                     f"(red = MaxIter reached)")
        fig.tight_layout()
        fig.savefig(f"img/{functionID}_heatmap.pdf", bbox_inches="tight")
        print(f"saved img/{functionID}_heatmap.pdf", flush=True)
