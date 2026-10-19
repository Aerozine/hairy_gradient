
import inspect

import numpy as np
import matplotlib.pyplot as plt

from getObjFVal import getObjFVal, getObjFHess
from optimization import (steepest_descent, conjugate_gradients_Fletcher_Reeves,
                          BFGS, crit_2_norm, MaxIterError)

MaxIter = 2000
STARTS = [(0.0, 0.0), (10.0, -10.0)]

methods = {
    "Steepest descent": steepest_descent,
    "BFGS": BFGS,
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
}

# colours checked against the light background; each series also has its own
# marker, so the colour never carries the identity on its own
styles = {
    "Steepest descent": dict(color="#2a78d6", marker="o"),
    "Conjugate gradients": dict(color="#c2410c", marker="^"),
    "BFGS": dict(color="#117a5e", marker="s"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "savefig.format": "pdf",
})


# cf slide 73
def rho():
    w = np.linalg.eigvalsh(getObjFHess(np.zeros(2), 1))
    e, E = w.min(), w.max()
    return ((E - e) / (E + e)) ** 2


# the pure method, without re-initialization: that is what the course analyses
def run(method, functionID, x0, Epsilon):
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    kw = {"verbose": False}
    if "restart" in inspect.signature(method).parameters:
        kw["restart"] = 0
    return method(x, 2, functionID, MaxIter, Epsilon, crit_2_norm, **kw)


# the tightest tolerance that still converges, to get the longest trajectory
def trajectory(method, functionID, x0):
    for Epsilon in (1e-12, 1e-10, 1e-8, 1e-5):
        try:
            return run(method, functionID, x0, Epsilon)
        except MaxIterError:
            continue
    return None


if __name__ == "__main__":

    rho = rho()
    # f1 is quadratic, its minimiser is the solution of A x + b = 0
    A = getObjFHess(np.zeros(2), 1)
    b = np.array([8.0, 7.0])
    fstar_f1 = getObjFVal(np.linalg.solve(A, -b), 1)

    fig, axes = plt.subplots(len(STARTS), 2, figsize=(12, 9))

    for row, x0 in enumerate(STARTS):
        for col, functionID in enumerate((1, 2)):
            ax = axes[row, col]
            anchor = None

            # gathered first: the floor of the panel depends on every series
            series = []
            for name, method in methods.items():
                path = trajectory(method, functionID, x0)
                if path is None:
                    continue
                f = np.array([getObjFVal(path[:, j], functionID)
                              for j in range(path.shape[1])])
                fstar = fstar_f1 if functionID == 1 else f[-1]
                series.append((name, f - fstar))

            # an iterate landing on f(x*) gives d = 0, and a log axis has no
            # place for it. rather than dropping it, which hides the very
            # result, it is drawn as a star on a floor below every real value.
            positive = [d[d > 0] for _, d in series if (d > 0).any()]
            floor = np.concatenate(positive).min() / 30 if positive else 1e-16
            exact = False

            for name, d in series:
                k = np.arange(len(d))
                keep = d > 0

                ax.semilogy(k[keep], d[keep], label=f"{name} ({len(d)} it.)",
                            linewidth=1.8, markersize=5, **styles[name])
                done = ~keep
                if done.any():
                    exact = True
                    if keep.any():  # dotted drop from the last real value
                        ax.semilogy([k[keep][-1], k[done][0]],
                                    [d[keep][-1], floor], linestyle=":",
                                    linewidth=1.2, color=styles[name]["color"])
                    ax.semilogy(k[done], np.full(done.sum(), floor),
                                linestyle="none", marker="*", markersize=13,
                                color=styles[name]["color"])
                if name == "Steepest descent" and keep.any():
                    anchor = d[keep][0]

            ax.set_ylim(bottom=floor / 4)

            # the course gives a rate only for the quadratic
            if functionID == 1 and anchor is not None:
                kk = np.arange(0, 12)
                ax.semilogy(kk, anchor * rho ** kk, color="0.45", linestyle="--",
                            linewidth=1.5, label=rf"$\rho = {rho:.3f}$")

            ax.set_title(rf"$f_{functionID}$, $x^0$ = ({x0[0]:g}, {x0[1]:g})")
            ax.set_xlabel("iteration k [-]")
            if col == 0:
                ax.set_ylabel(r"$f(x^k) - f(x^*)$  [-]")
            ax.grid(alpha=0.35, which="both")

            handles, labels = ax.get_legend_handles_labels()
            if exact:
                handles.append(plt.Line2D([], [], linestyle="none", marker="*",
                                          color="0.35", markersize=13))
                labels.append(r"$f(x^k) = f(x^*)$ exactly")
            ax.legend(handles, labels, fontsize=9)

    fig.suptitle("Rate of convergence, methods without re-initialization\n"
                 r"(a star marks an iterate that reached $f(x^*)$ exactly, so "
                 r"$f(x^k) - f(x^*) = 0$ and the log axis has no place for it)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("img/rate_of_convergence.pdf", bbox_inches="tight")
