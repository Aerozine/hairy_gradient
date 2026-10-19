
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
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
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


# the rate of slide 73 only involves the Hessian of the quadratic, which is
# constant, so any point gives it
def rho_slide73():
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

    rho = rho_slide73()
    # f1 is quadratic, its minimiser is the solution of A x + b = 0
    A = getObjFHess(np.zeros(2), 1)
    b = np.array([8.0, 7.0])
    fstar_f1 = getObjFVal(np.linalg.solve(A, -b), 1)
    print(f"slide 73: rho = {rho:.6f}   f1: f(x*) = {fstar_f1:.10f}")

    fig, axes = plt.subplots(len(STARTS), 2, figsize=(12, 9))

    for row, x0 in enumerate(STARTS):
        for col, functionID in enumerate((1, 2)):
            ax = axes[row, col]
            anchor = None

            for name, method in methods.items():
                path = trajectory(method, functionID, x0)
                if path is None:
                    print(f"f{functionID} x0={x0} {name}: no convergence, skipped")
                    continue

                f = np.array([getObjFVal(path[:, k], functionID)
                              for k in range(path.shape[1])])
                fstar = fstar_f1 if functionID == 1 else f[-1]
                d = f - fstar
                k = np.arange(len(d))
                keep = d > 0  # the last iterates reach f* exactly, log needs > 0

                ax.semilogy(k[keep], d[keep], label=f"{name} ({len(d)} it.)",
                            linewidth=1.8, markersize=5, **styles[name])
                if name == "Steepest descent" and keep.any():
                    anchor = d[keep][0]
                    # measured ratio over the tail, to compare with rho
                    dd = d[keep]
                    if len(dd) > 4:
                        ratios = dd[1:] / dd[:-1]
                        print(f"   f{functionID} x0={x0} steepest descent: "
                              f"measured ratio (median of the last 5) = "
                              f"{np.median(ratios[-5:]):.4f}")

            # the course gives a rate only for the quadratic
            if functionID == 1 and anchor is not None:
                kk = np.arange(0, 12)
                ax.semilogy(kk, anchor * rho ** kk, color="0.45", linestyle="--",
                            linewidth=1.5, label=f"slide 73: rho = {rho:.3f}")

            ax.set_title(f"f{functionID}, x0 = {x0}")
            ax.set_xlabel("iteration k [-]")
            if col == 0:
                ax.set_ylabel("f(x^k) - f(x*) [-]")
            ax.grid(alpha=0.35, which="both")
            ax.legend(fontsize=10)

    fig.suptitle("Rate of convergence, methods without re-initialization")
    fig.tight_layout()
    fig.savefig("img/rate_of_convergence.pdf", bbox_inches="tight")
    print("saved img/rate_of_convergence.pdf")
