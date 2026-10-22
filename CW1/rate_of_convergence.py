
import inspect

import numpy as np
import matplotlib.pyplot as plt

from getObjFVal import getObjFVal, getObjFGrad, getObjFHess
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


# f(x*) of the basin the method actually reached.
# taking f[-1] instead makes the last point of the curve exactly 0 BY
# DEFINITION, which a log axis cannot show and which says nothing about the
# convergence; and each method would be measured against its own last iterate.
# Newton on grad f = 0 (125) starts from that iterate and converges in a couple
# of steps, so x* is obtained to machine precision instead.
def polish(x0, functionID, MaxStep=50, tol=1e-14):
    x = np.array(x0, float)
    for _ in range(MaxStep):
        g = getObjFGrad(x, functionID)
        if np.linalg.norm(g) < tol:
            break
        try:
            step = np.linalg.solve(getObjFHess(x, functionID), g)
        except np.linalg.LinAlgError:  # singular Hessian, keep the last x
            break
        x -= step
    return x


# f2 has several local minima, so x* is the one of the basin reached, and two
# methods started at the same point may well not share it
def fstar_of(path, functionID):
    x_last = path[:, -1]
    x_star = polish(x_last, functionID)
    moved = np.linalg.norm(x_star - x_last)
    g = np.linalg.norm(getObjFGrad(x_star, functionID))
    # the polish has to stay in the basin: a jump means the Newton step left it
    # and f(x*) would belong to another minimum
    if not np.isfinite(x_star).all() or moved > 1.0 or g > 1e-8:
        print(f"   f{functionID}: polish moved {moved:.2e} and left "
              f"||grad|| = {g:.2e}, falling back on the last iterate")
        return getObjFVal(x_last, functionID), x_last
    return getObjFVal(x_star, functionID), x_star


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
    # the panels show f(x^k) itself, so f(x*) is reported here rather than
    # subtracted. on f1 it is exact (-353/44), on f2 it is the minimum of the
    # basin each method reached, polished by Newton
    print(f"f1: f* = {fstar_f1:.10f}, theoretical rate rho = {rho:.4f} (73)")

    fig, axes = plt.subplots(len(STARTS), 2, figsize=(12, 9))

    for row, x0 in enumerate(STARTS):
        for col, functionID in enumerate((1, 2)):
            ax = axes[row, col]

            for name, method in methods.items():
                path = trajectory(method, functionID, x0)
                if path is None:
                    continue
                # f(x^k) itself cannot go on a log axis: it is negative
                # around the minima of both functions. the norm of the
                # gradient is positive, needs no reference value, and is what
                # the stopping criterion reads, so the decay shows as a line
                g = np.array([np.linalg.norm(getObjFGrad(path[:, j], functionID))
                              for j in range(path.shape[1])])
                ax.semilogy(np.arange(len(g)), g, label=f"{name} ({len(g)} it.)",
                            linewidth=1.8, markersize=5, **styles[name])
                if functionID == 2:
                    fstar, x_star = fstar_of(path, functionID)
                    print(f"   f2 {name}, x0 = {x0}: x* = ({x_star[0]:.6f}, "
                          f"{x_star[1]:.6f}), f* = {fstar:.10f}")

            ax.set_xlabel("iteration k [-]")
            ax.set_ylabel(rf"$\|\nabla f_{functionID}(x^k)\|$  [-]")
            ax.grid(alpha=0.35, which="both")
            ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("img/rate_of_convergence.pdf", bbox_inches="tight")
