import numpy as np
import matplotlib.pyplot as plt

import optimization
from getObjFVal import getObjFVal, getObjFGrad, reset_counters
from optimization import steepest_descent, crit_2_norm, MaxIterError

TOL = 1e-5
MaxIter = 300
lb, up = -15, 15  # the domain of the statement
N = 8  # N x N grid of starting points in [lb, up]^2

HS = np.logspace(-3, 1, 17)  # 1e-3 ... 10
# rho = 0 gives a = a_max and rho = 1 gives a = a_min: the interval is never
# cut and the loop does not end, so only the open interval is swept
RHOS = np.linspace(0.05, 0.95, 19)

H_DEFAULT, RHO_DEFAULT = 0.1, 0.5  # what linear.py uses

# two functions, two fixed hues + a marker each, so the identity never rests
# on the colour alone
fstyles = {
    1: dict(color="blue", marker="o"),
    2: dict(color="red", marker="^"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "savefig.format": "pdf",
})


# the dichotomy works on phi' and the quadratic interpolation on phi, so they
# do not spend their calls on the same function: both counters are returned
# and each panel reads the one that carries its cost
def run_method(functionID, h, rho, search="dichotomy"):
    base = optimization.linear_search
    #rebranding basically
    optimization.linear_search = (
        lambda x, s, fid, tol, *a, **k: base(x, s, fid, tol, h, rho,
                                             method=search)
    )
    xi = np.linspace(lb, up, N)
    grad, fval, failed = [], [], 0
    try:
        for x0 in ((a, b) for a in xi for b in xi):
            reset_counters()
            x = np.zeros((2, MaxIter))
            x[:, 0] = x0
            try:
                steepest_descent(x, 2, functionID, MaxIter, TOL, crit_2_norm,
                                 verbose=False)
            except MaxIterError:  # stopped at MaxIter, not a converged result
                failed += 1
                continue
            grad.append(getObjFGrad.calls)
            fval.append(getObjFVal.calls)
    finally:
        optimization.linear_search = base

    # a setting is only reported when EVERY starting point converged, otherwise
    # the median would be taken on an easier problem than the other settings
    if failed or not grad:
        return np.nan, np.nan, failed
    return float(np.median(grad)), float(np.median(fval)), failed


# cost of the whole descent for each value of the swept parameter
def sweep(functionID, values, fixed, vary, search="dichotomy", counter="grad"):
    cost, dropped = [], 0
    for v in values:
        h, rho = (v, fixed) if vary == "h" else (fixed, v)
        med_grad, med_fval, fails = run_method(functionID, h, rho, search)
        if fails:
            dropped += 1
            print(f"f{functionID} {search}: {vary}={v:g} -> {fails}/{N * N} "
                  f"hit MaxIter, point dropped")
        cost.append(med_grad if counter == "grad" else med_fval)
    return cost, dropped


if __name__ == "__main__":

    fig, (ax_rho, ax_h) = plt.subplots(1, 2, figsize=(13, 5.5))

    # the cut rho, at the h of linear.py
    for functionID in (1, 2):
        # a setting that did not converge from every start leaves a gap in
        # the curve, and is named on the console
        cost, dropped = sweep(functionID, RHOS, H_DEFAULT, "rho")
        ax_rho.plot(RHOS, cost, linewidth=1.8, markersize=6,
                    label=rf"$f_{functionID}$", **fstyles[functionID])

    ax_rho.axvline(RHO_DEFAULT, color="0.45", linestyle="--", linewidth=1.3,
                   label=rf"default($\rho$ = {RHO_DEFAULT:g})")
    ax_rho.set_xlabel(r"cut $\rho$ [-]")
    ax_rho.set_ylabel(r"$\nabla F$ calls of the whole descent [-]")
    ax_rho.grid(alpha=0.35)
    ax_rho.legend(fontsize=10, title="dichotomy")

    # the first interval h, at the rho of linear.py
    for functionID in (1, 2):
        cost, dropped = sweep(functionID, HS, RHO_DEFAULT, "h")
        ax_h.semilogx(HS, cost, linewidth=1.8, markersize=6,
                      label=rf"$f_{functionID}$", **fstyles[functionID])

    ax_h.axvline(H_DEFAULT, color="0.45", linestyle="--", linewidth=1.3,
                 label=f"default (h = {H_DEFAULT:g})")
    ax_h.set_xlabel("first interval h [-]")
    ax_h.set_ylabel(r"$\nabla F$ calls of the whole descent [-]")
    ax_h.grid(alpha=0.35, which="both")
    ax_h.legend(fontsize=10, title="dichotomy")

    fig.tight_layout()
    fig.savefig("img/rhoh.pdf", bbox_inches="tight")

    # the same first interval h for the quadratic interpolation (128-133),
    # which has no rho: it reads phi and not phi', so its cost is in the F calls
    fig, ax_quad = plt.subplots(figsize=(7, 5.5))
    for functionID in (1, 2):
        cost, dropped = sweep(functionID, HS, RHO_DEFAULT, "h",
                              search="quadratic", counter="fval")
        ax_quad.semilogx(HS, cost, linewidth=1.8, markersize=6,
                         label=rf"$f_{functionID}$", **fstyles[functionID])

    ax_quad.axvline(H_DEFAULT, color="0.45", linestyle="--", linewidth=1.3,
                    label="default")
    ax_quad.set_xlabel("first interval h [-]")
    ax_quad.set_ylabel(r"$F$ calls of the whole descent [-]")
    ax_quad.grid(alpha=0.35, which="both")
    ax_quad.legend(fontsize=10, title="quadratic interpolation")

    fig.tight_layout()
    fig.savefig("img/quadratich.pdf", bbox_inches="tight")
