#!/usr/bin/env python3

# Effect of the preconditioner H^0 on BFGS (162).
# The course starts from H^0 = I; any SPD matrix is admissible, and the closer
# H^0 is to the inverse Hessian, the closer the first step is to a Newton step.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from getObjFVal import getObjFGrad
from optimization import BFGS, crit_2_norm, MaxIterError, preconditioners

MaxIter = 300
Epsilon = 1e-5
CRITERION = crit_2_norm
X0 = (10.0, -7.0)  # the starting point of both panels

# each series has its own marker, so the colour never carries the identity alone
styles = {
    "identity": dict(color="green", marker="o"),
    "jacobi": dict(color="red", marker="^"),
    "inverse hessian": dict(color="blue", marker="s"),
}

plt.rcParams.update({
    "savefig.format": "pdf",
})


def run(functionID, x0, precond):
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    try:
        return BFGS(x, 2, functionID, MaxIter, Epsilon, CRITERION,
                    precond=precond, verbose=False)
    except MaxIterError:
        return None


if __name__ == "__main__":

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, functionID in zip(axes, (1, 2)):
        for name in preconditioners:
            path = run(functionID, X0, name)
            if path is None:
                print(f"f{functionID} {name}: MaxIter reached, series dropped")
                continue
            g = np.array([np.linalg.norm(getObjFGrad(path[:, j], functionID))
                          for j in range(path.shape[1])])
            ax.semilogy(np.arange(len(g)), g, linewidth=1.8, markersize=6,
                        label=name, **styles[name])

        # crit_2_norm stops on ||grad||^2 < Epsilon, i.e. ||grad|| < sqrt(Epsilon)
        ax.axhline(np.sqrt(Epsilon), color="0.45", linestyle="--", linewidth=1.2)
        ax.text(0.99, np.sqrt(Epsilon), r" $\sqrt{\epsilon}$ ", color="0.35",
                fontsize=10, va="bottom", ha="right",
                transform=ax.get_yaxis_transform())

        ax.set_xlabel("iteration k [-]")
        ax.set_ylabel(rf"$\|\nabla f_{functionID}(x^k)\|$  [-]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # k is an index
        ax.grid(alpha=0.35, which="both")
        ax.legend(fontsize=10, title=rf"BFGS, $x^0$ = ({X0[0]:g}, {X0[1]:g})")

    fig.tight_layout()
    fig.savefig("img/precond.pdf", bbox_inches="tight")
