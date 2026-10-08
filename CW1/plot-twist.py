# because paralellism is fun
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from getObjFVal import *
from optimization import *

# based on plotOptimizationPath
functionID = 2
lb, up = -15, 15
N = 80
MaxIter = 300
Epsilon = 1e-5

# mapping from name to function name basically
methods = {
    "Steepest descent": steepest_descent,
    "Conjugate gradients": conjugate_gradients_Fletcher_Reeves,
    "BFGS": BFGS,
}

metrics = ["Iterations", "getObjFVal calls", "getObjFGrad calls", "getObjFHess calls"]

# same switch as plotOptimizationPath
automatic_contour_levels = True
switch_dict = {
    1: [0, 100, 200, 400, 800, 1200, 1600, 2000, 2800],
    2: np.arange(-20, 250, 10),
}


# on point job take a position and a name
# execute the algo
def run_point(task):
    name, x0 = task
    reset_counters()  # init counter to 0
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    # returns number of iterations and the calls
    nit = methods[name](x, 2, functionID, MaxIter, Epsilon, verbose=False).shape[1]
    return nit, getObjFVal.calls, getObjFGrad.calls, getObjFHess.calls


if __name__ == "__main__":
    xi = np.linspace(lb, up, N)
    # pts is basically all combination possible from xi X xi
    pts = [(a, b) for b in xi for a in xi]
    # for topology contour
    # contour of f, as in plotOptimizationPath
    xc = np.arange(lb, up + 0.02, 0.02)
    X, Y = np.meshgrid(xc, xc)  # f[j,i] = getObjFVal([xc[i], xc[j]])
    F = getObjFVal([X, Y], functionID)

    for name in methods:
        # map is a function that will call run_point on each element of the array , so all the point
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(run_point, [(name, p) for p in pts], chunksize=50))
        # wizard plotting stuff based from older project
        data = np.array(res, float).reshape(N, N, 4)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for k, ax in enumerate(axes.flat):
            im = ax.imshow(
                data[:, :, k],
                origin="lower",
                extent=[lb, up, lb, up],
                aspect="equal",
                cmap="viridis",
            )
            if automatic_contour_levels:
                C = ax.contour(xc, xc, F, cmap="viridis", extend="both")
            else:
                C = ax.contour(
                    xc,
                    xc,
                    F,
                    sorted(switch_dict[functionID]),
                    cmap="viridis",
                    extend="both",
                )
            ax.clabel(C, inline=True, fontsize=8, fmt="%1.1f")
            fig.colorbar(im, ax=ax, shrink=0.85, norm="log")
            ax.set(xlabel="x", ylabel="y", title=metrics[k])
        fig.suptitle(f"{name} on f{functionID}")
        fig.tight_layout()
        fig.savefig(f"img/{functionID}_{name.split()[0]}.pdf")
        print(".", end="")

    plt.show()
#TODO 3D heatmap ? 
