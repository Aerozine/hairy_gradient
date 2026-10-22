import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from getObjFVal import (getObjFVal, getObjFGrad, getObjFHess,
                        reset_counters)
from linear import dichotomy, newton_raphson, quadratic, LineSearchError

TOL = 1e-5
# once a method sits exactly on its limit the distance is 0, which a log axis
# cannot place: the curve simply leaves the bottom of the panel there
BOTTOM = 1e-12

# (functionID, x^0)
# f1 is quadratic: phi is a parabola, the three methods converge to the same
# alpha. f2 is only C1: phi has several local minima and they do NOT.
CASES = [(1, (10.0, -7.0)),
         (2, (12.0, 12.0))]

searches = {
    "dichotomy": dichotomy,
    "newton-raphson": newton_raphson,
    "quadratic": quadratic,
}

styles = {
    "dichotomy": dict(color="blue", marker="o"),
    "newton-raphson": dict(color="red", marker="^"),
    "quadratic": dict(color="green", marker="s"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "savefig.format": "pdf",
})


# the iterates of one method, and what it cost to produce them
def run(search, x0, s, functionID, tol=TOL):
    trace = []
    reset_counters()
    try:
        search(x0, s, functionID, tol, trace=trace)
    except LineSearchError:
        return None, None
    calls = getObjFVal.calls + getObjFGrad.calls + getObjFHess.calls
    # the dichotomy traces its bracket, the other two their current estimate
    iterates = np.array([0.5 * (t[0] + t[1]) if isinstance(t, tuple) else t
                         for t in trace])
    return iterates, calls


if __name__ == "__main__":

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (functionID, start) in zip(axes, CASES):
        x0 = np.array(start, float)
        s = -getObjFGrad(x0, functionID)

        for name, search in searches.items():
            iterates, calls = run(search, x0, s, functionID)
            if iterates is None:
                # NR needs phi'' > 0 and alpha > 0 (119, 114); on f2 it meets
                # neither, so the iteration runs away instead of converging
                ax.plot([], [], linestyle="none",
                        label=f"{name}: diverges", **styles[name])
                continue

            # the three methods do not converge to the same alpha on f2, so
            # each one is measured against ITS OWN limit, obtained by rerunning
            # it at a much tighter tolerance
            alpha_inf = search(x0, s, functionID, TOL * 1e-3)

            err = np.abs(iterates - alpha_inf)
            ax.semilogy(np.arange(len(err)), err, linewidth=1.8, markersize=6,
                        label=f"{name}: {len(err)} iterations, {calls} calls",
                        **styles[name])

        ax.set_ylim(bottom=BOTTOM)
        ax.axhline(TOL, color="0.45", linestyle="--", linewidth=1.2)
        ax.text(0.99, TOL, f" tol = {TOL:g} ", color="0.35", fontsize=9,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())

        ax.set_xlabel("iteration of the line search [-]")
        ax.set_ylabel(rf"$|\alpha_k - \alpha_\infty|$ on $f_{functionID}$"
                      r"  [-]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.35, which="both")
        ax.legend(fontsize=9, title=rf"$x^0$ = ({start[0]:g}, {start[1]:g}), "
                                    rf"$s = -\nabla f_{functionID}(x^0)$")

    fig.tight_layout()
    fig.savefig("img/linesearch.pdf", bbox_inches="tight")
