# because paralellism is fun
import inspect
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from getObjFVal import *
from linear import linear_search
from optimization import *

# based on plotOptimizationPath
FUNCTIONS = (1, 2)        # one row per function
lb, up = -15, 15
N = 80                    # N x N starting points, ALWAYS over [lb, up]^2 (the zoom never changes it)
MaxIter = 300
Epsilon = 1e-5
CRITERION = crit_2_norm
OUT_FILE = "img/heatmaps.pdf"

FIND_ONE_IT = False       # True: print the grid points giving exactly 1 iteration

# starting points whose path is drawn on each map: (functionID, method)
PATH_STARTS = {
    (1, "Steepest descent"):    [(-8, -12.5), (-2.5, -14)],
    (1, "Conjugate gradients"): [(-15, 5.886), (-15, 5)],
    (1, "BFGS"):                [(-15, 5.886), (-15, 5)],
    (2, "Steepest descent"):    [(4, 12), (-1.5, 12.5)],
    (2, "Conjugate gradients"): [(4, 12), (-1.5, 12.5)],
    (2, "BFGS"):                [(4, 12), (-1.5, 12.5)],
}
PATH_COLORS = ["red", "white"]
TEXT_COLORS = ["red", "black"]

# zone shown for each map: (xmin, xmax, ymin, ymax). It only crops the map.
PLOT_LIMITS = {
    (1, "Steepest descent"):    (-15, 0, -15, 0),
    (1, "Conjugate gradients"): (-15, 0, -5, 10),
    (1, "BFGS"):                (-15, 0, -5, 10),
    (2, "Steepest descent"):    (-5.25, 9.25, -2, 13),
    (2, "Conjugate gradients"): (-5.25, 9.25, -2, 13),
    (2, "BFGS"):                (-5.25, 9.25, -2, 13),
}

MARKER_SIZE = 5           # numbered balls along the paths
MARKER_FONT = 4

SMALL_AFTER = {
    (1, "Steepest descent"):    [1, 7],
    (1, "Conjugate gradients"): [1, 2],
    (1, "BFGS"):                [1, 2],
    # grand (0-2) -> petit (3-7) -> grand (8-11) -> petit (12-fin)
    (2, "Steepest descent"):    [3, [(4, 13), (16, None)]],
    (2, "Conjugate gradients"): [4, 9],
    (2, "BFGS"):                [2, 6],
}

SMALL_MARKER_SIZE = 2      
ZOOM_SMALL_MARKER_SIZE = 4


# ---- magnifier (zoom inset) -----------------------------------------------
# zone to magnify on each map: (xmin, xmax, ymin, ymax), in data coordinates,
# same convention as PLOT_LIMITS. A map with no entry (or None) gets no magnifier.
# The values below are PLACEHOLDERS: set them to the overlapping zones you care about.
ZOOM_BOXES = {
    (1, "Steepest descent"):    (-1.9, -1.6, -2.55, -2.2),
    (1, "Conjugate gradients"): (-3, -1, -3.5, -1.5),
    (1, "BFGS"):                (-3, -1, -3.5, -1.5),
    (2, "Steepest descent"):    (-0.55, -0.475, 1.66, 1.72),
    (2, "Conjugate gradients"): (-0.64, -0.5, 1.58, 1.72),
    (2, "BFGS"):                (-0.85, -0.45, 1.5, 1.9),
}
ZOOM_ENABLED = True
ZOOM_DEFAULT_LOC = "lower right"   # where the magnifier sits inside the map
# per-map override of the corner: "upper left", "upper right", "lower left", "lower right"
ZOOM_LOCS = {
    (1, "Steepest descent"):    "upper left",
    (1, "Conjugate gradients"): "upper right",
    (1, "BFGS"):                "upper right",
    (2, "Steepest descent"):    "lower right",
    (2, "Conjugate gradients"): "lower right",
    (2, "BFGS"):                "lower right",
}
ZOOM_SIZE = 0.42          # magnifier width, as a fraction of the map width
ZOOM_MAX_H = 0.60         # magnifier height can't exceed this fraction of the map height
ZOOM_PAD = 0.03           # gap between the magnifier and the map border (axes fraction)
ZOOM_EDGE = "black"       # colour of the rectangle, the connectors and the magnifier frame
ZOOM_MARKER_SIZE = 9      # balls inside the magnifier (every iteration is numbered there)
ZOOM_MARKER_FONT = 6

# ---- layout, all in inches -------------------------------------------------
CELL_W, CELL_H = 4.0, 3.75   # one cell = map + its colorbar
GAP_H = 1.10                 # between two columns: colorbar labels + next y tick labels
LEFT_IN = 1.25               # left margin: row label + "y" label + y tick labels
RIGHT_IN = 0.70              # right margin: last colorbar ticks + "Iterations"
TOP_IN = 1.00                # column titles (2 lines)
CAPTION_OFFSET_IN = 0.75     # bottom of the map -> first caption (clears ticks + "x" label)
CAPTION_LINE_IN = 0.22       # spacing between two captions
ROW_LABEL_OFFSET_IN = 1.25   # left edge of the map -> centre of the vertical f_i label

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "savefig.format": "pdf",
})

def small_after_for(functionID, name, n_paths):
    """Liste des n (un par chemin), complétée par None si besoin."""
    s = SMALL_AFTER.get((functionID, name))
    if s is None or isinstance(s, int):
        return [s] * n_paths
    return list(s) + [None] * (n_paths - len(s))

def is_small(i, spec):
    """spec : None | int (petit dès n) | liste d'intervalles [(a, b), ...] (b=None : jusqu'à la fin)."""
    if spec is None:
        return False
    if isinstance(spec, (int, np.integer)):
        return i >= spec
    return any(a <= i and (b is None or i < b) for a, b in spec)

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
        ).shape[1] - 1
    except MaxIterError:
        return np.nan  # did not converge, drawn in red on the map
    return nit


def get_path(name, functionID, x0):
    x = np.zeros((2, MaxIter))
    x[:, 0] = x0
    try:
        return methods[name](x, 2, functionID, MaxIter, Epsilon, CRITERION, verbose=False)
    except MaxIterError:
        print(f"f{functionID} {name}: no path to draw from {x0}, MaxIter reached", flush=True)
        return None


def draw_paths(ax, paths, marker_size, marker_font, box=None,
               small_after=None, small_size=2):
    """Paths with numbered balls.

    small_after : une spec par chemin (voir is_small) : les itérations
    concernées ont une petite balle sans texte.
    """
    if small_after is None:
        small_after = [None] * len(paths)

    for k, p in enumerate(paths):
        if p is None:
            continue
        color = PATH_COLORS[k % len(PATH_COLORS)]
        n_pts = p.shape[1]
        spec = small_after[k]
        ax.plot(p[0], p[1], "-", color=color, linewidth=1.5, zorder=3)

        if box is None:
            step = max(1, n_pts // 15)  # at most ~15 labelled points
            idx = set(range(0, n_pts, step)) | {n_pts - 1}
            # les petites balles sont dessinées à TOUTES les itérations concernées
            idx |= {i for i in range(n_pts) if is_small(i, spec)}
        else:
            bx0, bx1, by0, by1 = box
            idx = {i for i in range(n_pts)
                   if bx0 <= p[0, i] <= bx1 and by0 <= p[1, i] <= by1}

        for i in sorted(idx):
            if is_small(i, spec):
                ax.plot(p[0, i], p[1, i], "o", markersize=small_size,
                        markerfacecolor=color, markeredgewidth=0, zorder=4)
                continue  # pas de texte
            ax.plot(p[0, i], p[1, i], "o", markersize=marker_size,
                    markerfacecolor=color, markeredgewidth=0, zorder=4)
            ax.text(p[0, i], p[1, i], str(i), ha="center", va="center",
                    fontsize=marker_font, color="black", zorder=5)
            

def add_zoom_inset(ax, v, extent, cmap, norm, functionID, paths, limits, zoom_box, loc,
                   small_after=None):
    """Magnifier of `zoom_box` inside `ax`, with the rectangle + connectors on the map."""
    zx0, zx1, zy0, zy1 = zoom_box
    x0_, x1_, y0_, y1_ = limits

    # magnifier size in axes fractions, keeping the aspect ratio of the zoomed zone
    # (the map itself is aspect="equal", so data units are comparable in x and y)
    w = ZOOM_SIZE
    h = w * ((zy1 - zy0) / (zx1 - zx0)) * ((x1_ - x0_) / (y1_ - y0_))
    if h > ZOOM_MAX_H:                      # very tall zone: shrink, keep the ratio
        w *= ZOOM_MAX_H / h
        h = ZOOM_MAX_H

    vert, horiz = loc.split()
    left = ZOOM_PAD if horiz == "left" else 1 - ZOOM_PAD - w
    bottom = ZOOM_PAD if vert == "lower" else 1 - ZOOM_PAD - h
    axins = ax.inset_axes([left, bottom, w, h])

    axins.imshow(v, origin="lower", extent=extent, cmap=cmap, norm=norm)

    xc = np.linspace(zx0, zx1, 200)
    yc = np.linspace(zy0, zy1, 200)
    X, Y = np.meshgrid(xc, yc)
    F = getObjFVal([X, Y], functionID)
    C = axins.contour(xc, yc, F, 6, colors="white", linewidths=0.5, alpha=0.55)
    axins.clabel(C, inline=True, fontsize=6, fmt="%.0f")

    draw_paths(axins, paths, ZOOM_MARKER_SIZE, ZOOM_MARKER_FONT, box=zoom_box,
               small_after=small_after, small_size=ZOOM_SMALL_MARKER_SIZE)

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor(ZOOM_EDGE)
        s.set_linewidth(1.2)

    # rectangle on the global map + lines to the magnifier
    ax.indicate_inset_zoom(axins, edgecolor=ZOOM_EDGE, linewidth=1.2, alpha=1)


def draw_panel(fig, ax, name, functionID, v, extent, limits, paths, show_ylabel):
    """One heat map (one function, one method), cropped on `limits`."""
    x0_, x1_, y0_, y1_ = limits

    # a scale of its own per panel, computed on the whole map
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    lo = max(vmin, 1)
    hi = max(vmax, lo + 1)
    norm = LogNorm(vmin=lo, vmax=hi)
    
    sa = small_after_for(functionID, name, len(paths))

    cmap = plt.get_cmap("viridis").copy()
    # starting points that never converged. red is absent from viridis
    cmap.set_bad("#d03b3b")

    ax.imshow(v, origin="lower", extent=extent, cmap=cmap, norm=norm)

    # contour lines of f on the shown area, in a neutral colour
    xc = np.linspace(x0_, x1_, 300)
    yc = np.linspace(y0_, y1_, 300)
    X, Y = np.meshgrid(xc, yc)
    F = getObjFVal([X, Y], functionID)
    C = ax.contour(xc, yc, F, 8, colors="white", linewidths=0.6, alpha=0.55)
    ax.clabel(C, inline=True, fontsize=7, fmt="%.0f")

    # optimization paths with numbered balls
    draw_paths(ax, paths, MARKER_SIZE, MARKER_FONT,
               small_after=sa, small_size=SMALL_MARKER_SIZE)

    # the zoom: crops the map, set after everything is drawn
    ax.set_xlim(x0_, x1_)
    ax.set_ylim(y0_, y1_)
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xlabel("x")
    if show_ylabel:               # only the first column, the ticks stay everywhere
        ax.set_ylabel("y")

    # magnifier on the zone chosen in ZOOM_BOXES
    zoom_box = ZOOM_BOXES.get((functionID, name))
    if ZOOM_ENABLED and zoom_box is not None:
        loc = ZOOM_LOCS.get((functionID, name), ZOOM_DEFAULT_LOC)
        add_zoom_inset(ax, v, extent, cmap, norm, functionID, paths, limits,
                       zoom_box, loc, small_after=sa)

    # a few log-spaced integer ticks, otherwise the default log locator
    # writes every minor tick in scientific notation
    ticks = np.unique(np.round(np.geomspace(lo, max(hi, 2), 5)).astype(int))
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, ticks=ticks, format=ScalarFormatter())
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    cb.ax.minorticks_off()
    cb.set_label("Iterations")

    # only when there is something red to explain
    if np.isnan(v).any():
        ax.legend(handles=[Patch(facecolor="#d03b3b", label="MaxIter reached")],
                  loc="upper right", fontsize=10, framealpha=0.9)


def compute(functionID, name, xi, pts):
    """Heat map values (whole N x N grid) and example paths of one (function, method)."""
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run_point,
                          [(name, functionID, p) for p in pts],
                          chunksize=50))
    v = np.array(res, float).reshape(N, N)  # v[j, i] <-> (xi[i], xi[j])
    print(f"f{functionID} {name:20s} done, {np.isnan(v).sum()}/{N * N} hit MaxIter",
          flush=True)

    starts = PATH_STARTS.get((functionID, name), [])
    paths = [get_path(name, functionID, p) for p in starts]
    return dict(v=v, starts=starts, paths=paths)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    xi = np.linspace(lb, up, N)
    # pts is basically all combination possible from xi X xi
    pts = [(a, b) for b in xi for a in xi]
    # imshow centres each pixel on its grid point: the pixels are h wide,
    # so the drawn area is [lb - h/2, up + h/2]
    h = (up - lb) / (N - 1)
    extent = [lb - h / 2, up + h / 2, lb - h / 2, up + h / 2]

    # 1) all the computations, on the same global grid
    results = {(f, name): compute(f, name, xi, pts)
               for f in FUNCTIONS for name in methods}

    # optional: grid points giving exactly 1 iteration, to copy (all decimals) in PATH_STARTS
    if FIND_ONE_IT:
        for (f, name), r in results.items():
            for j, i in zip(*np.where(r["v"] == 1)):   # v[j, i] <-> (xi[i], xi[j])
                print(f"f{f} {name:20s} 1 it. at ({xi[i]:.6f}, {xi[j]:.6f})")

    # 2) layout: the figure size comes from the margins, not the other way round
    n_rows, n_cols = len(FUNCTIONS), len(methods)
    n_cap = max(1, max(len(r["starts"]) for r in results.values()))
    # height needed under a map for the captions (the "x" label is inside the offset)
    caption_block = CAPTION_OFFSET_IN + CAPTION_LINE_IN * (n_cap - 1) + 0.20
    gap_v = caption_block + 0.15      # between two rows: captions of the upper row
    bottom_in = caption_block + 0.05  # under the last row: its captions

    fig_w = LEFT_IN + n_cols * CELL_W + (n_cols - 1) * GAP_H + RIGHT_IN
    fig_h = TOP_IN + n_rows * CELL_H + (n_rows - 1) * gap_v + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols,
                          left=LEFT_IN / fig_w, right=1 - RIGHT_IN / fig_w,
                          top=1 - TOP_IN / fig_h, bottom=bottom_in / fig_h,
                          wspace=GAP_H / CELL_W, hspace=gap_v / CELL_H)

    axes = {}
    for r, f in enumerate(FUNCTIONS):
        for c, name in enumerate(methods):
            ax = fig.add_subplot(gs[r, c])
            limits = PLOT_LIMITS.get((f, name), (lb, up, lb, up))
            res = results[(f, name)]
            draw_panel(fig, ax, name, f, res["v"], extent, limits, res["paths"],
                       show_ylabel=(c == 0))
            axes[(f, name)] = ax

    # the real position of each map is only known once the aspect is applied
    fig.canvas.draw()
    pos = {k: ax.get_position() for k, ax in axes.items()}

    # column titles (first row only), centred on the map, just above the highest one
    top_row = max(pos[(FUNCTIONS[0], n)].y1 for n in methods)
    for name in methods:
        p = pos[(FUNCTIONS[0], name)]
        fig.text((p.x0 + p.x1) / 2, top_row + 0.10 / fig_h,
                 f"{name}\n{reinit_label(methods[name])}",
                 ha="center", va="bottom", fontsize=15)

    # row labels: vertical, no box, close to the first map of the row
    for f in FUNCTIONS:
        p0 = pos[(f, list(methods)[0])]
        y_lo = min(pos[(f, n)].y0 for n in methods)
        y_hi = max(pos[(f, n)].y1 for n in methods)
        x = max(p0.x0 - ROW_LABEL_OFFSET_IN / fig_w, 0.15 / fig_w)
        fig.text(x, (y_lo + y_hi) / 2, rf"$f_{{{f}}}$",
                 ha="center", va="center", rotation=90, fontsize=22)

    # captions under each map, centred on it. Their baseline is the lowest map
    # of the row, so they are aligned and can never touch the tick labels
    for f in FUNCTIONS:
        y_base = min(pos[(f, n)].y0 for n in methods)
        for name in methods:
            res = results[(f, name)]
            p = pos[(f, name)]
            xm = (p.x0 + p.x1) / 2
            for k, s in enumerate(res["starts"]):
                path = res["paths"][k]  # same order as starts, None if MaxIter was reached
                nit = "MaxIter" if path is None else f"{path.shape[1] - 1} it."
                y = y_base - (CAPTION_OFFSET_IN + CAPTION_LINE_IN * k) / fig_h
                fig.text(xm, y,
                         f"$x_{{0,{k + 1}}} = ({s[0]:g}, {s[1]:g})$ ({nit})",
                         ha="center", va="top", fontsize=11,
                         color=TEXT_COLORS[k % len(TEXT_COLORS)])

    # no bbox_inches="tight": everything is placed in figure coordinates
    fig.savefig(OUT_FILE)
    print(f"saved {OUT_FILE}", flush=True)