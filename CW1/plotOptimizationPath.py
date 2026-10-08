import numpy as np
import matplotlib.pyplot as plt
from getObjFVal import getObjFVal


def plotOptimizationPath(x, functionID):

    lb = -15
    up = 15
    xi = np.arange(lb, up + 0.02, 0.02)
    f = np.zeros((len(xi), len(xi)))

    for i in range(len(xi)):
        for j in range(len(xi)):
            f[j, i] = getObjFVal([xi[i], xi[j]], functionID)

    plt.figure(figsize=(10, 8))
    plt.title("Optimization Path and Contour Plot")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.axis([lb, up, lb, up])
    plt.axis("square")

    automatic_contour_levels = True  # You can let pyplot choose automatically the contour levels, or you can specify them yourself

    if automatic_contour_levels:
        # Automatically calculate contour levels
        C = plt.contour(xi, xi, f, cmap="viridis", extend="both")

    else:
        switch_dict = {
            1: (
                [0, 100, 200, 400, 800, 1200, 1600, 2000, 2800],
                "Contour for Function 1",
            ),  # CHOOSE HERE THE CONTOUR LEVELS
            2: (np.arange(-20, 250, 10), "Contour for Function 2"),
        }

        levels, title_text = switch_dict.get(functionID, ([], "Unknown Function"))
        levels = sorted(levels)

        C = plt.contour(xi, xi, f, levels, cmap="viridis", extend="both")

    # Add labels to contour lines
    plt.clabel(C, inline=True, fontsize=8, fmt="%1.1f")

    ind = 1
    for i in range(x.shape[1] - 1):
        # Use different marker styles for optimization path points
        plt.plot(
            x[0, i],
            x[1, i],
            "o",
            markersize=9,
            markerfacecolor="c",
            markeredgecolor="black",
        )
        plt.plot([x[0, i], x[0, i + 1]], [x[1, i], x[1, i + 1]], "c", linewidth=2)
        plt.text(
            x[0, i],
            x[1, i],
            str(ind - 1),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
        )
        ind += 1

    # Mark the last point differently
    plt.plot(
        x[0, -1],
        x[1, -1],
        "o",
        markersize=9,
        markerfacecolor="red",
        markeredgecolor="black",
    )
    plt.text(
        x[0, -1],
        x[1, -1],
        str(ind - 1),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=8,
    )

    plt.show()
