import matplotlib.pyplot as plt
import numpy as np
import getObjFVal as F
from matplotlib.ticker import LinearLocator

# based on https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
functionID = 1
lb, up = -15, 15

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(lb, up + 0.25, 0.25)
Y = np.arange(lb, up + 0.25, 0.25)
X, Y = np.meshgrid(X, Y)
Z = F.getObjFVal([X, Y], functionID)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(Z.min(), Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter("{x:.0f}")

ax.set_xlabel("x_1")
ax.set_ylabel("x_2")
ax.set_title(f"f{functionID}")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
