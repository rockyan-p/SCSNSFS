import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox

def calculate_z(x, y, A_B2, A_B, B, B_2):
    term1 = (B/x + A_B/x**2) / (B/x + A_B/x**2 + 1 + (B + A_B)/y)
    term2 = (B_2/x + A_B2/x**2) / (B_2/x + A_B2/x**2 + 1 + (B_2 + A_B2)/y)
    return -term1 + term2

A_B2 = 1
A_B = 1
B = 0
B_2 = (23.6 - 12.6) / 12.6

x = np.linspace(0.015, 1, 100)
y = np.linspace(0.015, 1, 100)
X, Y = np.meshgrid(x, y)

Z = calculate_z(X, Y, A_B2, A_B, B, B_2)

fig = plt.figure(figsize=(3.2, 3.2), dpi=300)
ax = fig.add_subplot(111, projection='3d')
max_abs = max(abs(Z.min()), abs(Z.max()))
vmin, vmax = -max_abs, max_abs

norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=Z.max())

surf = ax.plot_surface(Y, X, Z, cmap='RdBu_r', norm=norm, antialiased=False)

ax.set_xlabel(r"$K_{bb}'$", fontsize=12, labelpad=-5)
ax.set_ylabel(r"$K_{ab}'$", fontsize=12, labelpad=-5)
ax.tick_params(axis='both', which='major', labelsize=7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_zticks([0.05, 0, -0.05, -0.10])

cbar = fig.colorbar(surf, ax=ax, aspect=15, pad=0.05, shrink=0.6)
ax.tick_params(axis='x', which='major', labelsize=7, pad=-2.5)
ax.tick_params(axis='y', which='major', labelsize=7, pad=-2.5)
ax.tick_params(axis='z', which='major', labelsize=7, pad=-0.5)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()

cbar.formatter = ticker.FuncFormatter(lambda x, p: f"{x:.2f}")
cbar.set_ticks([0.05, 0, -0.05, -0.10, -0.13])
cbar.set_ticklabels(['0.05', '0', '-0.05', '-0.10', '-0.13'])
cbar.ax.tick_params(labelsize=7)
cbar.ax.yaxis.set_label_position('left')
cbar.set_label('$\Delta P_{\mathrm{active}}$', fontsize=12, rotation=90, labelpad=0.3)
ax.view_init(elev=30, azim=225)
title_text = r"${\mathrm{\{Mtb\}}} = 0.87$ vs. ${\mathrm{\{Mtb\}}} = 0$"
plt.suptitle(title_text, fontsize=12, x= 0.38, y=0.77)  # Adjust the y value as needed
# Adjust layout to minimize white space
plt.tight_layout(rect=[0, 0, 1, 0.95])
# renderer = fig.canvas.get_renderer()
# tight_bbox = fig.get_tightbbox(renderer)
# new_bbox = Bbox.from_bounds(
#     tight_bbox.x0,
#     tight_bbox.y0,
#     tight_bbox.width,
#     tight_bbox.height - 0.5
# )
plt.savefig("Fig5F.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=-0.01)
plt.savefig('Fig5F.png', format='png', dpi=600, bbox_inches='tight', pad_inches=-0.01)
plt.close()