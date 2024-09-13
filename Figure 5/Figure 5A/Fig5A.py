import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox

def log_tick_formatter(val, pos=None):
    return f"$10^{{{val}}}$"

Kbb = 0.01
x = np.logspace(-6, 0, 1000)
y = np.logspace(-4, 0, 1000)
D, Kba = np.meshgrid(x, y)
Z = (D / Kba**2) / ((D / Kba**2) + D / Kbb + 1)

fig = plt.figure(figsize=(3.6, 3.4), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=45)

#log plot
log_D = np.log10(D)
log_Kba = np.log10(Kba)

surf = ax.plot_surface(log_D, log_Kba, Z, cmap='magma_r', edgecolor='none', rcount=16, ccount=40)

ax.set_xlabel(r'$\mathrm{\{Mtb_{tot}\}}$', fontsize=11, labelpad=-9)
ax.set_ylabel(r"$K_{ab}'$", fontsize=11, labelpad=-6)

ax.set_ylim(-4, 0)
ax.set_xlim(0, -6)
ax.set_zlim(0, 1)
ax.set_zticks([ 0.25, 0.5, 0.75, 1.0])

ax.set_yticks([ -4, -3,-2,-1])
ax.set_xticks([-5, -3, -1, 0])

ax.tick_params(axis='z', which='major', labelsize=7, pad=-1)
ax.tick_params(axis='y', which='major', labelsize=7, pad=-3)
ax.tick_params(axis='x', which='major', labelsize=7, pad=-5)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.xaxis._axinfo['axisline']['color'] = (0, 0, 0, 0)
ax.yaxis._axinfo['axisline']['color'] = (0, 0, 0, 0)
ax.zaxis._axinfo['axisline']['color'] = (0, 0, 0, 0)
cbar = fig.colorbar(surf, shrink=0.66, aspect=15, pad=0.03, extend='neither')
cbar.ax.tick_params(labelsize=7, pad=0)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.set_ylabel(r'$P_{\mathrm{active}}$', rotation=90, labelpad=3, fontsize=12)
ax.zaxis.set_rotate_label(False)
fig = plt.gcf()
renderer = fig.canvas.get_renderer()
tight_bbox = fig.get_tightbbox(renderer)
new_bbox = Bbox.from_bounds(
    tight_bbox.x0-0.2,
    tight_bbox.y0+0.025,
    tight_bbox.width+0.2,
    tight_bbox.height-0.150
)
plt.savefig("Fig5A.pdf", format='pdf', dpi=300, bbox_inches=new_bbox, pad_inches=0)
plt.savefig("Fig5A.png", format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
