import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

B = 0
A_B = 1
epsilon = 1e-10
mesh_size = 2000
x = np.linspace(epsilon, 1, mesh_size)
y = np.linspace(epsilon, 1, mesh_size)
X, Y = np.meshgrid(x, y)
numerator = (B/Y) + (A_B / Y**2)
denominator = (B/Y) + (A_B / Y**2) + 1 + (B + A_B) / X
P_bound = (numerator / denominator)**2
P_bound = np.clip(P_bound, 0, 1)
fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
im = ax.imshow(P_bound, cmap='magma_r', origin='lower', extent=[0, 1, 0, 1], aspect='equal')
ax.set_xlabel(r"$K_{bb}'$", fontsize=12, labelpad=2)
ax.set_ylabel(r"$K_{ab}'$", rotation=0, fontsize=12, labelpad=8, va="center")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.tick_params(axis='both', which='major', labelsize=8)
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)

def text_box(ax, x, y, text, fontsize=9):
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.5)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, bbox=bbox_props, zorder=5)

text_box(ax, 0.2, 0.875, 'Self-\ninhibition')
text_box(ax, 0.8, 0.125, 'Mating')
text_box(ax, 0.2, 0.125, 'Selfing')
text_box(ax, 0.76, 0.875, 'Non-\nrecognition')
cbar = fig.colorbar(im, ax=ax, shrink=0.62, aspect=15, pad=0.07)
cbar.ax.tick_params(labelsize=8)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_label('$P^{\mathregular{2}}_{\mathrm{active}}$', fontsize=12, rotation=90, labelpad=0)
plt.tight_layout()
plt.savefig("Fig5B.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig("Fig5B.png", format='png', dpi=600, bbox_inches='tight')
plt.close(fig)
