import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Roboto'
rcParams['figure.figsize'] = (3/2.54, 5.5/2.54)  
rcParams['figure.dpi'] = 300
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

def equation1(x, B, A_B, s):
    numerator = B/s + A_B/(s**2)
    denominator = numerator + 1 + (B + A_B)/x
    return np.where(x > 0, numerator / denominator, 0) * (x < 1)

def equation2(x, B2, A_B2, s):
    numerator = B2/s + A_B2/(s**2)
    denominator = numerator + 1 + (B2 + A_B2)/x
    return np.where(x > 0, numerator / denominator, 0) * (x < 1)

x = np.logspace(-4, 0, 1000)

s = 0.1
B, A_B = 0, 1  
B2, A_B2 = 10/12.6, 1  

y1 = equation1(x, B, A_B, s)
y2 = equation2(x, B2, A_B2, s)

fig, ax = plt.subplots()

ax.semilogx(x, y1, color='black', linestyle='-', linewidth=1, label='Equation 1')
ax.semilogx(x, y2, color='black', linestyle='--', linewidth=1, label='Equation 2')

ax.fill_between(x, y1, y2, where=(y2 > y1), color='lightgray', alpha=0.5)
ax.fill_between(x, y1, y2, where=(y1 > y2), color='darkgray', alpha=0.5)

ax.set_xlabel(r"$K_{bb}'$", fontsize=12, labelpad=0)
ax.set_xlim(1e-4, 1)
ax.set_ylim(0, 1)

ax.set_title("$P_{\mathrm{active}}$ Difference \n at $K_{ab}' = 0.1$", fontsize=10, pad=5)

xticks = [1e-4, 0.013, 0.1, 1]
ax.set_xticks(xticks)

ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_xticklabels(['{:.4f}'.format(x) if x < 1e-3 else '{:.3f}'.format(x) if x < 0.1 else '{:.1f}'.format(x) for x in xticks])

ax.tick_params(axis='both', which='major', labelsize=5, pad=2, length=3, width=0.5)

ax.axvline(0.013, color='gray', linestyle=':', linewidth=1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(pad=0.1)
plt.savefig('Fig5G.pdf', format='pdf', bbox_inches='tight', dpi=330 , pad_inches=0)
plt.savefig('Fig5G.png', format='png', bbox_inches='tight', dpi=300)
plt.close()
