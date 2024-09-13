import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.transforms import Bbox
from scipy.stats import ttest_ind
data = [
    [-18.7, -10.5, -8.6, -9.8, -9.9, -14.6, -10.7, -14.4, -14.7, -12.9, -14.7, -15.1],
    [0, -14.9, -12.8, -10.2, -10.9, -13, -13.7, 0, -8.3, -11.2, -17.1, 0],
    [-13.2, -10.9, -11.8, -10.8, -24.6, -11.1, -15.5, -11.9, -11.5, -11.3, -13.7, -12],
    [-19, -10.6, -17.7, -10.1, -15.3, -15.5, -19.6, -11.6, -14, -8.2, -13.4, -17.5],
    [-11.9, -16.9, -10.9, -17.6, -13, -13.1, -12.7, -11.1, -15.5, -12.3, -9.4, -14.6],
    [-9.2, -18.2, -6.8, -14.4, -10.3, -13.5, -15.7, -15, -8.2, -13.4, -12.9, -10.7]
] #kcal mol^_1
data = [[-val for val in row] for row in data]
diagonal = [data[i][i*2] - data[i][i*2 + 1] for i in range(6)]
non_diagonal = [data[i][j] - data[i][j+1] for i in range(6) for j in range(0, 12, 2) if j != i*2]
data_dict = {
    'Group': [''] * len(diagonal) + [' '] * len(non_diagonal),
    'Difference': diagonal + non_diagonal
}
t_stat, p_value = ttest_ind(diagonal, non_diagonal)
print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')
df = pd.DataFrame(data_dict)
mean_diagonal = np.mean(diagonal)
mean_non_diagonal = np.mean(non_diagonal)
plt.figure(figsize=(2, 5.2))  
sns.set(style="whitegrid")
sns.stripplot(x='Group', y='Difference', data=df, jitter=True, palette=['black'], size=8, alpha=0.7)
plt.plot([-0.2, 0.25], [mean_diagonal, mean_diagonal], color='000000', lw=2)
plt.plot([0.75, 1.2], [mean_non_diagonal, mean_non_diagonal], color='000000', lw=2)
plt.title("Apparent\nInhibition\n$\\Delta\\Delta G$ (kcal/mol)", fontsize=16)
plt.xlabel("")  
plt.ylabel("")  
sns.despine()
p_value_text = f'p = {p_value:.3f}'
plt.figtext(0.5, 0.05, p_value_text, ha="center", fontsize=16)

ax = plt.gca()
ax.spines['left'].set_position(('data', 0.5))
ax.spines['left'].set_color('000000')
ax.spines['left'].set_linewidth(1)
plt.tight_layout()
plt.savefig('Fig4B.pdf', format='pdf', dpi=300*3.3/5.08, bbox_inches=Bbox.from_bounds(0.1, 0.17, 1.8, 4.89))
plt.savefig('Fig4B.png', format='png', dpi=900*3.3/5.08, bbox_inches=Bbox.from_bounds(0.1, 0.17, 1.8, 4.89))
plt.close()
