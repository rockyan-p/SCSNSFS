import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.legend_handler import HandlerBase

# Read values obtained from Yan, G. et al. Evolution of the mating type gene pair and multiple sexes in Tetrahymena. iScience 24, 101950 (2020) doi:10.1016/j.isci.2020.101950.
# T. the	12.6	23.6
# T mal	    16.3	19.3
# T. bor	8.41	10.6
# T. can	6.81	5.06
# T sha	    8.97	5.8
# T. ame	5.95	10.2
# T. pig	21.8	6.73

data = {
    "num_mating_types": [7, 6, 7, 5, 1, 9, 3],
    "fold_change": [1.873, 1.184, 1.260, 0.743, 0.647, 1.714, 0.309]
}

df = pd.DataFrame(data)

outlier = df[df['num_mating_types'] == 1]
non_outliers = df[df['num_mating_types'] != 1]

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'

plt.figure(figsize=(5.2, 6))
sns.regplot(data=non_outliers, x='num_mating_types', y='fold_change', ci=95, 
            scatter_kws={'s': 100, 'color': '#4b9fd3'}, 
            line_kws={'color': '#4b9fd3'})

plt.text(outlier['num_mating_types'], outlier['fold_change'], '*', color='black', fontsize=30, fontweight='bold', ha='center', va='center')

class TextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [plt.text(width / 2, height / 2 - height / 3, '*', fontsize=30, fontweight='bold', ha="center", va="center", color='black', transform=trans)]

plt.legend(handles=[plt.Line2D([], [], color='black', label='Selfer')],
           handler_map={plt.Line2D: TextHandler()}, loc='upper left', fontsize=24, bbox_to_anchor=(0.02, 0.93))

plt.axhline(y=1, color='black', linestyle='--')

plt.title("Ratio: Mtb/Mta RNA (FPKM)    ", fontsize=26, ha="center")
plt.xlabel("# of MT per Species", fontsize=24)
plt.xticks(range(10), fontsize=18)
plt.yticks([0.5, 1.0, 1.5, 2.0, 2.5], fontsize=18)
plt.ylim(0, 2.6)
plt.xlim(0, 9.25)
plt.tight_layout()
plt.ylabel('Fold', fontsize=18,)
plt.text(2.74, 1.45, r'$R^2 = 0.82$', fontsize=24, color='#4b9fd3')
plt.gca().grid(False)
plt.savefig('Fig5E.pdf', format='pdf', dpi=120, pad_inches = 0)
plt.savefig('Fig5E.png', format='png', dpi=240)

