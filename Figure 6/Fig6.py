import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
import math
import string
import pandas as pd
import fitz
from matplotlib.lines import Line2D 
from matplotlib.transforms import Bbox

# Digitized data from Kitamura et al., J Cell Sci (1986) 82 (1): 223–234., Fig. 7.
data = np.array([
    [0, 0, 0],  
    [5.217391304, 15.76787808, 50.93786635],
    [10.31344793, 16.47127784, 59.26143025],
    [20.14155713, 9.78898007, 67.70222743],
    [30.45500506, 3.927315358, 78.25322392],
    [61.51668352, 0.41031653, 78.6049238],
    [120.8190091, 0, 86.57678781]
])
times = data[:, 0]
total_data = data[:, 2] / 100
homo_data = np.array([
    [0, 0],  
    [4.85338726, 16.11957796],
    [9.70677452, 16.23681125],
    [20.02022245, 9.78898007],
    [30.45500506, 3.927315358],
    [61.03134479, 0.527549824],
    [120.8190091, 0.175849941]
])

homotimes = homo_data[:, 0]
X2_data = homo_data[:, 1] / 100
delayed_data = np.array([
    [20.02022245,	0],
    [29.84833165,	1.113716295],
    [40.16177958,	11.54747948],
    [49.8685541,	59.73036342],
    [60.42467139,	80.83235639],
    [70.85945399,	85.87338804],
    [80.80889788,	85.99062134],
    [142.2042467,	89.50762016]
])

delayed_times = delayed_data[:, 0] 
delayed_total_data = delayed_data[:, 1] / 100

# Physical Constants
diameter_microns = 15 #from images
speed_microns_per_sec = 450 * 2**(1/2)
concentration_particles_per_ml = 500000 
radius_microns = diameter_microns / 2
concentration_particles_per_micron3 = concentration_particles_per_ml / (10**12)
area_micron2 = math.pi * (radius_microns ** 2)
volume_swept_per_sec_micron3 = area_micron2 * speed_microns_per_sec
particles_encountered_per_sec = volume_swept_per_sec_micron3 * concentration_particles_per_micron3
collisions_per_min = particles_encountered_per_sec * 60 

k_coll = collisions_per_min

adhesive_diameter = 15
area_micron2 = math.pi * (adhesive_diameter/2) ** 2
surface_area_sphere = 4*math.pi * (diameter_microns/2)**2

k_f_max = k_coll * (area_micron2/surface_area_sphere) #steric factor of adhesive (conA) surface area  

# integration Time range
dense_t = np.linspace(0, max(times), 10000)

# Initial conditions for the delayed system
AB0 = 0.0
X20 = 0.0
P0 = 0.0
lambda_0 = 0.0
initial_conditions = [AB0, lambda_0, X20, P0]

# Differential equations system for the disrupted system
def system(y, t, k_r, k_p):
    AB, X2, P = y
    X = 1 - AB - P - X2/2 #symmetry in X and Y cell concentration makes X = Y
    dAB_dt = k_f_max * X**2 - k_r * AB - k_p * AB
    dX2_dt = k_f_max * X**2 - k_r * X2 
    dP_dt = k_p * AB
    return [dAB_dt, dX2_dt, dP_dt]

# Differential equations system for the delayed system before stimulation
def system_delayed(y, t, k_coll, c, k_f_max, k_p, k_r_max):
    AB, lambda_t, X2, P = y
    X = 1 - AB - P -  X2/2 
    if lambda_t > 0:
        S = (0.5 * (1 - erf((c - lambda_t) / np.sqrt(2 * lambda_t)))) #proportion of stimluated cells right tail CDF of normally distributed collisions
    else:
        S = 0
    k_r = k_r_max 
    dAB_dt = k_f_max * (S*X)**2 - k_r * AB - k_p * AB
    dlambda_dt =  k_coll * (X)  
    dX2_dt = k_f_max * (S*X)**2 - k_r * X2
    dP_dt = k_p * AB
    return [dAB_dt, dlambda_dt, dX2_dt ,dP_dt]

# Minimize objective function
def residuals(params, t, homotimes, times, delayed_times, X2_data, total_data, delayed_total_data):
    k_r, k_p, c = params

    max_time = max(max(times), max(delayed_times))
    t_model = np.linspace(0, max_time, 10000)
    solution = odeint(system, [0, 0, 0], t_model, args=(k_r, k_p))
    AB, X2, P = solution.T

    interp_X2 = interp1d(t_model, X2, kind='cubic', bounds_error=False, fill_value="extrapolate")(homotimes)
    interp_total = interp1d(t_model, AB + P + X2, kind='cubic', bounds_error=False, fill_value="extrapolate")(times)
    
    solution_delayed = odeint(system_delayed, initial_conditions, np.linspace(0, 150, 15000), args=(k_coll, c, k_f_max, k_p, k_r))
    ABd, lambduh, X2d, Pd = solution_delayed.T
    interp_delayed = interp1d(np.linspace(0, 150, 15000), ABd + Pd + X2d, kind='cubic', bounds_error=False, fill_value="extrapolate")(delayed_times)
    
    res_X2 = (interp_X2 - X2_data)*0.58 #weighted less to fit better with later plots also because X2 is a subset of the total count in the original data, so the uncertainty propogates
    res_total = interp_total - total_data
    res_delayed = interp_delayed - delayed_total_data
    
    return np.concatenate([res_X2, res_total, res_delayed])

# Initial optimizatoin parameter guesses
initial_params = [1, 1, 100]

# Perform optimization
result = least_squares(residuals, initial_params, 
                       args=(dense_t, homotimes, times, delayed_times, X2_data, total_data, delayed_total_data))

print("Optimized Parameters with k_f fixed:")
print("k_r = {:.4f}, k_p = {:.4f}, c = {:.4f}".format(*result.x))

solution = odeint(system, [0, 0, 0], np.linspace(0, 120, 15000), args=(tuple(result.x))[0:2])
AB, X2, P = solution.T
model_X2 = interp1d(np.linspace(0, 150, 15000), X2, kind='cubic')(homotimes)
model_total = interp1d(np.linspace(0, 150, 15000), AB + P + X2, kind='cubic')(times)

solution_delayed = odeint(system_delayed, initial_conditions, np.linspace(0, 210, 21000), args=(k_coll, result.x[2], k_f_max, result.x[1], result.x[0]))
ABd, lambduh, X2d, Pd = solution_delayed.T
model_total_d = interp1d(np.linspace(0, 210, 21000), ABd + Pd + X2d, kind='cubic')(delayed_times)
k_p_universal =  result.x[1]
#confirm assumption of 90 minute stimulation
#print((0.5 * (1 - erf((result.x[2] - lambduh[9000]) / np.sqrt(2 * lambduh[9000]))))) #approaches 1.0

# Plotting
sns.set(style="whitegrid", font_scale=0.8)
fig1 = plt.figure(figsize=(3.3, 4.7))
gs1 = fig1.add_gridspec(3, 1, height_ratios=[5, 1, 5], hspace=0.2)

# First subplot
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.set_axisbelow(True)
colors = ['#000000', '#D32F2F', '#2ca02c']  
markers = ['o', 's', '^']
linestyles = ['-', '-', '--']

ax1.scatter(delayed_times, delayed_total_data, color=colors[0], marker=markers[0], s=30, label='Pairs', zorder=5)
ax1.plot(np.linspace(0, 210, 21000), ABd + X2d + Pd, color=colors[0], linestyle=linestyles[0], zorder=4)

ax1.scatter(times + (90), total_data, color=colors[1], marker=markers[1], s=30, label='Pairs (Disrupted)', zorder=5)
ax1.plot(np.linspace(90, 210, 15000), AB + X2 + P, color=colors[1], linestyle=linestyles[1], zorder=4)

ax1.scatter(homotimes + (90), X2_data, color=colors[2], marker=markers[2], s=30, label='Same-MT Pairs (Disrupted)', zorder=5)
ax1.plot(np.linspace(90, 210, 15000), X2, color=colors[2], linestyle=linestyles[2], zorder=4, alpha = 0.7)

ax1.set_xlabel('Time (minutes)', size=9 , labelpad=2)  
ax1.set_title('Paired Proportion, Disrupted at 90 min', size=10)

ax1.tick_params(axis='y', which='major', labelsize=8, pad=-4)  
ax1.tick_params(axis='x', which='major', labelsize=8, pad=-2)  
ax1.spines['bottom'].set_visible(False)
ax1.set_xlim(0, 220)
ax1.set_ylim(-00.03, 1)
ax1.set_xticks(np.arange(0, 221, 30))

params = {
    '$c$': result.x[2],
    '$k_f$': k_f_max,
    '$k_r$': result.x[0],
    '$k_p$': result.x[1]
}
param_text = '\n'.join([f'{k} = {v:.2f}' for k, v in params.items()])
ax1.text(0.95, 0.25, param_text, transform=ax1.transAxes, fontsize=9, 
        verticalalignment='bottom', horizontalalignment='right', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Legend subplot
leg = fig1.add_subplot(gs1[1, 0])
leg.axis('off')
legend = leg.legend(*ax1.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.45, 0.5), ncol=3, fontsize='x-small',
                    handletextpad=0.0, columnspacing=0.25, borderaxespad=0.0, frameon=False)

# Second subplot (Kitamura et al., Fig 3. data)
ax2 = fig1.add_subplot(gs1[2, 0])
ax2.set_axisbelow(True)

data_array_1 = np.array([
    [28.04733728, 0],
    [35.0295858, 11.82669789],
    [40.1183432, 24.3559719],
    [45.0887574, 36.8852459],
    [49.6470588, 56.99052132],
    [54.55621302, 65.69086651],
    [64.85207101, 78.33723653],
    [95.0295858, 85.94847775]
])
data_array_2 = np.array([
    [28.04733728, 0],
    [35.38461538, 5.620608899],
    [40, 7.611241218],
    [44.49704142, 5.386416862],
    [49.64705882, 7.345971564],
    [55.0295858, 6.674473068],
    [64.85207101, 6.323185012],
    [95.0295858, 1.170960187]
])

t_end = 210
t = np.linspace(0, t_end, 10000)

solution_fig3 = odeint(system_delayed, initial_conditions, t, args=(k_coll, result.x[2], k_f_max, result.x[1], result.x[0]))
AB_k, lambda_t_k, X2_k, P_k = solution_fig3.T

ax2.plot(t, P_k + AB_k + X2_k, color=colors[0], linestyle=linestyles[0], label='Fig. 6A Curve', zorder=4)
ax2.plot(t, X2_k, color=colors[2], linestyle=linestyles[2], label='Same-MT Pairs', zorder=6, alpha = 0.7)
ax2.scatter(data_array_2[:, 0], data_array_2[:, 1]/100, color=colors[2], marker=markers[2], s=30, zorder=5)
ax2.scatter(data_array_1[:, 0], data_array_1[:, 1]/100, color=colors[0], marker=markers[0], s=30,  zorder=4)

ax2.set_title('Predict Replicate Experiment', size=10)
ax2.set_xlabel('Time (minutes)', size=9, labelpad=2)
ax2.legend(fontsize='small', bbox_to_anchor=(.70, 0.65), loc='upper center')
ax2.set_xlim(0, 220)
ax2.set_ylim(-00.03, 1)

ax2.tick_params(axis='y', which='major', labelsize=8, pad=-4)
ax2.tick_params(axis='x', which='major', labelsize=8, pad=-2)
ax2.spines['bottom'].set_visible(False)
ax2.set_xticks(np.arange(0, 221, 30))

ax1.text(-0.09, 1.05, 'A', transform=ax1.transAxes, size=14, weight='bold')
ax2.text(-0.09, 1.05, 'B', transform=ax2.transAxes, size=14, weight='bold')
plt.tight_layout()

fig2 = plt.figure(figsize=(3.4, 4.4))
gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.1)

# Load the CSV file (digitzied data from Yan et al., eLife, 2024)
file_path = 'data serieses.csv'
data_series_df = pd.read_csv(file_path)

# Extracting time series for "diff" and "same"
time_diff = data_series_df['t (min)'].dropna().values
time_same = data_series_df['t (min).1'].dropna().values

# Extracting data series for "diff"
ctrl_diff = data_series_df['ctrl diff'].dropna().values
mtaxc_diff = data_series_df['mtaxc diff'].dropna().values
mtbxc_diff = data_series_df['mtbxc diff'].dropna().values
both_diff = data_series_df['both diff'].dropna().values

# Extracting data series for "same"
ctrl_same = data_series_df['ctrl same'].dropna().values
mtaxc_same = data_series_df['mtaxc same'].dropna().values
mtbxc_same = data_series_df['mtbxc same'].dropna().values
both_same = data_series_df['both same'].dropna().values

# Extracting concentration data for mta
time_mta = data_series_df['t (for mta)'].dropna().values
conc_mta_0 = data_series_df['conc. 0'].dropna().values
conc_mta_3 = data_series_df['conc. 3'].dropna().values
conc_mta_30 = data_series_df['conc. 30'].dropna().values
conc_mta_300 = data_series_df['conc. 300'].dropna().values

# Extracting concentration data for mtb
time_mtb = data_series_df['t (for mtb)'].dropna().values
conc_mtb_0 = data_series_df['conc. 0.1'].dropna().values
conc_mtb_3 = data_series_df['conc. 3.1'].dropna().values
conc_mtb_30 = data_series_df['conc. 30.1'].dropna().values
conc_mtb_300 = data_series_df['conc. 300.1'].dropna().values

diff_arrays = {
    "time_diff": time_diff,
    "ctrl_diff": ctrl_diff,
    "mtaxc_diff": mtaxc_diff,
    "mtbxc_diff": mtbxc_diff,
    "both_diff": both_diff
}

same_arrays = {
    "time_same": time_same,
    "ctrl_same": ctrl_same,
    "mtaxc_same": mtaxc_same,
    "mtbxc_same": mtbxc_same,
    "both_same": both_same
}

conc_mta_arrays = {
    "time_mta": time_mta,
    "conc_0": conc_mta_0,
    "conc_3": conc_mta_3,
    "conc_30": conc_mta_30,
    "conc_300": conc_mta_300
}

conc_mtb_arrays = {
    "time_mtb": time_mtb,
    "conc_0": conc_mtb_0,
    "conc_3": conc_mtb_3,
    "conc_30": conc_mtb_30,
    "conc_300": conc_mtb_300
}
print(k_coll)
#new parameters:
# Physical Constants
diameter_microns_new = 20 #estimated from images
new_concentration_particles_per_ml = 250000 
new_radius_microns = diameter_microns_new / 2
new_concentration_particles_per_micron3 = new_concentration_particles_per_ml / (10**12)
new_area_micron2 = math.pi * (new_radius_microns ** 2)
new_volume_swept_per_sec_micron3 = new_area_micron2 * speed_microns_per_sec
new_particles_encountered_per_sec = new_volume_swept_per_sec_micron3 * new_concentration_particles_per_micron3
new_collisions_per_min = new_particles_encountered_per_sec * 60 

new_k_coll = new_collisions_per_min

new_surface_area_sphere = 4*math.pi * (diameter_microns_new/2)**2

new_k_f_max = new_k_coll * (area_micron2/new_surface_area_sphere) #steric factor of adhesive (conA) surface area  

def optimize_kp(time, data, new_k_coll, c, new_k_f_max, k_r):
    def residuals(k_p):
        k_p = k_p[0]  
        solution = odeint(system_delayed, initial_conditions, time, args=(new_k_coll, c, new_k_f_max, k_p, k_r))
        AB, _, X2, P = solution.T
        model_data = AB + X2 + P
        return model_data - data

    result = least_squares(residuals, x0=[k_p_universal/2], bounds=(0, np.inf))
    return result.x[0]

# Create the 2x2 plot for time series data
categories = [conc_mtb_arrays, conc_mta_arrays, diff_arrays, same_arrays]
category_labels = ['Conc. B', 'Conc. A', 'DIFF', 'SAME']
colors = ['black', '#1A5F7A', '#FFA41B', '#7A4069']
markers = ['o', 's', '^', 'D'] 
category_title_labels = ['Soluble sMtb', 'Soluble sMta', 'Different-MT', 'Same-MT']
for idx, (category, label) in enumerate(zip(categories, category_labels)):
    ax = fig2.add_subplot(gs2[idx // 2, idx % 2])
    
    legend_elements = []
    
    for i, key in enumerate([k for k in category.keys() if not k.startswith('time')]):
        time_key = 'time_' + key.split('_')[-1] if 'time' not in key else key
        time = category.get(time_key, np.linspace(0, 120, 9))  # Use default if time_key does not exist
        data = category[key]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Optimize k_p for this specific dataset
        optimized_kp = optimize_kp(time, data, new_k_coll, result.x[2], new_k_f_max, result.x[0])
        
        # Modify scatter legend labels
        if idx < 2:  # For Conc. B and Conc. A plots
            scatter_label = f"{key.split('_')[-1]} pg/mL"
        else:  # For DIFF and SAME plots
            if 'ctrl' in key:
                scatter_label = "Control"
            elif 'mtaxc' in key:
                scatter_label = "+sMta"
            elif 'mtbxc' in key:
                scatter_label = "+sMtb"
            elif 'both' in key:
                scatter_label = "+sMta+sMtb"
            else:
                scatter_label = key

        # Modify trendline legend labels
        trendline_label = f"$k_p$ = {optimized_kp:.2f}"
        if i == 0:
            zeeorder = 15
        else:
            zeeorder=4+i%3
        scatter = ax.scatter(time, data, color=color, marker=marker, s=15, alpha=0.75, edgecolors='none', zorder=1+2*zeeorder)
        
        # Use the optimized k_p in the plotted solution
        t_dense = np.linspace(0, 250, 10000)
        solution_fig3 = odeint(system_delayed, initial_conditions, t_dense, 
                               args=(new_k_coll, result.x[2], new_k_f_max, optimized_kp, result.x[0]))
        AB_e, _, X2_e, P_e = solution_fig3.T
        line = ax.plot(t_dense, P_e + AB_e + X2_e, color=color, alpha=0.75, zorder=2*zeeorder)
        
        # Create custom legend entries with very short line for trendline
        legend_elements.append(Line2D([0], [0], marker=marker, markeredgecolor='none', color='w', alpha=0.75 , markerfacecolor=color, markersize=5, label=scatter_label))
        legend_elements.append(Line2D([0], [0], color=color,  markeredgecolor='none', alpha=0.75, lw=2, solid_capstyle='butt', label=trendline_label))
        
        print(f"Optimized k_p for {scatter_label}: {optimized_kp}")
    
    legend_pos = (-0.075,1.03)
    ax.set_title(category_title_labels[idx], size=10)
    if idx // 2 == 1:  # Only set x-label for bottom row
        ax.set_xlabel('Time (min)', size=9, labelpad=2)
        if idx == 3:  # Only set x-label for bottom row
            legend_pos  = (0.475,0.6)
    
     # Create legend with tight spacing and custom entries
    legend = ax.legend(handles=legend_elements, fontsize='xx-small', loc='upper left', bbox_to_anchor=legend_pos, 
              frameon=True, ncol=1, handletextpad=-0.2, columnspacing=-0.2, labelspacing=-0.05)
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.65)

    # Remove the edge (border) of the legend
    legend.get_frame().set_edgecolor('none')
    # Adjust the length of the line in the legend
    for text, line in zip(legend.get_texts(), legend.get_lines()):
        if text.get_text().startswith('$k_p$'):  # Only adjust trendline entries
            line.set_linestyle('-')
            line.set_linewidth(1)
            line.set_markersize(0)
            line.set_markeredgewidth(0)
            line.set_marker('')
            line.set_dash_capstyle('butt')
            line.set_solid_capstyle('butt')
            line.set_dash_joinstyle('miter')
            line.set_solid_joinstyle('miter')
            line.set_visible(True)
            # Set a very short line length with consistent shape
            line.set_xdata([3, 8])
            line.set_ydata([2, 2])  # Ensure y-data has the same length as x-data
    
    ax.set_xlim(0, 130)
    ax.set_xticks(np.arange(0, 121, 30))
    ax.set_ylim(-0.03, 1)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks(np.arange(0, 1.00001, 0.2))
    if idx % 2 == 0:  # Only set y-label for left col
        ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
    else:
        ax.set_yticklabels([]) 
    if (idx % 2 == 1) and (idx // 2 == 1):
        ax.set_xlim(0, 250)
        ax.set_xticks([0, 30, 60, 90, 120, 240])
        ax.set_xticklabels([0, '', 60, '', 120, 240])
    ax.tick_params(axis='both', which='major', labelsize=8, pad=-2)

    # Add subplot label
    ax.text(-0.1, 1.05, string.ascii_uppercase[2 + idx], transform=ax.transAxes, 
            size=14, weight='bold')

#  custom padding (in inches)
left_padding = 0.075  
right_padding = -0.025
fig = plt.gcf()
renderer = fig.canvas.get_renderer()
tight_bbox = fig.get_tightbbox(renderer)
new_bbox = Bbox.from_bounds(
    tight_bbox.x0 - left_padding,
    tight_bbox.y0,
    tight_bbox.width + left_padding + right_padding,
    tight_bbox.height
)
fig1.savefig('Fig6_part1.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
fig2.savefig('Fig6_part2.pdf', format='pdf', dpi=300, bbox_inches=new_bbox)

fig1.savefig('Fig6_part1.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.05)
fig2.savefig('Fig6_part2.png', format='png', dpi=600, bbox_inches=new_bbox)

plt.close(fig1)
plt.close(fig2)

# stack PDF pages into one figure
def stack_pdf_pages(input_pdf1, input_pdf2, output_pdf):
    doc1 = fitz.open(input_pdf1)
    doc2 = fitz.open(input_pdf2)
    width1, height1 = doc1[0].rect.width, doc1[0].rect.height
    width2, height2 = doc2[0].rect.width, doc2[0].rect.height
    combined_page = fitz.open()
    combined_rect = fitz.Rect(0, 0, max(width1, width2), height1 + height2)
    combined_page.new_page(width=max(width1, width2), height=height1 + height2)
    combined_page[0].show_pdf_page(fitz.Rect(0, 0, width1, height1), doc1, 0)
    combined_page[0].show_pdf_page(fitz.Rect(0, height1, width2, height1 + height2), doc2, 0)
    combined_page.save(output_pdf)
    combined_page.close()
    doc1.close()
    doc2.close()
stack_pdf_pages('Fig6_part1.pdf', 'Fig6_part2.pdf', 'Figure 6.pdf')
