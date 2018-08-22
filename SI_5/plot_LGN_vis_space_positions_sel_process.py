import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.patches as patches


type_colors = { 'transient_ON': 'green', 'transient_OFF': 'magenta', 'transient_ON_OFF': 'blue' }
type_colors_light = { 'transient_ON': 'lightgreen', 'transient_OFF': 'lightpink', 'transient_ON_OFF': 'lightblue' }

map_df = pd.read_csv('../build/ll2_inputs_from_LGN.csv', sep=' ')
gids = np.array(list(set(map_df['index'].values)))

cells = pd.read_csv('../build/ll2.csv', sep=' ')

filters_data_fname = '../../6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN2_visual_space_positions_and_cell_types.dat'
filters_data = pd.read_csv(filters_data_fname, sep=' ')
filters_data.columns = ['LGN_type', 'x', 'y', 'x_offset', 'y_offset', 'sigma_c', 'sigma_s', 'r0', 'scaling_factor', 'k_alpha']
types_list = list(set(filters_data['LGN_type'].values))

# Compute the position of the center of the visual field, using coordinates of the filters.
vis_center_x = filters_data['x'].mean()
vis_center_y = filters_data['y'].mean()

# Visual space size for the target cells system.
vis_tar_L_x = 140.0
vis_tar_L_y = 70.0


cell_gid = 107
cell_xlim = (84.0, 120.0)
cell_ylim = (54.0, 90.0)
cell_el_a = 8.0/2.0
cell_el_aspect_r = 2.8
cell_el_width = 2.0 * cell_el_a
cell_el_height = cell_el_width * cell_el_aspect_r
cell_el_centers_d = 11.0

tmp_df = map_df.loc[map_df['index'] == cell_gid]

# Get coordinates of the target cell.
tar_cell_df = cells[cells['index'] == cell_gid]
tar_x = tar_cell_df['x'].values[0]
tar_y = tar_cell_df['y'].values[0]

# Obtain retinotopic coordinates for the target cell by mapping the physical (x, y) coordinates of the cell
# to the visual space.
tar_center_x = cells['x'].mean()
tar_center_y = cells['y'].mean()
tar_L_x = cells['x'].max() - cells['x'].min()
tar_L_y = cells['y'].max() - cells['y'].min()

tar_vis_x = vis_center_x + ( (tar_x - tar_center_x) / tar_L_x ) * vis_tar_L_x
tar_vis_y = vis_center_y + ( (tar_y - tar_center_y) / tar_L_y ) * vis_tar_L_y 

# Information about ellipses representing the receptive subfields.
cell_el_tuning_angle = float(tar_cell_df['tuning'].values[0])
cell_el_centers_dv = (cell_el_centers_d*np.cos(np.deg2rad(cell_el_tuning_angle))/2.0, cell_el_centers_d*np.sin(np.deg2rad(cell_el_tuning_angle))/2.0)
center_ON = (tar_vis_x + cell_el_centers_dv[0], tar_vis_y + cell_el_centers_dv[1])
center_OFF = (tar_vis_x - cell_el_centers_dv[0], tar_vis_y - cell_el_centers_dv[1])
center_ON_OFF = (tar_vis_x, tar_vis_y)
cell_el_D_ON_OFF = cell_el_centers_d # For the ON/OF filters, we are using the diameter equal to the distance between the centers of the subfields for ON and OFF filters.

# Plot positions of all the LGN filters and of the selected cell (in the visual space).
fig, ax = plt.subplots()
for type in types_list:
    x = filters_data[filters_data['LGN_type'] == type]['x'].values
    y = filters_data[filters_data['LGN_type'] == type]['y'].values
    ax.scatter(x, y, s=500, c=type_colors_light[type], lw=0, label=type)

ax.scatter(tar_vis_x, tar_vis_y, marker='^', s=500, c='gray', label=type)

# Add the ellipses.
ell_ON = patches.Ellipse(xy=center_ON, width=cell_el_width, height=cell_el_height, angle = 180+cell_el_tuning_angle, facecolor='none', edgecolor='green', linestyle='dashed')
ax.add_patch(ell_ON)

ell_OFF = patches.Ellipse(xy=center_OFF, width=cell_el_width, height=cell_el_height, angle = 180+cell_el_tuning_angle, facecolor='none', edgecolor='magenta', linestyle='dashed')
ax.add_patch(ell_OFF)

ell_ON_OFF = patches.Ellipse(xy=center_ON_OFF, width=cell_el_D_ON_OFF, height=cell_el_D_ON_OFF, angle = 180+cell_el_tuning_angle, facecolor='none', edgecolor='blue', linestyle='dashed')
ax.add_patch(ell_ON_OFF)

ax.set_aspect('equal')
#ax.autoscale()
#plt.gca().set_aspect('equal')
ax.set_xlim(cell_xlim)
ax.set_ylim(cell_ylim)

#plt.legend()
ax.set_title('Cell gid %d' % (cell_gid))


# Plot positions of the input LGN filters for the selected cell.
src_types = list(set(tmp_df['src_type'].values))
for type in src_types:
    x = tmp_df[tmp_df['src_type'] == type]['src_vis_x'].values
    y = tmp_df[tmp_df['src_type'] == type]['src_vis_y'].values
    ax.scatter(x, y, s=500, c=type_colors[type], label=type)

plt.savefig('plot_LGN_vis_space_positions/sel_process_ll2_LGN_src.eps', format='eps')
plt.show()

# Plot the approximate extent of the receptive fields of individual source filters.
fig, ax = plt.subplots()
ax.scatter(tar_vis_x, tar_vis_y, marker='^', s=500, c='gray', label=type)
for type in ['transient_ON', 'transient_OFF', 'transient_ON_OFF']: # Use a list of type names to make sure transient_ON_OFF is being used the last, for better visibility.
    x = tmp_df[tmp_df['src_type'] == type]['src_vis_x'].values
    y = tmp_df[tmp_df['src_type'] == type]['src_vis_y'].values
    src_gid = tmp_df[tmp_df['src_type'] == type]['src_gid'].values
    src_sigma_c = filters_data.ix[src_gid]['sigma_c'].values
    src_sigma_s = filters_data.ix[src_gid]['sigma_s'].values
    src_x_offset = filters_data.ix[src_gid]['x_offset'].values
    src_y_offset = filters_data.ix[src_gid]['y_offset'].values
    # Plot the receptive fields of the individual filters, using circles to approximate double Gaussians.
    # Note that the double Gaussian we are using (Ac = 1.2, As = 0.2, sigma_s = 2 * sigma_c) reaches zero before turning negative
    # at ~2*sigma_c and approaches zero again after reaching the minimum (negative) at ~5*sigma_s.
    for i in range(x.size):
        ell_tmp = patches.Circle((x[i] + src_x_offset[i]/2.0, y[i] + src_y_offset[i]/2.0), radius=2*src_sigma_c[i], alpha=0.1, color=type_colors[type]) #, hatch='///')
        ax.add_patch(ell_tmp)
        ell_tmp = patches.Circle((x[i] - src_x_offset[i]/2.0, y[i] - src_y_offset[i]/2.0), radius=2*src_sigma_c[i], linewidth=1, alpha=0.1, color=type_colors[type]) #, hatch='///')
        ax.add_patch(ell_tmp)

ax.set_aspect('equal')
ax.set_xlim(cell_xlim)
ax.set_ylim(cell_ylim)

plt.savefig('plot_LGN_vis_space_positions/sel_process_ll2_LGN_src_RF.eps', format='eps')

plt.show()

