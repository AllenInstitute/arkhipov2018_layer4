import pickle
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scp_stats

import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 15})


N_trials = 10

# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/gratings/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Rmax/ll1_Rmax.csv', 'grating_ids': range(6, 240, 30)+range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Rmax/ll2_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/gratings/output_ll3_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Rmax/ll3_Rmax.csv', 'grating_ids': range(8, 240, 30) }
#sys_dict['ll1_LIF'] = { 'cells_file': '../build/ll1.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll1/output_ll1_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Rmax_LIF/ll1_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2_LIF'] = { 'cells_file': '../build/ll2.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll2/output_ll2_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Rmax_LIF/ll2_Rmax.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3_LIF'] = { 'cells_file': '../build/ll3.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll3/output_ll3_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Rmax_LIF/ll3_Rmax.csv', 'grating_ids': range(8, 240, 30) }
sys_dict['ll1_LGN_only_no_con'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/gratings/output_ll1_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Rmax/ll1_Rmax_LGN_only_no_con.csv', 'grating_ids': [7, 37]+range(97, 240, 30)+range(8, 240, 30) }
sys_dict['ll2_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Rmax/ll2_Rmax_LGN_only_no_con.csv', 'grating_ids': range(7, 240, 30)+range(8, 240, 30) }
sys_dict['ll3_LGN_only_no_con'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll3/gratings/output_ll3_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Rmax/ll3_Rmax_LGN_only_no_con.csv', 'grating_ids': range(7, 240, 30)+[8, 38]+range(98, 240, 30) }


result_fname_prefix = 'Rmax/new_Rmax_ll_LGN_only_no_con' #'Rmax_LIF/Rmax_by_type'
result_fig_fname = result_fname_prefix + '.eps'


# Load simulation data and obtain Rmax for each cell.
for sys_name in sys_dict.keys():
    gratings_rates = np.array([])
    print gratings_rates.shape
    for grating_id in sys_dict[sys_name]['grating_ids']:
        rates_tmp = np.array([])
        for i_trial in xrange(0, N_trials):
            f_name = '%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, i_trial, sys_dict[sys_name]['f_3'])
            print 'Processing file %s.' % (f_name)
            tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
            if (rates_tmp.size == 0):
                rates_tmp = tmp
            else:
                rates_tmp = rates_tmp + tmp
        rates_tmp = rates_tmp / (1.0 * N_trials)
        if (gratings_rates.size == 0):
            gratings_rates = rates_tmp
        else:
            gratings_rates = np.vstack((gratings_rates, rates_tmp))
    Rmax = np.amax(gratings_rates, axis = 0)

    sys_df = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    # Write Rmax to a file, with gid and cell type information.
    f_out = open(sys_dict[sys_name]['f_out'], 'w')
    f_out.write('gid type Rmax\n')
    for gid in xrange(0, Rmax.size):
        f_out.write('%d %s %f\n' % (gid, sys_df['type'].ix[gid], Rmax[gid]))
    f_out.close()






type_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan', 'AnL4E': 'gray', 'AwL4E': 'gray', 'AnI': 'gray', 'AwI': 'gray'}
type_order = ['Scnn1a', 'Rorb', 'Nr5a1', 'AnL4E', 'AwL4E', 'PV1', 'PV2', 'AnI', 'AwI']

# Read files with Rmax from simulations.
sim_df = pd.DataFrame()
for sys_name in sys_dict.keys():
    tmp_df = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    # Combine Rmax from all systems into one file.
    sim_df = pd.concat([sim_df, tmp_df], axis=0)


# Read files with Rmax from experiments.
exp_f = { 'AnL4E': 'exp_gratings/ANL4Exc.csv',
          'AwL4E': 'exp_gratings/AWL4Exc.csv',
          'AnI': 'exp_gratings/ANInh.csv',
          'AwI': 'exp_gratings/AWInh.csv' }
exp_df = pd.DataFrame()
for exp_key in exp_f:
    tmp_df = pd.read_csv(exp_f[exp_key], sep=',')
    tmp_df['type'] = exp_key
    tmp_df['gid'] = -1
    exp_df = pd.concat([exp_df, tmp_df], axis=0)

exp_df_1 = pd.DataFrame()
exp_df_1['gid'] = exp_df['gid'].values
exp_df_1['type'] = exp_df['type'].values
exp_df_1['Rmax'] = exp_df['Rmax'].values

tot_df = pd.concat([sim_df, exp_df_1], axis=0)


# Arrange data into a list of numpy arrays.
type_data = []
for type_key in type_order:
    type_data.append(tot_df[tot_df['type']==type_key]['Rmax'].values)

# Plot the results.
y_lim_top = 27.0

fig, ax = plt.subplots(figsize = (7, 5))

box = ax.boxplot(type_data, patch_artist=True, sym='c.') # notch=True
for patch, color in zip(box['boxes'], [type_color[type_key] for type_key in type_order]):
    patch.set_facecolor(color)

for i, type_key in enumerate(type_order):
    ax.errorbar([i+1], [type_data[i].mean()], yerr=[type_data[i].std() / np.sqrt(1.0 * type_data[i].size)], marker='o', ms=8, color='k', linewidth=2, capsize=5, markeredgewidth=2, ecolor='k', elinewidth=2)
    ind = np.where(type_data[i] > y_lim_top)[0]
    ax.annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, type_data[i].size), xy=(i+1.2, 1.0*y_lim_top), fontsize=12)

ax.set_ylim((0.0, y_lim_top))
ax.set_xticks(range(1, len(type_order)+1))
ax.set_xticklabels(type_order)

ax.set_ylabel('Rmax (Hz)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(size=10)

plt.savefig(result_fig_fname, format='eps')

plt.show()


