import pickle
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scp_stats

import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 15})


def box_plot_data(tot_df, label, units, type_order, type_color, y_lim_top, out_fig_name):

    # Drop NaN elements.
    tmp_df = tot_df[tot_df[label].notnull()]

    # Arrange data into a list of numpy arrays.
    type_data = []
    for type_key in type_order:
        type_data.append(tmp_df[tmp_df['type']==type_key][label].values)

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

    if (units == ''):
        ax.set_ylabel('%s' % (label))
    else:
        ax.set_ylabel('%s (%s)' % (label, units))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(size=10)

    plt.savefig(out_fig_name, format='eps')

    plt.show()



# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_out': 'Ori/ll1_rates.npy', 'f_out_pref': 'Ori/ll1_pref_stat.csv'}
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_out': 'Ori/ll2_rates.npy', 'f_out_pref': 'Ori/ll2_pref_stat.csv'}
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_out': 'Ori/ll3_rates.npy', 'f_out_pref': 'Ori/ll3_pref_stat.csv'}
#sys_dict['rl1'] = { 'cells_file': '../build/rl1.csv', 'f_out': 'Ori/rl1_rates.npy', 'f_out_pref': 'Ori/rl1_pref_stat.csv'}
#sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'f_out': 'Ori/rl2_rates.npy', 'f_out_pref': 'Ori/rl2_pref_stat.csv'}
#sys_dict['rl3'] = { 'cells_file': '../build/rl3.csv', 'f_out': 'Ori/rl3_rates.npy', 'f_out_pref': 'Ori/rl3_pref_stat.csv'}
#sys_dict['lr1'] = { 'cells_file': '../build/lr1.csv', 'f_out': 'Ori/lr1_rates.npy', 'f_out_pref': 'Ori/lr1_pref_stat.csv'}
#sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'f_out': 'Ori/lr2_rates.npy', 'f_out_pref': 'Ori/lr2_pref_stat.csv'}
#sys_dict['lr3'] = { 'cells_file': '../build/lr3.csv', 'f_out': 'Ori/lr3_rates.npy', 'f_out_pref': 'Ori/lr3_pref_stat.csv'}
#sys_dict['rr1'] = { 'cells_file': '../build/rr1.csv', 'f_out': 'Ori/rr1_rates.npy', 'f_out_pref': 'Ori/rr1_pref_stat.csv'}
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_out': 'Ori/rr2_rates.npy', 'f_out_pref': 'Ori/rr2_pref_stat.csv'}
#sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'f_out': 'Ori/rr3_rates.npy', 'f_out_pref': 'Ori/rr3_pref_stat.csv'}
#sys_dict['ll2_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_out': 'Ori/ll2_rates_4Hz.npy', 'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv' }
sys_dict['ll1_LGN_only_no_con'] = { 'cells_file': '../build/ll1.csv', 'f_out': 'Ori/ll1_LGN_only_no_con_rates.npy', 'f_out_pref': 'Ori/ll1_LGN_only_no_con_pref_stat.csv' }
sys_dict['ll2_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_out': 'Ori/ll2_LGN_only_no_con_rates.npy', 'f_out_pref': 'Ori/ll2_LGN_only_no_con_pref_stat.csv' }
sys_dict['ll3_LGN_only_no_con'] = { 'cells_file': '../build/ll3.csv', 'f_out': 'Ori/ll3_LGN_only_no_con_rates.npy', 'f_out_pref': 'Ori/ll3_LGN_only_no_con_pref_stat.csv' }
#sys_dict['ll1_LIF'] = { 'cells_file': '../build/ll1.csv', 'f_out': 'Ori_LIF/ll1_rates.npy', 'f_out_pref': 'Ori_LIF/ll1_pref_stat.csv'}
#sys_dict['ll2_LIF'] = { 'cells_file': '../build/ll2.csv', 'f_out': 'Ori_LIF/ll2_rates.npy', 'f_out_pref': 'Ori_LIF/ll2_pref_stat.csv'}
#sys_dict['ll3_LIF'] = { 'cells_file': '../build/ll3.csv', 'f_out': 'Ori_LIF/ll3_rates.npy', 'f_out_pref': 'Ori_LIF/ll3_pref_stat.csv'}
#sys_dict['rl1_LIF'] = { 'cells_file': '../build/rl1.csv', 'f_out': 'Ori_LIF/rl1_rates.npy', 'f_out_pref': 'Ori_LIF/rl1_pref_stat.csv'}
#sys_dict['rl2_LIF'] = { 'cells_file': '../build/rl2.csv', 'f_out': 'Ori_LIF/rl2_rates.npy', 'f_out_pref': 'Ori_LIF/rl2_pref_stat.csv'}
#sys_dict['rl3_LIF'] = { 'cells_file': '../build/rl3.csv', 'f_out': 'Ori_LIF/rl3_rates.npy', 'f_out_pref': 'Ori_LIF/rl3_pref_stat.csv'}
#sys_dict['lr1_LIF'] = { 'cells_file': '../build/lr1.csv', 'f_out': 'Ori_LIF/lr1_rates.npy', 'f_out_pref': 'Ori_LIF/lr1_pref_stat.csv'}
#sys_dict['lr2_LIF'] = { 'cells_file': '../build/lr2.csv', 'f_out': 'Ori_LIF/lr2_rates.npy', 'f_out_pref': 'Ori_LIF/lr2_pref_stat.csv'}
#sys_dict['lr3_LIF'] = { 'cells_file': '../build/lr3.csv', 'f_out': 'Ori_LIF/lr3_rates.npy', 'f_out_pref': 'Ori_LIF/lr3_pref_stat.csv'}
#sys_dict['rr1_LIF'] = { 'cells_file': '../build/rr1.csv', 'f_out': 'Ori_LIF/rr1_rates.npy', 'f_out_pref': 'Ori_LIF/rr1_pref_stat.csv'}
#sys_dict['rr2_LIF'] = { 'cells_file': '../build/rr2.csv', 'f_out': 'Ori_LIF/rr2_rates.npy', 'f_out_pref': 'Ori_LIF/rr2_pref_stat.csv'}
#sys_dict['rr3_LIF'] = { 'cells_file': '../build/rr3.csv', 'f_out': 'Ori_LIF/rr3_rates.npy', 'f_out_pref': 'Ori_LIF/rr3_pref_stat.csv'}



result_fig_prefix = 'Ori/new_Ori_ll_LGN_only_no_con'
result_fig_CV_ori = result_fig_prefix + '_CV_ori.eps'
result_fig_DSI = result_fig_prefix + '_DSI.eps'


type_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan', 'AnL4E': 'gray', 'AwL4E': 'gray', 'AnI': 'gray', 'AwI': 'gray'}
type_order = ['Scnn1a', 'Rorb', 'Nr5a1', 'AnL4E', 'AwL4E', 'PV1', 'PV2', 'AnI', 'AwI']

# Read files with OSI and DSI from simulations.
sim_df = pd.DataFrame()
for sys_name in sys_dict.keys():
    tmp_df = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')

    cells_df = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    cells_df_1 = pd.DataFrame()
    cells_df_1['id'] = cells_df['index'].values
    cells_df_1['type'] = cells_df['type'].values

    tmp_df = pd.merge(tmp_df, cells_df_1, on='id', how='inner')

    # Combine dataframes from all systems into one file.
    sim_df = pd.concat([sim_df, tmp_df], axis=0)


sim_df_1 = pd.DataFrame()
sim_df_1['gid'] = sim_df['id'].values
sim_df_1['type'] = sim_df['type'].values
sim_df_1['CV_ori'] = sim_df['CV_ori'].values
sim_df_1['DSI'] = sim_df['DSI'].values


# Read file with OSI and DSI from experiments.
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
exp_df_1['CV_ori'] = exp_df['CV_ori'].values
exp_df_1['DSI'] = exp_df['DSI'].values

tot_df = pd.concat([sim_df_1, exp_df_1], axis=0)


label = 'CV_ori'
units = ''
y_lim_top = 1.0
out_fig_name = result_fig_CV_ori
box_plot_data(tot_df, label, units, type_order, type_color, y_lim_top, out_fig_name)

label = 'DSI'
units = ''
y_lim_top = 1.0
out_fig_name = result_fig_DSI
box_plot_data(tot_df, label, units, type_order, type_color, y_lim_top, out_fig_name)


