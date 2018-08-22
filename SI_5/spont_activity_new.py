import pickle
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scp_stats

import pandas as pd

import matplotlib
matplotlib.rcParams.update({'font.size': 20})

N_trials = 20

type_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan', 'Exp.E.': 'gray', 'Exp.I.': 'gray'}
type_order = ['Scnn1a', 'Rorb', 'Nr5a1', 'Exp.E.', 'PV1', 'PV2', 'Exp.I.']

# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/spont/output_ll1_spont_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'spont_activity/ll1_spont.csv', 'types': [] }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/spont/output_ll2_spont_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'spont_activity/ll2_spont.csv', 'types': [] }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/spont/output_ll3_spont_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'spont_activity/ll3_spont.csv', 'types': [] }
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_1': '../output_rr2_spont_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'spont_activity/rr2_spont.csv', 'types': [] }


result_fname_prefix = 'spont_activity/new_av_spont_rates_by_type'
result_fig_fname = result_fname_prefix + '.eps'


# Read files containing firing rates for each trial, average over all trials, and save to file.
for i_sys, sys_name in enumerate(sys_dict.keys()):
    # Obtain information about cell types.
    cells  = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    gids = cells['index'].values
    out_df = pd.DataFrame({'gid': gids, 'type': cells['type'].values})

    # Process the firing rate files.
    rates = np.zeros(gids.size)
    for i_trial in xrange(0, N_trials):
        f_name = '%s%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_3'])
        print 'Processing file %s.' % (f_name)
        tmp_rates = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.        
        rates += tmp_rates
    rates = rates / (1.0 * N_trials)
    out_df['%s_frate' % (sys_name)] = rates
    out_df.to_csv(sys_dict[sys_name]['f_out'], sep=' ', index=False)


# Read files with firing rate averages over trials for simulations.
rates_df = pd.DataFrame()
for sys_name in sys_dict.keys():
    tmp_df = pd.read_csv(sys_dict[sys_name]['f_out'], sep=' ')
    # Combine firing rates from all systems into one file.
    tmp_df.rename(columns={'%s_frate' % (sys_name): 'frate'}, inplace=True)
    rates_df = pd.concat([rates_df, tmp_df], axis=0)

# Read file with firing rate averages over trials for experiments.
exp_df = pd.read_csv('exp_spont/f_avg_per_cell_Spont.csv', sep=' ', header=None)
exp_df.columns = ['frate', 'EI']
exp_df['gid'] = -1
exp_df['type'] = ''
exp_df.ix[exp_df['EI']==0.0, 'type'] = 'Exp.E.'
exp_df.ix[exp_df['EI']==1.0, 'type'] = 'Exp.I.'
del exp_df['EI']

rates_df = pd.concat([rates_df, exp_df], axis=0)


# Arrange data into a list of numpy arrays.
type_rates = []
for type_key in type_order:
    type_rates.append(rates_df[rates_df['type']==type_key]['frate'].values)

# Plot the results.
y_lim_top = 2.5

fig, ax = plt.subplots(figsize = (10, 5))

box = ax.boxplot(type_rates, patch_artist=True, sym='c.') # notch=True
for patch, color in zip(box['boxes'], [type_color[type_key] for type_key in type_order]):
    patch.set_facecolor(color)

for i, type_key in enumerate(type_order):
    ax.errorbar([i+1], [type_rates[i].mean()], yerr=[type_rates[i].std() / np.sqrt(1.0 * type_rates[i].size)], marker='o', ms=8, color='k', linewidth=2, capsize=5, markeredgewidth=2, ecolor='k', elinewidth=2)
    ind = np.where(type_rates[i] > y_lim_top)[0]
    ax.annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, type_rates[i].size), xy=(i+1.2, 1.0*y_lim_top), fontsize=12)

ax.set_ylim((0.0, y_lim_top))
ax.set_xticks(range(1, len(type_order)+1))
ax.set_xticklabels(type_order)

ax.set_ylabel('Spontaneous rate (Hz)')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(size=10)

plt.savefig(result_fig_fname, format='eps')

plt.show()


