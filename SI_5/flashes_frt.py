import pickle
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import f_rate_t_by_type_functions as frtbt

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import matplotlib.gridspec as gspec

bin_start = 0.0
bin_stop = 1500.0
bin_size = 2.0

N_trials = 10

list_of_types = []

#result_fname_prefix = 'spont_activity/av_spont_rates_by_type'
#result_fname = result_fname_prefix + '.csv'
#result_fig_fname = result_fname_prefix + '.eps'
#t_av_start = 500.0
#t_av_stop = 1000.0


# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/flashes/output_ll2_flash_2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'flashes/ll2_flash_2.pkl', 'types': [] }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/flashes/output_ll3_flash_2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'flashes/ll3_flash_2.pkl', 'types': [] }
sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'f_1': '../simulations_rl2/flashes/output_rl2_flash_2_', 'f_2': '_sd285/spk.dat', 'f_3': '_sd285/tot_f_rate.dat', 'f_out': 'flashes/rl2_flash_2.pkl', 'types': [] }
sys_dict['rl3'] = { 'cells_file': '../build/rl3.csv', 'f_1': '../simulations_rl3/flashes/output_rl3_flash_2_', 'f_2': '_sd285/spk.dat', 'f_3': '_sd285/tot_f_rate.dat', 'f_out': 'flashes/rl3_flash_2.pkl', 'types': [] }
sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'f_1': '../simulations_lr2/flashes/output_lr2_flash_2_', 'f_2': '_sd287_cn0/spk.dat', 'f_3': '_sd287_cn0/tot_f_rate.dat', 'f_out': 'flashes/lr2_flash_2.pkl', 'types': [] }
sys_dict['lr3'] = { 'cells_file': '../build/lr3.csv', 'f_1': '../simulations_lr3/flashes/output_lr3_flash_2_', 'f_2': '_sd287_cn0/spk.dat', 'f_3': '_sd287_cn0/tot_f_rate.dat', 'f_out': 'flashes/lr3_flash_2.pkl', 'types': [] }
sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_1': '../simulations_rr2/flashes/output_rr2_flash_2_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'flashes/rr2_flash_2.pkl', 'types': [] }
sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'f_1': '../simulations_rr3/flashes/output_rr3_flash_2_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'flashes/rr3_flash_2.pkl', 'types': [] }


for sys_name in sys_dict.keys():
    # Obtain information about cell types.
    gids_by_type = frtbt.construct_gids_by_type_dict(sys_dict[sys_name]['cells_file'])
    sys_dict[sys_name]['types'] = gids_by_type.keys()
'''
    # Process the spike files and save computed firing rates in files.
    f_list = []
    for i_trial in xrange(0, N_trials):
        f_list.append('%s%d%s' % (sys_dict[sys_name]['f_1'], i_trial, sys_dict[sys_name]['f_2']))
    frtbt.f_rate_t_by_type(gids_by_type, bin_start, bin_stop, bin_size, f_list, sys_dict[sys_name]['f_out'])
'''
'''
# Compute averages and standard deviations by type and save to file.
result_f = open(result_fname, 'w')
result_f.write('system cell_type av_rate std\n')

for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_data = pickle.load(f)
    f.close()

    ind = np.intersect1d( np.where( rates_data['t_f_rate'] > t_av_start ), np.where( rates_data['t_f_rate'] < t_av_stop ) )

    for type in sys_dict[sys_name]['types']:
        result_string = '%s %s %f %f' % (sys_name, type, rates_data['mean'][type][ind].mean(), rates_data['mean'][type][ind].std())
        result_f.write(result_string + '\n')
        print result_string

result_f.close()
'''

# Plot time series of average (over cells) firing rates by type.
rates_dict = {}
for sys_name in sys_dict.keys():
    f = open(sys_dict[sys_name]['f_out'], 'r')
    rates_dict[sys_name] = pickle.load(f)
    f.close()

plot_types_color = {'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan'}
'''
# Plot results for different systems together, separately for each type.
#for type in sys_dict[sys_dict.keys()[0]]['types']: # Use types from the first system; assume that all systems have those types.
#    for sys_name in sys_dict.keys():
#        plt.plot(rates_dict[sys_name]['t_f_rate'], rates_dict[sys_name]['mean'][type], c='darkorange', label=sys_name)
#    plt.title('Type %s' % (type))

# Plot results for several types together, separately for each system.
for sys_name in sys_dict.keys():
    for type in ['Scnn1a', 'Rorb', 'Nr5a1']:
        plt.plot(rates_dict[sys_name]['t_f_rate'], rates_dict[sys_name]['mean'][type], c=plot_types_color[type], label=type)
    plt.title('%s' % (sys_name))

    plt.xlabel('Time (ms)')
    plt.ylabel('Mean firing rate (Hz)')
    plt.legend()
    plt.xlim((600.0, 1050.0))
    plt.show()
'''

# Compute the average trace (over excitatory types and over models).
# Also, compute the average peak and time to peak.
N_av = 0
t_flash_onset = 600.0

# Decide how many peaks we would like to process.
N_peaks = 2
peaks = []
peaks_t = []
t_start_peaks = []
t_stop_peaks = []
for i in xrange(N_peaks):
    peaks.append([])
    peaks_t.append([])
# Parameters for computing the first peak.
t_start_peaks.append(600.0)
t_stop_peaks.append(700.0)
# Parameters for computing the second peak.
t_start_peaks.append(700.0)
t_stop_peaks.append(900.0)


#fig_out = 'flashes/flashes_frt_ll2_ll3_exc_flash_2.eps'
#sys_list = ['ll2', 'll3']
#type_list = ['Scnn1a', 'Rorb', 'Nr5a1']
#sim_color = 'firebrick'
#exp_color = 'gray'
#y_lim_top = [55.0, 380.0]
#y_ticks_psth = np.arange(0.0, 31.0, 10.0)
# Load the experimental peak and time-to-peak values.
#exp_peaks = []
#exp_peaks_t = []
#exp_peaks.append(np.load('exp_flashes/average_peaks_1.npy'))
#exp_peaks.append(np.load('exp_flashes/average_peaks_2.npy'))
#exp_peaks_t.append(np.load('exp_flashes/average_time_to_peak_1.npy'))
#exp_peaks_t.append(np.load('exp_flashes/average_time_to_peak_2.npy'))

fig_out = 'flashes/flashes_frt_ll2_ll3_inh_flash_2.eps'
sys_list = ['ll2', 'll3']
type_list = ['PV1', 'PV2']
sim_color = 'steelblue'
exp_color = 'gray'
y_lim_top = [100.0, 450.0]
y_ticks_psth = np.arange(0.0, 90.0, 20.0)
# Load the experimental peak and time-to-peak values.
exp_peaks = []
exp_peaks_t = []
exp_peaks.append(np.load('exp_flashes/average_peaks_1_inh.npy'))
exp_peaks.append(np.load('exp_flashes/average_peaks_2_inh.npy'))
exp_peaks_t.append(np.load('exp_flashes/average_time_to_peak_1_inh.npy'))
exp_peaks_t.append(np.load('exp_flashes/average_time_to_peak_2_inh.npy'))


for sys_name in sys_list:
    for type in type_list:
        tmp = rates_dict[sys_name]['mean'][type]
        t_tmp = rates_dict[sys_name]['t_f_rate']
        if ('av_rate' not in dir()):
            av_rate = np.zeros(tmp.size)
        av_rate = av_rate + tmp
        N_av += 1
        for k_peak in xrange(len(peaks)):
            ind = np.intersect1d( np.where( t_tmp >= t_start_peaks[k_peak] ), np.where( t_tmp <= t_stop_peaks[k_peak] ) )
            tmp1 = tmp[ind]
            t_tmp1 = t_tmp[ind]
            ind_peak = tmp1.argmax()
            peaks[k_peak].append(tmp1[ind_peak])
            peaks_t[k_peak].append(t_tmp1[ind_peak])

av_rate = av_rate / (1.0 * N_av)

for k_peak in xrange(len(peaks)):
    peaks[k_peak] = np.array(peaks[k_peak])
    peaks_t[k_peak] = np.array(peaks_t[k_peak])
    peaks_t[k_peak] = peaks_t[k_peak] - t_flash_onset # Compute the time to peak relative to the onset of the flash.


# Plot the time-dependent average firing rate, average peak, and average time to peak.
fig = plt.figure(figsize = (12, 7))

gs = gspec.GridSpec(2, 2)
ax = []
ax.append(fig.add_subplot(gs[:,0]))
ax.append(fig.add_subplot(gs[0,1]))
ax.append(fig.add_subplot(gs[1,1]))

ax[0].plot(rates_dict['ll2']['t_f_rate'], av_rate, c=sim_color)
ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Mean firing rate (Hz)')
ax[0].set_xlim((550.0, 920.0))

ax[0].set_xticks(np.arange(600.0, 901.0, 100.0))
ax[0].set_yticks(y_ticks_psth)

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].yaxis.set_ticks_position('left')
ax[0].xaxis.set_ticks_position('bottom')
ax[0].tick_params(size=10)


exp_displ = 0.25 # Displacement of the experimental numbers in comparison with simulation numbers for plotting along the x-axis (which simply reflects the order in which items are plotted).

for i_peak in xrange(len(peaks)):
    ax[1].scatter(i_peak*np.ones(peaks[i_peak].size), peaks[i_peak], s=150, lw=0, facecolor=sim_color)
    ind = np.where(peaks[i_peak] > y_lim_top[0])[0]
    ax[1].annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, peaks[i_peak].size), xy=(i_peak, 1.0*y_lim_top[0]), fontsize=12)

    ax[1].scatter(i_peak*np.ones(exp_peaks[i_peak].size)+exp_displ, exp_peaks[i_peak], s=150, lw=0, facecolor=exp_color)
    ind = np.where(exp_peaks[i_peak] > y_lim_top[0])[0]
    ax[1].annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, exp_peaks[i_peak].size), xy=(i_peak+exp_displ, 1.0*y_lim_top[0]), fontsize=12)

    ax[2].scatter(i_peak*np.ones(peaks_t[i_peak].size), peaks_t[i_peak], s=150, lw=0, facecolor=sim_color)
    ind = np.where(peaks_t[i_peak] > y_lim_top[1])[0]
    ax[2].annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, peaks_t[i_peak].size), xy=(i_peak, 1.0*y_lim_top[1]), fontsize=12)

    ax[2].scatter(i_peak*np.ones(exp_peaks_t[i_peak].size)+exp_displ, exp_peaks_t[i_peak], s=150, lw=0, facecolor=exp_color)
    ind = np.where(exp_peaks_t[i_peak] > y_lim_top[1])[0]
    ax[2].annotate(u'$\u2191$'+'\n%d/%d' % (ind.size, exp_peaks_t[i_peak].size), xy=(i_peak+exp_displ, 1.0*y_lim_top[1]), fontsize=12)

ax[1].errorbar(range(len(peaks)), [x.mean() for x in peaks], yerr = [x.std()/np.sqrt(1.0*x.size) for x in peaks], marker='o', ms=5, color='k', linewidth=0, capsize=8, markeredgewidth=2, ecolor='k', elinewidth=2)
ax[1].errorbar(np.array(range(len(exp_peaks))) + exp_displ, [x.mean() for x in exp_peaks], yerr = [x.std()/np.sqrt(1.0*x.size) for x in exp_peaks], marker='o', ms=5, color='k', linewidth=0, capsize=8, markeredgewidth=2, ecolor='k', elinewidth=2)
ax[1].set_xticks([])
#ax[1].set_xticklabels([])
ax[1].set_xlim([-0.5, len(peaks) - 0.5])
ax[1].set_ylim([0.0, y_lim_top[0]])
ax[1].set_yticks(np.arange(0.0, y_lim_top[0], 20.0))
ax[1].set_ylabel('Peak (Hz)')

ax[2].errorbar(range(len(peaks_t)), [x.mean() for x in peaks_t], yerr = [x.std()/np.sqrt(1.0*x.size) for x in peaks_t], marker='o', ms=5, color='k', linewidth=0, capsize=8, markeredgewidth=2, ecolor='k', elinewidth=2)
ax[2].errorbar(np.array(range(len(exp_peaks_t))) + exp_displ, [x.mean() for x in exp_peaks_t], yerr = [x.std()/np.sqrt(1.0*x.size) for x in exp_peaks_t], marker='o', ms=5, color='k', linewidth=0, capsize=8, markeredgewidth=2, ecolor='k', elinewidth=2)
ax[2].set_xticks([])
#ax[2].set_xticklabels([])
ax[2].set_xlim([-0.5, len(peaks) - 0.5])
ax[2].set_ylim([0.0, y_lim_top[1]])
ax[2].set_yticks(np.arange(0.0, y_lim_top[1], 100.0))
ax[2].set_ylabel('Time to peak (ms)')

for i in [1, 2]:
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].yaxis.set_ticks_position('left')
    ax[i].xaxis.set_ticks_position('bottom')
    ax[i].tick_params(size=10)

plt.savefig(fig_out, format='eps')
plt.show()



