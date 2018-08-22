import oscillations
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 20})

bin_start = 500.0
bin_stop = 3000.0
bin_size = 1.0

electrode_pos = [0.0, 0.0, 0.0]
r_cutoff = 10.0 # Distance, in um, below which the weights for 1/r contributions are set to 0.

N_trials = 10





# Decide which systems we are doing analysis for.
sys_dict = {}

#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/gratings/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_out_prefix': 'oscillations/ll1_spectrum', 'grating_ids': [range(6, 240, 30), range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '--' }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278/spk.dat', 'f_out_prefix': 'oscillations/ll2_spectrum', 'grating_ids': [range(7, 240, 30), range(8, 240, 30), range(9, 240, 30)], 'marker': '-' }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/gratings/output_ll3_', 'f_2': '_sd278/spk.dat', 'f_out_prefix': 'oscillations/ll3_spectrum', 'grating_ids': [range(8, 240, 30)], 'marker': ':' }

output_fig = 'oscillations/oscillations_ll.eps'


# Process the data and obtain spectra.
for sys_name in sys_dict.keys():
    for gratings_list in sys_dict[sys_name]['grating_ids']:
        f_spk_list = []
        cells_file_list = []
        for grating_id in gratings_list:
            for trial in xrange(N_trials):
                f_spk_list.append('%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, trial, sys_dict[sys_name]['f_2']))
                cells_file_list.append(sys_dict[sys_name]['cells_file'])
        tmp = oscillations.av_r_weighted_mu_activity(f_spk_list, cells_file_list, electrode_pos, r_cutoff, bin_start, bin_stop, bin_size)
        f = open('%s_%d.pkl' % (sys_dict[sys_name]['f_out_prefix'], gratings_list[0]), 'w')
        pickle.dump(tmp, f)
        f.close()

# Plot the results.
for sys_name in sys_dict.keys():
    grating_start = 8
    f_name = '%s_%d.pkl' % (sys_dict[sys_name]['f_out_prefix'], grating_start)
    f = open(f_name, 'r')
    freq_fft_abs, av_fft_abs, std_fft_abs = pickle.load(f)
    f.close()
    ind = np.intersect1d( np.where(freq_fft_abs > 0.0), np.where(freq_fft_abs < 100.0) )
    #plt.errorbar(freq_fft_abs[ind], av_fft_abs[ind], yerr=std_fft_abs[ind], marker=sys_dict[sys_name]['marker'], ms=10, markevery=5, color='k', linewidth=2, capsize=0, ecolor='lightgray', elinewidth=5, label=f_name)
    plt.errorbar(freq_fft_abs[ind], 1000.0*av_fft_abs[ind], yerr=1000.0*std_fft_abs[ind], ls=sys_dict[sys_name]['marker'], color='k', linewidth=2, capsize=0, ecolor='lightgray', elinewidth=5, label=f_name)

plt.legend()

plt.ylabel('Power (arb. u.)')
plt.xlabel('Frequency (Hz)')

plt.savefig(output_fig, format='eps')

plt.show()


