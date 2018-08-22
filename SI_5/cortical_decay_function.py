import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

def exp_function(t, params):
    (f_min, f_max, tau, t0) = params
    return f_min + f_max * np.exp( (t0 - t) / tau )


def exp_function_fit( params, t, y_av ):
    return (y_av - exp_function( t, params ))


def exp_fit_within_t_bounds(t, a_mean, t_fit_start, t_fit_end, tau_start):
    ind = np.where( t >= t_fit_start )[0]
    t_tmp = t[ind]
    a_tmp = a_mean[ind]
    ind1 = np.where( t_tmp <= t_fit_end )[0]
    t_tmp = t_tmp[ind1]
    a_tmp = a_tmp[ind1]

    params_start = [a_tmp[-1], a_tmp[0], 3.0, t_fit_start]

    return t_tmp, leastsq( exp_function_fit, params_start, args=(t_tmp, a_tmp) )


def comp_mu_activity(f_spk, cells_file, types, bin_start, bin_stop, bin_size):
    bins = np.arange(bin_start, bin_stop+bin_size, bin_size)
    # Originally bins include both leftmost and rightmost bin edges.
    # Here we remove the rightmost edge to make the size consistent with the result computed below.
    t_out = bins[:-1]

    print 'Processing file %s.' % (f_spk)
    data = np.genfromtxt(f_spk, delimiter=' ')
    if (data.size == 0):
        t = np.array([])
        gids = np.array([])
    elif (data.size == 2):
        t = np.array([data[0]])
        gids = np.array([data[1]])
    else:
        t = data[:, 0]
        gids = data[:, 1]
    # The gids array is obtained as float64.  We should convert it to int.
    gids = gids.astype(np.int64)

    # Read information about cells.
    cells_db = pd.read_csv(cells_file, sep=' ')
    # Sort cells by gid ('index') to make sure the gids can be used as indices of numpy arrays later.
    # For those later operations, we assume that gids are increasing strictly by 1 and start from 0.
    cells_db = cells_db.sort('index', ascending=True)
    # Find cells that belong to the types of interest.
    gids_sel = cells_db[cells_db['type'].isin(types)]['index'].values

    mask = np.in1d(gids, gids_sel)

    a_out = np.histogram(t[mask], bins=bins)[0] # Here, np.histogram returns a tuple, where the first element is the histogram itself and the second is bins.
    a_out = 1000.0 * a_out / ( gids_sel.size * bin_size ) # Use this conversion to ensure that the result is the mean firing rate in Hz (time is in ms).

    return t_out, a_out





sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../output_ll1_', 'f_2': '_stop_1s_sd278/spk.dat', 'f_out': 'cortical_decay_function/ll1_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll1.eps', 'grating_ids': [8] }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278/spk.dat', 'f_out': 'cortical_decay_function/ll2_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2.eps', 'grating_ids': [8] }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../output_ll3_', 'f_2': '_stop_1s_sd278/spk.dat', 'f_out': 'cortical_decay_function/ll3_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll3.eps', 'grating_ids': [8] }

sys_dict['ll2_varEtE_1.2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278_varEtE_1.2/spk.dat', 'f_out': 'cortical_decay_function/ll2_varEtE_1.2_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2_varEtE_1.2.eps', 'grating_ids': [8] }
sys_dict['ll2_varEtE_1.4'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278_varEtE_1.4/spk.dat', 'f_out': 'cortical_decay_function/ll2_varEtE_1.4_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2_varEtE_1.4.eps', 'grating_ids': [8] }
sys_dict['ll2_varEtE_1.6'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278_varEtE_1.6/spk.dat', 'f_out': 'cortical_decay_function/ll2_varEtE_1.6_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2_varEtE_1.6.eps', 'grating_ids': [8] }
sys_dict['ll2_varEtE_1.8'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278_varEtE_1.8/spk.dat', 'f_out': 'cortical_decay_function/ll2_varEtE_1.8_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2_varEtE_1.8.eps', 'grating_ids': [8] }
sys_dict['ll2_varEtE_2.0'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_stop_1s_sd278_varEtE_2.0/spk.dat', 'f_out': 'cortical_decay_function/ll2_varEtE_2.0_cdf.npy', 'figname': 'cortical_decay_function/cdf_ll2_varEtE_2.0.eps', 'grating_ids': [8] }

fig_out_name = 'cortical_decay_function/av_cdf_ll1_ll2_ll3.eps'

# Process all the files and obtain multi-unit activity as a function of time; save that to files.
for sys_name in sys_dict.keys():
    N_files = 0
    for grating_id in sys_dict[sys_name]['grating_ids']:
        for trial in xrange(0, 10):
            t, a = comp_mu_activity('%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, trial, sys_dict[sys_name]['f_2']), sys_dict[sys_name]['cells_file'], ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2', 'LIF_exc', 'LIF_inh'], 0.0, 2000.0, 1.0)
            if 'a_mean' in dir():
                a_mean = a_mean + a
            else:
                a_mean = a
            N_files += 1

    a_mean = a_mean / (1.0 * N_files)
    #plt.plot(t, a_mean)
    #plt.show()
    np.save( sys_dict[sys_name]['f_out'], np.vstack((t, a_mean)) )

# Read the multi-unit activity from files and fit the decay with an exponent.
t_fit_start = 1000.0 # Time for beginning of fitting, in ms.
t_fit_end = 1050.0 # Time for end of fitting, in ms.
tau_start = 3.0 # Initial value for the time constant, before optimization.

for sys_name in sys_dict.keys():
    data = np.load(sys_dict[sys_name]['f_out'])
    t = data[0]
    a_mean = data[1]
    if 'a_av_sys' in dir():
        a_av_sys = a_av_sys + a_mean
    else:
        a_av_sys = a_mean

    t_tmp, ((f_min, f_max, tau, t0), tmp) = exp_fit_within_t_bounds(t, a_mean, t_fit_start, t_fit_end, tau_start)

    plt.plot(t, a_mean)
    plt.plot(t_tmp, f_min + f_max * np.exp( (t0 - t_tmp) / tau ))
    plt.title('tau = %f ms' % (tau))
    plt.annotate('f_min = %f, f_max = %f, t0 = %f' % (f_min, f_max, t0), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=8)
    plt.xlim((990.0, 1020.0))
    plt.ylim((0.0, 8.0))
    plt.savefig(sys_dict[sys_name]['figname'], format='eps')
    plt.show()

# Average from all the systems.
a_av_sys = a_av_sys / (1.0 * len(sys_dict.keys()))
t_tmp, ((f_min, f_max, tau, t0), tmp) = exp_fit_within_t_bounds(t, a_av_sys, t_fit_start, t_fit_end, tau_start)

plt.plot(t, a_av_sys)
plt.plot(t_tmp, f_min + f_max * np.exp( (t0 - t_tmp) / tau ))
plt.title('tau = %f ms' % (tau))
plt.annotate('f_min = %f, f_max = %f, t0 = %f' % (f_min, f_max, t0), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=8)
plt.xlim((990.0, 1020.0))
plt.ylim((0.0, 8.0))
plt.savefig(fig_out_name, format='eps')
plt.show()

