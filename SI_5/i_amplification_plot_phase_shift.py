import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

# A function to wrap the values from a 1D array around periodic boundary.
# Here, x should be  a numpy 1D array, and x_start and  x_end are scalars.
def periodic_shift(x, x_start, x_end):
    tmp_L = x_end - x_start
    if (tmp_L <= 0.0):
        print 'periodic_shift: tmp_L <= 0; change x_start and x_end; exiting.'
        quit()
    return (x - x_start) % tmp_L + x_start


t_av = [500.0, 3000.0]

sys_dict = {}

#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'tot_path': '../simulations_ll1/gratings/output_ll1_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll1/gratings/output_ll1_g', 'LGN': '_sd278_LGN_only_no_con' }
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_sd278_LGN_only_no_con' }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'tot_path': '../simulations_ll3/gratings/output_ll3_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll3/gratings/output_ll3_g', 'LGN': '_sd278_LGN_only_no_con' }
#df_bln = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#df_summary = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv', sep=' ')
#N_trials = 10
#TF = 2.0
#SF = 0.05
#fig_n_1 = 'i_amplification/i_amplification_plot_phase_shift_all_current_unnormalized'
#fig_n_2 = 'i_amplification/i_amplification_plot_phase_shift_i_all_t_av'
#y_lim_2 = (-0.3, 0.0)

#sys_dict['ll2_inh_current'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../output_ll2_g', 'tot': '_sd278_SEClamp_e0/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30) }
#df_bln = pd.read_csv('../5_cells_measure_i_baseline_SEClamp_e0/i_SEClamp_baseline.csv', sep=' ')
#df_summary = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_inh_current_bln_from_5_cell_test.csv', sep=' ')
#N_trials = 10
#TF = 2.0
#SF = 0.05
#fig_n_1 = 'i_amplification/i_amplification_plot_phase_shift_all_current_unnormalized_inh_current'
#fig_n_2 = 'i_amplification/i_amplification_plot_phase_shift_i_all_t_av_inh_current'
#y_lim_2 = (0.0, 1.5)

sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../output_ll2_g', 'tot': '_sd278_remove_inputs_from_SEClamp_cells/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_sd278_LGN_only_no_con' }
df_bln = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
df_summary = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv', sep=' ')
N_trials = 1
TF = 2.0
SF = 0.05
fig_n_1 = 'i_amplification/i_amplification_plot_phase_shift_all_current_unnormalized_remove_inputs_from_SEClamp_cells'
fig_n_2 = 'i_amplification/i_amplification_plot_phase_shift_i_all_t_av_remove_inputs_from_SEClamp_cells'
y_lim_2 = (-0.3, 0.0)


sel_dict = {'exc': range(0, 8500), 'inh': range(8500, 10000)}

color_dict = {'tot': 'red', 'LGN': 'blue', 'tot_LGN_dif': 'black'}

ref_lbl = 'LS_F1_LGN'
ref_phase = 'phase_LGN'


# Determine which entries in the summary data frame we should be working with.
df_summary = df_summary[df_summary['TF']==TF]
df_summary = df_summary[df_summary['SF']==SF]

TF_per = 1000.0 / TF # Period of the grating, in ms.

for sel_lbl in sel_dict:
    gids = np.unique(df_summary[df_summary['gid'].isin(sel_dict[sel_lbl])]['gid'].values)

    i_sel = {}
    i_sel['tot'] = np.array([])
    i_sel['LGN'] = np.array([])
    i_sel['tot_LGN_dif'] = np.array([])
    i_av_sel = {}
    i_av_sel['tot'] = np.array([])
    i_av_sel['LGN'] = np.array([])
    i_av_sel['tot_LGN_dif'] = np.array([])
    for sys_name in sys_dict:
        df_sys = df_summary[df_summary['model']==sys_name]
        for gid in gids:
            i_combined = {}
            i_combined['tot'] = np.array([])
            i_combined['LGN'] = np.array([])

            df_gid = df_sys[df_sys['gid']==gid]
            gid_type = df_gid['type'].values[0]
            i_bln = df_bln[df_bln['type']==gid_type]['i_bln'].values[0]

            # Find the preferred grating direction.
            ref_tmp = df_gid[ref_lbl].values
            ind_pref = np.abs(ref_tmp).argmax() # Make sure to apply abs() here, because some of the entries in the data frame are stored as negative current values.
            df_tmp = df_gid[df_gid[ref_lbl] == ref_tmp[ind_pref]]
            grating_id = df_tmp['grating_id'].values[0]

            for trial in xrange(0, N_trials):
                for f_label in ['tot', 'LGN']:
                    f_name = '%s%d_%d%s/i_SEClamp-cell-%d.h5' % (sys_dict[sys_name]['%s_path' % (f_label)], grating_id, trial, sys_dict[sys_name][f_label], gid)
                    print 'Processing file %s.' % (f_name)
                    h5 = h5py.File(f_name, 'r')
                    values = h5['values'][...]
                    tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']
                    h5.close()
                    if (i_combined[f_label].size == 0):
                        i_combined[f_label] = values
                    else:
                        i_combined[f_label] = i_combined[f_label] + values

            # Use true baseline.
            i_combined['tot'] = i_combined['tot'] / N_trials - i_bln
            i_combined['LGN'] = i_combined['LGN'] / N_trials - i_bln
            i_combined['tot_LGN_dif'] = i_combined['tot'] - i_combined['LGN']

            #plt.plot(tarray, i_combined['tot'], c='r')
            #plt.plot(tarray, i_combined['LGN'], c='b')
            #plt.plot(tarray, i_combined['tot_LGN_dif'], c='k')
            #plt.show()

            # Assume that tarray is the same between 'tot' and 'LGN' and for all trials.
            ind_av = np.intersect1d( np.where(tarray > t_av[0]), np.where(tarray < t_av[1]) )
            t_ind_av = tarray[ind_av]
            # Shift the arrays by the phase relative to the reference.
            t_ind_av = periodic_shift(t_ind_av + df_tmp[ref_phase].values[0] * TF_per/(2 * np.pi), t_av[0],  t_av[1] )
            # Very importantly, the arrays need to be sorted in the ascending order of t_ind_av, so that arrays from different cells can be combined.
            ind_av_sorted = t_ind_av.argsort()
            t_ind_av = t_ind_av[ind_av_sorted]

            # Prepare an array containing indices corresponding to the start and end times of each grating cycle.
            ind_cy_av = []
            for t_k_tmp in np.arange(t_av[0], t_av[1]+0.000001*TF_per, TF_per)[:-1]:
                ind_cy_av.append( np.intersect1d( np.where(tarray > t_k_tmp), np.where(tarray < t_k_tmp + TF_per) ) )

            # Use the time within the cycle average limits.
            t_cy_av = tarray[ind_cy_av[0]] - t_av[0]
            # Shift the arrays by the phase relative to the reference.
            t_cy_av = periodic_shift(t_cy_av + df_tmp[ref_phase].values[0] * TF_per/(2 * np.pi), 0.0,  TF_per )
            # Very importantly, the arrays need to be sorted in the ascending order of t_cy_av, so that arrays from different cells can be combined.
            ind_sorted = t_cy_av.argsort()
            t_cy_av = t_cy_av[ind_sorted]

            for f_label in ['tot', 'LGN', 'tot_LGN_dif']:
                i_tmp = i_combined[f_label]

                # Accumulate current over the whole time of the stimulus presentation.
                if (i_av_sel[f_label].size == 0):
                    i_av_sel[f_label] = i_tmp[ind_av][ind_av_sorted]
                else:
                    i_av_sel[f_label] = np.vstack((i_av_sel[f_label], i_tmp[ind_av][ind_av_sorted]))


                # First, prepare a cycle average.
                i_cy_av = np.array([])
                for ind_tmp in ind_cy_av:
                    if (i_cy_av.size == 0):
                        i_cy_av = i_tmp[ind_tmp]
                    else:
                        i_cy_av = i_cy_av +  i_tmp[ind_tmp]
                i_cy_av = i_cy_av / len(ind_cy_av)

                '''
                # Shift the cycle average by its mean.
                i_cy_av = i_cy_av - i_cy_av.mean()
                # Normalize.
                i_cy_av = i_cy_av / i_cy_av.max()
                '''
                '''
                # Take the absolute values of the current and keep the mean unsubtracted.
                i_cy_av = np.abs(i_cy_av)
                # Normalize.
                i_cy_av = i_cy_av / i_cy_av.max()
                '''

                # Order according to the order of t_cy_av.
                i_cy_av = i_cy_av[ind_sorted]

                if (i_sel[f_label].size == 0):
                    i_sel[f_label] = i_cy_av
                else:
                    i_sel[f_label] = np.vstack((i_sel[f_label], i_cy_av))

                #plt.plot(t_cy_av, i_cy_av)

            #plt.title('gid = %d' % (gid))
            #plti.show()

    for f_label in ['tot', 'LGN', 'tot_LGN_dif']:
        i_sel[f_label] = i_sel[f_label].mean(axis=0)
        plt.plot(t_cy_av, i_sel[f_label], c=color_dict[f_label], label=f_label)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (arb. u.)')
    plt.legend()
    #plt.savefig('i_amplification/i_amplification_plot_phase_shift_ref_%s_ref_phase_%s_%s.eps' % (ref_lbl, ref_phase, sel_lbl), format='eps')
    #plt.savefig('i_amplification/i_amplification_plot_phase_shift_mean_kept_ref_%s_ref_phase_%s_%s.eps' % (ref_lbl, ref_phase, sel_lbl), format='eps')
    plt.savefig('%s_ref_%s_ref_phase_%s_%s.eps' % (fig_n_1, ref_lbl, ref_phase, sel_lbl), format='eps')
    plt.show()

    for f_label in ['tot', 'LGN', 'tot_LGN_dif']:
        i_av_sel[f_label] = i_av_sel[f_label].mean(axis=0)
        plt.plot(t_ind_av, i_av_sel[f_label], c=color_dict[f_label], label=f_label)
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    plt.legend()
    try:
        plt.ylim(y_lim_2)
    except NameError:
        print 'y_lim_2 does not exist; use default ylim.'
    plt.savefig('%s_ref_%s_ref_phase_%s_%s.eps' % (fig_n_2, ref_lbl, ref_phase, sel_lbl), format='eps')
    plt.show()

