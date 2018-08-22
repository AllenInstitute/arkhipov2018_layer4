import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

from scipy.optimize import leastsq

def sin_fun( params, x, omega ):
    return params[0] * np.sin( omega * x + params[1] )

def sin_fit( params, x, y, omega ):
    return (y - sin_fun( params, x, omega))

#def sin_fit( params, x, y, omega ):
#    if sin_fit_bounds( params ):
#        return (y - sin_fun( params, x, omega))
#    else:
#        return 1e20

#def sin_fit_bounds( params ):
#    if ((params[0] >= 0.0) and (params[1] >= 0.0) and (params[1] <= 2.0*np.pi)):
#        return True
#    else:
#        return False

sys_dict = {}

#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'tot_path': '../simulations_ll1/gratings/output_ll1_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll1/gratings/output_ll1_g', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'tot_path': '../simulations_ll3/gratings/output_ll3_g', 'tot': '_sd278/', 'LGN_path': '../simulations_ll3/gratings/output_ll3_g', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30)+range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv'

#sys_dict['ll2_ctr30'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_ctr30_sd278/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_ctr30_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_ctr30_bln_from_5_cell_test.csv'

#sys_dict['ll2_ctr10'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_ctr10_sd278/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_ctr10_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_ctr10_bln_from_5_cell_test.csv'

#sys_dict['rl1'] = { 'cells_file': '../build/rl1.csv', 'tot_path': '../simulations_rl1/gratings/output_rl1_g', 'LGN_path': '../simulations_ll1/gratings/output_ll1_g', 'tot': '_sd285/',  'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'tot_path': '../simulations_rl2/gratings/output_rl2_g', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_sd285/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['rl3'] = { 'cells_file': '../build/rl3.csv', 'tot_path': '../simulations_rl3/gratings/output_rl3_g', 'LGN_path': '../simulations_ll3/gratings/output_ll3_g', 'tot': '_sd285/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_RL_bln_from_5_cell_test.csv'

#sys_dict['lr1'] = { 'cells_file': '../build/lr1.csv', 'tot_path': '../simulations_lr1/gratings/output_lr1_g', 'LGN_path': '../simulations_ll1/gratings/output_ll1_g', 'tot': '_sd287_cn0/',  'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'tot_path': '../simulations_lr2/gratings/output_lr2_g', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_sd287_cn0/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['lr3'] = { 'cells_file': '../build/lr3.csv', 'tot_path': '../simulations_lr3/gratings/output_lr3_g', 'LGN_path': '../simulations_ll3/gratings/output_ll3_g', 'tot': '_sd287_cn0/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_LR_bln_from_5_cell_test.csv'

#sys_dict['rr1'] = { 'cells_file': '../build/rr1.csv', 'tot_path': '../simulations_rr1/gratings/output_rr1_g', 'LGN_path': '../simulations_ll1/gratings/output_ll1_g', 'tot': '_sd282_cn0/',  'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'tot_path': '../simulations_rr2/gratings/output_rr2_g', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'tot': '_sd282_cn0/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'tot_path': '../simulations_rr3/gratings/output_rr3_g', 'LGN_path': '../simulations_ll3/gratings/output_ll3_g', 'tot': '_sd282_cn0/', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(8, 240, 30) }
#bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')
#f_out = 'i_amplification/i_amplification_process_LGN_crt_RR_bln_from_5_cell_test.csv'

sys_dict['ll2_inh_current'] = { 'cells_file': '../build/ll2.csv', 'tot_path': '../output_ll2_g', 'tot': '_sd278_SEClamp_e0/', 'LGN_path': '../simulations_ll2/gratings/output_ll2_g', 'LGN': '_sd278_LGN_only_no_con', 'grating_id': range(7, 240, 30) }
bln_df = pd.read_csv('../5_cells_measure_i_baseline_SEClamp_e0/i_SEClamp_baseline.csv', sep=' ')
f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_inh_current_bln_from_5_cell_test.csv'


g_metadata = pd.read_csv('/allen/aibs/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt', sep=' ', header=None)
g_metadata.columns = ['path', 'ori', 'SF', 'TF', 'do_not_use']
g_metadata['grating_id'] = g_metadata['path'].str.split('.').str[-2].str.split('_').str[-1].astype(int)


N_trials = 10

t_av = [500.0, 3000.0]
'''
t_bln = [100.0, 500.0]
data_range_bln = 0.05 # Range of the data to be included in the baseline calculation.
'''

gids = range(2, 10000, 200)

crt_av_list = []
LGN_av_list = []
crt_sub_LGN_av_list = []
F0_crt = []
F0_LGN = []
F0_crt_sub_LGN = []
LS_F1_crt = []
LS_F1_LGN = []
LS_F1_crt_sub_LGN = []
r2_crt = []
r2_LGN = []
r2_crt_sub_LGN = []
phase_crt = []
phase_LGN = []
phase_crt_sub_LGN = []
gid_list = []
model_list = []
type_list = []
Ntrial_list = []
grating_id_list = []
TF_list = []
SF_list = []
ori_list = []

for sys_name in sys_dict:
    cells = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    for gid in gids:
        i_combined = {}
        i_combined['tot'] = np.array([])
        i_combined['LGN'] = np.array([])
        gid_type = cells[cells['index']==gid]['type'].values[0]

        i_bln = bln_df[bln_df['type']==gid_type]['i_bln'].values[0]

        for grating_id in sys_dict[sys_name]['grating_id']:
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

            # Find out what the TF and other parameters of the current grating are.
            TF = g_metadata[g_metadata['grating_id'] == grating_id]['TF'].values[0]
            TF_per = 1000.0 / TF # In ms.
            TF_omega = 2.0 * np.pi * TF / 1000.0 # In ms^(-1).
            SF = g_metadata[g_metadata['grating_id'] == grating_id]['SF'].values[0]
            ori = g_metadata[g_metadata['grating_id'] == grating_id]['ori'].values[0]

            # Prepare an array containing indices corresponding to the start and end times of each grating cycle.
            ind_cy_av = []
            for t_k_tmp in np.arange(t_av[0], t_av[1]+0.000001*TF_per, TF_per)[:-1]:
                ind_cy_av.append( np.intersect1d( np.where(tarray > t_k_tmp), np.where(tarray < t_k_tmp + TF_per) ) )

            av = {}
            F0 = {}
            LS_F1 = {}
            sin_phase = {}
            r2 = {}
            for f_label in ['tot', 'LGN', 'tot_LGN_dif']:
                '''
                # Compute baseline. For that, use the current values within certain window.  Then, choose the bottom 5 percentile
                # of values, as those represent the values closest to true baseline, unaffected by spontaneous activity (this follows
                # Lien and Scanziani, Nat. Neurosci., 2013).  Since stronger current is more negative, we actually choose the top 5 percentile.
                ind_bln = np.intersect1d( np.where(tarray > t_bln[0]), np.where(tarray < t_bln[1]) )
                i_tmp = i_combined[f_label][ind_bln]
                i_tmp_cutoff = i_tmp.max() - (i_tmp.max() - i_tmp.min()) * data_range_bln
                ind_bln_1 = np.where(i_tmp > i_tmp_cutoff)[0]

                av[f_label] = i_combined[f_label][ind_av].mean() - i_tmp[ind_bln_1].mean()
                '''
                # Use true baseline.
                #av[f_label] = i_combined[f_label][ind_av].mean() - i_bln

                i_tmp = i_combined[f_label] 
                av[f_label] = i_tmp[ind_av].mean()

                # Compute the F1 component following Lien and Scanziani, Nature Neurosci. (2013), that is, using sine function fitting (see Fig. 3a of that paper).
                # First, prepare a cycle average.
                i_cy_av = np.array([])
                for ind_tmp in ind_cy_av:
                    if (i_cy_av.size == 0):
                        i_cy_av = i_tmp[ind_tmp]
                    else:
                        i_cy_av = i_cy_av +  i_tmp[ind_tmp]
                i_cy_av = i_cy_av / len(ind_cy_av)

                # Shift the cycle average by its mean.
                F0[f_label] = i_cy_av.mean()
                i_cy_av = i_cy_av - F0[f_label]
                t_cy_av = tarray[ind_cy_av[0]] - t_av[0]

                params_start = (0.01, np.pi)
                fit_params, cov, infodict, mesg, ier = leastsq( sin_fit, params_start, args=(t_cy_av, i_cy_av, TF_omega), full_output=True )
                ss_err = (infodict['fvec']**2).sum()
                ss_tot = ((i_cy_av - i_cy_av.mean())**2).sum()
                if (ss_tot != 0.0):
                    rsquared = 1 - (ss_err/ss_tot)
                else:
                    rsquared = 0.0

                # Compute F1 according to Lien and Scanziani.  Note that the multiplier in front of the sinusoid (fit_params[0]) can be positive or negative.
                # We should convert its sign to the 0 or 180 degrees additional phase shift, which should be added to the phase shift from the fitting
                # procedure (fit_params[1]).  The F1 value itself needs to be >= 0, and thus we should take the absolute value
                # of fit_params[0].  Furthermore, this parameter determines the amplitude of the sinusoid, whereas Lien and Scanziani's definition
                # of F1 uses the total height from the minimum to the maximum of the sinusoid.  Therefore, we need to use a factor of 2 here.
                LS_F1[f_label] = np.abs(fit_params[0]) * 2.0
                if (fit_params[0] >= 0):
                    add_phase = 0.0
                else:
                    add_phase = np.pi
                sin_phase[f_label] = fit_params[1] + add_phase
                r2[f_label] = rsquared

                #plt.plot(tarray, i_combined[f_label])

                #plt.plot(tarray[ind_av], i_tmp[ind_av])
                #plt.plot(tarray[ind_cy_av[0]], i_cy_av)

                #plt.plot(t_cy_av, i_cy_av)
                #plt.plot(t_cy_av, sin_fun( fit_params, t_cy_av, TF_omega))
                #plt.title('System %s, gid %d, lbl=%s, A=%.2f, phi=%.1f deg, r2=%.2f' % (sys_name, gid, f_label, fit_params[0], 180.0*fit_params[1]/np.pi, rsquared))
                #plt.xlabel('Time (ms)')
                #plt.ylabel('Current (nA)')
                #plt.ylim([-0.15, 0.15])

                #plt.plot(tarray[ind_bln], i_combined[f_label][ind_bln])
                #plt.plot(tarray[ind_bln][ind_bln_1], i_tmp[ind_bln_1])
                #plt.ylim((0.0, 0.1))
                #plt.title('System %s, gid %d' % (sys_name, gid))

            #plt.show()

            # Average current.
            crt_av_list.append(av['tot'])
            LGN_av_list.append(av['LGN'])
            crt_sub_LGN_av_list.append(av['tot_LGN_dif'])

            # Mean from the cycle average, i.e., F0.  This may not be equal to the average above, if, for example, the time for averaging cannot be composed of an integer number of gratign cycles.
            F0_crt.append(F0['tot'])
            F0_LGN.append(F0['LGN'])
            F0_crt_sub_LGN.append(F0['tot_LGN_dif'])

            # F1 computed according to Lien adn Scanziani.
            LS_F1_crt.append(LS_F1['tot'])
            LS_F1_LGN.append(LS_F1['LGN'])
            LS_F1_crt_sub_LGN.append(LS_F1['tot_LGN_dif'])

            # Sinusoid phase.
            phase_crt.append(sin_phase['tot'])
            phase_LGN.append(sin_phase['LGN'])
            phase_crt_sub_LGN.append(sin_phase['tot_LGN_dif'])

            # Quality of the fit.
            r2_crt.append(r2['tot'])
            r2_LGN.append(r2['LGN'])
            r2_crt_sub_LGN.append(r2['tot_LGN_dif'])

            gid_list.append(gid)
            model_list.append(sys_name)
            type_list.append(gid_type)
            Ntrial_list.append(N_trials)
            grating_id_list.append(grating_id)
            TF_list.append(TF)
            SF_list.append(SF)
            ori_list.append(ori)

df = pd.DataFrame()
df['model'] = model_list
df['gid'] = gid_list
df['type'] = type_list
df['N_trials'] = Ntrial_list
df['grating_id'] = grating_id_list
df['TF'] = TF_list
df['SF'] = SF_list
df['ori'] = ori_list
df['LGN'] = LGN_av_list
df['crt'] = crt_av_list
df['crt_sub_LGN'] = crt_sub_LGN_av_list
df['F0_crt'] = F0_crt
df['F0_LGN'] = F0_LGN
df['F0_crt_sub_LGN'] = F0_crt_sub_LGN
df['LS_F1_crt'] = LS_F1_crt
df['LS_F1_LGN'] = LS_F1_LGN
df['LS_F1_crt_sub_LGN'] = LS_F1_crt_sub_LGN
df['r2_crt'] = r2_crt
df['r2_LGN'] = r2_LGN
df['r2_crt_sub_LGN'] = r2_crt_sub_LGN
df['phase_crt'] = phase_crt
df['phase_LGN'] = phase_LGN
df['phase_crt_sub_LGN'] = phase_crt_sub_LGN

df.to_csv(f_out, sep=' ', index=False)

