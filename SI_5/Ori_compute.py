import pickle
import numpy as np
import matplotlib.pyplot as plt

import operator

import pandas as pd

import os.path

import f_rate_t_by_type_functions as frtbt


N_trials = 10


# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll1_rates.npy', 'f_out_std': 'Ori/ll1_rates_std.npy', 'f_out_pref': 'Ori/ll1_pref_stat.csv', 'grating_id': range(6, 240, 30)+range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates.npy', 'f_out_std': 'Ori/ll2_rates_std.npy', 'f_out_pref': 'Ori/ll2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../output_ll3_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll3_rates.npy', 'f_out_std': 'Ori/ll3_rates_std.npy', 'f_out_pref': 'Ori/ll3_pref_stat.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['rl1'] = { 'cells_file': '../build/rl1.csv', 'f_1': '../simulations_rl1/gratings/output_rl1_', 'f_2': '_sd285/spk.dat', 'f_3': '_sd285/tot_f_rate.dat', 'f_out': 'Ori/rl1_rates.npy', 'f_out_std': 'Ori/rl1_rates_std.npy', 'f_out_pref': 'Ori/rl1_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['rl2'] = { 'cells_file': '../build/rl2.csv', 'f_1': '../output_rl2_', 'f_2': '_sd285/spk.dat', 'f_3': '_sd285/tot_f_rate.dat', 'f_out': 'Ori/rl2_rates.npy', 'f_out_std': 'Ori/rl2_rates_std.npy', 'f_out_pref': 'Ori/rl2_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['rl3'] = { 'cells_file': '../build/rl3.csv', 'f_1': '../simulations_rl3/gratings/output_rl3_', 'f_2': '_sd285/spk.dat', 'f_3': '_sd285/tot_f_rate.dat', 'f_out': 'Ori/rl3_rates.npy', 'f_out_std': 'Ori/rl3_rates_std.npy', 'f_out_pref': 'Ori/rl3_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['lr1'] = { 'cells_file': '../build/lr1.csv', 'f_1': '../simulations_lr1/gratings/output_lr1_', 'f_2': '_sd287_cn0/spk.dat', 'f_3': '_sd287_cn0/tot_f_rate.dat', 'f_out': 'Ori/lr1_rates.npy', 'f_out_std': 'Ori/lr1_rates_std.npy', 'f_out_pref': 'Ori/lr1_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['lr2'] = { 'cells_file': '../build/lr2.csv', 'f_1': '../output_lr2_', 'f_2': '_sd287_cn0/spk.dat', 'f_3': '_sd287_cn0/tot_f_rate.dat', 'f_out': 'Ori/lr2_rates.npy', 'f_out_std': 'Ori/lr2_rates_std.npy', 'f_out_pref': 'Ori/lr2_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['lr3'] = { 'cells_file': '../build/lr3.csv', 'f_1': '../simulations_lr3/gratings/output_lr3_', 'f_2': '_sd287_cn0/spk.dat', 'f_3': '_sd287_cn0/tot_f_rate.dat', 'f_out': 'Ori/lr3_rates.npy', 'f_out_std': 'Ori/lr3_rates_std.npy', 'f_out_pref': 'Ori/lr3_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['rr1'] = { 'cells_file': '../build/rr1.csv', 'f_1': '../simulations_rr1/gratings/output_rr1_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'Ori/rr1_rates.npy', 'f_out_std': 'Ori/rr1_rates_std.npy', 'f_out_pref': 'Ori/rr1_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'f_1': '../simulations_rr2/gratings/output_rr2_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'Ori/rr2_rates.npy', 'f_out_std': 'Ori/rr2_rates_std.npy', 'f_out_pref': 'Ori/rr2_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'f_1': '../simulations_rr3/gratings/output_rr3_', 'f_2': '_sd282_cn0/spk.dat', 'f_3': '_sd282_cn0/tot_f_rate.dat', 'f_out': 'Ori/rr3_rates.npy', 'f_out_std': 'Ori/rr3_rates_std.npy', 'f_out_pref': 'Ori/rr3_pref_stat.csv', 'grating_id': range(8, 240, 30)}
#sys_dict['ll2_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['ll1_LGN_only_no_con'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/gratings/output_ll1_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Ori/ll1_LGN_only_no_con_rates.npy', 'f_out_std': 'Ori/ll1_LGN_only_no_con_rates_std.npy', 'f_out_pref': 'Ori/ll1_LGN_only_no_con_pref_stat.csv', 'grating_id': range(7, 240, 30) + range(8, 240, 30) }
#sys_dict['ll2_LGN_only_no_con'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/gratings/output_ll2_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Ori/ll2_LGN_only_no_con_rates.npy', 'f_out_std': 'Ori/ll2_LGN_only_no_con_rates_std.npy', 'f_out_pref': 'Ori/ll2_LGN_only_no_con_pref_stat.csv', 'grating_id': range(7, 240, 30) + range(8, 240, 30) }
sys_dict['ll3_LGN_only_no_con'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/gratings/output_ll3_', 'f_2': '_sd278_LGN_only_no_con/spk.dat', 'f_3': '_sd278_LGN_only_no_con/tot_f_rate.dat', 'f_out': 'Ori/ll3_LGN_only_no_con_rates.npy', 'f_out_std': 'Ori/ll3_LGN_only_no_con_rates_std.npy', 'f_out_pref': 'Ori/ll3_LGN_only_no_con_pref_stat.csv', 'grating_id': range(7, 240, 30) + range(8, 240, 30) }
#sys_dict['ll2_ctr30_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr30_sd278/spk.dat', 'f_3': '_ctr30_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr30_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr30_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr30_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['ll2_ctr10_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr10_sd278/spk.dat', 'f_3': '_ctr10_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr10_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr10_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr10_pref_stat_4Hz.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['ll1_LIF'] = { 'cells_file': '../build/ll1.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll1/output_ll1_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/ll1_rates.npy', 'f_out_std': 'Ori_LIF/ll1_rates_std.npy', 'f_out_pref': 'Ori_LIF/ll1_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2_LIF'] = { 'cells_file': '../build/ll2.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll2/output_ll2_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/ll2_rates.npy', 'f_out_std': 'Ori_LIF/ll2_rates_std.npy', 'f_out_pref': 'Ori_LIF/ll2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll3_LIF'] = { 'cells_file': '../build/ll3.csv', 'f_1': '/data/mat/ZiqiangW/simulation_ll_syn_data_lif_z102/simulation_ll3/output_ll3_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/ll3_rates.npy', 'f_out_std': 'Ori_LIF/ll3_rates_std.npy', 'f_out_pref': 'Ori_LIF/ll3_pref_stat.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['rl1_LIF'] = { 'cells_file': '../build/rl1.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rl_final_syn_data_lif_z101/simulation_rl1/output_rl1_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rl1_rates.npy', 'f_out_std': 'Ori_LIF/rl1_rates_std.npy', 'f_out_pref': 'Ori_LIF/rl1_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['rl2_LIF'] = { 'cells_file': '../build/rl2.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rl_final_syn_data_lif_z101/simulation_rl2/output_rl2_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rl2_rates.npy', 'f_out_std': 'Ori_LIF/rl2_rates_std.npy', 'f_out_pref': 'Ori_LIF/rl2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['rl3_LIF'] = { 'cells_file': '../build/rl3.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rl_final_syn_data_lif_z101/simulation_rl3/output_rl3_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rl3_rates.npy', 'f_out_std': 'Ori_LIF/rl3_rates_std.npy', 'f_out_pref': 'Ori_LIF/rl3_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['lr1_LIF'] = { 'cells_file': '../build/lr1.csv', 'f_1': '/data/mat/ZiqiangW/simulation_lr_final_syn_data_lif_z101/simulation_lr1/output_lr1_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/lr1_rates.npy', 'f_out_std': 'Ori_LIF/lr1_rates_std.npy', 'f_out_pref': 'Ori_LIF/lr1_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['lr2_LIF'] = { 'cells_file': '../build/lr2.csv', 'f_1': '/data/mat/ZiqiangW/simulation_lr_final_syn_data_lif_z101/simulation_lr2/output_lr2_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/lr2_rates.npy', 'f_out_std': 'Ori_LIF/lr2_rates_std.npy', 'f_out_pref': 'Ori_LIF/lr2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['lr3_LIF'] = { 'cells_file': '../build/lr3.csv', 'f_1': '/data/mat/ZiqiangW/simulation_lr_final_syn_data_lif_z101/simulation_lr3/output_lr3_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/lr3_rates.npy', 'f_out_std': 'Ori_LIF/lr3_rates_std.npy', 'f_out_pref': 'Ori_LIF/lr3_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['rr1_LIF'] = { 'cells_file': '../build/rr1.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rr_final_data_lif_z103/simulation_rr1/output_rr1_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rr1_rates.npy', 'f_out_std': 'Ori_LIF/rr1_rates_std.npy', 'f_out_pref': 'Ori_LIF/rr1_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['rr2_LIF'] = { 'cells_file': '../build/rr2.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rr_final_data_lif_z103/simulation_rr2/output_rr2_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rr2_rates.npy', 'f_out_std': 'Ori_LIF/rr2_rates_std.npy', 'f_out_pref': 'Ori_LIF/rr2_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['rr3_LIF'] = { 'cells_file': '../build/rr3.csv', 'f_1': '/data/mat/ZiqiangW/simulation_rr_final_data_lif_z103/simulation_rr3/output_rr3_', 'f_2': '_sdlif_z101/spk.dat', 'f_3': '_sdlif_z101/tot_f_rate.dat', 'f_out': 'Ori_LIF/rr3_rates.npy', 'f_out_std': 'Ori_LIF/rr3_rates_std.npy', 'f_out_pref': 'Ori_LIF/rr3_pref_stat.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }





# Load gratings metadata.
grating_par = pd.read_csv('/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/movies_gratings/res_192_metadata.txt', sep=' ', header=None)
grating_par.columns = ['path', 'ori', 'SF', 'TF', 'ignore']
tmp_par = grating_par['path'].str.split('.').str[-2].str.split('_').str[-1]
grating_par['grating_id'] = tmp_par


# Load simulation data, average over trials, and save to file.
for sys_name in sys_dict.keys():
    rates_array = np.array([])
    std_rates_array = np.array([])
    for grating_id in sys_dict[sys_name]['grating_id']:
        rates_tmp = np.array([])
        std_rates_tmp = np.array([])
        N_existing_f = 0
        for i_trial in xrange(0, N_trials):
            f_name = '%sg%d_%d%s' % (sys_dict[sys_name]['f_1'], grating_id, i_trial, sys_dict[sys_name]['f_3'])
            if not (os.path.isfile(f_name)):
                continue
            print 'Processing file %s.' % (f_name)
            N_existing_f += 1
            tmp = np.genfromtxt(f_name, delimiter=' ')[:, 1] # Assume all files have the same columns of gids; use the 2nd column for rates.
            if (rates_tmp.size == 0):
                rates_tmp = tmp
                std_rates_tmp = tmp**2
            else:
                rates_tmp = rates_tmp + tmp
                std_rates_tmp = std_rates_tmp + tmp**2
        if (rates_array.size == 0):
            rates_array = rates_tmp / (1.0 * N_existing_f)
            std_rates_array = np.sqrt(std_rates_tmp / (1.0 * N_existing_f) - rates_array**2)
        else:
            rates_array = np.vstack( (rates_array, rates_tmp / (1.0 * N_existing_f)) )
            std_rates_array = np.vstack( (std_rates_array, np.sqrt(std_rates_tmp / (1.0 * N_existing_f) - (rates_tmp / (1.0 * N_existing_f))**2)) )
    np.save(sys_dict[sys_name]['f_out'], rates_array)
    np.save(sys_dict[sys_name]['f_out_std'], std_rates_array)


# Determine the preferred ori, SF, and TF.
for sys_name in sys_dict.keys():
    print 'Working with the data from %s' % (sys_dict[sys_name]['f_out'])
    rates_array = np.load(sys_dict[sys_name]['f_out'])
    std_rates_array = np.load(sys_dict[sys_name]['f_out_std'])

    # Build an ordered list of ori.
    ori_list = []
    for grating_id in sys_dict[sys_name]['grating_id']:
        current_grating_par = grating_par[grating_par['grating_id'] == str(grating_id)]
        ori_list.append(float(current_grating_par['ori'].values[0]))
    ori_list = sorted(list(set(ori_list)))

    # Average rates for each cell over TF and SF, for each ori.
    av_for_ori = np.zeros((len(ori_list), rates_array.shape[1])) # Make an array of the size determined by the nubmer of ori and the number of cells.
    N_av_for_ori = np.zeros(len(ori_list))
    for index_0, grating_id in enumerate(sys_dict[sys_name]['grating_id']):
        current_grating_par = grating_par[grating_par['grating_id'] == str(grating_id)]
        ind_ori = ori_list.index(float(current_grating_par['ori'].values[0]))
        av_for_ori[ind_ori, :] = av_for_ori[ind_ori, :] + rates_array[index_0]
        N_av_for_ori[ind_ori] = N_av_for_ori[ind_ori] + 1
    for ind_ori in xrange(av_for_ori.shape[0]):
        av_for_ori[ind_ori] = av_for_ori[ind_ori] / (1.0 * N_av_for_ori[ind_ori])

    # For each cell, find the maximal response and obtain the indices corresponding to that in the ori dimension.
    pref_ori_index = av_for_ori.argmax(axis=0)
    # Create a dictionary for preferred ori, SF, and TF.
    pref_dict = {'ori': np.empty(av_for_ori.shape[1]), 'SF': np.empty(av_for_ori.shape[1]), 'TF': np.empty(av_for_ori.shape[1]), 'tuning_curve': {}, 'tuning_curve_std': {}, 'CV_ori': {}, 'OSI_modulation': {}, 'DSI': {}}
    pref_dict['ori'][:] = np.NAN
    pref_dict['SF'][:] = np.NAN
    pref_dict['TF'][:] = np.NAN
    for k_cell in xrange(av_for_ori.shape[1]):
        pref_dict['ori'][k_cell] = ori_list[pref_ori_index[k_cell]]

    # For each cell, find the SF and TF that evoke the largest response at the preferred ori; those are assigned as preferred SF and TF.
    # Obtain the tuning curve for responses at these preferred SF and TF.
    f_out_pref = open(sys_dict[sys_name]['f_out_pref'], 'w')
    f_out_pref.write('id ori SF TF CV_ori OSI_modulation DSI')
    for ori in ori_list:
        f_out_pref.write(' %f' % (ori))
    f_out_pref.write('\n')
    for k_cell in xrange(av_for_ori.shape[1]):
        if (k_cell % 100 == 0):
            print 'System %s; computing tuning statistiscs for cell %d' % (sys_name, k_cell)
        ori_tmp = pref_dict['ori'][k_cell]
        current_grating_par = grating_par[grating_par['ori'] == ori_tmp]
        current_grating_par = current_grating_par[current_grating_par['grating_id'].isin([str(x) for x in sys_dict[sys_name]['grating_id']])]
        ind = []
        for grating_id_str in current_grating_par['grating_id'].values:
            ind.append(sys_dict[sys_name]['grating_id'].index(int(grating_id_str)))
        ind = np.array(ind)
        # Select the rows in rates_array that correspond to the grating_id at the preferred ori, and the column that corresponds to the current cell;
        # find the index of the maximum in that slice using argmax and from that infer the index of the grating from sys_dict;
        # based on that, look up grating_id.
        grating_id_pref = sys_dict[sys_name]['grating_id'][ind[rates_array[ind, k_cell].argmax()]]
        pref_SF = grating_par[grating_par['grating_id'] == str(grating_id_pref)]['SF'].values[0]
        pref_TF = grating_par[grating_par['grating_id'] == str(grating_id_pref)]['TF'].values[0]
        pref_dict['SF'][k_cell] = pref_SF
        pref_dict['TF'][k_cell] = pref_TF

        # Find all gratings with the preferred SF and TF.
        pref_dict['tuning_curve'][k_cell] = {}
        pref_dict['tuning_curve_std'][k_cell] = {}
        CV_ori_1 = 0.0
        CV_ori_2 = 0.0
        CV_ori_norm = 0.0
        current_grating_par = grating_par[grating_par['TF'] == pref_TF]
        current_grating_par = current_grating_par[current_grating_par['SF'] == pref_SF]
        for index_0, grating_id in enumerate(sys_dict[sys_name]['grating_id']):
            grating_par_row = current_grating_par[current_grating_par['grating_id'] == str(grating_id)]
            if not grating_par_row.empty:
                ori = grating_par_row['ori'].values[0]
                tmp_rate = rates_array[index_0, k_cell]
                pref_dict['tuning_curve'][k_cell][ori] = tmp_rate
                pref_dict['tuning_curve_std'][k_cell][ori] = std_rates_array[index_0, k_cell]
                ori_rad_t2 = 2.0 * np.deg2rad(ori)
                CV_ori_1 += tmp_rate * np.cos(ori_rad_t2)
                CV_ori_2 += tmp_rate * np.sin(ori_rad_t2)
                CV_ori_norm += tmp_rate
        if (CV_ori_norm == 0.0):
            CV_ori = 0.0
        else:
            CV_ori = np.sqrt(CV_ori_1**2 + CV_ori_2**2) / CV_ori_norm
        pref_dict['CV_ori'][k_cell] = CV_ori

        # Compute the OSI_modulation and DSI.
        tcurve = pref_dict['tuning_curve'][k_cell]
        ori_pref, rate_pref =  max(tcurve.iteritems(), key=operator.itemgetter(1)) # Obtain the tuple with the key corresponding to the maximum and the maximum value itself.
        # Find the values at orthogonal and opposite directions.
        # NOTE that this assumes our data always contain values at +/-90 degrees from any other orientation.  If that's not
        # the case, we can use this:  min(myList, key=lambda x:abs(x-myNumber)) -- finding the element in myList which has the minimum distance from myNumber.
        ori_ortho_1 = (ori_pref + 90.0) % 360.0
        rate_ortho_1 = tcurve[ori_ortho_1]
        ori_ortho_2 = (ori_pref - 90.0) % 360.0
        rate_ortho_2 = tcurve[ori_ortho_2]
        rate_ortho = (rate_ortho_1 + rate_ortho_2) / 2.0
        ori_opposite = (ori_pref + 180.0) % 360.0
        rate_opposite = tcurve[ori_opposite]
        rate_p_o = (rate_pref + rate_opposite) / 2.0
        if (rate_pref > 0.0):
            OSI_modulation = (rate_p_o - rate_ortho) / (rate_p_o + rate_ortho)
            DSI = (rate_pref - rate_opposite) / (rate_pref + rate_opposite)
        else:
            OSI_modulation = 0.0
            DSI = 0.0
        pref_dict['OSI_modulation'][k_cell] = OSI_modulation
        pref_dict['DSI'][k_cell] = DSI

        # Write the results to file.
        f_out_pref.write('%d %f %f %f %f %f %f' % (k_cell, pref_dict['ori'][k_cell], pref_dict['SF'][k_cell], pref_dict['TF'][k_cell], pref_dict['CV_ori'][k_cell], pref_dict['OSI_modulation'][k_cell], pref_dict['DSI'][k_cell]))
        for ori in sorted(pref_dict['tuning_curve'][k_cell].keys()):
            f_out_pref.write(' (%f,%f)' % (pref_dict['tuning_curve'][k_cell][ori], pref_dict['tuning_curve_std'][k_cell][ori]))
        f_out_pref.write('\n')
    f_out_pref.close()

