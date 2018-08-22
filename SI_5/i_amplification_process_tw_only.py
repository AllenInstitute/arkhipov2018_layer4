import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/tw_only/output_ll1_tw_only_', 'f_2': '_sd278/'}
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/tw_only/output_ll2_tw_only_', 'f_2': '_sd278/'}
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/tw_only/output_ll3_tw_only_', 'f_2': '_sd278/'}

f_out = 'i_amplification/i_amplification_process_tw_only_LL_bln_from_5_cell_test.csv'

N_trials = 10

t_av = [500.0, 3000.0]
'''
t_bln = [100.0, 500.0]
data_range_bln = 0.05 # Range of the data to be included in the baseline calculation.
'''

bln_df = pd.read_csv('../5_cells_measure_i_baseline/i_SEClamp_baseline.csv', sep=' ')

gids = range(2, 10000, 200)

av_list = []
gid_list = []
model_list = []
type_list = []
Ntrial_list = []

for sys_name in sys_dict:
    cells = pd.read_csv(sys_dict[sys_name]['cells_file'], sep=' ')
    for gid in gids:
        i_combined = np.array([])
        gid_type = cells[cells['index']==gid]['type'].values[0]
        i_bln = bln_df[bln_df['type']==gid_type]['i_bln'].values[0]
        for trial in xrange(0, N_trials):
            f_name = '%s%d%s/i_SEClamp-cell-%d.h5' % (sys_dict[sys_name]['f_1'], trial, sys_dict[sys_name]['f_2'], gid)
            print 'Processing file %s.' % (f_name)
            h5 = h5py.File(f_name, 'r')
            values = h5['values'][...]
            tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']
            h5.close()
            if (i_combined.size == 0):
                i_combined = values
            else:
                i_combined = i_combined + values
        i_combined = i_combined / N_trials

        ind_av = np.intersect1d( np.where(tarray > t_av[0]), np.where(tarray < t_av[1]) )
        '''
        # Compute baseline. For that, use the current values within certain window.  Then, choose the bottom 5 percentile
        # of values, as those represent the values closest to true baseline, unaffected by spontaneous activity (this follows
        # Lien and Scanziani, Nat. Neurosci., 2013).  Since stronger current is more negative, we actually choose the top 5 percentile.
        ind_bln = np.intersect1d( np.where(tarray > t_bln[0]), np.where(tarray < t_bln[1]) )
        i_tmp = i_combined[ind_bln]
        i_tmp_cutoff = i_tmp.max() - (i_tmp.max() - i_tmp.min()) * data_range_bln
        ind_bln_1 = np.where(i_tmp > i_tmp_cutoff)[0]

        av = i_combined[ind_av].mean() - i_tmp[ind_bln_1].mean()
        #print i_tmp[ind_bln_1].mean()
        '''
        # Use true baseline.
        av = i_combined[ind_av].mean() - i_bln

        #plt.plot(tarray, i_combined)
        #plt.plot(tarray[ind_bln], i_combined[ind_bln])
        #plt.plot(tarray[ind_bln][ind_bln_1], i_tmp[ind_bln_1])
        #plt.ylim((0.0, 0.1))
        #plt.title('System %s, gid %d' % (sys_name, gid))
        #plt.show()

        av_list.append(av)
        gid_list.append(gid)
        model_list.append(sys_name)
        type_list.append(gid_type)
        Ntrial_list.append(N_trials)

#        ind = np.intersect1d( np.where(tarray > t_av[0]), np.where(tarray < t_av[1]) )
#        plt.plot(tarray[ind], i_combined[ind])
#        plt.xlabel('Time (ms)')
#        plt.ylabel('Current (nA)')
#        plt.show()

df = pd.DataFrame()
df['model'] = model_list
df['gid'] = gid_list
df['type'] = type_list
df['N_trials'] = Ntrial_list
df['tw_only'] = av_list

df.to_csv(f_out, sep=' ', index=False)

