import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'f_1': '../simulations_ll1/bars/output_ll1_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'bars/ll1_frate_max.csv', 'stims': ['Bbar_v50pixps_hor', 'Bbar_v50pixps_vert', 'Wbar_v50pixps_hor', 'Wbar_v50pixps_vert'] }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../simulations_ll2/bars/output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'bars/ll2_frate_max.csv', 'stims': ['Bbar_v50pixps_hor', 'Bbar_v50pixps_vert', 'Wbar_v50pixps_hor', 'Wbar_v50pixps_vert'] }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'f_1': '../simulations_ll3/bars/output_ll3_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'bars/ll3_frate_max.csv', 'stims': ['Bbar_v50pixps_hor', 'Bbar_v50pixps_vert', 'Wbar_v50pixps_hor', 'Wbar_v50pixps_vert'] }


t0 = 500.0
t1 = 3000.0
hist_step = 50.0
hist_bins = np.arange(t0, t1, hist_step)

Ntrial = 10

for sys in sys_dict:

    cells_df = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ')
    gids = cells_df['index'].values

    frate_max = pd.DataFrame(gids, columns=['gid'])

    for stim in sys_dict[sys]['stims']:
        df_tot = pd.DataFrame(columns=['t', 'gid'])
        for i_trial in xrange(0, Ntrial):
            f_name = '%s%s_%d%s' % (sys_dict[sys]['f_1'], stim, i_trial, sys_dict[sys]['f_2'])
            df = pd.read_csv(f_name, sep=' ', header=None)
            df.columns = ['t', 'gid']
            df = df[df['t'] >= t0]
            df = df[df['t'] <= t1]
            df_tot = df_tot.append(df)

        tmp_max = np.zeros(gids.size)
        grouped = df_tot.groupby('gid')
        for group in grouped:
            gid = int(group[0])
            t = group[1]['t'].values
            frate_hist = np.histogram(t, bins=hist_bins)[0] * 1000.0 / (hist_step * Ntrial) # Make sure the resulting units are Hz (time is in ms).
            #plt.plot(hist_bins[:-1], frate_hist); plt.show()
            tmp_max[gid] = frate_hist.max()
            if (gid % 1000 == 0):
                print 'Processing system %s, stim %s, gid %d' % (sys, stim, gid)

        frate_max[stim] = tmp_max

    #    plt.plot(frate_max['gid'], frate_max[stim])
    #plt.show()

    # For each cell, find the stim name corresponding to the strongest response.
    frate_max['pref'] = frate_max[sys_dict[sys]['stims']].idxmax(axis=1)

    frate_max.to_csv(sys_dict[sys]['f_out'], sep=' ', index=False)


