import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})


sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'bars_pref': 'bars/ll1_frate_max.csv', 'gratings_pref': 'Ori/ll1_pref_stat.csv' }
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'bars_pref': 'bars/ll2_frate_max.csv', 'gratings_pref': 'Ori/ll2_pref_stat.csv' }
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'bars_pref': 'bars/ll3_frate_max.csv', 'gratings_pref': 'Ori/ll3_pref_stat.csv' }

fig_example = 'bars/ll2_max_frate_for_bars.eps'
fig_av = 'bars/av_delta_pref_ori_gratings_bars.eps'

delta_bg_sys = np.array([x for x in sys_dict])
delta_bg_av = np.zeros(delta_bg_sys.size)
delta_bg_std = np.zeros(delta_bg_sys.size)

for sys_i, sys in enumerate(sys_dict):

    df_bars = pd.read_csv(sys_dict[sys]['bars_pref'], sep=' ')
    # Plot examples.
    if (sys == 'll2'):
        for stim in ['Bbar_v50pixps_hor', 'Bbar_v50pixps_vert', 'Wbar_v50pixps_hor', 'Wbar_v50pixps_vert']:
            plt.plot(df_bars['gid'], df_bars[stim], label=stim)
        plt.xlim([0, 10000])
        plt.xlabel('Neuron ID')
        plt.ylabel('Max firing rate')
        plt.legend()
        plt.savefig(fig_example, format = 'eps')
        plt.show()

    df_g = pd.read_csv(sys_dict[sys]['gratings_pref'], sep=' ')

    df = pd.DataFrame(df_g['id'].values, columns=['gid'])
    df['pref_b'] = df_bars['pref']
    df['pref_g'] = df_g['ori']

    # Find cells that prefer gratings with ori 0, 90, 180, or 270 degrees.
    df = df[df['pref_g'].isin([0.0, 90.0, 180.0, 270.0])]

    # Select only biophysical cells.
    df = df[df['gid'] < 10000]

    # Turn the strings in 'pref_b' into numbers.
    df['pref_b'] = df['pref_b'].str.split('_').str[-1]
    df[df['pref_b'] == 'hor'] = 0.0
    df[df['pref_b'] == 'vert'] = 90.0

    # Reduce the preferred grating orientations to either 0 or 90 degrees (0 and 180 turn into 0, 90 and 270 turn into 90). 
    df[df['pref_g'].isin([0.0, 180.0])] = 0.0
    df[df['pref_g'].isin([90.0, 270.0])] = 90.0

    df['delta_b_g'] = (df['pref_g'] - df['pref_b']) #.abs()

    delta_bg_av[sys_i] = df['delta_b_g'].mean()
    delta_bg_std[sys_i] = df['delta_b_g'].std()

    #plt.scatter(df['pref_g'], df['pref_b'])
    #plt.show()

#    plt.plot(df['delta_b_g'].values)
#plt.show()

fig, ax = plt.subplots(figsize = (8, 3))
ind_sorted = np.argsort(delta_bg_sys)
ax.errorbar(ind_sorted, delta_bg_av[ind_sorted], yerr = delta_bg_std[ind_sorted], linestyle='None', marker='o', markersize=10, capsize=20)
ax.set_xticks(np.arange(0, delta_bg_av.size))
ax.set_xticklabels(delta_bg_sys[ind_sorted])
ax.set_xlim([-0.5, delta_bg_sys.size - 0.5])
ax.set_ylabel('delta_ori (degrees)')
ax.set_title('delta_ori = pref_ori_gratings - pref_ori_bars')
plt.savefig(fig_av, format = 'eps')
plt.show()

