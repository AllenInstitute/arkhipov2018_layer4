import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from matplotlib import gridspec


# Process firing rates.
N_trials = 10
for sys in ['ll1', 'll2', 'll3']:
    for grating_id in [8]:
        f_rate = np.array([])
        for trial_id in xrange(N_trials):
            df = pd.read_csv('../simulations_%s/gratings/output_%s_g%d_%d_sd278/tot_f_rate.dat' % (sys, sys, grating_id, trial_id), sep=' ')
            df.columns = ['gid', 'frate', 'frate_include_equilibration']
            del df['frate_include_equilibration']
            if (f_rate.size == 0):
                f_rate = df['frate'].values
            else:
                f_rate = f_rate + df['frate'].values
        df['frate'] = f_rate / (1.0 * N_trials)
        df.to_csv('log_normal_rates/f_rate_mean_%s_g%d.csv' % (sys, grating_id), sep=' ', index=False)


# Load the firing rates from simulations.
# Need to run spont_activity_new.py first, to obtain the csv files used below.
sys_dict = {}
sys_dict['ll1'] = {'g8': 'log_normal_rates/f_rate_mean_ll1_g8.csv', 'spont': 'spont_activity/ll1_spont.csv'}
sys_dict['ll2'] = {'g8': 'log_normal_rates/f_rate_mean_ll2_g8.csv', 'spont': 'spont_activity/ll2_spont.csv'}
sys_dict['ll3'] = {'g8': 'log_normal_rates/f_rate_mean_ll3_g8.csv', 'spont': 'spont_activity/ll3_spont.csv'}
sim_df = pd.DataFrame()
for sys in sys_dict:
    df1 = pd.read_csv(sys_dict[sys]['g8'], sep=' ')
    df1 = df1.rename(columns={'frate': 'g8'})
    df2 = pd.read_csv(sys_dict[sys]['spont'], sep=' ')
    df2 = df2.rename(columns={'%s_frate' % (sys): 'Spont'})
    df = pd.merge(df1, df2, on='gid', how='inner')
    df['model'] = sys
    sim_df = pd.concat([sim_df, df], axis=0)


tmp_sim_df = sim_df[sim_df['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2'])]
sim_df_pos = tmp_sim_df[tmp_sim_df['Spont']!=0.0]

bins = np.arange(-0.001, 100.0, 0.1)
av_bin_rate = np.zeros(bins.size - 1)
for i, bin in enumerate(bins[:-1]):
    bin1 = bins[i+1]
    df1 = tmp_sim_df[tmp_sim_df['Spont']>=bin]
    df1 = df1[df1['Spont']<bin1]
    if (df1.shape[0] > 0):
        av_bin_rate[i] = df1['g8'].mean()

# Extract only the non-zero entries from the bin-averaged firing rate.
ind = np.where(av_bin_rate > 0.0)
av_bin_rate1 = av_bin_rate[ind]
bins1 = bins[ind]

fig = plt.figure(figsize=(8, 12)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5]) 
ax = [plt.subplot(gs[0]), plt.subplot(gs[1])]

ax[1].scatter(sim_df_pos['Spont'], sim_df_pos['g8'], c='firebrick', s=2, lw=0)
ax[1].scatter(bins1, av_bin_rate1, c='red', s=50, lw=0)
ax[1].set_xlim((0.08, 60.0))
ax[1].set_yscale('log')
ax[1].set_xscale('log')

sim_df_0 = tmp_sim_df[tmp_sim_df['Spont']==0.0]
ax[0].scatter(sim_df_0['Spont'], sim_df_0['g8'], c='firebrick', s=3, lw=0)
ax[0].scatter([0.0], [av_bin_rate[0]], c='red', s=50, lw=0)
ax[0].set_yscale('log')

ax[0].set_xticks([0.0])
ax[1].set_yticklabels([])

ax[1].set_xlabel('Spont. rate (Hz)')
ax[0].set_ylabel('Drifting grating response (Hz)')

ax[0].set_ylim((0.03, 60.0))
ax[1].set_ylim((0.03, 60.0))

plt.tight_layout()
plt.savefig('log_normal_rates/log_normal_rates_spont_g8.eps', format='eps')
plt.show()



