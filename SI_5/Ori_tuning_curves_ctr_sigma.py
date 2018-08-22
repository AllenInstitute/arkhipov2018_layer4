import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pandas as pd


cells_file = '../build/ll2.csv'
cells_db = pd.read_csv(cells_file, sep=' ')

# Use sigma from the pre-computed Gaussian fits for tuning curves as a measure of the tuning width.
sim_data_ctr80 = pd.read_csv('Ori/ll2_pref_stat_and_Gfit_4Hz.csv', sep=' ')
sim_data_ctrlow = pd.read_csv('Ori/ll2_ctr10_pref_stat_and_Gfit_4Hz.csv', sep=' ')
fig_basename = 'Ori/sigma_ctr80_ctr10_ll2_'

# Select those cells that have sufficiently high orientation selectivity.
# Do that based on the selectivity observed for contrast of 80%.
gids_with_high_CV_ori = sim_data_ctr80[sim_data_ctr80['CV_ori'] > 0.5]['id'].values

# Also, select only the cells for which goodness of fit was fairly high.
fit_gids_ctr80 = sim_data_ctr80[sim_data_ctr80['Gfit_goodness_r'] > 0.95]['id'].values
fit_gids_ctrlow = sim_data_ctrlow[sim_data_ctrlow['Gfit_goodness_r'] > 0.95]['id'].values

gids_sel_ctr80 = np.intersect1d(gids_with_high_CV_ori, fit_gids_ctr80)
gids_sel_ctrlow = np.intersect1d(gids_with_high_CV_ori, fit_gids_ctrlow)

gids_sel = np.intersect1d(gids_sel_ctr80, gids_sel_ctrlow)

for type_list in [['Scnn1a'], ['Rorb'], ['Nr5a1'], ['Scnn1a', 'Rorb', 'Nr5a1']]:
    type_list_str = '_'.join(type_list)
    fig_name = fig_basename + '%s.eps' % (type_list_str)
    fig_name_hist = fig_basename + '%s_hist.eps' % (type_list_str)

    type_gids = cells_db[cells_db['type'].isin(type_list)]['index'].values
    current_gids = np.intersect1d(gids_sel, type_gids)

    sigma_data_ctr80 = sim_data_ctr80[sim_data_ctr80['id'].isin(current_gids)]['Gfit_sigma']
    sigma_data_ctrlow = sim_data_ctrlow[sim_data_ctrlow['id'].isin(current_gids)]['Gfit_sigma']

    # Compute the coefficient of determination, R^2, assuming that ideally the relationship should be sigma_data_ctrlow = sigma_data_ctr80.
    # Compute the residual sum of squares.
    SS_res = np.sum((sigma_data_ctrlow - sigma_data_ctr80)**2)
    # Compute the total sum of squares.
    SS_tot = np.sum((sigma_data_ctrlow - sigma_data_ctrlow.mean())**2)
    # Combine them to obtain R^2.
    R2 = 1.0 - SS_res/SS_tot
    # Average root mean square deviation.
    RMSD = np.sqrt(SS_res / sigma_data_ctrlow.size)
 
    plt.scatter(sigma_data_ctr80, sigma_data_ctrlow, s=20, lw=0, facecolor='k')
    plt.plot([0.0, 100.0], [0.0, 100.0], '--', c='gray')
    plt.xlim((10.0, 50.0))
    plt.ylim((10.0, 50.0))
    plt.xlabel('Contrast 80% HWHH (Degrees)')
    plt.ylabel('Contrast 10% HWHH (Degrees)')
    plt.title('%s' % (type_list_str))
    plt.savefig(fig_name, format='eps')
    plt.show()

    fit_sigma_dif = sigma_data_ctr80.values - sigma_data_ctrlow.values
    print '%s: average HWHH difference (Degrees): %f +/- %f, R^2 = %f, RMSD = %f degrees.' % (type_list_str, fit_sigma_dif.mean(), fit_sigma_dif.std(), R2, RMSD)

    plt.hist(fit_sigma_dif, bins=np.arange(-18.0, 18.0, 3.0), normed=True, rwidth=0.8, facecolor='k')
    plt.xlim((-23.0, 23.0))
    plt.xlabel('HWHH difference (Degrees)')
    plt.ylabel('Density (1/Degree)')
    plt.title('%s' % (type_list_str))
    plt.savefig(fig_name_hist, format='eps')
    plt.show()    


