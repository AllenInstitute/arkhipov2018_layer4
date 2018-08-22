import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

from scipy.optimize import leastsq

grating_id_color = {8: 'red', 38: 'orange', 68: 'yellow', 98: 'yellowgreen', 128: 'green', 158: 'cyan', 188: 'blue', 218: 'purple', 7: 'red', 37: 'orange', 67: 'yellow', 97: 'yellowgreen', 127: 'green', 157: 'cyan', 187: 'blue', 217: 'purple'}

# If sel_dir = 0, use all directions; otherwise, choose one direction for the analysis.
sel_dir = 0

# If sel_ctr_TF = 0, use all contrasts and TFs; otherwise, choose one contrast and TF.
sel_ctr_TF = 1

# All directions.
fig_out_summary = 'i_amplification/i_amplification_fit_for_cells_summary.eps'
fig_out_summary_A_ylim = [0.3, 0.75]
fig_out_summary_B_ylim = [0, 0.04]
fig_out = 'i_amplification/i_amplification_fit_for_cells_example.eps'
fig_out_zoom = 'i_amplification/i_amplification_fit_for_cells_example_zoomed_in.eps'

f_data_out = 'i_amplification/i_amplification_fit_for_cells.csv'


# Select one direction.
if (sel_dir != 0):
    fig_out_summary = 'i_amplification/i_amplification_fit_for_cells_summary_ori90.eps'
    fig_out_summary_A_ylim = [0.3, 0.75]
    fig_out_summary_B_ylim = [0, 0.04]
    fig_out = 'i_amplification/i_amplification_fit_for_cells_example_ori90.eps'
    fig_out_zoom = 'i_amplification/i_amplification_fit_for_cells_example_zoomed_in_ori90.eps'

    f_data_out = 'i_amplification/i_amplification_fit_for_cells_ori90.csv'


# Select one contrast and TF.
if (sel_ctr_TF != 0):
    fig_out_summary = 'i_amplification/i_amplification_fit_for_cells_summary_TF2Hz_ctr80.eps'
    fig_out_summary_A_ylim = [-15.0, 10.0]
    fig_out_summary_B_ylim = [-0.5, 1.5]
    fig_out = 'i_amplification/i_amplification_fit_for_cells_example_TF2Hz_ctr80.eps'
    fig_out_zoom = 'i_amplification/i_amplification_fit_for_cells_example_zoomed_in_TF2Hz_ctr80.eps'

    f_data_out = 'i_amplification/i_amplification_fit_for_cells_TF2Hz_ctr80.csv'


def lin_mpl(x, A, B_const):
    return A * x + B_const

def lin_mpl_fit( params, x, y, B_const ):
    return (y - lin_mpl( x, params[0], B_const ))

def lin_mpl_fit_2( params, x, y ):
    return (y - lin_mpl( x, params[0], params[1] ))


df_tw = pd.read_csv('i_amplification/i_amplification_process_tw_only_LL_bln_from_5_cell_test.csv', sep=' ')

df_LGN_crt = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv', sep=' ')
df_ctr30 = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_ctr30_bln_from_5_cell_test.csv', sep=' ')
df_ctr10 = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_ctr10_bln_from_5_cell_test.csv', sep=' ')

df_LGN_crt['contrast'] = 0.8 * np.ones(df_LGN_crt.shape[0])
df_ctr30['contrast'] = 0.3 * np.ones(df_ctr30.shape[0])
df_ctr10['contrast'] = 0.1 * np.ones(df_ctr10.shape[0])

df_LGN_crt = pd.concat([df_LGN_crt, df_ctr30], axis=0)
df_LGN_crt = pd.concat([df_LGN_crt, df_ctr10], axis=0)
df_LGN_crt = df_LGN_crt.reset_index(drop=True)
df_LGN_crt.ix[df_LGN_crt['model']=='ll2_ctr30', 'model'] = 'll2'
df_LGN_crt.ix[df_LGN_crt['model']=='ll2_ctr10', 'model'] = 'll2'

# Select one direction.
if (sel_dir != 0):
    df_LGN_crt = df_LGN_crt[df_LGN_crt['grating_id'].isin([67, 68, 69])] # Direction 90 degrees.

# Select one contrast and TF.
if (sel_ctr_TF != 0):
    df_LGN_crt = df_LGN_crt[df_LGN_crt['contrast'] == 0.8] # Contrast 80%.
    df_LGN_crt = df_LGN_crt[df_LGN_crt['grating_id'].isin([7, 37, 67, 97, 127, 157, 187, 217])] # TF = 2 Hz.

#model_list = ['ll1', 'll2', 'll3']
# RESTRICT ANALYSIS TO LL2 ONLY, BECAUSE LL1 AND LL3 DO NOT HAVE CONTRASTS OTHER THAN 80%. 
model_list = ['ll2']

gid_sel_dict = { 'biophys_exc': {'sel': range(2, 8500, 200), 'color': 'firebrick'}, 'biophys_inh': {'sel': range(8602, 10000, 200), 'color': 'steelblue'} }

save_data = np.array([])

for gid_sel_key in gid_sel_dict:
    gid_sel_dict[gid_sel_key]['A_param'] = []
    gid_sel_dict[gid_sel_key]['B_param'] = []
    gid_sel_dict[gid_sel_key]['rsquared'] = []

    for model_name in model_list:
        tmp_model = df_LGN_crt[df_LGN_crt['model']==model_name]
        tmp_tw_model = df_tw[df_tw['model']==model_name]
        for gid in gid_sel_dict[gid_sel_key]['sel']:
            tmp = tmp_model[tmp_model['gid']==gid]
            tmp_tw = tmp_tw_model[tmp_tw_model['gid']==gid]

            # Linear fit.  Combine the LGN_crt and tw datasets to get a fit for both A and B in the function y = A * x + B.
            # Invert the current, so that stronger excitation corresponds to larger values.
            if (sel_ctr_TF == 0):
                current_LGN = np.hstack((np.zeros(tmp_tw.shape[0]), -1.0*tmp['LGN'].values)) 
                current_crt = np.hstack((-1.0*tmp_tw['tw_only'].values, -1.0*tmp['crt'].values))
            else: # Don't use the values from tw_only.
                current_LGN = -1.0*tmp['LGN'].values
                current_crt = -1.0*tmp['crt'].values

            # Use only one parameter for fitting, the coefficient A in y = A * x + B.
            # The offset B can be computed from the currents obtained in the case of tw-only inputs (zero LGN current).
            if (sel_ctr_TF == 0):
                params_start = (2.0)
                B_const = -1.0 * tmp_tw['tw_only'].values.mean() # Invert the current.  Also, currently, this is supposed to be just one number, but just in case, take a mean here in case a bigger array is provided.
                #fit_params, ier = leastsq( lin_mpl_fit, params_start, args=(current_LGN, current_crt, B_const) )
                fit_params, cov, infodict, mesg, ier = leastsq( lin_mpl_fit, params_start, args=(current_LGN, current_crt, B_const), full_output=True )
            else: # Don't use the values from tw_only.
                params_start = (2.0, 0.0)
                fit_params, cov, infodict, mesg, ier = leastsq( lin_mpl_fit_2, params_start, args=(current_LGN, current_crt), full_output=True )
                B_const = fit_params[1]

            ss_err = (infodict['fvec']**2).sum()
            ss_tot = ((current_crt - current_crt.mean())**2).sum()
            if (ss_tot != 0.0):
                rsquared = 1 - (ss_err/ss_tot)
            else:
                rsquared = 0.0

            tmp_save = np.array([model_name, gid, fit_params[0], B_const, rsquared])
            if (save_data.size == 0):
                save_data = tmp_save
            else:
                save_data = np.vstack( (save_data, tmp_save) )

            gid_sel_dict[gid_sel_key]['A_param'].append(1.0/fit_params[0])
            gid_sel_dict[gid_sel_key]['B_param'].append(B_const)
            gid_sel_dict[gid_sel_key]['rsquared'].append(rsquared)
            if ((gid in [4602]) and (model_name in ['ll2'])):
                #plt.scatter(-1.0*tmp['LGN'], -1.0*tmp['crt'], s=100, c='b', edgecolors='none') # Invert the current, so that stronger excitation corresponds to larger values.
                # Color the dots according to the orientation.
                for g_tmp in grating_id_color:
                    tmp1 = tmp[tmp['grating_id'] == g_tmp]
                    plt.scatter(-1.0*tmp1['LGN'], -1.0*tmp1['crt'], s=100, c=grating_id_color[g_tmp], edgecolors='none') # Invert the current, so that stronger excitation corresponds to larger values.
                if (sel_ctr_TF == 0):
                    plt.scatter(np.zeros(tmp_tw.shape[0]), -1.0*tmp_tw['tw_only'], s=100, c='gray', edgecolors='none') # Invert the current, so that stronger excitation corresponds to larger values.
                plt.plot(current_LGN, lin_mpl(current_LGN, fit_params[0], B_const), c='k')
                plt.title('Model %s, gid = %d; y=Ax+B, A=%f, B=%f' % (model_name, gid, fit_params[0], B_const), fontsize=12)
                plt.xlim(left=-0.001)
                plt.ylim(bottom=0.0)
                plt.xlabel('LGN-only current (nA)')
                plt.ylabel('Total current (nA)')
                plt.savefig(fig_out, format='eps')

                plt.xlim((0.045, 0.075))
                plt.ylim((0.100, 0.220))
                plt.savefig(fig_out_zoom, format='eps')

                plt.show()

save_data_df = pd.DataFrame(save_data, columns=['model', 'gid', 'A', 'B', 'rsquared'])
save_data_df.to_csv(f_data_out, sep=' ', index=False)





fig, axes = plt.subplots(1, 2, figsize=(10, 5))

box0 = axes[0].boxplot([gid_sel_dict[gid_sel_key]['A_param'] for gid_sel_key in gid_sel_dict], patch_artist=True) # notch=True
box1 = axes[1].boxplot([gid_sel_dict[gid_sel_key]['B_param'] for gid_sel_key in gid_sel_dict], patch_artist=True) # notch=True

for patch, color in zip(box0['boxes'], [gid_sel_dict[gid_sel_key]['color'] for gid_sel_key in gid_sel_dict]):
    patch.set_facecolor(color)
for patch, color in zip(box1['boxes'], [gid_sel_dict[gid_sel_key]['color'] for gid_sel_key in gid_sel_dict]):
    patch.set_facecolor(color)

axes[0].set_ylabel('1/A')
axes[1].set_ylabel('B (nA)')
for i, param_key in enumerate(['A_param', 'B_param']):
    tmp_str = ''
    if (param_key == 'A_param'):
        param_key_usage = '1/A_param'
    else:
        param_key_usage = param_key
    for gid_sel_key in gid_sel_dict:
        tmp_str = tmp_str + '%s: %s=%.3f+/-%.3f, n=%d\n' % (gid_sel_key, param_key_usage, np.mean(gid_sel_dict[gid_sel_key][param_key]), np.std(gid_sel_dict[gid_sel_key][param_key]), len(gid_sel_dict[gid_sel_key][param_key]))
    axes[i].annotate(tmp_str, xy=(-0.1, 0.8), xycoords='axes fraction', fontsize=12)

axes[0].set_ylim(fig_out_summary_A_ylim)
axes[1].set_ylim(fig_out_summary_B_ylim)

# Add the information about goodness of fit.
tmp_str = ''
for gid_sel_key in gid_sel_dict:
    tmp_str = tmp_str + '%s: r^2=%.2f+/-%.2f\n' % ( gid_sel_key, np.mean(gid_sel_dict[gid_sel_key]['rsquared']), np.std(gid_sel_dict[gid_sel_key]['rsquared']) )
axes[0].annotate(tmp_str, xy=(0.3, 0.5), xycoords='axes fraction', fontsize=12)

for ax in axes:
    ax.set_xticklabels(gid_sel_dict.keys())

plt.savefig(fig_out_summary, format='eps')
plt.show()


