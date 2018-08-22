import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

# A function to wrap the values from a 1D array around periodic boundary.
# Here, x should be  a numpy 1D array, and x_start and  x_end are scalars.
def periodic_shift(x, x_start, x_end):
    tmp_L = x_end - x_start
    if (tmp_L <= 0.0):
        print 'periodic_shift: tmp_L <= 0; change x_start and x_end; exiting.'
        quit()
    return (x - x_start) % tmp_L + x_start

def comp_mean_norm_tuning_curve(df, norm_flag, lbl_list, lbl_ref, TF, SF, models, gids):
    data_av = {}
    for lbl in lbl_list:
        data_av[lbl] = np.array([])

    df1 = df[df['TF']==TF]
    df1 = df1[df1['SF']==SF]

    for model in models:
        df_2 = df1[df1['model']==model]
       
        for gid in gids:
            df_tmp = df_2[df_2['gid']==gid]
            ori_x = df_tmp['ori'].values
            data = {}
            for lbl in lbl_list:
                data[lbl] = np.abs(df_tmp[lbl].values) # Make sure to apply abs() here, because some of these are stored as negative current values.
                #plt.plot(ori_x, data[lbl], '-o')
            #plt.show()

            # Shift the tuning curve so that the maximum is at ori = 0; use the maximum of lbl_ref for all data entries.
            data_m_ind = data[lbl_ref].argmax()
            ori_m = ori_x[data_m_ind]
            data_m = data[lbl_ref][data_m_ind]
            ori_x = periodic_shift(ori_x - ori_m, -90.0, 270.0) # Make sure ori values run from -90 to 270.0 degrees.

            # Very importantly, the arrays need to be sorted in the ascending order of ori, so that arrays from different cells can be combined.
            ind_sorted = ori_x.argsort()
            ori_x = ori_x[ind_sorted]
            for lbl in lbl_list:
                data[lbl] = data[lbl][ind_sorted]

                #plt.plot(ori_x, data[lbl], '-o')
                #plt.show()

                # Normalize.
                if norm_flag:
                    data[lbl] = data[lbl] / data_m
                #plt.plot(ori_x, data[lbl], c='gray')

                if (data_av[lbl].size == 0):
                    data_av[lbl] = data[lbl]
                else:
                    data_av[lbl] = np.vstack((data_av[lbl], data[lbl]))#data_av[lbl] + data[lbl]

                #plt.plot(ori_x, data[lbl])
            #plt.show()

    data_err = {}
    for lbl in lbl_list:
        data_err[lbl] = data_av[lbl].std(axis=0) / np.sqrt(data_av[lbl].shape[0]) # SEM.
        data_av[lbl] = data_av[lbl].mean(axis=0) #data_av[lbl] / (len(gids) * len(models))

        #plt.plot(ori_x, data_av[lbl], '-o', c='r')
    #plt.show()

    return data_av, data_err, ori_x


def plot_mean_norm_tuning_curve(f_in, ax, norm_flag, TF, SF, gids, models, lbl_list, lbl_ref, c_lbl, display_lbl, xlabel, ylabel, xticks):
    df = pd.read_csv(f_in, sep=' ')

    data_av, data_err, ori_x = comp_mean_norm_tuning_curve(df, norm_flag, lbl_list, lbl_ref, TF, SF, models, gids)

    for lbl in lbl_list:
        #ax.plot(ori_x, data_av[lbl], '-o', c=c_lbl[lbl], label=display_lbl[lbl])
        ax.errorbar(ori_x, data_av[lbl], yerr=data_err[lbl], marker='o', ms=8, color=c_lbl[lbl], capsize=10, markeredgewidth=1, label=display_lbl[lbl])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_ticks(xticks)

    ax.legend()


#TF = 2
#SF = 0.05
#norm_flag = True
#models = ['ll1', 'll2', 'll3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_2Hz_LL'

TF = 2
SF = 0.05
norm_flag = True
models = ['ll2_inh_current']
f_in = 'i_amplification/i_amplification_process_LGN_crt_LL_inh_current_bln_from_5_cell_test.csv'
f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_2Hz_LL_inh_current'


#TF = 4
#SF = 0.05
#norm_flag = True
#models = ['ll1', 'll2', 'll3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_4Hz_LL'

#TF = 4
#SF = 0.05
#norm_flag = True
#models = ['rl1', 'rl2', 'rl3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_RL_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_4Hz_RL'

#TF = 4
#SF = 0.05
#norm_flag = True
#models = ['lr1', 'lr2', 'lr3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_LR_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_4Hz_LR'

#TF = 4
#SF = 0.05
#norm_flag = True
#models = ['rr1', 'rr2', 'rr3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_RR_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_norm_tuning_curves_TF_4Hz_RR'

#TF = 4
#SF = 0.05
#norm_flag = False
#models = ['ll1', 'll2', 'll3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_tuning_curves_TF_4Hz_LL'

#TF = 4
#SF = 0.05
#norm_flag = False
#models = ['rl1', 'rl2', 'rl3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_RL_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_tuning_curves_TF_4Hz_RL'

#TF = 4
#SF = 0.05
#norm_flag = False
#models = ['lr1', 'lr2', 'lr3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_LR_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_tuning_curves_TF_4Hz_LR'

#TF = 4
#SF = 0.05
#norm_flag = False
#models = ['rr1', 'rr2', 'rr3']
#f_in = 'i_amplification/i_amplification_process_LGN_crt_RR_bln_from_5_cell_test.csv'
#f_fig_out_base = 'i_amplification/i_amplification_mean_tuning_curves_TF_4Hz_RR'

xlabel = 'Direction (degrees)'
xticks = range(-90, 271, 90) #45)
sel_dict = {'exc': range(2, 8500, 200), 'inh': range(8602, 10000, 200)}

for sel in sel_dict:
    gids = sel_dict[sel]
    f_fig_out = '%s_%s.eps' % (f_fig_out_base, sel)

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))

    lbl_list = ['crt', 'LGN', 'crt_sub_LGN']
    lbl_ref = 'crt'
    c_lbl = {'crt': 'red', 'LGN': 'blue', 'crt_sub_LGN': 'black'}
    display_lbl = {'crt': 'tot', 'LGN': 'LGN', 'crt_sub_LGN': 'sub'}
    if norm_flag:
        ylabel = 'Mean normalized current'
    else:
        ylabel = 'Mean current (nA)'
    plot_mean_norm_tuning_curve(f_in, ax[0], norm_flag, TF, SF, gids, models, lbl_list, lbl_ref, c_lbl, display_lbl, xlabel, ylabel, xticks)


    lbl_list = ['LS_F1_crt', 'LS_F1_LGN', 'LS_F1_crt_sub_LGN']
    lbl_ref = 'LS_F1_crt'
    c_lbl = {'LS_F1_crt': 'red', 'LS_F1_LGN': 'blue', 'LS_F1_crt_sub_LGN': 'black'}
    display_lbl = {'LS_F1_crt': 'tot', 'LS_F1_LGN': 'LGN', 'LS_F1_crt_sub_LGN': 'sub'}
    if norm_flag:
        ylabel = 'Mean normalized F1'
    else:
        ylabel = 'Mean F1 (nA)'
    plot_mean_norm_tuning_curve(f_in, ax[1], norm_flag, TF, SF, gids, models, lbl_list, lbl_ref, c_lbl, display_lbl, xlabel, ylabel, xticks)

    ax[0].set_ylim(bottom=0.0)
    ax[1].set_ylim(bottom=0.0)
    if norm_flag:
        ax[0].set_ylim((0, 1.1))
        ax[1].set_ylim((0, 1.4))
    plt.savefig(f_fig_out, format='eps')
    plt.show()


