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

def comp_mean_norm_tuning_curve(sys_dict, norm_flag, gids):
    data_av = {}

    for sys in sys_dict:
        df_dict = {}
        for key in sys_dict[sys]:
            if (key != 'ref'):

                if (key not in data_av.keys()):
                    data_av[key] = np.array([])

                df_dict[key] = pd.read_csv(sys_dict[sys][key], sep=' ')
                # Assume that the grating ori columns are the same for all files for the same system.
                ori_list = [x for x in df_dict[key].columns.values if x not in ['id', 'ori', 'SF', 'TF', 'CV_ori', 'OSI_modulation', 'DSI']]
                ori_float = np.array([float(x) for x in ori_list])
            else:
                lbl_ref = sys_dict[sys][key]
      
        for gid in gids:
            if (gid % 100 == 0):
                print 'Processing gid %d' % (gid)
            f_rate = {}
            for key in df_dict:
                df_tmp = df_dict[key][df_dict[key]['id'] == gid]
                f_rate[key] = []
                for ori in ori_list:
                    tmp = df_tmp[ori].values[0]
                    f_rate[key].append(tmp[1:][:-1].split(',')[0]) # This is a string with the form '(av,std)', and so we can remove the brackets and comma to get strings 'av' and 'std', where av and std are numbers.
                f_rate[key] = np.array([float(x) for x in f_rate[key]])

            # Shift the tuning curve so that the maximum is at ori = 0; use the maximum of lbl_ref for all data entries.
            data_m_ind = f_rate[lbl_ref].argmax()
            ori_m = ori_float[data_m_ind]
            data_m = f_rate[lbl_ref][data_m_ind]
            ori_x = periodic_shift(ori_float - ori_m, -90.0, 270.0) # Make sure ori values run from -90 to 270.0 degrees.

            # Very importantly, the arrays need to be sorted in the ascending order of ori, so that arrays from different cells can be combined.
            ind_sorted = ori_x.argsort()
            ori_x = ori_x[ind_sorted]
            for key in f_rate:
                f_rate[key] = f_rate[key][ind_sorted]

                # Normalize.
                if norm_flag:
                    if data_m > 0:
                        f_rate[key] = f_rate[key] / data_m

                if (data_av[key].size == 0):
                    data_av[key] = f_rate[key]
                else:
                    data_av[key] = np.vstack((data_av[key], f_rate[key]))

    data_err = {}
    for key in data_av:
        data_err[key] = data_av[key].std(axis=0) / np.sqrt(data_av[key].shape[0]) # SEM.
        data_av[key] = data_av[key].mean(axis=0) #data_av[lbl] / (len(gids) * len(models))

        #plt.plot(ori_x, data_av[lbl], '-o', c='r')
    #plt.show()

    return data_av, data_err, ori_x


def plot_mean_norm_tuning_curve(sys_dict, ax, norm_flag, gids, c_lbl, xlabel, ylabel, xticks):

    data_av, data_err, ori_x = comp_mean_norm_tuning_curve(sys_dict, norm_flag, gids)

    for key in data_av:
        ax.errorbar(ori_x, data_av[key], yerr=data_err[key], marker='o', ms=8, color=c_lbl[key], capsize=10, markeredgewidth=1, label=key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_ticks(xticks)

    ax.legend()


norm_flag = True
sys_dict = {}
sys_dict['ll1'] = {'tot': 'Ori/ll1_pref_stat.csv', 'LGN_only': 'Ori/ll1_LGN_only_no_con_pref_stat.csv', 'ref': 'LGN_only'}
sys_dict['ll2'] = {'tot': 'Ori/ll2_pref_stat.csv', 'LGN_only': 'Ori/ll2_LGN_only_no_con_pref_stat.csv', 'ref': 'LGN_only'}
sys_dict['ll3'] = {'tot': 'Ori/ll3_pref_stat.csv', 'LGN_only': 'Ori/ll3_LGN_only_no_con_pref_stat.csv', 'ref': 'LGN_only'}
fig_out_base = 'Ori/Ori_plot_combined_tuning_curve_LL'

c_lbl = {'tot': 'red', 'LGN_only': 'blue'}

xlabel = 'Direction (degrees)'
xticks = range(-90, 271, 90) #45)
sel_dict = {'exc': range(0, 8500), 'inh': range(8500, 10000)}

for sel in sel_dict:
    gids = sel_dict[sel]
    f_fig_out = '%s_%s.eps' % (fig_out_base, sel)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    if norm_flag:
        ylabel = 'Mean normalized rate'
    else:
        ylabel = 'Mean rate (Hz)'
    plot_mean_norm_tuning_curve(sys_dict, ax, norm_flag, gids, c_lbl, xlabel, ylabel, xticks)

    ax.set_ylim(bottom=0.0)
    #if norm_flag:
    #    ax.set_ylim((0, 1.2))
    plt.savefig(f_fig_out, format='eps')
    plt.show()


