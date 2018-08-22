import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams.update({'font.size': 15})

import pandas as pd

from scipy.optimize import leastsq

# Treat ori_pref as one of the parameters.

# Double Gaussian function.
def double_gaussian(x, params):
    (ori_pref, A1, A2, B, sigma) = params
    ori_pref2 = (ori_pref + 180.0) % 360.0

    # For a simple Gaussian, we may run into boundary effects, where x close to 360 degrees may not have the corresponding y values
    # that represent properly the effect from a peak close to 0 degrees.  It would be better to use a wrapped normal distribution here
    # (a Gaussian on a circle), but it is an infinite series; we approximate it by just three terms, with shifts at 0 degrees,
    # 360 degrees, and -360 degrees.  In cases with relatively small sigma, this approximation should hold well.  However, for large sigma
    # it may not be quite as reasonable.
    # NOTE THAT B+A1 AND B+A2 DO NOT CORRESPOND EXACTLY TO THE HEIGHTS OF THE PEAKS WITH THIS CHOICE OF THE FUNCTION.
    # For small sigma, the peaks should be very close to B+A1 and B+A2, but for large sigma they will deviate from those values significantly.
    r1 = A1 * np.exp( - (x - ori_pref)**2.0 / (2.0 * sigma**2.0) ) + A2 * np.exp( - (x - ori_pref2)**2.0 / (2.0 * sigma**2.0) )
    r2 = A1 * np.exp( - (x - ori_pref + 360.0)**2.0 / (2.0 * sigma**2.0) ) + A2 * np.exp( - (x - ori_pref2 + 360.0)**2.0 / (2.0 * sigma**2.0) )
    r3 = A1 * np.exp( - (x - ori_pref - 360.0)**2.0 / (2.0 * sigma**2.0) ) + A2 * np.exp( - (x - ori_pref2 - 360.0)**2.0 / (2.0 * sigma**2.0) )

    return B + r1 + r2 + r3

def double_gaussian_fit( params, x, y_av ):
    if within_bounds(params, y_av):
        return (y_av - double_gaussian( x, params ))
    else:
        return 1.0e6

def within_bounds(params, y_av):
    (ori_pref, A1, A2, B, sigma) = params
    y_max = y_av.max()
    y_min = y_av.min()
    delta_y = y_max - y_min

    res = True
    if ((ori_pref < -30.0) or (ori_pref > 360.0) or (A1 < 0.0) or (A1 > 1.2*delta_y) or (A2 < 0.0) or (A2 > 1.2*delta_y) or (B < 0.0) or (sigma < 5.0) or (sigma > 500.0)):
    #if ((ori_pref < 0.0) or (ori_pref > 360.0) or (A1 < 0.0) or (A1 > 2.0*y_max) or (A2 < 0.0) or (A2 > 2.0*y_max) or (B < 0.0) or (B > y_min) or (sigma < 5.0) or (sigma > 180.0)):
        res = False

    return res


# Decide which systems we are doing analysis for.
sys_dict = {}
#sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates.npy', 'f_out_pref': 'Ori/ll2_pref_stat.csv', 'f_out_Gfit': 'Ori/ll2_pref_stat_and_Gfit.csv', 'grating_id': range(7, 240, 30)+range(8, 240, 30)+range(9, 240, 30) }
#sys_dict['ll2_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_sd278/spk.dat', 'f_3': '_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_pref_stat_4Hz.csv', 'f_out_Gfit': 'Ori/ll2_pref_stat_and_Gfit_4Hz.csv', 'grating_id': range(8, 240, 30) }
#sys_dict['ll2_ctr30_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr30_sd278/spk.dat', 'f_3': '_ctr30_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr30_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr30_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr30_pref_stat_4Hz.csv', 'f_out_Gfit': 'Ori/ll2_ctr30_pref_stat_and_Gfit_4Hz.csv', 'grating_id': range(8, 240, 30) }
sys_dict['ll2_ctr10_TF4Hz'] = { 'cells_file': '../build/ll2.csv', 'f_1': '../output_ll2_', 'f_2': '_ctr10_sd278/spk.dat', 'f_3': '_ctr10_sd278/tot_f_rate.dat', 'f_out': 'Ori/ll2_ctr10_rates_4Hz.npy', 'f_out_std': 'Ori/ll2_ctr10_rates_std_4Hz.npy', 'f_out_pref': 'Ori/ll2_ctr10_pref_stat_4Hz.csv', 'f_out_Gfit': 'Ori/ll2_ctr10_pref_stat_and_Gfit_4Hz.csv', 'grating_id': range(8, 240, 30) }


# Compute the Gaussian fits for tuning curves.

# Load simulation data.
sim_data = {}
for sys_name in sys_dict.keys():
    sim_data[sys_name] = pd.read_csv(sys_dict[sys_name]['f_out_pref'], sep=' ')
    ori_list = [x for x in sim_data[sys_name].columns.values if x not in ['id', 'ori', 'SF', 'TF', 'CV_ori', 'OSI_modulation', 'DSI']]
    ori_float = np.array([float(x) for x in ori_list])
    ind_ori_sort = ori_float.argsort()
    ori_float_sorted = ori_float[ind_ori_sort]

    A1_list = []
    A2_list = []
    B_list = []
    sigma_list = []
    ori_pref_list = []
    goodness_r_list = []
    for k_cell, gid in enumerate(sim_data[sys_name]['id'].values):
        av = []
        std = []
        if (k_cell % 100 == 0):
            print 'System %s; processing cell %d.' % (sys_name, gid)
        for ori in ori_list:
            tmp = sim_data[sys_name][sim_data[sys_name]['id'] == gid][ori].values[0]
            tmp = tmp[1:][:-1].split(',') # This is a string with the form '(av,std)', and so we can remove the brackets and comma to get strings 'av' and 'std', where av and std are numbers.
            av.append(float(tmp[0]))
            std.append(float(tmp[1]))
        # Convert av and std to numpy array and change the sequence of elements according to the sorted ori.
        av = np.array(av)[ind_ori_sort]
        std = np.array(std)[ind_ori_sort]

        # Treat ori_pref as one of the parameters.
        params_start = [ori_float_sorted[av.argmax()], av.max()-av.min(), av.max()-av.min(), av.min(), 30.0]
        fit_params, cov, infodict, mesg, ier = leastsq( double_gaussian_fit, params_start, args=(ori_float_sorted, av), full_output=True )
        ori_pref = fit_params[0]

        ss_err = (infodict['fvec']**2).sum()
        ss_tot = ((av - av.mean())**2).sum()
        if (ss_tot != 0.0):
            rsquared = 1 - (ss_err/ss_tot)
        else:
            rsquared = 0.0
        A1_list.append(fit_params[1])
        A2_list.append(fit_params[2])
        B_list.append(fit_params[3])
        sigma_list.append(fit_params[4])
        ori_pref_list.append(ori_pref)
        goodness_r_list.append(rsquared)

    sim_data[sys_name]['Gfit_A1'] = np.array(A1_list)
    sim_data[sys_name]['Gfit_A2'] = np.array(A2_list)
    sim_data[sys_name]['Gfit_B'] = np.array(B_list)
    sim_data[sys_name]['Gfit_sigma'] = np.array(sigma_list) 
    sim_data[sys_name]['Gfit_ori_pref'] = np.array(ori_pref_list)
    sim_data[sys_name]['Gfit_goodness_r'] = np.array(goodness_r_list)

    sim_data[sys_name].to_csv(sys_dict[sys_name]['f_out_Gfit'], sep=' ')    


