import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

f_out = 'i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.eps'

model_list = ['ll1', 'll2', 'll3']
gid_sel_dict = { 'biophys_exc': {'sel': range(2, 8500, 200), 'color': 'r'}, 'biophys_inh': {'sel': range(8602, 10000, 200), 'color': 'b'} }

df_LGN_crt = pd.read_csv('i_amplification/i_amplification_process_LGN_crt_LL_bln_from_5_cell_test.csv', sep=' ')
tmp = df_LGN_crt[df_LGN_crt['grating_id'].isin([7, 37, 67, 97, 127, 157, 187, 217])]

tmp_str = ''
for gid_sel_key in gid_sel_dict:
    amp_list = []
    for model_name in model_list:
        tmp_model = tmp[tmp['model']==model_name]
        for gid in gid_sel_dict[gid_sel_key]['sel']:

            tmp_sel = tmp_model[tmp_model['gid']==gid]
            tmp_sel = tmp_sel[tmp_sel['crt'] == tmp_sel['crt'].min()] # Choose the condition corresponding to the strongest cortical current; since the current is negative, this requires taking the minimum among values.

            amp_factor = tmp_sel['LGN'].values[0] / tmp_sel['crt'].values[0]
            amp_list.append(amp_factor)

    tmp_str = tmp_str + 'LGN contrbution, %s: %.3f+/-%.3f\n' % (gid_sel_key, np.mean(amp_list), np.std(amp_list))

    amp_hist, amp_bins = np.histogram(amp_list, bins=np.linspace(0, 1.0, 30))
    amp_hist = amp_hist / (1.0 * amp_hist.sum()) # Normalize.

    plt.step(amp_bins[1:], amp_hist, c=gid_sel_dict[gid_sel_key]['color'], label=gid_sel_key)

plt.annotate(tmp_str, xy=(0.01, 0.3), xycoords='axes fraction', fontsize=10) 
plt.xlim((0.0, 0.6))
plt.legend(loc='upper left')
plt.ylabel('Fraction of cells')
plt.xlabel('LGN contribution to exc. current')
plt.savefig(f_out, format='eps')
plt.show()

