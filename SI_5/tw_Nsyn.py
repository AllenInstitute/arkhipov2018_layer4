import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

f_list = ['../tw_data/ll2_tw_build/mapping_2_tw_src.csv', '../tw_data/ll1_tw_build/mapping_tw_src_0.csv', '../tw_data/ll3_tw_build/mapping_3_tw_src.csv']

Nsyn_all = np.array([])
for f_name in f_list:
    tmp_df = pd.read_csv(f_name, sep=' ')
    Nsyn = tmp_df['N_syn'].values
    Nsyn_all = np.concatenate( (Nsyn_all, Nsyn) )

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(1,1,1)

weights = np.ones_like(Nsyn_all)/float(len(Nsyn_all))
ax.hist(Nsyn_all, weights=weights, bins=np.logspace(-0.5, 5.5, 7, base=2))
plt.gca().set_xscale('log', basex=2)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlim((0.5, 32))
ax.set_ylim((0, 0.83))
ax.set_xlabel('Number of synapses per background connection')
ax.set_ylabel('Fraction of connections')
plt.savefig('tw_Nsyn/LL_tw_Nsyn.eps', format='eps')
plt.show()



