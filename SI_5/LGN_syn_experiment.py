import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

syn_df = pd.read_csv('exp_LGN_syn/Disector_counts_synapse_percentages_only.csv', sep=' ')


m_ID_unq = np.unique(syn_df['Mouse_ID'].values)

fig, ax = plt.subplots(figsize = (3, 6))
ax.set_aspect(25.0)

for i, m_ID in enumerate(m_ID_unq):
    syn_frac = syn_df[syn_df['Mouse_ID']==m_ID]['labeled_syn_fraction'].values
    ax.scatter((i+1)*np.ones(syn_frac.size) + 0.0 * (np.random.rand(syn_frac.size) - 0.5), syn_frac, s=20, lw=0, facecolor='gray')

    ax.errorbar([i+1], [syn_frac.mean()], yerr=[syn_frac.std() / np.sqrt(1.0 * syn_frac.size)], marker='o', ms=8, color='k', linewidth=1, capsize=5, markeredgewidth=1, ecolor='k', elinewidth=1)

ax.set_xlabel('Mouse #')
ax.set_ylabel('Proportion of VGLUT2+ synapses')

ax.set_ylim(bottom=0.0)
ax.set_yticks(np.arange(0.0, 0.33, 0.1))
ax.set_xticks(np.arange(1, len(m_ID_unq)+1, 1))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(size=10)

plt.savefig('LGN_syn_experiment.eps', format='eps')
plt.show()



