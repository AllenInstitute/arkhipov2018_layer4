import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Decide which systems we are doing analysis for.
sys_dict = {}
sys_dict['ll1'] = { 'cells_file': '../build/ll1.csv', 'con_path': '../build/ll1_connections', 'f_out': 'connectivity/ll1_Nsyn.csv'}
sys_dict['ll2'] = { 'cells_file': '../build/ll2.csv', 'con_path': '../build/ll2_connections', 'f_out': 'connectivity/ll2_Nsyn.csv'}
sys_dict['ll3'] = { 'cells_file': '../build/ll3.csv', 'con_path': '../build/ll3_connections', 'f_out': 'connectivity/ll3_Nsyn.csv'}
sys_dict['rr1'] = { 'cells_file': '../build/rr1.csv', 'con_path': '../build/rr1_connections', 'f_out': 'connectivity/rr1_Nsyn.csv'}
sys_dict['rr2'] = { 'cells_file': '../build/rr2.csv', 'con_path': '../build/rr2_connections', 'f_out': 'connectivity/rr2_Nsyn.csv'}
sys_dict['rr3'] = { 'cells_file': '../build/rr3.csv', 'con_path': '../build/rr3_connections', 'f_out': 'connectivity/rr3_Nsyn.csv'}


# Read the connectivity information and save number of synapses received by each neuron.
N_tar_gid_per_file = 100

for sys in sys_dict:
    cells_db = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ')

    exc_gids = cells_db.loc[cells_db['type'].isin(['Scnn1a', 'Rorb', 'Nr5a1', 'LIF_exc'])]['index'].values
    inh_gids = cells_db.loc[cells_db['type'].isin(['PV1', 'PV2', 'LIF_inh'])]['index'].values

    N_tot = cells_db['index'].values.size

    f_key_for_tar_gid = ''
    f_out = open(sys_dict[sys]['f_out'], 'w')
    f_out.write('gid N_exc_syn N_inh_syn\n')
    for k, tar_gid in enumerate(cells_db['index'].values):
        if ( tar_gid % 100 == 0 ):
            print "System %s; processing cell %d of %d." % (sys, tar_gid, N_tot)

        target_type = cells_db.ix[k]['type']

        f_key_for_tar_gid_new = N_tar_gid_per_file * (tar_gid / N_tar_gid_per_file) # Note that integer division is used here.
        if (f_key_for_tar_gid_new != f_key_for_tar_gid): # Open the file and process its contents.
            f_key_for_tar_gid = f_key_for_tar_gid_new
            f_con_name = '%s/target_%d_%d.dat' % (sys_dict[sys]['con_path'], f_key_for_tar_gid, f_key_for_tar_gid+N_tar_gid_per_file)
            con_df = pd.read_csv(f_con_name, sep=' ', header=None)
            con_df.columns = ['tar', 'src', 'N_syn']
            con_exc_src = con_df.loc[con_df['src'].isin(exc_gids)]
            con_inh_src = con_df.loc[con_df['src'].isin(inh_gids)]

        # Compute the number of excitatory and inhibitory synapses received by the current neuron.
        N_exc_syn = con_exc_src.loc[con_exc_src['tar'] == tar_gid]['N_syn'].sum() # Sum all N_syn.
        N_inh_syn = con_inh_src.loc[con_inh_src['tar'] == tar_gid]['N_syn'].sum() # Sum all N_syn.
        f_out.write('%d %d %d\n' % (tar_gid, N_exc_syn, N_inh_syn))

    f_out.close()


# Plot distributions of N_exc_syn and N_inh_syn for each cell type.
bin_size = 20 # Make sure this is the same for all source and target types; otherwise, values of the histograms cannot be compared directly unless normalized by bin size.
hist_bins = { 'exc': np.arange(1500, 2500, bin_size), 'inh': np.arange(0, 700, bin_size) }
y_lim = {'exc': [0.0, 0.1], 'inh': [0.0, 0.2]}
x_lim = {'exc': [hist_bins['exc'][0], hist_bins['exc'][-1]], 'inh': [hist_bins['inh'][0], hist_bins['inh'][-1]]}
sel_types = ['Scnn1a', 'Rorb', 'Nr5a1', 'PV1', 'PV2']
colors_dict = { 'Scnn1a': 'darkorange', 'Rorb': 'red', 'Nr5a1': 'magenta', 'PV1': 'blue', 'PV2': 'cyan' }
type_gids = {}
for sys in sys_dict:
    cells_db = pd.read_csv(sys_dict[sys]['cells_file'], sep=' ').rename(columns = {'index': 'gid'})
    Nsyn_df = pd.read_csv(sys_dict[sys]['f_out'], sep=' ')
    df_merged = pd.merge(cells_db, Nsyn_df, on='gid', how='inner')

    for type in sel_types:
        type_df = df_merged.loc[df_merged['type'] == type]
        for exc_inh in ['exc', 'inh']:
            N_syn = type_df['N_%s_syn' % (exc_inh)].values
            Nsyn_hist = np.histogram(N_syn, bins=hist_bins[exc_inh])[0] / (1.0 * N_syn.size)
            plt.plot(hist_bins[exc_inh][:-1], Nsyn_hist, c=colors_dict[type])
            plt.title('%s; type %s; sources %s' % (sys, type, exc_inh))
            plt.xlabel('Number of synapses')
            plt.ylabel('Fraction of cells')
            plt.ylim(y_lim[exc_inh])
            plt.xlim(x_lim[exc_inh])
            fig_f_name = 'connectivity/Nsyn_hist_%s_%s_%s.eps' % (sys, type, exc_inh)
            print '%s' % (fig_f_name)
            plt.savefig(fig_f_name, format='eps')
            #plt.show()
            plt.close()

