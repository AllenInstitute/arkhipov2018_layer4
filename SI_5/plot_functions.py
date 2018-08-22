import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import h5py
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

# Check coordinates of the source to find the one close to (0, 0).  Use it
# for illustration of the dynamics of traveling waves near the center of the system.
'''
f = open('tw_data/ll1_tw_build/tw_src_0/sources.pkl', 'r')
#f = open('tw_data/ll2_tw_build/2_tw_src/sources.pkl', 'r')
src_dict = pickle.load(f)
f.close()

# Sources coordinates.
x_src = []
y_src = []
for src_id in src_dict:
  x_src.append(src_dict[src_id]['x'])
  y_src.append(src_dict[src_id]['y'])
x_src = np.array(x_src)
y_src = np.array(y_src)

ind = np.intersect1d(np.where(np.abs(x_src) < 15.0), np.where(np.abs(y_src) < 15.0))
print ind, x_src[ind], y_src[ind]

plt.scatter(x_src[ind], y_src[ind]); plt.show()
'''

def plot_spikes(spk_f_name, **kwargs):
    cells_file = kwargs.get('cells_file', None)
    types_color_list = kwargs.get('types_colors', ['black', 'brown', 'magenta', 'blue', 'cyan', 'red', 'darkorange'])
    if (cells_file != None):
        cells_db = pd.read_csv(cells_file, sep=' ')

    # Read spikes.
    series = np.genfromtxt(spk_f_name, delimiter=' ')

    # Plot spikes.
    fig, axes = plt.subplots(1, 1)
    if (series.size > 2):
        if (cells_file != None):
            types = sorted(list(set(list(cells_db['type'].values))))
            for k_type, type in enumerate(types):
                gids = cells_db[cells_db['type'] == type]['index'].values
                tf_mask = np.in1d(series[:, 1], gids)
                axes.scatter(series[tf_mask, 0], series[tf_mask, 1], s=5, lw=0, facecolor=types_color_list[k_type % len(types_color_list)])
        else:
            axes.scatter(series[:, 0], series[:, 1], s=1, c='k')
    axes.set_xlabel('Time (ms)')
    axes.set_ylabel('Neuron ID')

    return fig, axes





def plot_spikes_tw(spk_f_name, tw_f_name, tw_src_id, **kwargs):
    cells_file = kwargs.get('cells_file', None)
    tstop = kwargs.get('tstop', 1000.0)
    types_color_list = kwargs.get('types_colors', ['black', 'brown', 'magenta', 'blue', 'cyan', 'red', 'darkorange'])
    if (cells_file != None):
        cells_db = pd.read_csv(cells_file, sep=' ')

    # Read spikes.
    series = np.genfromtxt(spk_f_name, delimiter=' ')

    # Read the background activity trace.
    f = open(tw_f_name, 'r')
    tw_data = pickle.load(f)
    f.close()

    # Set up axes.
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 0.2, 1], hspace=0.05)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    # Plot spikes.
    if (series.size > 2):
        if (cells_file != None):
            types = sorted(list(set(list(cells_db['type'].values))))
            for k_type, type in enumerate(types):
                gids = cells_db[cells_db['type'] == type]['index'].values
                tf_mask = np.in1d(series[:, 1], gids)
                ax1.scatter(series[tf_mask, 0], series[tf_mask, 1], s=5, lw=0, facecolor=types_color_list[k_type % len(types_color_list)])
        else:
            ax1.scatter(series[:, 0], series[:, 1], s=5, lw=0, facecolor='k')
    ax1.set_ylabel('Neuron ID')

    # Plot the magnitude of background activity.
    ax3.plot(tw_data['t'], tw_data['cells'][tw_src_id], c='k', linewidth=2)
    ax3.set_ylabel('Bkg. activity (arb. u.)')

    # Define consistent limits.
    ax3.set_xlabel('Time (ms)')
    ax1.set_xlim((0, tstop))
    ax2.set_xlim((0, tstop))
    ax3.set_xlim((0, tstop))

    # Fix the appearance for the plot of background activity.
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    # Use the middle panel to show duration of the stimulus (e.g., the image) if necessary.
    ax2.axis('off')
    # This probably fits best in the external script that may be calling the whole function.
    #ax2.hlines(0.0, t_vis_stim[0], t_vis_stim[1], linewidth=10, color='c')

    return fig, [ax1, ax2, ax3]





def plot_series_tw(f_names, series_label, cell_gids, tw_f_name, tw_src_id, **kwargs):
    cells_file = kwargs.get('cells_file', None)
    tstop = kwargs.get('tstop', 1000.0)
    types_color_list = kwargs.get('types_colors', ['black', 'brown', 'magenta', 'blue', 'cyan', 'red', 'darkorange'])
    if (cells_file != None):
        cells_db = pd.read_csv(cells_file, sep=' ')

    # Read the background activity trace.
    f = open(tw_f_name, 'r')
    tw_data = pickle.load(f)
    f.close()

    # Set up axes.
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 0.2, 1], hspace=0.05)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    # Plot series.
    for k, f_name in enumerate(f_names):
        gid = cell_gids[k]

        h5 = h5py.File(f_name, 'r')
        values = h5[series_label][...]
        tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']
        if (tarray.size > 1):  
            if (cells_file != None):
                types = sorted(list(set(list(cells_db['type'].values))))
                k_type = types.index(cells_db[cells_db['index'] == gid]['type'].values[0])
                ax1.plot(tarray, values, color=types_color_list[k_type % len(types_color_list)], label=('Cell %d' % (gid)))
            else:
                ax1.plot(tarray, values, color='k', label=('Cell %d' % (gid)))
    ax1.legend()

    # Plot the magnitude of background activity.
    ax3.plot(tw_data['t'], tw_data['cells'][tw_src_id], c='k', linewidth=2)
    ax3.set_ylabel('Bkg. activity (arb. u.)')

    # Define consistent limits.
    ax3.set_xlabel('Time (ms)')
    ax1.set_xlim((0, tstop))
    ax2.set_xlim((0, tstop))
    ax3.set_xlim((0, tstop))

    # Fix the appearance for the plot of background activity.
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.yaxis.set_ticks_position('left')
    ax3.xaxis.set_ticks_position('bottom')

    # Use the middle panel to show duration of the stimulus (e.g., the image) if necessary.
    ax2.axis('off')
    # This probably fits best in the external script that may be calling the whole function.
    #ax2.hlines(0.0, t_vis_stim[0], t_vis_stim[1], linewidth=10, color='c')

    return fig, [ax1, ax2, ax3]


def plot_series(f_names, series_label, cell_gids, **kwargs):
    cells_file = kwargs.get('cells_file', None)
    tstop = kwargs.get('tstop', 1000.0)
    tstart = kwargs.get('tstart', 0.0)
    types_color_list = kwargs.get('types_colors', ['black', 'brown', 'magenta', 'blue', 'cyan', 'red', 'darkorange'])
    if (cells_file != None):
        cells_db = pd.read_csv(cells_file, sep=' ')

    # Set up axes.
    fig, ax = plt.subplots()

    # Plot series.
    for k, f_name in enumerate(f_names):
        gid = cell_gids[k]

        h5 = h5py.File(f_name, 'r')
        values = h5[series_label][...]
        tarray = np.arange(0, values.size) * 1.0 * h5.attrs['dt']
        if (tarray.size > 1):
            if (cells_file != None):
                types = sorted(list(set(list(cells_db['type'].values))))
                k_type = types.index(cells_db[cells_db['index'] == gid]['type'].values[0])
                ax.plot(tarray, values, color=types_color_list[k_type % len(types_color_list)], label=('Cell %d' % (gid)))
            else:
                ax.plot(tarray, values, color='k', label=('Cell %d' % (gid)))
    ax.legend()

    ax.set_xlabel('Time (ms)')
    ax.set_xlim((tstart, tstop))

    ind = np.intersect1d(np.where(tarray > tstart), np.where(tarray < tstop))
    val_max = values[ind].max()
    val_min = values[ind].min()
    val_range = val_max - val_min
    if (val_range == 0.0):
      val_range = 0.1 * val_max
    ax.set_ylim((val_min - val_range*0.05, val_max + val_range*0.05))


    return fig, ax

