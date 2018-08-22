import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def comp_r_weighted_mu_activity(f_spk, cells_file, electrode_pos, r_cutoff, bin_start, bin_stop, bin_size):
    bins = np.arange(bin_start, bin_stop+bin_size, bin_size)
    # Originally bins include both leftmost and rightmost bin edges.
    # Here we remove the rightmost edge to make the size consistent with the result computed below.
    t_out = bins[:-1]

    print 'Processing file %s.' % (f_spk)
    data = np.genfromtxt(f_spk, delimiter=' ')
    if (data.size == 0):
        t = np.array([])
        gids = np.array([])
    elif (data.size == 2):
        t = np.array([data[0]])
        gids = np.array([data[1]])
    else:
        t = data[:, 0]
        gids = data[:, 1]
    # The gids array is obtained as float64.  We should convert it to int.
    gids = gids.astype(np.int64)

    # Read information about cells.
    cells_db = pd.read_csv(cells_file, sep=' ')
    # Sort cells by gid ('index') to make sure the gids can be used as indices of numpy arrays later.
    # For those later operations, we assume that gids are increasing strictly by 1 and start from 0.
    cells_db = cells_db.sort('index', ascending=True)

    # Create an array, where, for each cells, we store the r_cutoff/r value (r is the distance to the electrode).
    x = cells_db['x'].values
    y = cells_db['y'].values
    z = cells_db['z'].values
    inv_r_array = np.sqrt((x - electrode_pos[0])**2 + (y - electrode_pos[1])**2 + (z - electrode_pos[2])**2)
    ind = np.where( inv_r_array < r_cutoff )[0]
    #inv_r_array = 1.0 + np.zeros(inv_r_array.size) # Use this for uniform weights.
    inv_r_array = r_cutoff/inv_r_array
    inv_r_array[ind] = 0.0

    # Create an array for weights, using inv_r_array values and gids from the spikes raster as indices.
    # The resulting array has the same dimensions as the array of gids form the spikes raster.
    r_weights = inv_r_array[gids]

    a_out = np.histogram(t, bins=bins, weights=r_weights)[0] # Here, np.histogram returns a tuple, where the first element is the histogram itself and the second is bins.
    a_out = 1000.0 * a_out / ( t.size * bin_size ) # Use this conversion  -- if all weights are 1, this ensures that the resulting units are Hz (time is in ms).

    # Use FFT on a_out.  Note that numpy.fft.fft(y) produces an array of the same length (perhaps +/- 1) as y; element 0 is the mean
    # of the signal (i.e., F0); the rest are components with positive and negative frequencies (the former until the middle, then latter).
    # We need ABSOLUTE VALUES of fft elements (the elements straight out of fft are complex numbers).
    fft_abs = np.abs( np.fft.fft(a_out) / a_out.size )

    # Get the set of frequencies for which the fft_abs has been computed.
    freq_fft_abs = np.fft.fftfreq(a_out.size, bin_size*0.001) # Make sure bin_size is converted from ms to seconds.

    return t_out, a_out, inv_r_array, freq_fft_abs, fft_abs





def av_r_weighted_mu_activity(f_spk_list, cells_file_list, electrode_pos, r_cutoff, bin_start, bin_stop, bin_size):
    all_fft_abs = np.array([])
    for i, f_spk in enumerate(f_spk_list):
        tmp = comp_r_weighted_mu_activity(f_spk, cells_file_list[i], electrode_pos, r_cutoff, bin_start, bin_stop, bin_size)
        fft_abs = tmp[4]
        if (all_fft_abs.size == 0):
            all_fft_abs = fft_abs
        else:
            all_fft_abs = np.vstack( (all_fft_abs, fft_abs) )
    av_fft_abs = all_fft_abs.mean(axis=0)
    std_fft_abs = all_fft_abs.std(axis=0)
    freq_fft_abs = tmp[3] # This should be the same for all f_spk.

    return freq_fft_abs, av_fft_abs, std_fft_abs

