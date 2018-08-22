from math import *
from random import *
import numpy as np

def f_rate_to_spike_train(t, f_rate, random_seed, p_spike_max):
  # t and f_rate are numpy array containing time stamps and corresponding firing rate values;
  # they are assumed to be of the same length and ordered with the time strictly increasing;
  # p_spike_max (should be <1!) is the maximal probability of spiking that we allow within the time bin; it is used to decide on the size of the time bin.

  spike_times = []

  # Use seed(...) to instantiate the random number generator.  Otherwise, current system time is used.
  seed(random_seed)

  # Assume here for each pair (t[k], f_rate[k]) that the f_rate[k] value applies to the time interval [t[k], t[k+1]).
  for k, t_k in enumerate(t[:-1]):
    delta_t = t[k+1] - t_k

    av_N_spikes = f_rate[k] / 1000.0 * delta_t # Average number of spikes expected in this interval (note that firing rate is in Hz and time is in ms).

    if (av_N_spikes > 0):
      N_bins = int(ceil(av_N_spikes / p_spike_max))
      t_base = t_k
      t_bin = 1.0 * delta_t / N_bins
      p_spike_bin = 1.0 * av_N_spikes / N_bins
      for i_bin in xrange(0, N_bins):
        rand_tmp = random()
        if rand_tmp < p_spike_bin:
          spike_t = t_base + random() * t_bin
          spike_times.append(spike_t)
        t_base += t_bin

  return spike_times


