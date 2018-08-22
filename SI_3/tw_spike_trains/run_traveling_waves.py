import traveling_waves_class as tw
import os
import random
import pickle
import sys

sys.path.insert(0, '../LGN_spike_trains/')
import LGN_f_rates_and_spikes_functions as f_spk_functions


tw_obj = tw.TravelingWave(t_total=5000.0, Lx_Ly=(1800.0, 1800.0), t_up_wave=700.0, t_up_fluct=1000.0, L_up_wave=2000.0, t_between=1000.0, t_between_fluct=1500.0)

# Create one set of sources (structure of the inputs corresponding to traveling waves).
# System 1.
#workdir = 'tw_src_0'
# System 2.
workdir = '2_tw_src'
# System 3.
#workdir = '3_tw_src'
if not os.path.exists(workdir):
  os.mkdir(workdir)

print 'Generating sources.'
# System 1.
#tw_obj.generate_sources(N=3000, rnd_seed=150, out_f_name=workdir+'/sources.pkl')
 System 2.
tw_obj.generate_sources(N=3000, rnd_seed=5, out_f_name=workdir+'/sources.pkl')
# System 3.
#tw_obj.generate_sources(N=3000, rnd_seed=12345, out_f_name=workdir+'/sources.pkl')

# Then, create many instantiations of activity (rates) generated via that structure.
# System 1.
#random.seed(15)
 System 2.
random.seed(5)
# System 3.
#random.seed(12345)
N = 600
for i in xrange(N):
  rnd_seed = random.randrange(2**30)
  print 'Generating rates; %d of %d' % (i, N)
  tw_obj.generate_rates(src_f_name=workdir+'/sources.pkl', out_name=(workdir+'/f_rates_%d.pkl' % (i)), rnd_seed=rnd_seed)


# Based on the generated rates, create spike trains.
spike_trains_t_window_start = 0.0 # Generate spikes from the very beginning of the available f_rate trace.
spike_trains_t_window_end = 1000000.0 # Make this very large, so that f_rate is included up to the very end of the trace.
spike_trains_p_spike_max = 0.1
# Number of spike trains we generate for each cell.
N_trains = 1 # Use N_trains = 1 here so that each spike train we might later choose is coming from a unique set of waves.  Otherwise, results of supposedly independent trials might become very strongly correlated.

for i in xrange(N):
  print 'Generating spikes; %d of %d' % (i, N)
  f = open(workdir+'/f_rates_%d.pkl' % (i), 'r')
  f_rate_dict = pickle.load(f)
  f.close()
  f_out_base = workdir+'/%d' % (i)
  f_spk_functions.run_f_rate_to_spike_train(f_rate_dict, spike_trains_t_window_start, spike_trains_t_window_end, spike_trains_p_spike_max, N_trains, f_out_base)

