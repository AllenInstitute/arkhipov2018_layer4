import pickle
from LGN_f_rates_and_spikes_functions import *

# System 1.
#outdir = 'output'

# System 2.
outdir = 'output2'

# System 3.
#outdir = 'output3'

# Generate spike trains from the firing rates.
spike_trains_t_window_start = 500.0
spike_trains_t_window_end = 1000000.0 # Make this very large, so that f_rate is included up to the very end of the trace.
spike_trains_p_spike_max = 0.1


#labels = ['flash_1']
labels = []
#labels.append('flash_2')
'''
for i in [8]: #range(8, 240, 30): #(range(6, 240, 30) + range(7, 240, 30) + range(8, 240, 30) + range(9, 240, 30) + range(10, 240, 30)):
  # Ignore gratings with numbers 20-30, 50-60, etc.  These correspond to sf=0.4 and 0.8, which are too fine.  We have generated
  # the sf=0.8 movies with x2 resolution in comparison with others (otherwise the grid failed to represent the fine grating
  # evenly), so it should be OK to use the movies.  However, because of the higher resolution they are very large, and
  # producing firing rates with them takes much longer.  At the same time, mouse cells are mostly unresponsive to such
  # fine gratings.  Thus, it is probably reasonable to skip them for now.
  # For sf=0.4, our preliminary tests show that responses of filters at this sf are already close to
  # baseline, as they are for even coarser sf=0.2.
  check_i = i % 30
  if ((check_i <= 20) and (check_i > 0)):
    labels.append( 'grating_' + str(i) )
'''
#labels = labels + ['Wbar_v50pixps_vert', 'Wbar_v50pixps_hor', 'Bbar_v50pixps_vert', 'Bbar_v50pixps_hor']
#labels = labels + ['8068', '108069', '130034', '163062', 'imk00895', 'imk01220', 'imk01261', 'imk01950', 'imk04208', 'pippin_Mex07_023']
#labels = labels + ['TouchOfEvil_frames_3600_to_3690', 'TouchOfEvil_frames_5550_to_5640', 'Protector2_frames_500_to_590', 'Protector2_frames_3050_to_3140', 'Protector2_frames_4700_to_4790']

#labels = labels + ['img011_BR', 'img019_BT', 'img024_BT', 'img049_BT', 'img057_BR', 'img062_VH', 'img069_VH', 'img071_VH', 'img090_VH', 'img101_VH']
'''
for imseq_id in xrange(0, 100):
    labels.append('imseq_%d' % (imseq_id))

labels = labels + ['TouchOfEvil_frames_1530_to_1680', 'TouchOfEvil_frames_3600_to_3750', 'TouchOfEvil_frames_5550_to_5700']
labels.append('spont')
'''

labels = labels + ['TouchOfEvil_frames_3600_to_3750']
#print labels

for label in labels:
  #f_name = outdir + '/' + label + '_LGN_f_rate.pkl'
  #f_out_base = outdir + '/' + label + '_LGN'

  #f_name = outdir + '/' + label + '_ctr10_LGN_f_rate.pkl'
  #f_out_base = outdir + '/' + label + '_ctr10_LGN'

  f_name = outdir + '/' + label + '_scrbl_xy_LGN_f_rate.pkl'
  f_out_base = outdir + '/' + label + '_scrbl_xy_LGN'

  #f_name = outdir + '/' + label + '_scrbl_t_LGN_f_rate.pkl'
  #f_out_base = outdir + '/' + label + '_scrbl_t_LGN'

  #f_name = outdir + '/' + label + '_LGN_f_rate.pkl'
  #f_out_base = outdir + '/' + label + '_stop_1s_LGN'

  f = open(f_name, 'r')
  f_rate_dict = pickle.load(f)
  f.close()

  # Define number of spike trains we generate for each cell.
  if ('grating' in label):
    N_trains = 10
  elif ('spont' in label):
    N_trains = 50
  else:
    N_trains = 20 # Use a larger N_trains for non-grating stimuli than for gratings, because good sampling of such stimuli (especially, natural images) seems to require more trials.

  run_f_rate_to_spike_train(f_rate_dict, spike_trains_t_window_start, spike_trains_t_window_end, spike_trains_p_spike_max, N_trains, f_out_base)

