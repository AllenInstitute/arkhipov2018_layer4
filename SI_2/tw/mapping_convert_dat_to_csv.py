import pandas as pd
import numpy as np
from os import path, makedirs

import random
random.seed(150)

N_spk_trains_per_src = 1 # Number of spike trains generated for each source; this is something we need to know from previous steps.
N_syn_dict = {}

source_type = 'tw_exc'
t_shift = 0.0

cells = pd.read_csv('../ll2.csv', sep=' ') #ll1.csv, ll3.csv
gid_array = np.array(cells['index'])

# Read the description of the mapping from sources to cells.
f_in = open('mapping_2_tw_src.dat', 'r') #mapping_1_tw_src.dat, mapping_3_tw_src.dat
map_dict = {}
for line in f_in:
  tmp_l = line.split()
  gid = int(tmp_l[0])
  map_dict[gid] = np.array([int(x) for x in tmp_l[1:]])
  x_rand = 100.0 * random.random()
  if (x_rand < 80.0):
    N_syn_dict[gid] = 1
  elif (x_rand < 85.0):
    N_syn_dict[gid] = 2
  elif (x_rand < 90.0):
    N_syn_dict[gid] = 4
  elif (x_rand < 95.0):
    N_syn_dict[gid] = 8
  else:
    N_syn_dict[gid] = 16
f_in.close()





# Go over target cells and choose source cells that should supply inputs to each target cell.
f_out = open('mapping_2_tw_src.csv', 'w') #mapping_1_tw_src.csv, mapping_3_tw_src.csv
f_out.write('index src_type src_gid presyn_type N_syn\n')

presyn_type = 'tw_exc'

N_tar = 0
for gid in map_dict:
  N_tar += 1

N_src_dict = {}
tar_count = 0
# Make sure the list of keys is sorted, so that the target cells are processed in ascending order.
# However, in principle this should work with any list of targed gids, whether it starts from 0 or not, has gaps, etc.
for tar_gid in sorted(cells['index']):
  N_src_dict[tar_gid] = 0
  if (tar_count % 100 == 0):
    print 'Choosing tw source cells; working on target ', tar_count, ' of ', N_tar
  tar_count += 1

  if (tar_gid in map_dict.keys()):
    for src_gid in map_dict[tar_gid]:
      src_type = 'tw'
      f_out.write('%d %s %d %s %d\n' % (tar_gid, src_type, src_gid, presyn_type, N_syn_dict[tar_gid]))

f_out.close()


