import pandas as pd
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

d_cutoff = 150.0 # Cutoff distance, in um.
d_cutoff_2 = d_cutoff**2

# Maximal and minimal number of allowed sources per cell.
N_src_min = 18
N_src_max = 24

cells = pd.read_csv('../ll2.csv', sep=' ') #ll1.csv, ll3.csv



src_system_id = 2 #1, 3
src_dir = '/data/mat/antona/network/14-simulations/9-network/tw_data/ll%d_tw_build/%d_tw_src' % (src_system_id, src_system_id)

f = open(src_dir + '/sources.pkl', 'r')
src_dict = pickle.load(f)
f.close()

# Cell coordinates.
gid_array = np.array(cells['index'])
x = np.array(cells['x'])
y = np.array(cells['y'])

# Sources coordinates.
x_src = []
y_src = []
for src_id in src_dict:
  x_src.append(src_dict[src_id]['x'])
  y_src.append(src_dict[src_id]['y'])
x_src = np.array(x_src)
y_src = np.array(y_src)


# For each cell, find sources that are within a cutoff distance from the cell and use them to provide inputs to that cell.
f_out = open('mapping_%d_tw_src.dat' % (src_system_id), 'w')
N_src_list = []
for gid in gid_array:                                                       
    src_ids = np.where( (x_src - x[gid])**2 + (y_src - y[gid])**2 <= d_cutoff_2 )[0]
    # Choose a number of sources (among those defined by src_ids) that will be actually used; disregard the rest for this cell.
    N_src = np.minimum(random.randrange(N_src_min, N_src_max), src_ids.size)
    src_ids = np.random.choice(src_ids, N_src, replace=False) # Here, 'replace=False' is essential, so that the same src_id is not choosen more than once.
    f_out.write('%d' % (gid))
    for src_id in src_ids:
      f_out.write(' %d' % (src_id))
    f_out.write('\n')
    N_src_list.append(src_ids.size)
f_out.close()


