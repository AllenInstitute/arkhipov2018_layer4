from math import *
import numpy as np
from os import path, makedirs

import matplotlib.pyplot as plt

from generate_inputs_functions import *


# HERE AND THROUGHOUT THE CODE WE USE psychopy CONVENTION OF LINEAR ANGLE APPROXIMATION.
# That is, the angles are defined not as real angles, but as ratios of the distance from
# the center of the screen to the given point (in the screen plane), d_xy, to the distance from
# the observer to the screen center, d.  Near the screen center, these "linear angles" are
# close to the real angles.  The difference is that the former are computed as d_xy/d,
# whereas the latter are computed as arctan(d_xy/d).
# To measure the "linear angles" in units consistent with the normal angles, we can use
# "linear degrees" (lindeg); to obtain that, one should compute d_xy/d * 180/pi.

# System 1.
#sys_name = 'll1'

# System 2.
#sys_name = 'll2'

# System 3.
sys_name = 'll3'


# Define the part of the visual space (in lindegs), where the centers of the ellipses representing the receptive fields of target cells should be placed.

# This should depend on the size of the cortex patch that we are simulating.  The cylinder layer 4 model with 10,000 biophysical neurons and 35,000 LIFs
# covers a circle of ~0.85 mm in radius in V1.  This seems to correspond to approximately 100 degrees in x and 50 degrees in y in the visual space
# (see Fig. 29.1D in Niell et al., chapter 29 of "The New Visual Neurosciences").
# That should be converted to lindegs.  For the center of the screen being in the middle of the angle alpha that we are dealing with,
# we can assume d_xy/d = tan(alpha/2).  Then, for lindegs, we can obtain d_xy/d * 180/pi = tan(alpha/2) * 180/pi for half of the total angle.
# Using this calculation, we arrive at ~140 lindegs in x and ~70 lindegs in y.
vis_tar_L_x = 140.0
vis_tar_L_y = 70.0

# Read the description of the source cells (filters in the visual space).
src_cells = {}
x_mean = 0.0
y_mean = 0.0
N = 0

# System 1.
#f_in = open('/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN_visual_space_positions_and_cell_types.dat', 'r')
# System 2.
#f_in = open('/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN2_visual_space_positions_and_cell_types.dat', 'r')
# System 3.
f_in = open('/data/mat/antona/network/14-simulations/6-LGN_firing_rates_and_spikes/LGN_spike_trains/LGN3_visual_space_positions_and_cell_types.dat', 'r')

for gid, line in enumerate(f_in):
  tmp_l = line.split()
  if (len(tmp_l) < 5):
    print 'Error: gaps in the source cells file; exiting.'
    quit()

  type = tmp_l[0]
  x = float(tmp_l[1])
  y = float(tmp_l[2])
  x_offset = float(tmp_l[3])
  y_offset = float(tmp_l[4])
  tmp_vec = np.array( [x_offset, y_offset] )
  if (tmp_vec.sum() == 0.0):
    tuning_angle = 'None'
  else:
    tmp_vec = tmp_vec / np.sqrt((tmp_vec**2).sum())
    tuning_angle = (360.0 + 180.0 * np.arctan2(tmp_vec[1], tmp_vec[0]) / np.pi) % 360.0 # Compute the tuning angle from the offset vector.
  tmp_dict = { gid : { 'x' : x, 'y' : y, 'x_offset' : x_offset, 'y_offset' : y_offset, 'tuning_angle' : tuning_angle } }
  x_mean += x
  y_mean += y
  N += 1
  if (type in src_cells.keys()):
    src_cells[type].update( tmp_dict )
  else:
    src_cells.update( {type : tmp_dict} )

f_in.close


# Compute the position of the center of the visual field.
x_mean = x_mean / (1.0 * N)
y_mean = y_mean / (1.0 * N)
vis_center_x = x_mean
vis_center_y = y_mean


# Read the description of the target cells (realistic cells in the physical space).
x_tot = []
y_tot = []
tar_cells = {}
f_in = open('%s.csv' % (sys_name), 'r')
for line_index, line in enumerate(f_in):
  tmp_l = line.split()
  if (len(tmp_l) < 8):
    print 'Error: gaps in the target cells file; exiting.'
    quit()

  if (line_index > 0): # Ignore the first line, as it is a header.
    gid = int(tmp_l[0])
    type = tmp_l[1]
    x = float(tmp_l[2])
    y = float(tmp_l[3])
    z = float(tmp_l[4])
    x_tot.append(x)
    y_tot.append(y)
    tar_cells.update( { gid : { 'type' : type, 'x' : x, 'y' : y, 'z' : z, 'tuning_angle' : tmp_l[5] } } )
f_in.close()

# Obtain retinotopic coordinates for the target cells by mapping the physical (x, y) coordinates of the cells
# to the visual space.
x_tot = np.array(x_tot)
y_tot = np.array(y_tot)
tar_center_x = x_tot.mean()
tar_center_y = y_tot.mean()
tar_L_x = x_tot.max() - x_tot.min()
tar_L_y = y_tot.max() - y_tot.min()

for gid  in tar_cells.keys():
  tar_cells[gid].update( { 'vis_x' : vis_center_x + ( (tar_cells[gid]['x'] - tar_center_x) / tar_L_x ) * vis_tar_L_x } )
  tar_cells[gid].update( { 'vis_y' : vis_center_y + ( (tar_cells[gid]['y'] - tar_center_y) / tar_L_y ) * vis_tar_L_y } )





# Go over target cells and choose source cells that should supply inputs to each target cell.
f_out = open('%s_inputs_from_LGN.csv' % (sys_name), 'w')
f_out.write('index src_type src_gid src_vis_x src_vis_y presyn_type N_syn\n')

f_out_el_c_d = open('%s_inputs_from_LGN_el_c_d.dat' % (sys_name), 'w')

presyn_type = 'LGN_exc'
N_syn = 30

N_tar = len(tar_cells)

N_src_dict = {}
tar_count = 0
# Make sure the list of keys is sorted, so that the target cells are processed in ascending order.
# However, in principle this should work with any list of targed gids, whether it starts from 0 or not, has gaps, etc.
for tar_gid in sorted(tar_cells.keys()):
  N_src_dict[tar_gid] = 0
  if (tar_count % 100 == 0):
    print 'Choosing source cells to create receptive fields; working on target ', tar_count, ' of ', N_tar
  tar_count += 1

  tar_type = tar_cells[tar_gid]['type']
  src_cells_selected, ellipses_centers_distance = select_source_cells(tar_cells, tar_type, tar_gid, src_cells)
  f_out_el_c_d.write('%d %g\n' % (tar_gid, ellipses_centers_distance))
  for src_type in src_cells_selected.keys():
    N_src_dict[tar_gid] += len(src_cells_selected[src_type])
    for src_gid in src_cells_selected[src_type]:
      f_out.write('%d %s %d %.3f %.3f %s %d\n' % (tar_gid, src_type, src_gid, src_cells[src_type][src_gid]['x'], src_cells[src_type][src_gid]['y'], presyn_type, N_syn))
    
f_out.close()
f_out_el_c_d.close()

f = open('tmp_N_src.dat', 'w')
for tar_gid in N_src_dict:
  f.write('%d %d\n' % (tar_gid, N_src_dict[tar_gid]))
f.close()

