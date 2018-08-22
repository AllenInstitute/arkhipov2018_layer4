import numpy as np
import pickle
from random import *
# Use seed(...) to instantiate the random number generator.  Otherwise, current system time is used.
# System 1.
#seed(150)

# System 2.
seed(5)

# System 3.
#seed(12345)

from LGN_f_rates_and_spikes_functions import *

# System 1.
#outdir = 'output'
#f_vis_space_name_prefix = 'LGN'

# System 2.
outdir = 'output2'
f_vis_space_name_prefix = 'LGN2'

# System 3.
#outdir = 'output3'
#f_vis_space_name_prefix = 'LGN3'





# HERE AND THROUGHOUT THE CODE WE USE psychopy CONVENTION OF LINEAR ANGLE APPROXIMATION.
# That is, the angles are defined not as real angles, but as ratios of the distance from
# the center of the screen to the given point (in the screen plane), d_xy, to the distance from
# the observer to the screen center, d.  Near the screen center, these "linear angles" are
# close to the real angles.  The difference is that the former are computed as d_xy/d,
# whereas the latter are computed as arctan(d_xy/d).
# To measure the "linear angles" in units consistent with the normal angles, we can use
# "linear degrees" (lindeg); to obtain that, one should compute d_xy/d * 180/pi.

# The numbers below are actually not exactly the screen width and height, but rather the
# width and height of the window that's 'cut out' from the larger visual space; these are
# assumed to be the dimensions of the movie that is read and processed.
screen_L_x = 240.0 #Screen width in lindegs.
screen_L_y = 120.0 #Screen height in lindegs.

# Define the region in the visual space to be tiled with LGN cells.
# AN ASSUMPTION HERE IS THAT MINIMAL X AND Y POSSIBLE ON THE SCREEN ARE X=0 and Y=0.
# THIS BASICALLY MEANS THAT WE WORK IN SCREEN COORDINATES.

# Center the region of interest on the center of the screen.
center_x = screen_L_x/2.0
center_y = screen_L_y/2.0

# Use the whole screen as a region of interest for current applications.
L_x = screen_L_x
L_y = screen_L_y

x_start = max(center_x - L_x/2.0, 0.0)
x_end = min(x_start + L_x/2.0, screen_L_x)
y_start = max(center_y - L_y/2.0, 0.0)
y_end = min(y_start + L_y/2.0, screen_L_y)


# Introduce LGN cell types and numbers of cells.
LGN_cells = {}

# Add transient ON cell type.
for gid in xrange(len(LGN_cells), len(LGN_cells) + 3000): 
  LGN_cells.update( {gid : {'cell_type_tag': 'transient_ON'}} )
# Add transient OFF cell type.
for gid in xrange(len(LGN_cells), len(LGN_cells) + 3000):
  LGN_cells.update( {gid : {'cell_type_tag': 'transient_OFF'}} )
# Add transient ON-OFF cell type.
for gid in xrange(len(LGN_cells), len(LGN_cells) + 3000):
  LGN_cells.update( {gid : {'cell_type_tag': 'transient_ON_OFF'}} )
# Continue adding more cell types if necessary...





# Tile LGN cells in the visual space.
define_cell_parameters(LGN_cells, x_start, y_start, L_x, L_y, f_vis_space_name_prefix)

#movies = [ {'name': 'flash_1', 'path': 'movies_flashes/flash_1.mat'} ]
movies = []
#movies.append( {'name': 'flash_2', 'path': 'movies_flashes/flash_2.mat'} )
'''
for i in range(8, 240, 30): #(range(6, 240, 30) + range(7, 240, 30) + range(8, 240, 30) + range(9, 240, 30) + range(10, 240, 30)):#xrange(1, 271):
  # Ignore gratings with numbers 20-30, 50-60, etc.  These correspond to sf=0.4 and 0.8, which are too fine.  We have generated
  # the sf=0.8 movies with x2 resolution in comparison with others (otherwise the grid failed to represent the fine grating
  # evenly), so it should be OK to use the movies.  However, because of the higher resolution they are very large, and
  # producing firing rates with them takes much longer.  At the same time, mouse cells are mostly unresponsive to such
  # fine gratings.  Thus, it is probably reasonable to skip them for now.
  # For sf=0.4, our preliminary tests show that responses of filters at this sf are already close to
  # baseline, as they are for even coarser sf=0.2.
  check_i = i % 30
  if ((check_i <= 20) and (check_i > 0)):
    movie_name = 'grating_' + str(i)
    movies.append( {'name': movie_name, 'path': 'movies_gratings/res_192/' + movie_name + '.mat'} )
'''
#movies = [ {'name' : 'grating_61', 'path': 'movies_gratings/res_192/grating_61.mat'} ]
'''
movies.append( {'name': 'Wbar_v50pixps_vert', 'path': 'movies_bars/Wbar_v50pixps_vert.mat'} )
movies.append( {'name': 'Wbar_v50pixps_hor', 'path': 'movies_bars/Wbar_v50pixps_hor.mat'} )
movies.append( {'name': 'Bbar_v50pixps_vert', 'path': 'movies_bars/Bbar_v50pixps_vert.mat'} )
movies.append( {'name': 'Bbar_v50pixps_hor', 'path': 'movies_bars/Bbar_v50pixps_hor.mat'} )
'''
'''
for movie in movies:
  mat_movie_fname = movie['path']
  mat_movie_objname = 'mov' #movie_name #'mov_fine'
  f_out_base = outdir + '/' + movie['name'] + '_LGN'
  #f_out_base = outdir + '/' + movie['name'] + '_ctr10' + '_LGN'
  #print mat_movie_fname, f_out_base

  # Obtain firing rates for each LGN cell using filters.
  movie_frame_rate = 1000
  f_rate_dict = mat_movie_to_f_rate(mat_movie_fname, mat_movie_objname, movie_frame_rate, LGN_cells, f_out_base, [screen_L_x, screen_L_y])
'''
'''
# Produce firing rates for presentation of individual images.
image_f_list = []
for image_name in ['img011_BR', 'img019_BT', 'img024_BT', 'img049_BT', 'img057_BR', 'img062_VH', 'img069_VH', 'img071_VH', 'img090_VH', 'img101_VH']: #['8068', '108069', '130034', '163062', 'imk00895', 'imk01220', 'imk01261', 'imk01950', 'imk04208', 'pippin_Mex07_023']:
  # The function for generating firing rates can use multiple images from a list to produce response to a movie consisting of sequential presentations
  # of these images.  At this point we use individual images.
  image_f_list = ['natural_images/' + image_name + '.tiff']
  f_out_base = outdir + '/' + image_name + '_LGN'

  # Obtain firing rates for each LGN cell using filters.
  movie_frame_rate = 1000
  f_rate_dict = image_sequence_to_f_rate(image_f_list, 192, 96, 500, 500, 250, movie_frame_rate, LGN_cells, f_out_base, [screen_L_x, screen_L_y])


# Produce firing rates for presentations of image sequences.
img_list0 = ['img011_BR', 'img019_BT', 'img024_BT', 'img049_BT', 'img057_BR', 'img062_VH', 'img069_VH', 'img071_VH', 'img090_VH', 'img101_VH']
imseq_dict = {}
for imseq_id in xrange(0, 100):
  img_list = img_list0[:] # Copy (instead of using a reference).
  shuffle(img_list) # Shuffle the sequence of images.
  imseq_dict.update( {imseq_id : img_list} )
f_img_list = open(outdir + '/imseq_metadata.pkl', 'w')
pickle.dump(imseq_dict, f_img_list)
f_img_list.close()

for imseq_id in sorted(imseq_dict.keys()):
  image_f_list = []
  for img_name in imseq_dict[imseq_id]:
    image_f_list.append('natural_images/' + img_name + '.tiff')
  f_out_base = '%s/imseq_%d_LGN' % (outdir, imseq_id)
  # Obtain firing rates for each LGN cell using filters.
  movie_frame_rate = 1000
  f_rate_dict = image_sequence_to_f_rate(image_f_list, 192, 96, 250, 500, 0, movie_frame_rate, LGN_cells, f_out_base, [screen_L_x, screen_L_y])
'''

for movie_name in ['TouchOfEvil_frames_3600_to_3750']: #['TouchOfEvil_frames_1530_to_1680', 'TouchOfEvil_frames_3600_to_3750', 'TouchOfEvil_frames_5550_to_5700']: #['TouchOfEvil_frames_3600_to_3690', 'TouchOfEvil_frames_5550_to_5640', 'Protector2_frames_500_to_590', 'Protector2_frames_3050_to_3140', 'Protector2_frames_4700_to_4790']:
  npy_movie_fname = 'natural_movies/' + movie_name + '.npy'
  #f_out_base = outdir + '/' + movie_name + '_LGN'
  #f_out_base = outdir + '/' + movie_name + '_scrbl_t_LGN'
  f_out_base = outdir + '/' + movie_name + '_scrbl_xy_LGN'

  # Obtain firing rates for each LGN cell using filters.
  # Note that in the actual movie the frame rate is 30 frames per second, but we need to make it higher, otherwise numerical artifacts in temporal
  # filters are too severe.
  original_frame_rate = 30 # Hz
  movie_frame_rate = 200 # Hz
  # We would like to add 500 ms of gray screen at the beginning, which translates to movie_frame_rate/2 frames.
  f_rate_dict = npy_movie_to_f_rate(npy_movie_fname, original_frame_rate, 192, 96, movie_frame_rate/2, movie_frame_rate, LGN_cells, f_out_base, [screen_L_x, screen_L_y])
'''
# Generate rates for "spontaneous" activity (corresponding to no stimulus, e.g., gray screen);
# do that by setting for each cell the rate f(t) = r0, where r0 is the baseline firing rate of that cell.
t_tot = 5000.0 # In ms.
dt = 1.0 # Time step for each frame, in ms.
t_list = [x * dt for x in range(0, int(t_tot/dt))]
f_rate_dict = uniform_f_rate(t_list, LGN_cells, outdir + '/' + 'spont_LGN')
#for gid in f_rate_dict['cells'].keys():
#  if (gid % 100 == 0):
#    plt.plot(f_rate_dict['t'], f_rate_dict['cells'][gid])
#plt.show()
'''
