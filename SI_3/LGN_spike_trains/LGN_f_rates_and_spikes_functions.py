from math import *
import numpy as np

from random import *

from filter_population import *
import scipy.io

from f_rate_to_spike_train import *

import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# NOTE THAT HERE WE USE "lindegs" THAT DESCRIBE ANGLES IN LINEAR APPROXIMATION.
# Essentially, it's the description of x,y positions on the screen divided
# by the distance from the observer to the screen.
# This measure is similar to the actual degrees for small angles, so it is
# a reasonable approximation near the center of the screen; further away it still can be
# used, but really have a different meaning from actual angles.





def define_cell_parameters(cells, x_start, y_start, L_x, L_y, f_out_base):
  # Assume that cells of all types are tiled equally in the visual space.

  f_out = open(f_out_base +  '_visual_space_positions_and_cell_types.dat', 'w')

  for gid in cells:
    cell_type_tag = cells[gid]['cell_type_tag']

    # Assume isotropic tiling in x and y dimensions in the visual space. Distribute cells randomly within the tiling region.  It seems like random distribution
    # makes sense as long as the centers of the cells' receptive fields are close enough to each other in the visual space; for example, if they are a few
    # degrees away, and the receptive fields are 10-20 degrees wide, the visual space should be completely tiled by highly overlapping receptive fields.
    x = x_start + random() * L_x
    y = y_start + random() * L_y
    x_y_offset = (0.0, 0.0)
    if (cell_type_tag in ['transient_ON_OFF']):
      phi = random() * 2.0 * np.pi
      offset_dist = 3.5 + 1.5 * random() # In lindegs.
      x_y_offset = (offset_dist * np.cos(phi), offset_dist * np.sin(phi))

    # Define the width of the spatial filter (use the same width for both ON and OFF subfields of ON/OFF cells).
    sigma_c_lindegs = 2.0 + random() * 1.0
    sigma_s_lindegs = 2.0 * sigma_c_lindegs

    # Define the magnitude of spontaneous firing rate.
    r0 = 4.5 * (1.0 + 0.1 * (2.0 * random() - 1.0)) # Currently the same for all cell types.

    # Define the scaling factor to get the evoked component of the firing rate.
    random_factor = 1.0 + 0.1 * (2.0 * random() - 1.0)
    if (cell_type_tag == 'transient_ON'):
      scaling_factor = 0.0051 * random_factor
    elif (cell_type_tag == 'transient_OFF'):
      scaling_factor = 0.0051 * random_factor
    elif (cell_type_tag == 'transient_ON_OFF'):
      scaling_factor = 0.003 * random_factor

    # Define the k_alpha parameter for the temporal filter.
    k_alpha_random_factor = 1.0 + 0.01 * (2.0 * random() - 1.0)
    if (cell_type_tag == 'transient_ON'):
      k_alpha = 0.038 * k_alpha_random_factor
    elif (cell_type_tag == 'transient_OFF'):
      k_alpha = 0.038 * k_alpha_random_factor
    elif (cell_type_tag == 'transient_ON_OFF'):
      k_alpha = 0.038 * k_alpha_random_factor

    cells[gid].update( {'x': x, 'y': y, 'x_y_offset': x_y_offset, 'sigma_c_lindegs': sigma_c_lindegs, 'sigma_s_lindegs': sigma_s_lindegs, 'r0': r0, 'scaling_factor': scaling_factor, 'k_alpha': k_alpha} )
    f_out.write(cells[gid]['cell_type_tag'] + ' ' + str(x) + ' ' + str(y) + ' ' + str(x_y_offset[0]) + ' ' + str(x_y_offset[1]) + ' ' + str(sigma_c_lindegs) + ' ' + str(sigma_s_lindegs) + ' ' + str(r0) + ' ' + str(scaling_factor) + ' ' + str(k_alpha) + '\n')

  f_out.close()





# Create an f(t) = const. time series for the firing rate,
# separately for each cell.
# Arguments are as follows:
# image_f_list is the list containing image file names;
# t_list is the list containing time stamps for the time series we need to construct;
# cells is the list containing dictionaries with information about each cell;
# f_out_base is the common portion of the file name for output.
def uniform_f_rate(t_list, cells, f_out_base):

  # Obtain firing rates for all cells.
  f_rate_dict = {}
  f_rate_dict['t'] = t_list

  f_rate_dict['cells'] = {}
  for gid in cells:
    if (gid % 100 == 0):
      print '%s: computing firing rate for cell %d out of %d' % (f_out_base, gid, len(cells))
    f_rate_dict['cells'][gid] = [cells[gid]['r0']] * len(t_list)

  f_f_rate = open(f_out_base + '_f_rate.pkl', 'w')
  pickle.dump(f_rate_dict, f_f_rate)
  f_f_rate.close()

  return f_rate_dict





def mat_movie_to_f_rate(mat_movie_fname, mat_movie_objname, movie_frame_rate, cells, f_out_base, screen_size_lindeg):
  # Read in the movie.
  print 'Reading and converting the movie...'
  mat_movie = scipy.io.loadmat(mat_movie_fname)

  # Movie with dimensions [Nx*Ny, Nt]
  m1 = mat_movie[mat_movie_objname] 

  # Change dimensions to [Nt, Nx*Ny].
  m1 = np.transpose(m1)

  # Reshape to [t,y,x] -- NOTE, it has to be y,x, and not x,y, because that's how the image is presented in the original file format.
  # Here we ASSUME that Nx = 2 Ny!
  Ny = int((np.shape(m1)[1] / 2)** (0.5))
  Nx  = 2 * Ny
  m1 = np.reshape(m1, (3000, Ny, Nx))

  # However, in all functions we use it is assumed that elements are listed as x,y; thus, we have to transpose the x-y dimensions of
  # the array, without affecting the time dimension.  This is achieved by the transpose command with the second argument (which
  # determines the new order of the axes -- here we are switching from (0, 1, 2) to (0, 2, 1)).
  m1 = np.transpose(m1, (0, 2, 1))

  # Now, convert the movie from the contrast scale (-1, 1) to the grayscale (0, 255).
  # Adjust contrast; in the original movie, the contrast is 0.8.
  m1 = 127.5 * (0.1 / 0.8) * m1 + 128
  m1 = m1.astype(int)
  print 'done.'


  print 'Loading the movie into filter population object...'
  # Create a filter object that can be reused for all (x,y) locations.
  # We need to supply pixel_per_lindeg here, because filters are implemented in pixel space; ASSUME that the ratio of pixels to lindegs is the same in x and y.
  movie_filter = FilterPopulation(movie_frames=m1, movie_frame_rate = movie_frame_rate, pixel_per_lindeg = 1.0*Nx/screen_size_lindeg[0])
  print 'done.'

  # Obtain firing rates for all cells.
  f_rate_dict = {}

  tmp_list = []
  for i in xrange(len(m1)):
    tmp_list.append(i * movie_filter.dt)
  f_rate_dict.update({'t': tmp_list})

  f_rate_cells = {}
  for gid in cells:
    tmp_dict = cells[gid]

    if (gid % 10 == 0):
      print mat_movie_fname + ': computing firing rate for cell ', gid, ' out of ', len(cells)

    # Add a small random number to the scaling factor here, so that it varies between each stimulus.  This helps to avoid systematic
    # differences between, e.g., gratings oriented at odd multiples of 45 degress vs. even multiples of 45 degrees (which is due to the
    # square grid and relatively large-size pixels used for the screen).  However, for some cases, where the time-averaged firing rate
    # is close to backround firing rate, this addition has to be much bigger to cancel such a systematic trend.  Those cases are perhaps
    # least important, and therefore we do not deal with them here.
    scaling_factor = tmp_dict['scaling_factor'] * ( 1.0 + 0.1 * (2.0 * random() - 1.0) )
    f_rate = movie_filter.calculate_f_rate(tmp_dict['x'], tmp_dict['y'], tmp_dict['x_y_offset'], tmp_dict['cell_type_tag'], tmp_dict['sigma_c_lindegs'], tmp_dict['sigma_s_lindegs'], tmp_dict['r0'], scaling_factor, tmp_dict['k_alpha'])

    f_rate_cells.update({gid : f_rate})

  f_rate_dict.update({'cells': f_rate_cells})

  f_f_rate = open(f_out_base + '_f_rate.pkl', 'w')
  pickle.dump(f_rate_dict, f_f_rate)
  f_f_rate.close()

  return f_rate_dict





# Read in a sequence of images and combine them into a movie showing static images separated by presentations of gray screen;
# use that movie to generate firing rates for cells. In the process, the images are reduced (downsampled) to a desired x-y size in pixels.
# Arguments are as follows:
# image_f_list is the list containing image file names;
# Nx is the desired x size of the screen (in pixels);
# Ny is the desired y size of the screen (in pixels);
# Nt is the number of frames for each image presentation;
# Nt1 is the number of frames for the first gray screen interval;
# Nt_p is the number of frames for the subsequent gray screen intervals between image presentations;
# movie_frame_rate is the desired frame rate;
# cells is the list containing dictionaries with information about each cell;
# f_out_base is the common portion of the file name for output;
# screen_size_lindeg is the size (in x and y) of the screen in lindegs.
def image_sequence_to_f_rate(image_f_list, Nx, Ny, Nt, Nt1, Nt_p, movie_frame_rate, cells, f_out_base, screen_size_lindeg):
  print 'Reading and combining the images...' 

  # Create a gray screen image.  NOTE that at this point axis 0 correponds to y and axis 1 to x.
  im_gray = np.full((Ny, Nx), [127.5])

  # Create arrays from this static image; in each case the first dimension is time, the second y, and the third x.
  # Create one array for the beginning of the sequence and the other for intervals between natural images.
  m1 = np.full( (Nt1, Ny, Nx), im_gray )
  m_gray_p = np.full( (Nt_p, Ny, Nx), im_gray )

  for image_f_name in image_f_list:
    # Read the image.
    im = plt.imread(image_f_name)

    # Downsample the image.  NOTE that at this point axis 0 correponds to y and axis 1 to x.
    Nlist = im.shape
    Nx_big = Nlist[1]
    Ny_big = Nlist[0]

    # Add gray pixels to make sure that the dimensions are multiples of Nx and Ny.
    if (Ny_big % Ny != 0):
      Ny_add = Ny - Ny_big % Ny
      tmp_gray = np.full((Ny_add, Nx_big), 127.5)
      im = np.concatenate( (im, tmp_gray), axis = 0)
    else:
      Ny_add = 0
    if (Nx_big % Nx != 0):
      Nx_add = Nx - Nx_big % Nx
      tmp_gray = np.full((Ny_big + Ny_add, Nx_add), 127.5)
      im = np.concatenate( (im, tmp_gray), axis = 1)
    else:
      Nx_add = 0

    # Measure the new dimensions.
    Nlist = im.shape
    Nx_big = Nlist[1]
    Ny_big = Nlist[0]
    # Get elements to be averaged "out" into the 1st and 3rd dimensions, and then take the mean over them.
    im_small = im.reshape([Ny, int(floor(1.0* Ny_big/Ny)), Nx, Nx_big/Nx]).mean(3).mean(1)

    # Create a long array from this static image, with the first dimension being time, the second y, and the third x.
    m_n_img = np.full( (Nt, Ny, Nx), im_small )

    m1 = np.concatenate((m1, m_n_img, m_gray_p))


  # In all functions we use it is assumed that elements are listed as x,y; thus, we have to transpose the x-y dimensions of
  # the array, without affecting the time dimension.  This is achieved by the transpose command with the second argument (which
  # determines the new order of the axes -- here we are switching from (0, 1, 2) to (0, 2, 1)).
  m1 = np.transpose(m1, (0, 2, 1))

  m1 = m1.astype(int)
  print 'done.'


  print 'Loading the movie into filter population object...'
  # Create a filter object that can be reused for all (x,y) locations.
  # We need to supply pixel_per_lindeg here, because filters are implemented in pixel space; ASSUME that the ratio of pixels to lindegs is the same in x and y.
  movie_filter = FilterPopulation(movie_frames=m1, movie_frame_rate = movie_frame_rate, pixel_per_lindeg = 1.0*Nx/screen_size_lindeg[0])
  print 'done.'

  # Obtain firing rates for all cells.
  f_rate_dict = {}

  tmp_list = []
  for i in xrange(len(m1)):
    tmp_list.append(i * movie_filter.dt)
  f_rate_dict.update({'t': tmp_list})

  f_rate_cells = {}
  for gid in cells:
    tmp_dict = cells[gid]

    if (gid % 10 == 0):
      print f_out_base + ': computing firing rate for cell ', gid, ' out of ', len(cells)

    # Add a small random number to the scaling factor here, so that it varies between each stimulus.  This helps to avoid systematic
    # differences between, e.g., gratings oriented at odd multiples of 45 degress vs. even multiples of 45 degrees (which is due to the
    # square grid and realtively large-size pixels used for the screen).  However, for some cases, where the time-averaged firing rate
    # is close to backround firing rate, this addition has to be much bigger to cancel such a systematic trend.  Those cases are perhaps
    # least important, and therefore we do not deal with them here.
    scaling_factor = tmp_dict['scaling_factor'] * ( 1.0 + 0.1 * (2.0 * random() - 1.0) )
    f_rate = movie_filter.calculate_f_rate(tmp_dict['x'], tmp_dict['y'], tmp_dict['x_y_offset'], tmp_dict['cell_type_tag'], tmp_dict['sigma_c_lindegs'], tmp_dict['sigma_s_lindegs'], tmp_dict['r0'], scaling_factor, tmp_dict['k_alpha'])

    f_rate_cells.update({gid : f_rate})

  f_rate_dict.update({'cells': f_rate_cells})

  f_f_rate = open(f_out_base + '_f_rate.pkl', 'w')
  pickle.dump(f_rate_dict, f_f_rate)
  f_f_rate.close()

  return f_rate_dict





# Read in a *.npy movie and use that movie to generate firing rates for cells. In the process,
# the images are reduced (downsampled) to a desired x-y size in pixels.
# Arguments are as follows:
# npy_movie_fname is the name of the file containing the movie;
# original_frame_rate is the frame rate in the movie;
# Nx is the desired x size of the screen (in pixels);
# Ny is the desired y size of the screen (in pixels);
# Nt1 is the number of frames for the first gray screen interval;
# movie_frame_rate is the desired frame rate;
# cells is the list containing dictionaries with information about each cell;
# f_out_base is the common portion of the file name for output;
# screen_size_lindeg is the size (in x and y) of the screen in lindegs.
def npy_movie_to_f_rate(npy_movie_fname, original_frame_rate, Nx, Ny, Nt1, movie_frame_rate, cells, f_out_base, screen_size_lindeg):
  print 'Reading and processing the movie...'
  m_tmp = np.load(npy_movie_fname)
  #NOTE that at this point axis 0 correponds to t, axis 1 to y, and axis 2 to x.
  Nlist = m_tmp.shape
  Nm_t = Nlist[0]
  Nm_y = Nlist[1]
  Nm_x = Nlist[2]

  # We need to increase the number of frames in the movie.  This is done by producing a new array with more frames and
  # populating each frame by an interpolation from the original movie frames.
  frame_rate_ratio = 1.0 * movie_frame_rate / original_frame_rate
  new_Nm_t = int( floor( (Nm_t - 1) * frame_rate_ratio) ) # Number of frames in the new movie.
  m_new = np.zeros( (new_Nm_t, Nm_y, Nm_x) )

  for i_new in xrange(new_Nm_t):
    tmp_k = i_new / frame_rate_ratio # This is a float, since frame_rate_ration is a float.
    k0 = int( floor(tmp_k) ) # Frame of the original movie that is right before the current frame of the new movie.
    k1 = k0 + 1 # Next frame of the original movie.
    dk = k1 - tmp_k
    # Interpolation.
    if (k1 < Nm_t):
      m_new[i_new] = m_tmp[k0, :, :] * dk + m_tmp[k1, :, :] * (1 - dk)
    else:
      m_new[i_new] = m_tmp[k0, :, :]

  # Now, create an array that will hold the new movie (which we need to downsample in x and y) and the initial gray screen presentation.
  m1 = np.zeros( (Nt1 + new_Nm_t, Ny, Nx) )
  # Create a gray screen image.  NOTE that at this point axis 0 correponds to y and axis 1 to x.
  im_gray = np.full((Ny, Nx), [127.5])
  # Create an array from this static image for the initial gray screen presentation; the first dimension is time, the second y, and the third x.
  for i in xrange(Nt1):
    m1[i] = im_gray

  # Downsample each frame of the movie to the size Nx x Ny (NOTE that at this point axis 0 of each frame correponds to y and axis 1 to x).
  Nx_c = Nm_x - Nm_x % Nx
  Nx_start = (Nm_x - Nx_c) / 2
  Nx_end = Nx_start + Nx_c
  Ny_c = Nm_y - Nm_y % Ny
  Ny_start = (Nm_y - Ny_c) / 2
  Ny_end = Ny_start + Ny_c
  dsmpl_x = Nx_c / Nx
  dsmpl_y = Ny_c / Ny
  for i in xrange(new_Nm_t):
    k_tot = i + Nt1
    # First, "cut out" the central portion of each frame with dimensions that are multiples of Nx and Ny.
    im = m_new[i, Ny_start:Ny_end, Nx_start:Nx_end]
    # Then, get elements to be averaged "out" into the 1st and 3rd dimensions, and then take the mean over them.
    m1[k_tot] = im.reshape([Ny, dsmpl_y, Nx, dsmpl_x]).mean(3).mean(1)


  # In all functions we use it is assumed that elements are listed as x,y; thus, we have to transpose the x-y dimensions of
  # the array, without affecting the time dimension.  This is achieved by the transpose command with the second argument (which
  # determines the new order of the axes -- here we are switching from (0, 1, 2) to (0, 2, 1)).
  m1 = np.transpose(m1, (0, 2, 1))

  m1 = m1.astype(int)
  print 'done.'

  # Shuffle the frames.
  #m2 = m1[:100] # Gray screen.
  #m3 = m1[100:] # Actual movie.
  #np.random.shuffle(m3)
  #m1 = np.concatenate((m2, m3), axis=0)

  # Shuffle pixels in each frame.
  #for i in xrange(0, m1.shape[0]):
  #    m1_f = m1[i].flatten()
  #    np.random.shuffle(m1_f)
  #    m1[i] = m1_f.reshape([Nx, Ny])


  print 'Loading the movie into filter population object...'
  # Create a filter object that can be reused for all (x,y) locations.
  # We need to supply pixel_per_lindeg here, because filters are implemented in pixel space; ASSUME that the ratio of pixels to lindegs is the same in x and y.
  movie_filter = FilterPopulation(movie_frames=m1, movie_frame_rate = movie_frame_rate, pixel_per_lindeg = 1.0*Nx/screen_size_lindeg[0])
  print 'done.'

  # Obtain firing rates for all cells.
  f_rate_dict = {}

  tmp_list = []
  for i in xrange(len(m1)):
    tmp_list.append(i * movie_filter.dt)
  f_rate_dict.update({'t': tmp_list})

  f_rate_cells = {}
  for gid in cells:
    tmp_dict = cells[gid]

    if (gid % 10 == 0):
      print f_out_base + ': computing firing rate for cell ', gid, ' out of ', len(cells)

    # Add a small random number to the scaling factor here, so that it varies between each stimulus.  This helps to avoid systematic
    # differences between, e.g., gratings oriented at odd multiples of 45 degress vs. even multiples of 45 degrees (which is due to the
    # square grid and realtively large-size pixels used for the screen).  However, for some cases, where the time-averaged firing rate
    # is close to backround firing rate, this addition has to be much bigger to cancel such a systematic trend.  Those cases are perhaps
    # least important, and therefore we do not deal with them here.
    scaling_factor = tmp_dict['scaling_factor'] * ( 1.0 + 0.1 * (2.0 * random() - 1.0) )
    f_rate = movie_filter.calculate_f_rate(tmp_dict['x'], tmp_dict['y'], tmp_dict['x_y_offset'], tmp_dict['cell_type_tag'], tmp_dict['sigma_c_lindegs'], tmp_dict['sigma_s_lindegs'], tmp_dict['r0'], scaling_factor, tmp_dict['k_alpha'])

    f_rate_cells.update({gid : f_rate})

  f_rate_dict.update({'cells': f_rate_cells})

  f_f_rate = open(f_out_base + '_f_rate.pkl', 'w')
  pickle.dump(f_rate_dict, f_f_rate)
  f_f_rate.close()

  return f_rate_dict





def run_f_rate_to_spike_train(f_rate_dict, spike_trains_t_window_start, spike_trains_t_window_end, spike_trains_p_spike_max, N_trains, f_out_base):

  f_spk = open(f_out_base + '_spk.dat', 'w')

  t = np.array(f_rate_dict['t'])
  ind_before_window = np.where(t < spike_trains_t_window_start)[0] # For these times, f_rate is set to a uniform value.
  ind_include = np.where(t <= spike_trains_t_window_end)[0] # All elements beyond this time are disregarded.

  t_include = t[ind_include]

  for cell_id in f_rate_dict['cells']:
    if (cell_id % 10 == 0):
      print f_out_base + ': generating spike trains for cell ', cell_id, ' out of ', len(f_rate_dict['cells'])

    f = np.array(f_rate_dict['cells'][cell_id])

    # Set the values of f to the baseline for all t <= spike_trains_t_window_start.
    # Consider f[0] in the original f array as the baseline.
    f[ind_before_window] = f[0]

    # Shut off the spikes after certain time.
    #f[ind_stop_spk] = 0.0

    # Keep only the portion of the f array that corresponds to t <= spike_trains_t_window_end.
    f = f[ind_include]

    # Generate a number (N_trains) of spike trains.
    for i_train in xrange(N_trains):
      random_seed = randrange(1e6)
      spike_train = f_rate_to_spike_train(t_include, f, random_seed, spike_trains_p_spike_max)

      for j in xrange(len(spike_train)):
        f_spk.write(' %.3f' % spike_train[j])
      f_spk.write('\n')

  f_spk.close()


