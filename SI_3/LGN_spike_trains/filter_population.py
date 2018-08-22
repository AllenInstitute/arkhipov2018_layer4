import numpy
from math import *
import pylab as pl
from scipy.signal.signaltools import convolve2d
import scipy.io
from copy import copy

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from filters import *


# NOTE THAT HERE WE USE "lindegs" THAT DESCRIBE ANGLES IN LINEAR APPROXIMATION.
# Essentially, it's the description of x,y positions on the screen divided
# by the distance from the observer to the screen.
# This measure is similar to the actual degrees for small angles, so it is
# a reasonable approximation near the center of the screen; further away it still can be
# used, but really have a different meaning from actual angles.


class FilterPopulation():
    default_movie = None
    
    def __init__(self, **kwargs):
        self.movie_frames = kwargs.get('movie_frames', FilterPopulation.default_movie)
        self.movie_frame_rate = kwargs.get('movie_frame_rate', 24)
        self.pixel_per_lindeg = kwargs.get('pixel_per_lindeg', 2.0)

        self.dt = 1000.0 / self.movie_frame_rate # Time step in ms.

                    
    def calculate_f_rate(self, x, y, x_y_offset, cell_type_tag, sigma_c_lindegs, sigma_s_lindegs, r0, scaling_factor, k_alpha):
        t_max = 500.0 # In ms.

        if (cell_type_tag in ['transient_ON_OFF']):
          # Note that the spatial filter widths are the same here for the ON and OFF filters of the transient_ON_OFF cells.

          x_displ = x_y_offset[0] / 2.0
          y_displ = x_y_offset[1] / 2.0

          l_xy_array = self.compute_spatially_filtered_images(x - x_displ, y - y_displ, sigma_c_lindegs, sigma_s_lindegs, light_sensitive=True)
          f_t_vec = alpha_temp_fil(self.dt, t_max, k_alpha, beta=1.0, n_filter=1)[0] #8
          l_t_vec1 = scaling_factor * self.dt * numpy.convolve(f_t_vec, l_xy_array)
          r0_1 = r0

          l_xy_array = self.compute_spatially_filtered_images(x + x_displ, y + y_displ, sigma_c_lindegs, sigma_s_lindegs, light_sensitive=False)
          f_t_vec = alpha_temp_fil(self.dt, t_max, k_alpha, beta=1.0, n_filter=1)[0] #8 # In principle, this could use parameters that would differ from the one above.
          l_t_vec2 = scaling_factor * self.dt * numpy.convolve(f_t_vec, l_xy_array)
          r0_2 = r0 # In principle, this could be different from r0_1.

          target_r_vec = numpy.maximum(numpy.maximum(r0_1 + l_t_vec1, 0.0) + numpy.maximum(r0_2 + l_t_vec2, 0.0) - (r0_1 + r0_2)/2.0, 0.0)

        else:
          if (cell_type_tag in ['transient_ON']):
            light_sensitive = True
          elif (cell_type_tag in ['transient_OFF']):
            light_sensitive = False
          l_xy_array = self.compute_spatially_filtered_images(x, y, sigma_c_lindegs, sigma_s_lindegs, light_sensitive)
          f_t_vec = alpha_temp_fil(self.dt, t_max, k_alpha, beta=1.0, n_filter=1)[0] #8
          l_t_vec = scaling_factor * self.dt * numpy.convolve(f_t_vec, l_xy_array)

          target_r_vec = numpy.maximum(r0 + l_t_vec, 0.0)

        return target_r_vec[0:len(self.movie_frames)] #Make sure the length of the returned vector is equal to the number of frames.


    def compute_spatially_filtered_images(self, x, y, sigma_c_lindegs, sigma_s_lindegs, light_sensitive):

        # The arguments to this function contain x and y in lindegs.
        # Convert them to pixels here, and make sure they are converted to integer numbers
        # (that makes it more convenient to center the filter at x and y).
        # AN ASSUMPTION HERE IS THAT MINIMAL X AND Y POSSIBLE ON THE SCREEN ARE X=0 and Y=0.
        # THIS BASICALLY MEANS THAT WE WORK IN SCREEN COORDINATES.
        x_pix = int(round(x * self.pixel_per_lindeg))
        y_pix = int(round(y * self.pixel_per_lindeg))

        # The current choice of parameters is such that the integral of this filter is >0.  The condition for that is sigma_s^2 < sigma_c^2 * A_c / A_s.
        # If the integral of the filter is negative, it essentially inverts the ON/OFF properties of the filter.
        filter_center = 0
        sigma_c = sigma_c_lindegs * self.pixel_per_lindeg
        sigma_s = sigma_s_lindegs * self.pixel_per_lindeg
        A_c = 1.2
        A_s = 0.2
        tol = 1e-8

        # Define the radius of the part of the visual field that's used for integration with the filter.
        # The filter array NEEDS to have the same dimensions as the subimage array (i.e., NxN, where N=2*subimage_radius+1).
        # If we are using a simple filter, like a single-Gaussian one, we can tie the subimage_radius to the width of the filter.
        subimage_radius = int(ceil(3.5 * sigma_s))
        filter_width = 2 * subimage_radius + 1

        gaussian_filter_tmp = gaussian2d_spatial_filter_center_surround(filter_width, filter_width, filter_center, A_c, sigma_c, A_s, sigma_s, tol)

        # Adjust the dimensions of the filter to make sure that they will coincide with the dimensions of the subimage
        # (the subimage dimensions may be truncated if the center is too close to the screen border).
        x_start = x_pix - subimage_radius
        x_end = x_pix + subimage_radius
        y_start = y_pix - subimage_radius
        y_end = y_pix + subimage_radius

        frames_shape = numpy.shape(self.movie_frames)
        max_x = frames_shape[1] - 1
        max_y = frames_shape[2] - 1

        filter_start_x = max(0, -x_start)
        filter_end_x = filter_width - max(0, x_end - max_x)
        filter_start_y = max(0, -y_start)
        filter_end_y = filter_width - max(0, y_end - max_y)

        gaussian_filter = gaussian_filter_tmp[filter_start_x:filter_end_x, filter_start_y:filter_end_y]
        gaussian_filter_sum = numpy.sum(gaussian_filter)

        # Prepare limiting start and end positions in x and y for the subimage.  Do this once here, rather than repeating for each time frame.
        # Note that the elements at end points in x and y are included in the subimage.
        x_start_subimage = max(x_start, 0)
        x_end_subimage = min(x_end, max_x)
        y_start_subimage = max(y_start, 0)
        y_end_subimage = min(y_end, max_y)
        
        l_xy_array = []

        for frame_index in range(0, len(self.movie_frames)):

            receptive_field_subimage = self.subimage(x_start_subimage, x_end_subimage, y_start_subimage, y_end_subimage, frame_index)

            if (light_sensitive == False):
              receptive_field_subimage = 255 - receptive_field_subimage

            image_filter_product = numpy.multiply(receptive_field_subimage, gaussian_filter)

            # Integrate the product of image and the filter.
            # Normally one would multiply this by dx*dy, but for that we need to define what dx and dy are, and also
            # a multiplicative factor is used later on anyway, so we can as well ignore the dx*dy factor here.
            l_xy = numpy.sum(image_filter_product) / (gaussian_filter_sum * self.pixel_per_lindeg ** 2) # Normalize to a common spatial scale.
            l_xy_array.append(l_xy)


        return l_xy_array       
         
         
    def time_to_frame_index(self, t):

        if (t <= 0):
          frame_index = 0
        else:
          frame_index = int(t / 1000.0 * self.movie_frame_rate) #Assume time is in ms.
        
        if frame_index >= len(self.movie_frames):
            frame_index = (self.movie_frames) - 1 

        return frame_index
    

    # Note, the subimage includes the elements at x_end and y_end!!!
    def subimage(self, x_start, x_end, y_start, y_end, frame_index):
        return self.movie_frames[frame_index][x_start:(x_end + 1), y_start:(y_end + 1)]


