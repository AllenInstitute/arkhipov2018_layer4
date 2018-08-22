from math import *
import numpy as np
import matplotlib.pyplot as plt
from random import *
# Use seed(...) to instantiate the random number generator.  Otherwise, current system time is used.

# System 1.
#seed(5)

# System 2.
seed(10)

# System 3.
#seed(15)


def plot_vis_space_mapping_targets(tar_cells):
  simple_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
  N_colors = len(simple_colors)
  cells_by_type = {}
  for gid in tar_cells.keys():
    type = tar_cells[gid]['type']
    if (type in cells_by_type.keys()):
      cells_by_type[type]['vis_x'].append( tar_cells[gid]['vis_x'] )
      cells_by_type[type]['vis_y'].append( tar_cells[gid]['vis_y'] )
    else:
      cells_by_type[type] = { 'vis_x' : [tar_cells[gid]['vis_x']], 'vis_y' : [tar_cells[gid]['vis_y']] } 
    
  for k, type in enumerate(cells_by_type.keys()):
    if (type not in ['PV1', 'PV2', 'LIF_inh']):
      m = '^'
    else:
      m = 'o'
    color_id = k % N_colors
    tmp_color = simple_colors[color_id]
    plt.scatter(cells_by_type[type]['vis_x'], cells_by_type[type]['vis_y'], s=50, marker=m, c=tmp_color, label=type)
  plt.legend()
  plt.axis('equal')
  plt.show()





# Function for selecting cells that should supply inputs to a target cell.
def select_source_cells(tar_cells, tar_type, tar_gid, src_cells):

  # Define the parameters of the ellipses that are used in the visual space to choose source cells supplying their inputs to target cells.
  # We assume below that the ON and OFF subfields for each target cell are described by two partially overlapping ellipses
  # with identical a and b radii (but those radii can vary between the target cells).

  # Define minimal and maximal distance in the visual space between the centers of the two subfields.
  subfields_centers_distance_min = 10.0
  subfields_centers_distance_max = 11.0
  subfields_centers_distance_L = subfields_centers_distance_max - subfields_centers_distance_min

  # Define maximal and minimal subfield width along the ON-OFF axis.
  subfields_ON_OFF_width_min = 6.0
  subfields_ON_OFF_width_max = 8.0
  subfields_ON_OFF_width_L = subfields_ON_OFF_width_max - subfields_ON_OFF_width_min

  # Define the aspect ratio -- the subfield width along the axis orthogonal to the ON-OFF axis divided by the subfield width along the ON-OFF axis.
  subfields_width_aspect_ratio_min = 2.8
  subfields_width_aspect_ratio_max = 3.0
  subfields_width_aspect_ratio_L = subfields_width_aspect_ratio_max - subfields_width_aspect_ratio_min


  # Define parameters of the ellipses.
  ellipse_center_x0 = tar_cells[tar_gid]['vis_x']
  ellipse_center_y0 = tar_cells[tar_gid]['vis_y']

  tuning_angle = tar_cells[tar_gid]['tuning_angle']
  if (tuning_angle == 'None'): #There is no tuning angle information for this target cell.
    # For target cells without tuning angle information we use two wide, non-elnogated, and strongly overlapping subfields.
    # According to Liu et al. (J. Neuroscience 29: 10520-10532, 2009), PV cells in the mouse visual cortex (although, that is in
    # L2/3 and in anesthetized mice) exhibit strongly overlapping ON and OFF subfields, with both subfields being rather wide and non-elongated.
    # Thus, make each subfield a simple circle. Also, scale the size of the receptive field a bit, so that it is significantly larger
    # than what we use for excitatory cells.
    ellipse_b0 = (subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L) / 2.0 # Divide by 2 to convert from width to radius.
    ellipse_b0 = 2.5 * ellipse_b0
    ellipse_a0 = ellipse_b0
    top_N_src_cells_subfield = 15
    ellipses_centers_halfdistance = 0.0

  else: #There is tuning angle information for this target cell.
    tuning_angle_value  = float(tuning_angle)
    ellipses_centers_halfdistance = (subfields_centers_distance_min + random() * subfields_centers_distance_L) / 2.0
    ellipse_b0 = (subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L) / 2.0 # Divide by 2 to convert from width to radius.
    ellipse_a0 = ellipse_b0 * (subfields_width_aspect_ratio_min + random() * subfields_width_aspect_ratio_L)

    # Assign angle of rotation of the ellipse based on the assumed preferred orientation of the target cell.  Because in the visual space the x direction is
    # from left to right, whereas y is from top to bottom (instead of bottom to top), we need to add 180 degrees to the actual preferred angle to map it
    # to the appropriate space.  We also need to add 90 degrees since the ellipse_phi angle describes the direction orthogonal to the ON-OFF direction.
    ellipse_phi = tuning_angle_value + 180.0 + 90.0 #Angle, in degrees, describing the rotation of the canonical ellipse away from the x-axis.
    ellipse_cos_mphi = cos(-radians(ellipse_phi))
    ellipse_sin_mphi = sin(-radians(ellipse_phi))
    # According to our in-house experimental data, and as reflected in the model of the LGN, the numbers of transient-ON and transient-OFF
    # LGN cells differ by a significant factor.  Because of that, we try to limit the number of LGN cells in each subfield to such a value
    # that all the extra cells in one of the subfields (above those in the other) are removed.
    top_N_src_cells_subfield = 8

  # Assume that each target cell recieves inputs from all types of source cells.
  src_cells_selected = {}
  for src_type in src_cells.keys():
    src_cells_selected[src_type] = []
    # Define the centers of all subfields.
    # For cells that do not possess the tuning angle property, all subfields are centered at the same point.
    # Also, the ellipse parameters a and b could differ between the subfields, so we need to define them here as well.
    if (tuning_angle == 'None'):
      ellipse_center_x = ellipse_center_x0
      ellipse_center_y = ellipse_center_y0
      ellipse_a = ellipse_a0
      ellipse_b = ellipse_b0
    else: 
      if (src_type == 'transient_ON'):
        ellipse_center_x = ellipse_center_x0 + ellipses_centers_halfdistance * ellipse_sin_mphi
        ellipse_center_y = ellipse_center_y0 + ellipses_centers_halfdistance * ellipse_cos_mphi
        ellipse_a = ellipse_a0
        ellipse_b = ellipse_b0
      elif (src_type == 'transient_OFF'):
        ellipse_center_x = ellipse_center_x0 - ellipses_centers_halfdistance * ellipse_sin_mphi
        ellipse_center_y = ellipse_center_y0 - ellipses_centers_halfdistance * ellipse_cos_mphi
        ellipse_a = ellipse_a0
        ellipse_b = ellipse_b0
      else:
        # Make this a simple circle.
        ellipse_center_x = ellipse_center_x0
        ellipse_center_y = ellipse_center_y0
        # Make the region from which source cells are selected a bit smaller for the transient_ON_OFF cells, since each
        # source cell in this case produces both ON and OFF responses.
        ellipse_b = ellipses_centers_halfdistance/2.0
        ellipse_a = ellipse_b0

    # Find those source cells of the appropriate type that have their visual space coordinates within the ellipse.
    for src_gid in src_cells[src_type]:
      x = src_cells[src_type][src_gid]['x']
      y = src_cells[src_type][src_gid]['y']

      # Shift the origin to the center of the ellipse.
      x = x - ellipse_center_x
      y = y - ellipse_center_y

      # Rotate the coordinate frame so that the ellipse is in a canonical orientation.
      # For that, we need to rotate the (x, y) by the -phi angle.
      if (tuning_angle != 'None'):
        x_new = x * ellipse_cos_mphi - y * ellipse_sin_mphi
        y_new = x * ellipse_sin_mphi + y * ellipse_cos_mphi
      else:
        x_new = x
        y_new = y

      if (((x_new / ellipse_a)**2 + (y_new / ellipse_b)**2) <= 1.0):
        if ((tuning_angle != 'None') and (src_type == 'transient_ON_OFF')):
          src_tuning_angle = float(src_cells[src_type][src_gid]['tuning_angle'])
          # Compute difference in the tuning angle between target and source.  Keep track of orientation (not direction) and only of
          # the absolute value of the difference.
          # Since the difference should only matter within [0, 90] degrees, convert the result to that scale.
          delta_tuning = abs(abs(abs(180.0 - abs(tuning_angle_value - src_tuning_angle) % 360.0) - 90.0) - 90.0)
          if (delta_tuning < 15.0):
            src_cells_selected[src_type].append(src_gid)
        else:
          src_cells_selected[src_type].append(src_gid)


    # Remove extra source cells from the subfield, if necesssary.  With the current choice of parameters for subfields,
    # this does not seem to be an issue for cells that have a tuning angle properties, but it may be useful to check
    # for this for all other cells.
    while len(src_cells_selected[src_type]) > top_N_src_cells_subfield:
      src_cells_selected[src_type].remove(choice(src_cells_selected[src_type]))

  ellipses_centers_distance = 2.0 * ellipses_centers_halfdistance
  return src_cells_selected, ellipses_centers_distance


