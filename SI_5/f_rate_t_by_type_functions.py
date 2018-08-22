import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

def f_rate_t_by_type(gids_by_type, bin_start, bin_stop, bin_size, f_list, out_f_name):
  bins = np.arange(bin_start, bin_stop+bin_size, bin_size)
  # Originally bins include both leftmost and rightmost bin edges.
  # Here we remove the rightmost edge to make the size consistent with that of f_rate computed below.
  t_f_rate = bins[:-1]


  f_rate_dict = {}
  f_rate_dict['t_f_rate'] = t_f_rate

  for f_name in f_list:
    f_rate_dict[f_name] = {}
    print 'Processing file %s.' % (f_name)
    data = np.genfromtxt(f_name, delimiter=' ')
    if (data.size == 0):
      t = np.array([])
      gids = np.array([])
    elif (data.size == 2):
      t = np.array([data[0]])
      gids = np.array([data[1]])
    else:
      t = data[:, 0]
      gids = data[:, 1]
    for type in gids_by_type:
      ind = np.where( np.in1d(gids, gids_by_type[type]) )[0] # np.in1d(A, B) produces a boolean array of length A.size, with True values where elements of A are in B.
      t1 = t[ind]
      f_rate = np.histogram(t1, bins)[0] # Here, np.histogram returns a tuple, where the first element is the histogram itself and the second is bins.
      f_rate_dict[f_name][type] = 1000.0 * f_rate / ( gids_by_type[type].size * bin_size ) # Convert to rate making sure the units are Hz (time is in ms).

      #plt.plot(t_f_rate, f_rate_dict[f_name][type])
      #plt.title('Type %s' % (type))
      #plt.show()

  # Obtain the average over all files.
  f_rate_dict['mean'] = {}
  for type in gids_by_type:
    f_rate_dict['mean'][type] = np.zeros(t_f_rate.size)
    for f_name in f_list:
      f_rate_dict['mean'][type] += f_rate_dict[f_name][type]
      #plt.plot(t_f_rate, f_rate_dict[f_name][type], c='gray')
    f_rate_dict['mean'][type] = f_rate_dict['mean'][type] / len(f_list)
    #plt.plot(t_f_rate, f_rate_dict['mean'][type])
    #plt.title('Type %s' % (type))
    #plt.show()

  f = open(out_f_name, 'w')
  pickle.dump(f_rate_dict, f)
  f.close()


def construct_gids_by_type_dict(cells_file):
  gids_by_type = {}
  f = open(cells_file, 'r')
  for i, line in enumerate(f):
    if (i > 0):
      tmp_l = line.split()
      gid = int(tmp_l[0])
      type = tmp_l[1]
      if (type not in gids_by_type.keys()):
        gids_by_type[type] = np.array([gid])
      else:
        gids_by_type[type] = np.append(gids_by_type[type], gid)
  f.close()
  return gids_by_type

