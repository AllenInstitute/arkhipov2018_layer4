import os
import h5py

from common import *
from cell_types import *

from allensdk.model.biophys_sim.config import Config
from utils import Utils

from shutil import copyfile

import pandas as pd

from mkcells import *
from connectcells import *
from external_inputs import *
from mkstim import *

import f_rate


def run_simulation(config_f_name):
  h('starttime = startsw()')
  config = Config().load(config_f_name)   # read configuration for the model
  utils_obj = Utils(config)   # Instantiate an object of a class Utils which configures NEURON and provides the interface to the necessary functions to set up the simulation.


  cells_db = utils_obj.load_cell_db() # Load the information about individual cells.

  utils_obj.set_run_params()  # set h.dt and h.tsop


  workdir_n = str(utils_obj.description.data['biophys'][0]['output_dir'])
  if (int(pc.id()) == 0):
    if not os.path.exists(workdir_n):
      os.mkdir(workdir_n)
    print 'Workdir: %s.' % workdir_n
    print ''

  pc.barrier() # Wait for all hosts to get to this point

  instantiate_cells_and_cell_types(cells_db)

  mkcells(cells_db, utils_obj)
  pc.barrier()

  utils_obj.load_syn_data() # Load synaptic parameters (for types of sources and targets).
  pc.barrier()

  if ('connections' in utils_obj.description.data.keys()):
    con_path = utils_obj.description.data['connections']
    connectcells(con_path, cells_db, utils_obj)
  pc.barrier()

  # Go over all source of external inputs and instantiate them for the appropriate cells.
  for ext_inp_path in reversed(utils_obj.description.data['ext_inputs'].keys()):
    if not os.path.exists(ext_inp_path):
      print "Error: the external inputs file %s does not exist; exiting." % (ext_inp_path)
      h.finish()  

    ext_inp_map = pd.read_csv(utils_obj.description.data['ext_inputs'][ext_inp_path]['map'], sep=' ')
    for tar_gid in cells:
      if pc.gid_exists(tar_gid):
        tar_ext_inp_map = ext_inp_map[ext_inp_map['index'] == tar_gid]
        external_inputs(tar_gid, ext_inp_path, tar_ext_inp_map, utils_obj.description.data['ext_inputs'][ext_inp_path], h.tstop, utils_obj)

  pc.barrier() # wait for all hosts to get to this point

  h.load_file('mkstim_SEClamp.hoc')
  SEClamp_gids = []
  if (utils_obj.description.data['cell_data_tracking']['SEClamp_insert'] == 'yes'):
    SEClamp_insert_first_cell = utils_obj.description.data['cell_data_tracking']['SEClamp_insert_first_cell']
    for gid in cells:
      if ((gid >= SEClamp_insert_first_cell) and ((gid - SEClamp_insert_first_cell) % utils_obj.description.data['cell_data_tracking']['SEClamp_insert_cell_gid_step'] == 0) and ( cell_types[type_index(gid)] not in ['LIF_exc', 'LIF_inh'])):
        tmp_line  = str(pc.gid2cell(gid).hname()) + '.soma[0]'
        h.mkstim_SEClamp(tmp_line, 0.5, -70.0, h.tstop)
        SEClamp_gids.append(gid)

  pc.barrier() # wait for all hosts to get to this point


  h.load_file("save_t_series.hoc")
  save_value_ID_list = []
  k = 0
  if (utils_obj.description.data['cell_data_tracking']['do_save_t_series'] == 'yes'):
    for cell_gid in cells:
      if ((cell_gid % utils_obj.description.data['cell_data_tracking']['id_step_save_t_series'] == 0) and (cell_types[type_index(cell_gid)] not in ['LIF_exc', 'LIF_inh'])):
        save_value  = str(pc.gid2cell(cell_gid).hname()) + '.soma[0].v(0)'
        save_value_ID_list.append(k)
        k += 1
        h.save_t_series_prep(save_value)
  pc.barrier() # wait for all hosts to get to this point


  h.load_file("spikefile.hoc")
  h.record_spikes(cell_displ[-1])
  pc.barrier() # wait for all hosts to get to this point

  pc.timeout(0)
  #h.cvode.debug_event(1)
  h.load_file('progress.hoc')
  fih = h.FInitializeHandler(2, "progress(0)")
  h.cvode.cache_efficient(1)
  pc.barrier() # wait for all hosts to get to this point

  h.prun(h.tstop)
  pc.barrier() # wait for all hosts to get to this point

  h.spike2file('%s/spk.dat' % (workdir_n))
  pc.barrier() # wait for all hosts to get to this point

  # Save t series values to files.
  k = 0
  if (utils_obj.description.data['cell_data_tracking']['do_save_t_series'] == 'yes'):
    for cell_gid in cells:
      if ((cell_gid % utils_obj.description.data['cell_data_tracking']['id_step_save_t_series'] == 0) and (cell_types[type_index(cell_gid)] not in ['LIF_exc', 'LIF_inh'])):
        tmp_value_list = []
        save_value_out_file_n = '%s/v_out-cell-%d.h5' % (workdir_n, cell_gid)
        for i_tmp in xrange(int(h.rec_value_vec_list[save_value_ID_list[k]].size())):
          tmp_value_list.append( h.rec_value_vec_list[save_value_ID_list[k]][i_tmp] )
        dt = h.rec_t_vec_list[save_value_ID_list[k]][1] - h.rec_t_vec_list[save_value_ID_list[k]][0]
        h5 = h5py.File(save_value_out_file_n, 'w', libver='latest')
        h5.attrs['dt']=dt
        h5.create_dataset('values',(len(tmp_value_list),),maxshape=(None,),chunks=True)
        h5['values'][0:len(tmp_value_list)] = tmp_value_list
        h5.close()
        k += 1

  # Save currents from SEClamps.
  if (len(h.rec_SEClamp_i_list) > 0):
    k = 0
    for cell_gid in SEClamp_gids:
      if (cell_types[type_index(cell_gid)] in ['LIF_exc', 'LIF_inh']):
        continue
      tmp_value_list = []
      save_value_out_file_n = '%s/i_SEClamp-cell-%d.h5' % (workdir_n, cell_gid)
      for i_tmp in xrange(int(h.rec_SEClamp_i_list[k].size())):
        tmp_value_list.append( h.rec_SEClamp_i_list[k][i_tmp] )
      dt = h.rec_SEClamp_t_list[k][1] - h.rec_SEClamp_t_list[k][0]
      h5 = h5py.File(save_value_out_file_n, 'w', libver='latest')
      h5.attrs['dt']=dt
      h5.create_dataset('values',(len(tmp_value_list),),maxshape=(None,),chunks=True)
      h5['values'][0:len(tmp_value_list)] = tmp_value_list
      h5.close()
      k += 1
  pc.barrier() # wait for all hosts to get to this point

  # Postprocessing.
  if (int(pc.id()) == 0):
    f_rate.tot_f_rate(workdir_n+'/spk.dat', workdir_n+'/tot_f_rate.dat', utils_obj.description.data['postprocessing']['in_t_omit'], (h.tstop - utils_obj.description.data['postprocessing']['post_t_omit']), h.tstop, cell_displ[-1])

  h.finish()

