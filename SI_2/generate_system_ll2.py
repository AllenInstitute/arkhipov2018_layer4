import cylinder
import orientation as orientation
import numpy as np
import random
import time

from network import Network
import utilities as nutil

network = Network()
network.seed(5)

sys_name = 'll2'

print "Generating cells."
Scnn1a = network.create_nodes(type_name='Scnn1a', N=3700)
Rorb = network.create_nodes(type_name='Rorb', N=3300)
Nr5a1 = network.create_nodes(type_name='Nr5a1', N=1500)
PV1 = network.create_nodes(type_name='PV1', N=800)
PV2 = network.create_nodes(type_name='PV2', N=700)
LIF_exc = network.create_nodes(type_name='LIF_exc', N=29750)
LIF_inh = network.create_nodes(type_name='LIF_inh', N=5250)

# Compute geometry parameters.
num_nodes = network.node_count()
cyl_center, cyl_height, cyl_radius = cylinder.cylinder_from_density(num_nodes, density = 0.0002, height = 100.0, center=np.array([0,0,0]))
radius_biophys = ((num_nodes - len(LIF_exc + LIF_inh)) / (num_nodes * 1.0) )**0.5 * cyl_radius # All neurons within this radius are represented by biophysical models.


# Assign morphologies, cell parameters, and positions.
print "Generating positions for Scnn1a cells."
positions = cylinder.generate_random_positions(N = len(Scnn1a), center = cyl_center, height = cyl_height, radius_outer = radius_biophys, radius_inner = 0.0)
network.update_nodes(array_data={ 'morphology': ['/data/mat/antona/cell_models/395830185/Scnn1a-Tg3-Cre_Ai14_IVSCC_-177300.01.02.01_473845048_m.swc'] * len(Scnn1a), 'cell_par': ['/data/mat/antona/cell_models/395830185/472363762_fit.json'] * len(Scnn1a), 'position': positions }, gids=Scnn1a)

print "Generating positions for Rorb cells."
positions = cylinder.generate_random_positions(N = len(Rorb), center = cyl_center, height = cyl_height, radius_outer = radius_biophys, radius_inner = 0.0)
network.update_nodes(array_data={ 'morphology': ['/data/mat/antona/cell_models/314804042/Rorb-IRES2-Cre-D_Ai14_IVSCC_-168053.05.01.01_325404214_m.swc'] * len(Rorb), 'cell_par': ['/data/mat/antona/cell_models/314804042/473863510_fit.json'] * len(Rorb), 'position': positions }, gids=Rorb)

print "Generating positions for Nr5a1 cells."
positions = cylinder.generate_random_positions(N = len(Nr5a1), center = cyl_center, height = cyl_height, radius_outer = radius_biophys, radius_inner = 0.0)
network.update_nodes(array_data={ 'morphology': ['/data/mat/antona/cell_models/318808427/Nr5a1-Cre_Ai14_IVSCC_-169250.03.02.01_471087815_m.swc']*len(Nr5a1), 'cell_par': ['/data/mat/antona/cell_models/318808427/473863035_fit.json']*len(Nr5a1), 'position': positions }, gids=Nr5a1)

print "Generating positions for PV1 cells."
positions = cylinder.generate_random_positions(N = len(PV1), center = cyl_center, height = cyl_height, radius_outer = radius_biophys, radius_inner = 0.0)
network.update_nodes(array_data={ 'morphology': ['/data/mat/antona/cell_models/330080937/Pvalb-IRES-Cre_Ai14_IVSCC_-176847.04.02.01_470522102_m.swc']*len(PV1), 'cell_par': ['/data/mat/antona/cell_models/330080937/472912177_fit.json']*len(PV1), 'position': positions }, gids=PV1)

print "Generating positions for PV2 cells."
positions = cylinder.generate_random_positions(N = len(PV2), center = cyl_center, height = cyl_height, radius_outer = radius_biophys, radius_inner = 0.0)
network.update_nodes(array_data={ 'morphology': ['/data/mat/antona/cell_models/318331342/Pvalb-IRES-Cre_Ai14_IVSCC_-169125.03.01.01_469628681_m.swc']*len(PV2), 'cell_par': ['/data/mat/antona/cell_models/318331342/473862421_fit.json']*len(PV2), 'position': positions }, gids=PV2)

print "Generating positions for LIF_exc cells."
positions = cylinder.generate_random_positions(N = len(LIF_exc), center = cyl_center, height = cyl_height, radius_outer = cyl_radius, radius_inner = radius_biophys)
network.update_nodes(array_data={ 'position': positions }, gids=LIF_exc)

print "Generating positions for LIF_inh cells."
positions = cylinder.generate_random_positions(N = len(LIF_inh), center = cyl_center, height = cyl_height, radius_outer = cyl_radius, radius_inner = radius_biophys)
network.update_nodes(array_data={ 'position': positions }, gids=LIF_inh)


# Assign tuning variables.
print "Generating tuning."
tunings = orientation.serial_tuning(len(Scnn1a))
network.update_nodes(array_data={ 'orientation_tuning': tunings }, gids=Scnn1a)

tunings = orientation.serial_tuning(len(Rorb))
network.update_nodes(array_data={ 'orientation_tuning': tunings }, gids=Rorb)

tunings = orientation.serial_tuning(len(Nr5a1))
network.update_nodes(array_data={ 'orientation_tuning': tunings }, gids=Nr5a1)

tunings = orientation.serial_tuning(len(LIF_exc))
network.update_nodes(array_data={ 'orientation_tuning': tunings }, gids=LIF_exc)


print "Writing the cells information to file."
# Write output to a text file.
f_prefix = '%s' % (sys_name) #'V1_L4'
f_out = open(f_prefix + '.csv', 'w')
f_out.write( 'index type x y z tuning morphology cell_par\n')
for node in network.nodes():
  id = node['id']
  type = node['type']
  position = node['position']
  orientation_tuning = node.get('orientation_tuning', None)
  morphology = node.get('morphology', None)
  cell_par = node.get('cell_par', None)
  f_out.write( '%d %s %.3f %.3f %.3f %s %s %s\n' % (id, type, position[0], position[1], position[2], str(orientation_tuning), morphology, cell_par) )
f_out.close()





# Define objects containing multiple cell types.
EXC_cells = Scnn1a + Rorb + Nr5a1 + LIF_exc
INH_cells = PV1 + PV2 + LIF_inh





# Define a function that tells the connection iterator what keep track of for a single source-target pair.
def distance_tuning_connection_handler(source, target,
                                       d_weight_min, d_weight_max, d_max, 
                                       t_weight_min, t_weight_max,
                                       nsyn_min, nsyn_max,
                                       center):
    # Avoid self-connections.
    if (source['id'] == target['id']):
      return None

    # first create weights by euclidean distance between cells
    # DO NOT use PERIODIC boundary conditions in x and y!
    dw = nutil.distance_weight(np.array(source['position'][:2]) - np.array(target['position'][:2]),
                               d_weight_min, d_weight_max, d_max)

    # drop the connection if the weight is too low
    if dw <= 0:
        return None
        
    # next create weights by orientation tuning [ aligned, misaligned ] --> [ 1, 0 ]
    # Check that the orientation tuning property exists for both cells; otherwise,
    # ignore the orientation tuning.
    if ( ('orientation_tuning' in source.keys()) and ('orientation_tuning' in target.keys()) ):
      tw = dw * nutil.orientation_tuning_weight(source['orientation_tuning'],
                                                target['orientation_tuning'],
                                                t_weight_min, t_weight_max)
    else:
      tw = dw
 
     
    # drop the connection if the weight is too low
    if tw <= 0:
        return None

    # filter out nodes by treating the weight as a probability of connection
    if random.random() > tw:
        return None

    # Add the number of synapses for every connection.

    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    return { 'nsyn': tmp_nsyn }




def distance_connection_handler(source, target,
                               d_weight_min, d_weight_max, d_max, 
                               nsyn_min, nsyn_max,
                               center):

    # Avoid self-connections.
    if (source['id'] == target['id']):
      return None

    # first create weights by euclidean distance between cells
    # DO NOT use PERIODIC boundary conditions in x and y!
    dw = nutil.distance_weight(np.array(source['position'][:2]) - np.array(target['position'][:2]),
                               d_weight_min, d_weight_max, d_max)

    # drop the connection if the weight is too low
    if dw <= 0:
        return None
        
    # filter out nodes by treating the weight as a probability of connection
    if random.random() > dw:
        return None

    # Add the number of synapses for every connection.

    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    return { 'nsyn': tmp_nsyn }





# Generate connections.
# Note differences in arguments provided to the connector functions for the different sets that we are connecting.

print "Generating E-to-E connections."
network.connect(EXC_cells,
                EXC_cells,
                weight_function=distance_tuning_connection_handler, 
                weight_function_params={
                    'd_weight_min':0.0, 'd_weight_max':0.34, 'd_max':300.0,
                    't_weight_min':0.5, 't_weight_max':1.0,
                    'nsyn_min':3, 'nsyn_max':7,
                    'center':cyl_center } )

print "Generating E-to-I connections."
network.connect(EXC_cells,
                INH_cells,
                weight_function=distance_connection_handler,
                weight_function_params={
                    'd_weight_min':0.0, 'd_weight_max':0.26, 'd_max':300.0,
                    'nsyn_min':3, 'nsyn_max':7,
                    'center':cyl_center } )

print "Generating I-to-E connections."
network.connect(INH_cells,
                EXC_cells,
                weight_function=distance_connection_handler,
                weight_function_params={
                    'd_weight_min':0.0, 'd_weight_max':1.0, 'd_max':160.0,
                    'nsyn_min':3, 'nsyn_max':7,
                    'center':cyl_center } )

print "Generating I-to-I connections."
network.connect(INH_cells,
                INH_cells,
                weight_function=distance_connection_handler,
                weight_function_params={
                    'd_weight_min':0.0, 'd_weight_max':1.0, 'd_max':160.0,
                    'nsyn_min':3, 'nsyn_max':7,
                    'center':cyl_center } )





# Write output using *.dat files.
# Due to restrictions on the number of files that can be simultaneously open, we lump multiple tar_gid into one file; the parameter
# N_tar_gid_per_file specifies how many tar_gid are reserved for each file.
N_tar_gid_per_file = 100
f_out = {}
nsyn_received = {}
for tid in xrange(0, network.node_count(), N_tar_gid_per_file):
  f_name = '%s_connections/target_%d_%d.dat' % (sys_name, tid, tid + N_tar_gid_per_file)
  f_out[tid] = open(f_name, 'w')

for sid, tid, v in network.connection_set:
  con_id = v['id']
  if (con_id % 10000 == 0):
    print 'Writing out connections; working on connection %d.' % (con_id)

  nsyn = v['nsyn']

  # Find the key for the correct file handle in the dictionary.
  tid_f_dict = N_tar_gid_per_file * (tid / N_tar_gid_per_file) # Note that integer division is used here.
  f_out[tid_f_dict].write('%d %d %d\n' % (tid, sid, nsyn) )

  if (tid in nsyn_received.keys()):
    nsyn_received[tid] += nsyn
  else:
    nsyn_received[tid] = nsyn

for tid in f_out.keys():
  f_out[tid].close()

# Write out the information about statistics of connections.
f_out = open('%s_connections_statistics.dat' % (sys_name), 'w')
for tid in nsyn_received.keys():
  f_out.write('%d %d\n' % (tid, nsyn_received[tid]))
f_out.close()

