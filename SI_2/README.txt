This directory contains code for building a model of the layer 4 (L4)
and providing external inputs to that model.

1. BUILDING CELL POSITIONS AND CONNECTIONS FOR THE L4 MODEL.

Relevant code files:

cylinder.py
orientation.py
nio.py
data_model.py
dict_data_model.py
mongodb_data_model.py
pandas_data_model.py
connection_set.py
connector.py
network.py
utilities.py

AND

generate_system_*.py


To generate an L4 model, run one of the "generate_system" python scripts:
generate_system_ll1.py for LL1, generate_system_ll2.py for LL2, generate_system_ll3.py for LL3,
generate_system_rr1.py for RR1, generate_system_rr2.py for RR2, generate_system_rr3.py for RR3.

USAGE:
python generate_system_ll1.py
etc.

NOTE -- generate_system_ll*.py files differ from each other in only two lines, the
random seed and the system name.  The same is true for generate_system_rr*.py files.

Files generate_system_ll1.py and generate_system_rr1.py only differ in the
system name, a few parameters for connectivity, and in the connectivity
function (one uses a function that takes into account both distance between the cells
and their orientation preference, "distance_tuning_connection_handler", and
the other -- a function that takes only the distance between cells into
account, "weight_function=distance_connection_handler").  The same is true for pairs
generate_system_ll2.py and generate_system_rr2.py
and
generate_system_ll3.py and generate_system_rr3.py.

NOTE -- the systems LR1, LR2, and LR3 are the same as LL1, LL2, and LL3,
respectively, in terms of the cell properties (including positions) and
connections.  The difference between these systems is in the way of how
synaptic weights are assigned.  This, however, is determined at run time
during simulation, and, thus, not by the code in this directory.  See instead
the simulation code (the files connectcells.py for like-to-like, or L,
synaptic weights, and connectcells0.py for random, or R, synaptic weights).

The same is true for the RL1, RL2, and RL3 systems vs. RR1, RR2, and RR3.

In other words, the model building code here determines the "like-to-like" vs.
"random" connection probability for the first "L" or "R" in the system name.
The second "L" or "R" in the system name is determined by whether the logics
of assignment of synaptic weights is "like-to-like" or "random", which happens
in the simulation code.


INPUTS to generate_system_*.py scripts are all determined in the scripts
themselves.

OUTPUTS: after running, the generate_system_*.py scripts create a cell table
that is stored in a csv file, a connectivity matrix that is stored in dat
files in a directory, and a summary of synapse numbers per cell that is stored
in a dat file.  The output files and directory are named
according to the name of the system (though, the user can change that in the
script).  For example, generate_system_ll2.py creates the output file ll2.csv
and writes out the output connectivity files to the directory
ll2_connections/, with the connectivity statistics saved in the file
ll2_connections_statistics.dat.  The connection statistics file contains two
columns.  The first column lists cell IDs (for those cells that receive any
synapses from the recurrent  connections) and the second column -- number
of synapses that the cell receives.

NOTE -- the scripts are not parallelized, and running one script for 45,000
cells make take from a few hours to a few days on a desktop computer.

IMPORTANT -- make sure the output directory is CREATE BEFORE THE SCRIPT RUNS!
The scripts do not check whether the output directory exists or not.  If it
does not exist, the script may spend a few days to establish all connections,
and then fail with an error because the output destination does not exist.
Always make sure the output directory exists by the time the script starts
writing output.

For convenience, the output directories for the LL and RR 1, 2, and 3 models
are already creatred in this directory.  The csv and connectivity dat files
are not stroed here, however, since they are too large.

NOTE -- in the portion of the generate_system_*.py scripts where cell
properties are assigned (see under the comment
"# Assign morphologies, cell parameters, and positions"),
the paths to morphologies and biophysical parameter files are specified
according to their original location in our file system.  If one aims to
generate models that can be simulated, these entries should be replaced by the
actual paths to the morphologies and biophysical parameters on the user's file
system.  As described in the instructions to the simulation code, the path can
be absolute or relative to the directory from which the simulation code is
being launched.


For convinience, a simple example script is provided (based on generate_system_ll2.py),
which builds a system with 450 cells and generates connectivity among them.
The output is saved in the file test.csv, directory test_connections/, and the
file test_connections_statistics.dat.

The connection statistics file contains two columns.  The first column lists
cell IDs (for those cells that receive any synapses from the recurrent
connections) and the second column -- number of synapses that the cell
receives.





2. BUILDING EXTERNAL INPUTS TO THE L4 MODEL FROM THE LGN.

Relevant code files:
generate_inputs_functions.py
generate_inputs_mapping.py

The LGN cells are represented by filters.  They are not used explicitly in
simulations.  Instead, visual stimuli are passed through these filters in
advance, resulting in an instantaneous firing rate trace as a function of time
for each LGN cell, which is then converted to spikes (multiple trials
corresponding to multiple spike trains generated from the same firing rate
trace).  The firing rate trace and spike trains are saved to files.  The
simulation code thenm uses the spike trains from those files to provide
external drive to the layer 4 cells that are explicitly modeled.

The code that produces LGN filters and spike trains is described separately.
Here, we provide the code that connects LGN cells to the layer 4 cells.

The connections from LGN cells to layer 4 cells are generated using the script
generate_inputs_mapping.py.
(It relies on functions from generate_inputs_functions.py).

The connections from LGN to layer 4 were generated in 3 instantiations in our model:
as "System 1" for LL1, LR1, RL1, and RR1,
as "System 2" for LL2, LR2, RL2, and RR2,
and as "System 3" for LL3, LR3, RL3, and RR3.

In other words, the models LL, LR, RL, and RR that have the same number (1, 2,
or 3) use exactly the same LGN filters and wiring of those filters to the
layer 4 cells.

To switch between these three cases in the generate_inputs_mapping.py script,
simply use comment out and uncomment the relevant lines marked by "# System
1", "# System 2", or "# System 3".

NOTE -- the code uses random seed in the file generate_inputs_functions.py.
This can be changed by the user.

For the layer 4 model, we ran the generate_inputs_mapping.py
script once for "# System 1" and used that one result for LL1, LR1, RL1, and RR1;
the same was done for "# System 2" and "# System 3".

NOTE -- the major difference between "# System 1", "# System 2", and "# System 3" is
that they use different sets of LGN filters (and, correspondingly, LGN spike
trains).  This is determined by providing the path to the appropriate set of
LGN filters,
f_in = open('...', 'r')

NOTE -- in the script, we use the absolute paths from the original file system where
the layer 4 model was generated.  One needs to replace these paths with those
from the user's file system.

USAGE:
python generate_inputs_mapping.py

INPUTS: the inputs are defined in the generate_inputs_mapping.py script itself.
They include paths to two files that need to be supplied: the cell
table for the model system (such as the layer 4 model, e.g., ll1.csv), and the
LGN cell table (the file containing information about the properties of the
LGN cells, e.g., LGN2_visual_space_positions_and_cell_types.dat; see the code
for distributing LGN filter models in the visual space).

OUTPUTS: the script generate_inputs_mapping.py produces three output files.
The main output is the csv file that contains the information about which
LGN cells (filters) connect to which cells in the layer 4 model.  The name of
that file is, e.g., ll2_inputs_from_LGN.csv (the name can be changed in the
generate_inputs_mapping.py script by the user).

The other two output files are auxiliary (can be useful for debugging).
Example names for these files are ll2_inputs_from_LGN_el_c_d.dat and
tmp_N_src.dat.  The former contains information about the distance between the
receptive subfield centers for each layer 4 target cell, and the latter --
information about how many LGN sources each layer 4 target cell receives
connections from.





3. BUILDING EXTERNAL INPUTS TO THE L4 MODEL FROM THE BACKGROUND SOURCES.

In the layer 4 model, we used external backround sources that provided inputs
to the layer 4 cells.  Similarly to the LGN sources, these background sources
were simulated explicitly.  Instead, they were pre-processed, spike trains
correspodning to multiple trials were generated for each such source and
saved, and during the layer 4 model simulations these spike trains were used
as inputs.

The model generating spike trains used a concept of traveling waves to produce
distinct states of background activity.  This portion of our model is then
referred to as "tw" (for "traveling waves").

The tw sources and their spike trains are generated using code in a different
directory.  Here we are providing code for connecting these sources to the
layer 4 cells.  The code is in the subdirectory tw/.

The relevant code files are

generate_mapping.py
mapping_convert_dat_to_csv.py

Generation of connections from tw to the layer 4 model is done in two steps.
(The way it is organized is partically due to historical reasons.)
In the first step, it is decided which tw sources connect to which layer 4
cells.  In the second step, each of these connections is assigned the number
of synapses are the result is saved in a csv file that is used in simulations.

NOTE -- like in the case of LGN sources, we created 3 instantiations of tw
sources.  The first one was used in identical way for models LL1, LR1, RL1,
and RR1; the second -- for models LL2, LR2, RL2, and RR2; and the third -- for
models LL3, LR3, RL3, and RR3.

NOTE -- this code does not use random seed, and thus each time it is run, the
configuration will be different.  The exact configurations used in our
simulations are provided for download separately.

To run the first stage, use the command

python generate_mapping.py

INPUTS: all inputs are defined in the script.  It refers to the cell tabels
csv file for the layer 4 model (e.g., ll2.csv) and to the file containing
definitions of the tw sources (e.g., sources.pkl).  The paths to these input
files should be updated to reflect the user's file system.  The user has full
control over these paths and file names; we used notations such as ll1, ll2,
and ll3, but they can be updated as neede as ll1, ll2, and ll3, but they can
be updated as needed.

OUTPUTS: this script produces a dat file that specifies IDs of the tw sources
connected to each layer 4 cell.


For the next stage, run the following command:

python mapping_convert_dat_to_csv.py

INPUTS: again, the paths to all input files are determined in the script and
should be updated depending on which system the user is working with (e.g.,
ll1, ll2, or ll3).  In particular, the script uses the dat file that is the
output of the previous stage.

OUTPUT: the script produces a csv file that contains five columns.  The first
column is the ID of the layer 4 target cells, the second -- the source type ("tw"),
the third -- the source ID, the fourth -- synaptic type associated with the
source, and the fifth -- the number of synapses for this connection.

This csv file is used in simulations.


