This is the code for running biophysically detailed simulations of a network of neurons
using the software NEURON.





PREREQUISITES.

python 2.7.9,
including numpy, pandas, and json.

NEURON 7.3.ansi-1078 or above (www.neuron.yale.edu/download/).
Make sure that NEURON is installed with python enabled!

Allen SDK (alleninstitute.github.io/AllenSDK/).

For simulations in parallel:
MPI (e.g., hydra 3.0.4, www.mpich.org/downloads/versions/).
If using this, make sure that NEURON is installed with parallel environment
enabled!





DIRECTORY CONTENTS.

The following files in this directory contain the simulation code:

build_all_d_distributions.py
cell.hoc
cell_types.py
common.py
connectcells0.py
connectcells.py
con_src_tar.py
external_inputs.py
f_rate.py
LIF_interneuron_1.hoc
LIF_pyramid_1.hoc
mkcells.py
mkstim.py
mkstim_SEClamp.hoc
parlib.hoc
progress.hoc
save_t_series.hoc
spikefile.hoc
start0.py
start.py
syn_uniform.py
utils.py

The following files are for running an example simulation:

run_1.py
config_1.json
modfiles/
example_1/
output_1/

The directory modfiles/ contains the MOD files that describe mechanisms
governing the biophysical processes (see below).

The directory example_1/ contains input files describing the model to be
simulated in this example.

The directory output_1/ contains the output of the simulaiton.





RUNNING SIMULATIONS.

IMPORTANT -- in general, to run simulations, execute the commands from the directory where the code is.
See instructions below.


IMPORTANT -- before running simulations, make sure your MOD files are
compiled!  See below for which path is being used for MOD files.  To compile
them, use the following command:

nrnivmodl modfiles_dir

(The command above assumes that the MOD files are in the directory modfiles_dir/;
it can be any directory, as long as the correct name is provided to the code
in the config file (see below) and the MOD files are compiled.)

The MOD files need to be compiled just once, unless the MOD files are changed,
in which case the MOD files need to be recompiled using the same command.


To run simulations in the parallel mode (this way, they can be submitted to a cluster)
execute the following command (in the example below, the code runs on 10 parallel processes):

mpiexec -np 10 nrniv -mpi run_1.py > output_1/log.txt

Here, run_1.py is an execution script (see below), output_1/ is a directory
for writing output (also see below), and output_1/log.txt is the log file.  All
three are supplied by the user (see the example in this directory).

In single-process mode, one can use the command above and specify '-np 1', or
execute the following command:

nrniv run_1.py > output_1/log.txt


NOTE that each simulation needs to have its own execution script, such as run_1.py
(for example, run_1.py, run_2.py, run_3.py, ..., run_ABC.py,
run_my_favorite_new_simulation.py, run_grating_125_trial_15.py, ...).
The name is actually arbitrary (does not have to start with 'run').

Each execution script refers to a unique config file, which determines all the
inputs and outputs of the simulation (see example in run_1.py and config_1.json).
Specifically, the execution script is simply two lines, such as in run_1.py:

import start as start

start.run_simulation('config_1.json')


The execution scripts use the main function from start.py.  In the case when
synaptic weights are not utilizing the like-to-like rule, one should replace
start.py by start0.py in the execution script
(i.e., 'import start as start' -> 'import start0 as start').
The only difference between these is that one sources connections.py and the
other, connections0.py, which, in turn, differ in one line.  That line assigns
scaling of synaptic weights according to the like-to-like rule (or not, in the
case of connections0.py).

The complete specification of the simulation is provided in the config file,
which is specified in the execution script.  The config file, such as
config_1.json in this directory, can be anywhere, as long as a valid path is
provided to this file in the execution script.  In this example, it is in the
same directory as the execution script itself.  All other files that
the example simulation needs are in the directory example_1/.
(There is also the directory with MOD files, modfiles/, which is discussed
above.)

IMPORTANT -- the config file specifies all paths relative to the directory of
the execution script; alternatively, absolute paths can be provided.

The config file specifies the output directory, which can be anywhere on the
file system.  In this example, it is also in the same directory as the
execution script:
output_1/





INPUTS AND OUTPUTS.

The inputs to the simulaiton are fully specified by the config file.  See the
example below.

The outputs are saved in the output directory, the path to which is specified by the user (see below).

The output includes the spike times saved in a text file as "cell_ID spike_time" pairs,
as well as somatic membrane voltage and currents from voltage clamp measurements for a subset of cells
(see below), saved in an h5 binary format (files named v_out-cell-i.h5 and i_SEClamp-cell-j.h5, where
i and j are cell IDs).  In addition, after the simulation is finished, the code processes all spike
times and computes the firing rate for each cell.  These rates are saved in a file called

tot_f_rate.dat

which contains three columns.  The first contains cell IDs, the second -- the firing
rates computed within the time window defined by the user (to enable
eliminating potentially irrelevant periods in the beginning and at the end of
the simulation), and the third -- the firing rates computed over the whole
duration of the simulation.

In addition, the NEURON messages can be piped into a log file as shown in the
example of running a simulation above (using "> output_1/log.txt").





CONTENTS OF THE CONFIG FILE.

Below we describe contents of a config file, using config_1.json as an
example.


The "biophys" section of the config file contains a reference to config files
(at the very least, this should include the main config file itself, but more json config files can be
added if necessary) as well as the path to the output directory.


The "run" section contains the total simulation time and time step (in ms).


The "syn_data_file" contains the path to the file that determines synaptic
weights for all connection types.


The "neuron" section can be kept with just the three entries below:
      "hoc": [
        "stdgui.hoc", 
        "import3d.hoc", 
        "cell.hoc"
      ]


The "manifest" section is partially ignored; the important portions are the
"MODFILE_DIR" and "CELL_DB".  The latter provides the path to a file that
contains information about all cels and their types.

IMPORTANT -- make sure MOD files in the directory specified here are compiled
(see above).

The "CELL_DB" parameter provides path to a file that contains information
about all cells in the model.  In the example model, this file is
example_1/1.csv

This file contains lines -- one line per cell -- describing cell properties.
The properties in the example are cell index; cell type (some usage of cell types is hard-coded
in the simulation code); x, y, and z positions of the cell soma; "tuning" -- the angle in the
visual space that the cell prefers (this is assigned by the user at the model
construction stage; whether cell's properties in simulations will actually respect this
assignment, or whether the assignment even makes sense, is not guaranteed); path to morphology;
and "cell_par" -- path to the file containing biophysical parameters of this cell type model.

Note that some of the parameter values can be "None".

Also note that all paths listed in the file example_1/1.csv are given relative
to the directory where the code is -- i.e., not relative to that csv file itself.
This is necessary for the code to be able to access the files without taking
the location of the csv file into account.  If one replaces these relative
paths by absolute paths, the code should work.

All morphologies and biophysical parameter files for this example are stored
in the directory
example_1/cell_models/

We have been assuming in this code that each cell type is represented by a
single morphology/biophysical parameters set pair.


The "postprocessing" section provides parameters for computing the firing
rates, as described above (for the second column in the output tot_f_rate.dat
file).  The parameter "in_t_omit" determines the time interval at the
beginning of the simulation that should eb ignored for rate calculation, and
the parameter "post_t_omit" -- the interval to be ignored at the end of
simulation.


The section "connections" provides the path to a directory that contains information
about connections between all explicitly modeled cells.  For example:
example_1/1_connections"

The connections directory contains files that list all connections to target
cells within the model; the connections are stored in separate files for
blocks of 100 target cells, with names
target_0_100.dat, target_100_200.dat, target_200_300.dat, and so on.

Within each file, there are three columns -- target cell ID, source cell ID,
and the number of synapses for that connection.  Note that source cells can be
any cell from the whole model, whereas the target IDs are only limited to the
range belonging to the specific file.  In the example in config_1.json, the
total number of cells is 14, and, thus, the only file with connections is
target_0_100.dat.


The section "ext_inputs" describes the sources of external spike trains
supplied to drive the simulation.  This section may contain multiple entries,
each of which comes with a key -- the key being the path to the file with
spike trains.  In the config_1.json example, there is just one such entry.
The key is "example_1/inputs_spk.dat".

The key path provides access to a text file that contains lines of numbers.
Each number is a spike time.  Each line corresponds to one trial for a given
source (which can be, for example, a thalamic cell that is not modeled
explicitly, but is used in the simulation, together woith other such thalamic
cells, simply as a source of spikes to the cortical cells that are modeled
explicitly).

The parameter "trials_in_file" determines how many trials we have in the spike
file for each source.  For example, if this parameter is 10 and there are 50
sources, the whole spike file should have 500 lines -- one for each cell and
for each trial.  The lines 1 to 10 correspond to the 10 trials for cell 1,
lines 11 to 20 -- to the 10 trials for cell 2, and so on.

The parameter "mode" tells the code that spikes are supplied from the files,
as described above.

The parameter "t_shift" can be used to shoft all the incoming spike times from
this spike file uniformly by the same time (it can be positive or negative).
If it is 0, spike times are used without any modifications.

The parameter "trial" indicates which trial should be used.  For example, if
"trial" is equal to 8 (and "trials_in_file" is 10), the code will read the
spike file and use line 8 for source 1, line 18 for source 2, and so on.

The parameter "map" provides a path to the file that describes which external
sources are connected to which cells in the model.  In this example, this file
with the mapping is example_1/inputs_map.csv.

The mapping file contains lines, each of which corresponds to a single source
connecting to a target cell.  Each lines provides the following parameters:
"index" -- index of the target cell; "src_type" -- type of the source;
"src_gid" -- index of the source; "src_vis_x", "src_vis_y" -- position of
the source in the visual space; "presyn_type" -- source type for the purposes
of determining the synaptic properties for this connection; "N_syn" -- number
of synapses in this connection.  Out of these, "src_type", "src_vis_x", and
"src_vis_y" are not used by the simulation code and, thus, can be disregarded.


The section "cell_data_tracking" is used to decide on saving the somatic
voltage traces and/or somatic excitatory currents (measured using NEURON's
voltage clamp function SEClamp).  The cells to be selected for saving these
traces are determined as

cell ID 0, id_step_save_t_series, 2*id_step_save_t_series, ...

for the voltage or

cell ID SEClamp_insert_first_cell, SEClamp_insert_first_cell + SEClamp_insert_cell_gid_step, SEClamp_insert_first_cell + 2*SEClamp_insert_cell_gid_step, ...

for the current.


The section "conditions" contains information about some of the conditions for
simulations, including the temperature.





NOTES.

This code is provided "as is", without explicit or implied warranties of
user support or further development.  It is released to accompany publication
of simulation results that it was originally used to produce.

The code employs a few minimal hoc files, but otherwise is fully implemented
in python.

The parallelization is organized using NEURON's parallel context, taking
advantage of the "round-robin" distribution of the cells among the parallel
processes.


