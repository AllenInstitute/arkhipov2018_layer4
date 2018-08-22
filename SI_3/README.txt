This directory provides code for building the sources of external
inputs to the layer 4 model -- the LGN and tw systems -- and for
generating activity (firing rates, and from those, spikes) of
these sources, which is then used in simulations of the layer 4
model as the external drive.

1. BUILDING THE LGN FILTERS AND GENERATING SPIKES FOR THEM.

The code is in the directory LGN_spike_trains/.

The code files are as follows.

filter_population.py
filters.py
f_rate_to_spike_train.py
LGN_f_rates_and_spikes_functions.py

The scripts for building the filters and generating spikes are

generate-LGN-rates.py

and

generate-LGN-spikes.py


The workflow is divided in two stages.  For the first, one needs to run

python generate-LGN-rates.py

This creates filters and distributes them in the visual space, and then
produces firings rates (instantaneous firing rate as a function of time)
in response to a variety of movies for each filter.

At the second stage, one needs to run

python generate-LGN-spikes.py

This produces spike trains from the previously created firing rates.


First we consider generate-LGN-rates.py.

INPUTS: defined in the script itself.

NOTE -- in our work, we used three instantiations of LGN filters, systems 1,
2, and 3, which were then re-used for all layer 4 models.  Specifically, LGN
system 1 was used for LL1, LR1, RL1, and RR1; system 2 -- for LL2, LR2, RL2,
and RR2, and system 3 -- for LL3, LR3, RL3, and RR3.  In the script, these systems
are initialized using different random seeds.

Uncomment and comment out the entries in the script according to whether it is
desired to build system 1, 2, or 3; alternatively, use another random seed
(and change other parameters if desired) to build a different system.

NOTE -- all paths in the script are provided according to the original
filesystem.  They should be changed to reflect the location of necessary files
in the actual user's filesystem.

The script instantiates LGN filters using the function define_cell_parameters().
Then, the script produces firing rates for a number of movies (which movies
are used can be controlled by commenting out or uncommenting the appropriate
entries, or adding new ones).  The firing rates are written out to files.

NOTE -- if the script has been run already and one needs to add firing rates
for new movies, one should simply rerun the script with new movies added (and
old ones commented out to reduce the run time, since those have been
processed already).  Because a specific random seed is used, it is guaranteed
that for given system the script will produce exactly the same LGN filters as
in the previous runs for this particular system.


OUTPUTS: the script produces the output file that contains information about
the LGN filters (e.g., LGN2_visual_space_positions_and_cell_types.dat) and a
set of files (one for each movie) that contain time series of firing rate for
each LGN filter; these files are saved in the output directory specified in
the script (the files might be saved, e.g., as output2/*_LGN_f_rate.pkl).

NOTE -- MAKE SURE THE OUTPUT DIRECTORY IS CREATED BEFORE RUNNING THE SCRIPT,
OTHERWISE IT WILL FAIL.


The next stage relies on the script generate-LGN-spikes.py.

INPUTS: defined in the script itself.

To switch between the systems 1, 2, or 3, or any other one that the user may
have created, provide the path to the appropriate directory containing the
firing rates (e.g., output2/).

NOTE -- as above, all paths in the script are provided according to the original
filesystem.  They should be changed to reflect the location of necessary files
in the actual user's filesystem.

The scripts processes firing rates for movies defined in the list 'labels'.
Add, change, or remove movies from the list to obtain spike trains as desired.

The script also determines how many trials will be produced for each movie.
For each trial, a separate spike train is produced for each LGN filter.  This
is determined by the parameter N_trains.

The function producing the spike trains from firing rates is run_f_rate_to_spike_train().

OUTPUTS: for each movie, a file is produced in the oiutput directory (e.g.,
output2/*_LGN_spk.dat).  This is a text file containing multiple lines -- one
line per spike train.  The first N_trains lines are for the first filter, the
next N_trains lines are for the second filter, and so on.  For example, if one
uses 9,000 filters and 10 trials (i.e., N_trains = 10), this output file will
contain 90,000 lines.  Each line is a sequence of spike times.





2. BUILDING THE TW FILTERS AND GENERATING SPIKES FOR THEM.

The code for generating tw systems is in the directory

tw_spike_trains/.

The code file is
traveling_waves_class.py

and the script for generating tw filters and their outputs is
run_traveling_waves.py.

Run the following command to perform all necessary tasks:

python run_traveling_waves.py

NOTE -- the run_traveling_waves.py script also relies on a function from
the file ../LGN_spike_trains/LGN_f_rates_and_spikes_functions.py.  From this
file, the function converting firing rate trace to spike trains is used.


The overall concept with the tw filters is very similar to what we did with
the LGN filters.  First, tw filters are generated and distributed in space
(here, the actual cortical space is used instead of the visual space, as is
the case with the LGN filters).  Then, a series of traveling waves are
generated and the time-dependent firing rates outputs of the tw filters are
produced based on that.  Finally, the firing rate traces are converted to
spike trains.


INPUTS: defined in the run_traveling_waves.py.  Like in the case of LGN
filters, use parameters for systems 1, 2, and 3 separately to generate
independent instantiations of the tw filters and their outputs.

NOTE -- these tw filters are totally independent of any input movies.  The
background activity generated in this model is assumed to be decoupled from
all other inputs (as well as from what is happening in the layer 4 network
itself).

NOTE -- unlike two scripts in the case of LGN filters, all operations for tw
filters are performed in run_traveling_waves.py.  It uses the class TravelingWave()
from the file traveling_waves_class.py.  The tw filters ("sources") are
generated using the command generate_sources().  The rates are created using
the command generate_rates().  Finally, the spike trains are produced using
the command run_f_rate_to_spike_train().

OUTPUTS: a pkl file containing the firing rate trace and a dat file containing spike trains
are written to a user-defined output directory for each instantiation of the
wave.  The format of these files is the same as in the case of the LGN
filters.  Thus, exactly the same portion of the simulation code is used to
process the LGN and tw inputs to the layer 4 model.

NOTE -- all paths in the script are provided according to the original
filesystem.  They should be changed to reflect the location of necessary files
in the actual user's filesystem.


