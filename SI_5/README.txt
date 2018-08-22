In this directory we provide scripts that were used to
carry out the types of analysis of the data that we report
with the layer 4 work.

NOTE -- check carefully all the scripts for inputs and outputs.

NOTE -- make sure that paths to necessary files are provided according to the user's file
system, and create output directories as necessary.

A common set of functions is used for some plotting procedures.  These
functions are found in the following file:
plot_functions.py.

Also, a common function for computing firing rates by cell type is used in
multiple scripts.  This function can be found in the file below:
f_rate_t_by_type_functions.py.


ANALYSIS SCRIPTS.

1. Computation of firing rates for spontaneous activitiy:
spont_activity_new.py.

2. Distribution of the number of synapses per background (tw) connection:
tw_Nsyn.py.

3. Computation of the maximal firing rates (for each cell) in response to
gratings:
Rmax_new.py.

4. Computation of the probability of connection between excitatory biophysical cells as a function of
the actual preferred orientation of those cells (the preferred orientation is pre-computed from the simulation data):
comp_pro_dis.py.

5. Distribution of the numbers of excitatory and inhibitory synapses recieved
by neurons in the model:
connectivity_syn_hist.py.

6. Compute the preferred spatial frequency, temporal frequency, and orientation, as well as the OSI and DSI,
and build a tuning curve for individual cells, based on gratings simulations:
Ori_compute.py.

7. Based on the information about OSI and DSI, etc., for individual cells, compute average OSI and DSI values
and associtated standard deviations; save and plot the results:
Ori_plot_new.py.
(NOTE -- the CV_Ori in this script and elsewhere is the OSI defined using
the circular variance).

8. Plot tuning curves for grating responses for a subset of cells:
Ori_tuning_curves.py.

9. Produce Gaussian fits for grating tuning curves for individual cells; save
the results:
Ori_tuning_curves_Gaussian_fit.py.

10. Build and plot histograms of tuning curve widths based on the Gaussian
fits of grating tuning curves for individual cells:
Ori_tuning_curves_ctr_sigma.py.

11. Plot spike rasters of responses to a grating:
gratings_plt_spk.py

12. Compute the weighted multiunit activity spectra and plot the results:
oscillations_run.py (relies on the functions in oscillations.py).

13. Plot spike rasters of responses to a natural movie:
movies_plt_spk.py.

14. Compute a "cortical decay function" describing the decay of the firing
rate in the cortex when LGN is silenced:
cortical_decay_function.py.

15. Plot the time-dependent average firing rate, average peak, and average
time to peak for full-field flash stimuli:
flashes_frt.py.

16. Find the prefered stimulus among the moving bars:
bars_pref_stim.py.

17. Compare the preferred orientations of cells based on gratings with those
based on moving bars:
bars_gratings_compare.py.

18. Plot positions of the input LGN filters (in the visual space) for the selected cell, and also, separately, illustrate the selection process
of the LGN inputs to a layer 4 cell, by visualizing the ellipses used to pool the LGN cells:
plot_LGN_vis_space_positions.py and plot_LGN_vis_space_positions_sel_process.py.

19. Plot the proportion of LGN synapses among all synapses in the neuropil based on the experimental EM data:
LGN_syn_experiment.py

20. Plot an averaged and normalized tuning curve of a group of cells based
on the firing rates of these cells:
Ori_plot_combined_tuning_curves.py

21. Characterize firing rates on the log scale, for spontaneous activity and responses to a grating:
log_normal_rates.py

22. Preprocess simulation data to obtain excitatory currents flowing through
cells where they were measured, relative to the baseline (for the latter,
currents obtained in isolated cell models, from the directory 5_cells_measure_i_baseline are used):
i_amplification_process_LGN_crt.py
and
i_amplification_process_tw_only.py.

23. Compare currents from the LGN with total currents, and obtain the
ampification factor for the preferred stimulus, as well as linear fits for all stimulus conditions:
i_amplification_fit_for_cells.py
and
i_amplification_ratio.py.

24. Plot averaged and normalized tuning curves of the synaptic currents
coming to cells from LGN and from all connections; also, plot averaged and
normalized time course of these currents:
i_amplification_combined_tuning_curves.py
and
i_amplification_plot_phase_shift.py

