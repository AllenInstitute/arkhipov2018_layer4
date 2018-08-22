import numpy as np
import random
import pickle
import matplotlib.pyplot as plt


class TravelingWave():
    
    def __init__(self, **kwargs):
        self.t_total = kwargs.get('t_total', 5000.0) # Total time we will be considering, in ms.
        self.delta_t = kwargs.get('delta_t', 10.0) # Time interval for keeping track of activity, in ms.
        self.Lx_Ly = kwargs.get('Lx_Ly', (2000.0, 2000.0)) # Size of the x-y space, in um.
        self.r_center = np.array( kwargs.get('r_center', [0.0, 0.0]) )
        self.f_min_f_max = kwargs.get('f_min_f_max', [5.0, 15.0]) # Minimal and maximal firing rates, in Hz.
        # Parameters for traveling waves.
        self.t_up_wave = kwargs.get('t_up_wave', 1000.0) # Average duration of the "up" state for the wave, in ms.
        self.t_up_fluct = kwargs.get('t_up_fluct', 500.0) # Window (in ms) within which the duration of the "up" state can vary.
        self.L_up_wave = kwargs.get('L_up_wave', 1000.0) # Average total spatial extent of the wave, in um.
        self.t_between = kwargs.get('t_between', 1000.0) # Average time between waves at a given cell (end of one wave to start of the next).
        self.t_between_fluct = kwargs.get('t_between_fluct', 1000.0) # A time window within which the timing of the waves varies.
        if (self.t_between_fluct > 2.0 * self.t_between):
            print 't_between_fluct = %f is greater than twice the t_between (t_between = %f); exiting.' % (self.t_between_fluct, self.t_between)
            quit()


    def generate_sources(self, **kwargs):
        N_src = kwargs.get('N', 100)
        rnd_seed = kwargs.get('rnd_seed', 0)
        out_f_name = kwargs.get('out_f_name', 'src_dict.pkl')
        random.seed(rnd_seed)

        Lx = self.Lx_Ly[0]
        Ly = self.Lx_Ly[1]
        f_min = self.f_min_f_max[0]
        f_max = self.f_min_f_max[1]

        src_dict = {}
        for src_id in xrange(N_src):
            src_dict[src_id] = { 'x': (self.r_center[0] + Lx*(random.random()-0.5)), 'y': (self.r_center[1] + Ly*(random.random()-0.5)) }
            f_amplitude = (f_min + random.random() * (f_max - f_min))
            src_dict[src_id]['f_amplitude'] = f_amplitude

        f = open(out_f_name, 'w')
        pickle.dump(src_dict, f)
        f.close()


    def generate_rates(self, **kwargs):
        src_f_name = kwargs.get('src_f_name', 'src_dict.pkl')
        out_name = kwargs.get('out_name', 'wave_f_rates.pkl')
        rnd_seed = kwargs.get('rnd_seed', 0)
        random.seed(rnd_seed)

        f = open(src_f_name, 'r')
        src_dict = pickle.load(f)
        f.close()

        t_array = np.arange(0.0, self.t_total, self.delta_t)

        Lx = self.Lx_Ly[0]
        Ly = self.Lx_Ly[1]
        v_wave = self.L_up_wave / self.t_up_wave # Average speed of the wave propagation, in um/ms.

        f_rate_dict = { 't': t_array, 'cells': {} } # Treat sources as "cells" here, just as a matter of naming convention.

        # Prepare arrays with individual waves.  Each wave is characterized by its spatial width (L_up)
        # and the t_wave, which is the time when this wave reaches the center of the system; it is used
        # to position the wave at an appropriate distance from the center at t=0.
        t_wave_array = np.array([])
        L_wave_array = np.array([])

        # Define the time of arrival of the initial wave to the center.  Select it randomly from a large interval and make
        # it negative, so that later waves that arrive at the center at positive times are well decorrelated.
        t = -3.0 * (self.t_up_wave + self.t_up_fluct + self.t_between + self.t_between_fluct) * random.random()
        print 'Start t = %g' % (t)
        L_up_current = v_wave * (self.t_up_wave + (random.random() - 0.5)*self.t_up_fluct)
        i_t = 0
        while ((t < (self.t_total + self.t_up_wave + self.t_up_fluct + self.t_between + self.t_between_fluct)) and (i_t <= 1000000)):
            t_wave_array = np.append(t_wave_array, t)
            L_wave_array = np.append(L_wave_array, L_up_current)
            t_up_current = self.t_up_wave + (random.random() - 0.5)*self.t_up_fluct
            t_between_current = self.t_between + (random.random() - 0.5) * self.t_between_fluct
            L_up_current = v_wave * t_up_current
            t = t + t_up_current + t_between_current
            i_t += 1

        # Determine the initial locations of the waves.
        # Compute that relative to the system's center first, and then convert that to absolute coordinates.
        v_vector_list = []
        v_n_vector_list = []
        r_start_list = []
        for t in t_wave_array:
            phi = random.random() * 2.0 * np.pi
            v_n_vector = np.array([np.cos(phi), np.sin(phi)])
            v_vector = v_wave *v_n_vector
            r_start = -1.0 * t * v_vector + self.r_center
            v_vector_list.append(v_vector)
            v_n_vector_list.append(v_n_vector)
            r_start_list.append(r_start)

        # Produce the output for each source.
        for src_id in src_dict:
            f_array = np.zeros(t_array.size)
            tmp_f = np.zeros(t_array.size)
            x = src_dict[src_id]['x']
            y = src_dict[src_id]['y']
            for i_wave, t_wave in enumerate(t_wave_array):
                r_start = r_start_list[i_wave]
                v_vector = v_vector_list[i_wave]
                v_n_vector = v_n_vector_list[i_wave]
                # Project coordinates of each source, relative to the current position of the wave front, onto the wave direction vector.
                r_proj = (x - r_start[0] - v_vector[0] * t_array) * v_n_vector[0] + (y - r_start[1] - v_vector[1] * t_array) * v_n_vector[1]
                ind = np.intersect1d( np.where(r_proj <= 0.0), np.where(r_proj >= -L_wave_array[i_wave]) )
                tmp_f.fill(0.0)
                tmp_f[ind] = 1.0
                f_array = f_array + tmp_f
            f_rate_dict['cells'][src_id] = f_array * src_dict[src_id]['f_amplitude']

        # Write output.
        f = open(out_name, 'w')
        pickle.dump(f_rate_dict, f)
        f.close()


