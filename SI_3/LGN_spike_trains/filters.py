import scipy as scp
from scipy.signal import convolve2d 
import numpy as np
import pylab as pl
from scipy.misc import factorial
import os

# A double Gaussian (center and negative surround) 2D filter.
def gaussian2d_spatial_filter_center_surround(N1, N2, mu, A_c, sigma_c, A_s, sigma_s, tol):

    # The mgrid assignment takes a floor of the first argument and is non-inclusive of the last argument; because of that, the way it's done above,
    # N1 = 3, mgrid[(-N1/2):(N1/2)] = [-2, -1, 0].  If we add 1 to each argument, we will have instead mgrid[(-N1/2+1):(N1/2+1)] = [-1, 0, 1].
    # This seems to be more reasonable.  For N1=4, we get mgrid[(-N1/2+1):(N1/2+1)] = [-1, 0, 1, 2].  This is assymetric, but that's OK, as
    # any small array with an even number of elements will be similarly asymmetric.  At least, for the case with N1=3 the array is centered on 0.
    # As the size of the array grows, this becomes less relevant, as there will be just one element that's producing the asymmetry, out of many elements.
    x,y = np.mgrid[(-N1/2+1):(N1/2+1),(-N2/2+1):(N2/2+1)]

    f = A_c * np.exp(-(x-mu)**2 / (2*sigma_c**2) - (y-mu)**2 / (2*sigma_c**2)) \
        - A_s * np.exp(-(x-mu)**2 / (2*sigma_s**2) - (y-mu)**2 / (2*sigma_s**2))
        
    f[np.where(np.fabs(f) < tol)] = 0
    
    return f


#An alpha-shaped temporal filter.
def alpha_temp_fil(dt, t_max, k_alpha, beta, n_filter):
    t_vec = np.arange(0,t_max,dt)
    f_t = (k_alpha * t_vec) ** n_filter * np.exp(-k_alpha * t_vec) * (1 / factorial(n_filter) - beta * ((k_alpha * t_vec) ** 2) / factorial(n_filter + 2))
    return f_t, t_vec

