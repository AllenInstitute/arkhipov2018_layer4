import numpy as np
import math

def lerp(v0, v1, t):
    return v0 * (1.0 - t) + v1 * t

def distance_weight(delta_p, w_min, w_max, r_max):
    r = np.linalg.norm(delta_p)

    if r >= r_max:
        return 0.0
    else:
        return lerp(w_max, w_min, r / r_max)

def periodic_distance_weight(pos1, pos2, w_min, w_max, r_max, box_size):
    ndims = pos1.size
    
    d = np.abs(pos2 - pos1)
    for i in xrange(ndims):
        hs = box_size[i]/2
        if d[i] > hs:
            d[i] = box_size[i] - d[i]

    return distance_weight(d, w_min, w_max, r_max)


def orientation_tuning_weight(tuning1, tuning2, w_min, w_max):

    # 0-180 is the same as 180-360, so just modulo by 180
    delta_tuning = math.fmod(abs(tuning1 - tuning2), 180.0)

    # 90-180 needs to be flipped, then normalize to 0-1
    delta_tuning = delta_tuning if delta_tuning < 90.0 else 180.0 - delta_tuning

    t = delta_tuning / 90.0

    return lerp(w_max, w_min, delta_tuning / 90.0)
