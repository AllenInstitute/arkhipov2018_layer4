import numpy as np

def serial_tuning(N):
    return np.linspace(0, 360.0, N, endpoint=False)
    
def random_tuning(N):
    return np.random.random(N) * 360.0


