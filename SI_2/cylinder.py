import random
import numpy as np
import math

def cylinder_from_density(N, density, height, center=None):
    '''
    Build a cylinder for given point density, center and height.
    N: number of points
    density: density of points
    height: height of the cylinder
    center: desired center of the cylinder
    '''

    if center is None:
        center = np.array([0,0,0])

    height = float(height)
    radius = math.sqrt((N / density) / (height * math.pi) )
    return center, height, radius

def generate_random_positions(N, center, height, radius_outer, radius_inner):
    ''' 
    Generate N random positions within a cylinder, between two values of radius;
    assume the cylinder's axis is along the z axis and top and bottom surfaces are parallel to the x-y plane.
    N: number of positions to generate
    center: center of the cylinder (numpy array)
    height: cylinder height
    radius_outer: outer radius, within which all positions are generated
    radius_inner: inner radius, within which no positions are generated
    '''    

    # Generate N random x and y values using polar coordinates;
    # for phi, use uniform distribution;
    # for r, the probability density is p(r)dr = r dr, so use
    # inverse transform sampling:
    # integral_R0_R p(r) dr = R^2/2 - R0^2/2; draw x = R^2/2 - R0^2/2 from a uniform distribution
    # with values of x between 0 and R1^2/2 - R0^2/2.
    phi = 2.0 * math.pi * np.random.random([N])
    r = np.sqrt( (radius_outer**2 - radius_inner**2) * np.random.random([N]) + radius_inner**2 )
    x = center[0] + r * np.cos(phi)
    y = center[1] + r * np.sin(phi)
    
    # Generate N random z values.
    z = center[2] + height * ( np.random.random([N]) - 0.5 )

    positions = np.column_stack((x, y, z))

    return positions

def random_positions_by_density(N, density, height, center=None):
    '''
    Draw random positions within a cylinder defined by its center, height, density of points and number of points.
    N: number of points
    density: density of points
    height: height of the cylinder
    center: center of the cylinder
    '''
    cyl_center, cyl_height, cyl_radius = cylinder_from_density(N, density, height, center)
    return generate_random_positions(N, cyl_center, cyl_height, cyl_radius, 0.0)
    
