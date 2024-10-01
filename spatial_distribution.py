#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate locations of observations
    
Author: Guannan Hu

"""

import numpy as np
import matplotlib.pyplot as plt
import great_circle_calculator.great_circle_calculator as gcc

def spatial_distribution(nx, ny, nobs, xrange, yrange, distrib, ifplot):
    """ Generate locations of observation in a 2-D domain with longitudes and 
    latitudes.
    
    Parameters
    ----------
    nx : int
        Number of model grid points in the East-West direction.
    ny : int
        Number of model grid points in the North-south direction.
    nobs : int
        Number of observations.
    xrange : tuple
        The east-west boundary of the domain in longitude.
    yrange : tuple
        The north-south boudary of the domain in latitude.
    distrib : int
        One of {'1', '2', '3', '4', '5'}
        Spatial distributions of observations: 1 - On randomly selected model 
        grid points; 2 - Uniform distribution; 3 - Gaussian distribution
        4 - Slightly off model grid; 5 - Regularly on model grid.
    ifplot : int
        '1' or '0'
        Illustration of model grid and observation location.

    Returns
    -------
    xlon : n x 1 array
        Longitudes of model grid points.
    xlat : n x 1 array
        Latitude of model grid points.
    ylon : nobs x 1 array
        Longitudes of observations.
    ylat : nobs x 1 array
        Latitudes of observations.
    x_grid_length : float
        Grid length in the East-West direction.
    y_grid_length : float
        Grid length in the North-South direction.
    nobs : int
        Updated number of observations after spatial thinning
    """
    
    # Model grids 
    n = nx * ny
    x = np.linspace(xrange[0], xrange[1], nx)
    y = np.linspace(yrange[0], yrange[1], ny)
    xlon_2d, xlat_2d = np.meshgrid(x, y)
    xlon = xlon_2d.flatten(order = 'C')
    xlat = xlat_2d.flatten(order = 'C')
    
    # Observation locations
    
    # 1 - On model grid
    if distrib == 1:
        pos = np.random.randint(0, n, nobs)
        ylon = xlon[pos]
        ylat = xlat[pos]
        
    # 2 - Uniform distribution
    if distrib == 2:
        ylon = np.random.uniform(xrange[0], xrange[1], nobs)
        ylat = np.random.uniform(yrange[0], yrange[1], nobs)
        
    # 3 - Gaussian distribution
    if distrib == 3:
        sd = 0.1
        ylon = np.random.normal(np.mean(xrange), (xrange[1] - xrange[0]) * sd, nobs)
        ylat = np.random.normal(np.mean(yrange), (yrange[1] - yrange[0]) * sd, nobs)
        
        # Spatial thinning to ensure each model grid box contain at most one 
        # observation
        index1 = np.zeros(nobs)
        ibox = 0
        for ix in range(nx-1):
            for iy in range(ny-1):
                for iobs in range(nobs):
                    if (ylon[iobs] >= x[ix] and ylon[iobs] < x[ix+1] and \
                        ylat[iobs] >= y[iy] and ylat[iobs] < y[iy+1]):
                        index1[iobs] = ibox
                ibox += 1
                
        # Index of selected observations
        index2 = [] 
        index1 = list(index1)
        # Remove duplicates from the list 
        short_index1 = list(dict.fromkeys(index1)) 
        for i in short_index1:
              index2.append(index1.index(i))  
             
        ylon = ylon[index2]
        ylat = ylat[index2]
        
    # 4 - Slightly move away from the grid
    if distrib == 4:
        pos = np.random.randint(0, n, nobs)
        ylon = xlon[pos] - 1
        ylat = xlat[pos] 
        
        # Ensure no observation is out of the model domain
        for i in range(nobs):
            if ylon[i] < xrange[0]:
                ylon[i] += 1
        
    # 5 - Regularly on grid
    if distrib == 5:            
        ylon_2d, ylat_2d = np.meshgrid(x[0::2], y[0::2])
        ylon = ylon_2d.flatten(order = 'C')
        ylat = ylat_2d.flatten(order = 'C')
        
    # Update the number of observations after spatial thinning
    nobs = len(ylon)
    
    # Calculate and print the grid length in East-West and North-South directions
    x_grid_length = gcc.distance_between_points((xlon[0], xlat[0]), (xlon[1], xlat[1]), unit='kilometers')
    y_grid_length = gcc.distance_between_points((xlon[0], xlat[0]), (xlon[nx], xlat[nx]), unit='kilometers')

    print('East-West grid length:', x_grid_length)
    print('North-South grid length:', y_grid_length)

    # Plot model grid points and observation locations 
    if ifplot == 1:
        plt.scatter(xlon, xlat, s=30, marker='.', color='black')
        plt.scatter(ylon, ylat, s=30, marker='x', color='red')
        plt.xlabel("lon")
        plt.ylabel("lat")    
        
    return(xlon, xlat, ylon, ylat, x_grid_length, y_grid_length, nobs)