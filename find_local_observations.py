#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the indices of observations used for each model grid point

Author: Guannan Hu
"""

import numpy as np
import great_circle_calculator.great_circle_calculator as gcc

def find_local_observations(n, nobs, xlon, xlat, ylon, ylat, r):
    """Find the indices of observations used for each model grid point
    
    Parameters
    ----------
    n : int 
        Number of model grid points.
    nobs : int
        Number of observations.
    xlon : n x 1 array
        Longitudes of model grid points.
    xlat : n x 1 array
        Latitude of model grid points.
    ylon : nobs x 1 array
        Longitudes of observations.
    ylat : nobs x 1 array
        Latitudes of observations.
    r : float or int
        Localization radius.

    Returns
    -------
    obs_indices : n x nobs matrix
        Contain 1s and 0s
        The (l,i)th element is 1 if the ith observation is used for the lth 
        grid point, and 0 otherwise.
    """
    obs_indices = np.zeros((n,nobs))

    for l in range(n):
        for i in range(nobs):
            if gcc.distance_between_points((xlon[l], xlat[l]), (ylon[i], ylat[i]), unit='kilometers') <= r:
                obs_indices[l, i] = 1
    
    return(obs_indices)