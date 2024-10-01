#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the ensemble perturbation matrix and ensemble perturbation matrix in
observation space

Author: Guannan Hu

"""

import numpy as np
from H_operator_linear import H_operator_linear

def XbYb(n, nobs, N, B, xlon, xlat, ylon, ylat, interpolator):
    """Generate the ensemble perturbation matrix (Xb) using the background 
    error covariance matrix and the ensemble perturbation matrix in observation
    space (Yb) using an observation operator.
    
    Parameters
    ----------
    n : int
        Number of model grid points.
    nobs : int
        Number of observations.
    N : int
        Ensemble size.
    B : numpy.ndarray
        The background error covariance matrix.
    xlon : n x 1 array
        Longitudes of model grid points.
    xlat : n x 1 array
        Latitude of model grid points.
    ylon : nobs x 1 array
        Longitudes of observations.
    ylat : nobs x 1 array
        Latitudes of observations.
    interpolator : str
        'linear' or 'nearest'
        Interpolation; Observation operator.

    Returns
    -------
    Xb : n x N matrix
        The ensemble perturbation matrix 
    Yb : nobs x N matrix
        The ensemble perturbation matrix in observation space
    """
    
    # The ensemble perturbation matrix
    Xb = np.random.multivariate_normal([0] * n, B, N).T
    sample_mean = np.mean(Xb, axis=1)
    
    # The ensemble perturbation matrix in observation space
    Yb = np.zeros(shape=(nobs, N))

    for k in range(N):
        # Remove bias due to sampling error 
        Xb[:, k] = Xb[:, k] - sample_mean  
        h = H_operator_linear(xlon, xlat, Xb[:, k], interpolator)
        Yb[:, k] = h(ylon, ylat)
        
    return(Xb, Yb)