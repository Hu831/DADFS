#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation of the background error vector, the observation error vector and 
the innovation vector

Author: Guannan Hu
"""

import numpy as np
from H_operator_linear import H_operator_linear

def innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator):
    """ Generate random numbers from Gaussian distribution with zero mean and 
    covariances given by matrices B and R.
    
    Parameters
    ----------
    n : int
        Number of model grid points.
    nobs : int
        Number of observations.
    B : n x n array
        The background error covariance matrix.
    R : nobs x nobs array
        The observation error covariance matrix.
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
    errb : n x 1 array
        Background error vector.
    erro : nobs x 1 array
        Observation error vector.
    yb : nobs x 1 array
        Background error vector in observation space.
    d : nobs x 1 array
        The innovation vector.
    """
    
    errb = np.random.multivariate_normal([0] * n, B)
    erro = np.random.multivariate_normal([0] * nobs, R)

    yb = H_operator_linear(xlon, xlat, errb, interpolator)

    d = erro - yb(ylon, ylat)
    return(errb, erro, yb, d)