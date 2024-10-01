#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ensemble of observation perturbations, ensemble of background error
vector and ensemble of innovation vector.

Author: Guannan Hu
"""

import numpy as np
from H_operator_linear import H_operator_linear

def generate_ensemble_members(n, nobs, N, xlon, xlat, ylon, ylat, Xb, R, d, errb, interpolator):
    """Generate ensemble of observation perturbations, ensemble of background error
    vector and ensemble of innovation vector.
    
    Parameters
    ----------
    n : int
        Number of model grid points.
    nobs : int
        Number of observations.
    N : int
        Ensemble size.
    xlon : n x 1 array
        Longitudes of model grid points.
    xlat : n x 1 array
        Latitude of model grid points.
    ylon : nobs x 1 array
        Longitudes of observations.
    ylat : nobs x 1 array
        Latitudes of observations.
    Xb : n x N array
        The background ensemble perturbation matrix.
    R : nobs x nobs array
        The observation error covariance matrix.
    d : nobs x 1 array
        The innovation vector.
    errb : n x 1 array
        Background error vector.
    interpolator : str
        'linear' or 'nearest'
        Interpolation; Observation operator.

    Returns
    -------
    obs_pert : nobs x N array
        The observation perturbations for each ensemble member.
    d_ens : nobs x N array
        The innovation vector for each ensemble member.
    errb_ens : n x N array
        The background error vector for each ensemble member.
    """

    # Generate observation perturbations for each ensemble member
    obs_pert = np.random.multivariate_normal([0] * nobs, R, N).T
    sample_mean = np.mean(obs_pert, axis=1)

    # Remove bias due to sampling error
    for k in range(N):
        obs_pert[:, k] = obs_pert[:, k] - sample_mean  
        
    # Generate the innovation vector for each ensemble member
    d_ens = np.zeros((nobs,N))
    for k in range(N):
        interp = H_operator_linear(xlon, xlat, Xb[:,k], interpolator)
        d_ens[:,k] = d + obs_pert[:,k] - interp(ylon,ylat)
        
    # Generate the background error vector for each ensemble member
    errb_ens = np.zeros((n,N))
    for k in range(N):
        errb_ens[:,k] = errb + Xb[:,k]
        
    return(obs_pert, d_ens, errb_ens)
