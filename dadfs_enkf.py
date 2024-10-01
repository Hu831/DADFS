#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform the EnKF and estimate the DFS

Author: Guannan Hu
"""

import numpy as np
from numpy.linalg import inv

from enkf import enkf
from innovation import innovation
from H_operator_linear import H_operator_linear
from generate_ensemble_members import generate_ensemble_members

def dadfs_enkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, option):
    """Perform the EnKF, obtain the innovation (O-B), residual (O-A) and 
    increment (A-B) vectors, and estimate the DFS using background ensemble 
    perturbations.
    
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
    Yb : nobs x N array
        The background ensemble perturbation matrix in observation space.
    B : n x n array
        The background error covariance matrix.
    R : nobs x nobs array
        The observation error covariance matrix.
    obs_indices : n x nobs matrix
        Contain 1s and 0s
        The (l,i)th element is 1 if the ith observation is used for the lth 
        grid point, and 0 otherwise.
    interpolator : str
        'linear' or 'nearest'
        Interpolation; Observation operator.
    option : str
        'ens' or 'vec'
        Estimate the DFS using ensemble perturbations, or estimate the DFS 
        using the weighting vector and return O-B, O-A and A-B vectors.
        
    Returns
    ------- 
    DFS : nobs x 1 array
        Theoretical DFS.
    count : nobs x 1 array
        How many times each observation has been used.
    omb : nobs x 1 array
        The innovation (O-B) vector.
    oma : nobs x 1 array
        The residual (O-A) vector.
    amb : nobs x 1 array
        The observation-space increment vector.    
    np.std(erra)) : float
        Standard deviation of the analysis error at model grid points.
    """
    
    DFS  = np.zeros(nobs)      

    # Times of each observations used 
    count = np.zeros((nobs))   
    
    # Generate the background error, observation error and innovation vectors
    errb, erro, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)
    
    erra = errb.copy()

    # Generate ensemble members
    obs_pert, d_ens, errb_ens = generate_ensemble_members(n, nobs, N, xlon, xlat, ylon, ylat, Xb, R, d, errb, interpolator)

    # Loop over model grid points
    for l in range(n):
        
        if obs_indices[l,:].any():
    
            # Indices of observations selected for the n-th grid point
            index = np.nonzero(obs_indices[l,:])[0]
            
            # Number of observations for the l-th grid point
            nobs_l = len(index) 
                      
            Yb_l = Yb[index,:]
            d_l = d[index]
            R_l = R[index,:][:,index]

            if option == "ens":
                YbYb = Yb_l @ Yb_l.T
                work_Yb = YbYb @ inv(YbYb + (N - 1) * R_l)
                for i in range(nobs_l):
                    count[index[i]] += 1
                    DFS[index[i]] += work_Yb[i,i]
                    
            if option == 'vec':
                d_ens_l = d_ens[index,:]
                _, erra[l], K = enkf(N, Xb[l,:], Yb_l, R_l, d_l, errb[l], d_ens_l, errb_ens[l,:])
                                    
    if option == "ens":
        return(DFS, count)
    
    if option == "vec":
        # The innovation, residual and increment vectors  
        ya = H_operator_linear(xlon, xlat, erra, interpolator)
        
        omb = d + np.mean(obs_pert,axis=1)
        oma = erro - ya(ylon, ylat) + np.mean(obs_pert, axis=1)
        # Require linear observation operator
        amb = d - oma 
            
        return(omb, oma, amb, np.std(erra))