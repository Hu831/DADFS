#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The innovation-based approach for estimating the DFS

Author: Guannan Hu
"""

import numpy as np
from numpy import linalg as LA

def d_approach(n, nobs, N, xlon, xlat, ylon, ylat, omb, amb, oma, obs_indices, Nd, R):
    """ Estimate the DFS using the innovation (O-B), residual (O-A) and 
    observation-space increment (A-B) vectors.
    
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
    Nd : int
        Size of the sample of the innovation (O-B), residual (O-A) and 
        observation-space increment (A-B) vectors.
    omb : Nd x nobs array
        The innovation (O-B) vector.
    oma : Nd x nobs array
        The residual (O-A) vector.
    amb : Nd x nobs array
        The observation-space increment vector.    
    obs_indices : n x nobs matrix
        Contain 1s and 0s
        The (l,i)th element is 1 if the ith observation is used for the lth 
        grid point, and 0 otherwise.
    R : nobs x nobs array
        The observation error covariance matrix.

    Returns
    -------
    DFS_d_theo : nobs x 1 array
        The theoretical DFS estimated using the original innovation-based 
        approach.
    DFS_d_act : nobs x 1 array
        The actual DFS estimated using the original innovation-based approach.
    DFS_d_alt : nobs x 1 array
        The theoretical DFS estimated using the new innovation-based approach.
    """
        
    DFS_d_theo = np.zeros(nobs)      
    DFS_d_act  = np.zeros(nobs)
    DFS_d_alt  = np.zeros(nobs)

    for l in range(n):
    
        if obs_indices[l,:].any():
        
            # Indices of observations selected for the l-th model grid point
            index = np.nonzero(obs_indices[l,:])[0]
                        
            exp_up = np.zeros((len(index),len(index)))
            exp_down = np.zeros((len(index),len(index)))
            
            exp_up_alt = np.zeros((len(index),len(index)))
            exp_down_alt =np.zeros((len(index),len(index)))
            
            R_l = R[index,:][:,index]
            val, vec = LA.eigh(R_l)
            sqrt_invR_l = vec @ np.diag(np.sqrt(1.0/val)) @ vec.T
                    
            for j in range(Nd):
                
                omb_l = sqrt_invR_l @ omb[j,index] 
                amb_l = sqrt_invR_l @ amb[j,index] 
                oma_l = sqrt_invR_l @ oma[j,index] 
                
                exp_up   += np.outer(amb_l, oma_l)
                exp_down += np.outer(omb_l, oma_l)
                
                exp_up_alt   += np.outer(amb_l, omb_l)
                exp_down_alt += np.outer(omb_l, omb_l)
        
            exp_up /= Nd
            exp_down /= Nd
            exp_up_alt /= Nd
            exp_down_alt /= Nd
        
            work = exp_up @ LA.inv(exp_down)
                     
            work_alt = exp_up_alt @ LA.inv(exp_down_alt)
            
            for i in range(len(index)):
                DFS_d_act[index[i]]  += exp_up[i,i]
                DFS_d_theo[index[i]] += work[i,i]
                DFS_d_alt[index[i]]  += work_alt[i,i]

    return(DFS_d_theo, DFS_d_act, DFS_d_alt)