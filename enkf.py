#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The EnKF data assimilation algorithm

Author: Guannan Hu
"""

import numpy as np
from numpy.linalg import eigh

def enkf(N, Xb, Yb, R, d, errb, d_ens, errb_ens):
    """ Perform the EnKF algorithm.
    
    Parameters
    ----------
    N : int
        Ensemble size.
    Xb : numpy.ndarray
        The background ensemble perturbation matrix.
    Yb : numpy.ndarray
        The background ensemble perturbation matrix in observation space.
    R : numpy.ndarray
        The observation error covariance matrix.
    d : numpy.ndarray
        The innovation vector.
    errb : numpy.ndarray
        The backgroud error vector.
    d_ens : numpy.ndarray
        An ensemble of the innovation vector.
    errb_ens : numpy.ndarray
        An ensemble of the background error vector.

    Returns
    -------
    erra_ens : numpy.ndarray
        An ensemble of the analysis error vector
    erra_mean : numpy.ndarray or float
        Ensemble mean of the analysis
    K : numpy.ndarray
        The Kalman gain
    """
    
    # Compute the Kalman gain
    val, vec = eigh(Yb @ Yb.T + (N - 1) * R)
    K = Xb @ Yb.T @ (vec @ np.diag(1.0/val) @ vec.T)
    
    # Calculate the analysis error
    if Xb.ndim == 1:
        erra_ens = np.zeros(N)
        for k in range(N):
            erra_ens[k] = errb_ens[k] + K @ d_ens[:,k]
        erra_mean = np.mean(erra_ens)
        return(erra_ens, erra_mean, K) 
    else:
        n = np.shape(Xb)[0]
        erra_ens = np.zeros((n,N))
        for k in range(N):
            erra_ens[:,k] = errb_ens[:,k] + K @ d_ens[:,k]
        erra_mean = np.mean(erra_ens,axis=1)
        return(erra_ens, erra_mean, K)
 
   
    



