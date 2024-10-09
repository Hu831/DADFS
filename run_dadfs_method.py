#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:06:48 2024

Author: Guannan Hu
"""

from dadfs_etkf import dadfs_etkf
from dadfs_enkf import dadfs_enkf

def run_dadfs_method(DA_method, n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, DFS_approach):
    """Run a data assimilation algorithm (LETKF or EnKF) and return the DFS with different approaches."""
    if DA_method == "LETKF":
        if DFS_approach == 'ens':
            return dadfs_etkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, DFS_approach)
        if DFS_approach == 'vec':
            return dadfs_etkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, DFS_approach)
    elif DA_method == "EnKF":
        if DFS_approach == 'ens':
            return dadfs_enkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, DFS_approach)
        if DFS_approach == 'vec':
            return dadfs_enkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, DFS_approach)