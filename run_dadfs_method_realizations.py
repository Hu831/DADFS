#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:06:48 2024

Author: Guannan Hu
"""

import numpy as np
from run_dadfs_method import run_dadfs_method 
from d_approach import d_approach

def run_dadfs_method_realizations(DA_method, n, nobs, N, Nd, r, avg, start_point, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator):
    """Run experiments with different random number seeds."""
    for i in range(avg):
        rseed_obs = i + start_point
        np.random.seed(rseed_obs)
        print(f'-----\nRandom number seed: {rseed_obs}')
        
        omb, oma, amb, erra, DFS_w = (np.zeros((Nd, nobs)) for _ in range(5))
        
        for k in range(Nd):
            result = run_dadfs_method(DA_method, n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'vec')
            if DA_method == "LETKF":
                DFS_w[k,:], omb[k,:], oma[k,:], amb[k,:], erra[k] = result
            elif DA_method == "EnKF":
                omb[k,:], oma[k,:], amb[k,:], erra[k] = result
        
        # Average over realisations of d
        DFS_w_avg = np.mean(DFS_w, axis=0)
        
        # Calculate DFS_d_act and DFS_d_theo
        DFS_d_theo, DFS_d_act, DFS_d_alt = d_approach(n, nobs, N, xlon, xlat, ylon, ylat, omb, amb, oma, obs_indices, Nd, R)
    
    return DFS_w_avg, DFS_d_theo, DFS_d_act, DFS_d_alt, erra, rseed_obs