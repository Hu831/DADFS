#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical experiments on the use of degrees of freedom for signal in the 
evaluation of observation influence in numerical weather forecasting

Author: Guannan Hu

"""

import numpy as np

from corr_func_2d import corr_func_2d
from d_approach import d_approach
from dadfs_etkf import dadfs_etkf
from dadfs_enkf import dadfs_enkf
from output import output

from spatial_distribution import spatial_distribution
from XbYb import XbYb
from find_local_observations import find_local_observations

#%% Set up parameters 

# Parameters that vary in our numerical experiments

# Data assimilation algorithm; "LETKF" or "EnKF"
DA_method = "LETKF"    

# Ensemble size 
N = 20       

# Sample size of the innovation-based and weighting-vector-based approaches
Nd = 100            

# Localization radius
r = 1300    

# Observation error correlation function; "diag" or "FOAR"
corr_o = "FOAR" 

# Observation error correlation lengthscale (km); used when corr_o = "FOAR"
l_o = 20

# Spatial distribution of observations (default 5)
# For detail see spatial_distribution.py
distrib = 5

# Random number seed to start with
start_point = 0      

# How many runs after the start_point
avg = 1         

# Fixed parameters in our numerical experiments

# Fixed random number seed for reproducibility (default 1)
rseed_ctrl = 1       

# Number of observations and model grid
nobs, nx, ny = 50, 20, 10  
xrange = (-5, 5)
yrange = (-2.5, 2.5)
n = nx * ny
        
# Backgroud error correlation function
corr_b = "SOAR" 

# Backgroud error correlation lengthscale in km (default 80)
l_b = 80              

# Background and observation error standard deviations (default 1.0, 1.0)
sigma_b, sigma_o = 1.0, 1.0  

# Whether to save the results; "0" means no and "1" means yes
save = 1                     

# Interpolation (observation operator); "linear" or "nearest" 
interpolator = 'linear'     

# Distance between two points; "great circle" or "chordal"
dist_type = "great circle"   

#%% Generate model grid points and observation locations
np.random.seed(rseed_ctrl) 

xlon, xlat, ylon, ylat, x_grid_length, y_grid_length, nobs = \
    spatial_distribution(nx, ny, nobs, xrange, yrange, distrib, 0)

#%% Generate observation and background error covariance matrices
_, R, cond_R = corr_func_2d(ylon, ylat, dist_type, sigma_o, l_o, corr_o)
_, B, cond_B = corr_func_2d(xlon, xlat, dist_type, sigma_b, l_b, corr_b)

# Print the condition number of the background error covariance matrix
print('-----')
print('Condition number of R:', cond_R)
print('Condition number of B:', cond_B)

#%% Estimate the theoretical DFS using ensemble perturbations
np.random.seed(rseed_ctrl) 

# Generate the ensemble perturbation matrices
Xb, Yb = XbYb(n, nobs, N, B, xlon, xlat, ylon, ylat, interpolator)

# Find observations used for each model grid point when domain localization is used
obs_indices = find_local_observations(n, nobs, xlon, xlat, ylon, ylat, r) 

# Estimate the DFS
if DA_method == "LETKF":
    DFS_Ya, count = dadfs_etkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'ens')

if DA_method == 'EnKF':
    DFS_Ya, count = dadfs_enkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'ens')

#%% Estimate the DFS using the innovation-based and weighting-vector-based approaches

# Loop over different DA runs 
for i in range(avg):
    
    rseed_obs = i + start_point
    print('-----')
    print('Random number seed:', rseed_obs)
        
    np.random.seed(rseed_obs) 

    omb = np.zeros((Nd, nobs))
    oma = np.zeros((Nd, nobs))
    amb = np.zeros((Nd, nobs))
    erra = np.zeros((Nd))
    
    DFS_w = np.zeros((Nd, nobs))
    for k in range(Nd):
        if DA_method == "LETKF":
            # Calculate DFS_w, and collect O-B, O-A and A-B vectors 
            DFS_w[k,:], omb[k,:], oma[k,:], amb[k,:], erra[k] = dadfs_etkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'vec')
            
        if DA_method == "EnKF":
            # Collect O-B, O-A and A-B vectors 
            omb[k,:], oma[k,:], amb[k,:], erra[k] = dadfs_enkf(n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'vec')
           
    # Average over realisations of d
    DFS_w_avg = np.mean(DFS_w, axis=0)
    
    # Calculate DFS_d_act and DFS_d_theo
    DFS_d_theo, DFS_d_act, DFS_d_alt = d_approach(n, nobs, N, xlon, xlat, ylon, ylat, omb, amb, oma, obs_indices, Nd, R)
    

#%% Print and save results
output(DFS_Ya, DFS_d_act, DFS_d_theo, DFS_w_avg, DFS_d_alt, count, erra, save, distrib, N, Nd, r, DA_method, rseed_obs, l_o, ylon, ylat)
