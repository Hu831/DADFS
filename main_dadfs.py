#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical experiments on the use of degrees of freedom for signal in the 
evaluation of observation influence in numerical weather forecasting

Author: Guannan Hu

"""

import numpy as np

from spatial_distribution import spatial_distribution
from corr_func_2d import corr_func_2d
from XbYb import XbYb
from find_local_observations import find_local_observations
from run_dadfs_method import run_dadfs_method
from run_dadfs_method_realizations import run_dadfs_method_realizations
from output import output

#%% Set up parameters 

# Parameters that vary in our experiments

DA_method = "LETKF"  # Data assimilation algorithm; "LETKF" or "EnKF"
N = 20               # Ensemble size 
Nd = 10              # Sample size of the innovation-based and weighting-vector-based approaches
r = 1300             # Localization radius
corr_o = "FOAR"      # Observation error correlation function; "diag" or "FOAR"
l_o = 20             # Observation error correlation lengthscale (km); used when corr_o = "FOAR"
distrib = 5          # Spatial distribution of observations (default 5); for detail see spatial_distribution.py
start_point = 0      # Random number seed to start with
avg = 1              # How many runs after the start_point

# Parameters that are fixed in our experiments

rseed_ctrl = 1               # Fixed random number seed for reproducibility (default 1)
save = 1                     # "1" - save the results
nobs, nx, ny = 50, 20, 10    # Number of observations and number of model grid points in x and y directions
xrange = (-5, 5)             # Western and eastern boundaries of the domain (longitudes)
yrange = (-2.5, 2.5)         # Southern and Northern boundaries of the domain (latitudes)
n = nx * ny                  # Total number of model grid points
corr_b = "SOAR"              # Backgroud error correlation function
l_b = 80                     # Backgroud error correlation lengthscale in km (default 80)
sigma_b, sigma_o = 1.0, 1.0  # Background and observation error standard deviations (default 1.0, 1.0)
interpolator = 'linear'      # Interpolation (observation operator); "linear" or "nearest" 
dist_type = "great circle"   # Distance between two points; "great circle" or "chordal"

#%% Generate model grid points and observation locations
np.random.seed(rseed_ctrl) 

xlon, xlat, ylon, ylat, x_grid_length, y_grid_length, nobs = spatial_distribution(nx, ny, nobs, xrange, yrange, distrib, 0)

# Generate observation and background error covariance matrices
_, R, cond_R = corr_func_2d(ylon, ylat, dist_type, sigma_o, l_o, corr_o)
_, B, cond_B = corr_func_2d(xlon, xlat, dist_type, sigma_b, l_b, corr_b)

# Print the condition number of the background error covariance matrix
print('-----')
print('Condition number of R:', cond_R)
print('Condition number of B:', cond_B)

# Fixed random number seed for background ensemble perturbations
np.random.seed(rseed_ctrl) 

# Generate the ensemble perturbation matrices
Xb, Yb = XbYb(n, nobs, N, B, xlon, xlat, ylon, ylat, interpolator)

# Find observations used for each model grid point when domain localization is used
obs_indices = find_local_observations(n, nobs, xlon, xlat, ylon, ylat, r) 

#%% Run data assimilation experiments and estimate the DFS

# Estimate the theoretical DFS using ensemble perturbations
DFS_Ya, count = run_dadfs_method(DA_method, n, nobs, N, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator, 'ens')
    
# Estimate the DFS using the innovation-based and weighting-vector-based approaches
DFS_w_avg, DFS_d_theo, DFS_d_act, DFS_d_alt, erra, rseed_obs = run_dadfs_method_realizations(DA_method, n, nobs, N, Nd, r, avg, start_point, xlon, xlat, ylon, ylat, Xb, Yb, B, R, obs_indices, interpolator)
    
#%% Print and save results
output(DFS_Ya, DFS_d_act, DFS_d_theo, DFS_w_avg, DFS_d_alt, count, erra, save, distrib, N, Nd, r, DA_method, rseed_obs, corr_o, l_o, ylon, ylat)
