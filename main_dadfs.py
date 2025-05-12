'''
====================================================================================
 Main Script: Estimation of Degrees of Freedom for Signal (DFS) in LETKF and EnKF 
              Data Assimilation Experiments

 This script implements multiple approaches for estimating the DFS, including 
 ensemble-based, innovation-based, and weight-vector-based approaches. 
 
 The code is used to generate the numerical results presented in:
 Hu et al. (2025), revised and submitted.

====================================================================================
'''

import numpy as np
import fixed_experiment_parameters as args
from spatial_distribution import spatial_distribution
from corr_func_2d import corr_func_2d
from XbYb import XbYb
from find_local_observations import find_local_observations
from dadfs_etkf import dadfs_etkf
from dadfs_enkf import dadfs_enkf
from dadfs_enkf_B import dadfs_enkf_B
from dadfs_enkf_HO import dadfs_enkf_HO
from d_approach import d_approach
from output import output

# -------------------------------
# Parameters to be changed in our experiments
# -------------------------------
DA_method = "EnKF"            # Choose: "EnKF" or "LETKF"
nens = 20                     # Ensemble size
Nd = 100                      # Number of innovation-based samples
r = 1300                      # Localization radius in km
distrib = 5                   # Observation spatial distribution type

# DFS estimation approaches (on/off switches)
approaches = {
    "Ya_approach":      1,
    "Ya_approach_B":    1,
    "w_approach":       1,
    "d_approaches":     1,
    "d_approaches_B":   1,
    "HO_approach":      1,
}

start_seed = 0                # Random number seed 
avg = 1                       # Number of independent trials to average
save = 0                      # Save results to CSV (1 = yes)
plot_obs_distribution = 0     # Whether or not to plot the distribution of observations

# -------------------------------
# Begin experiment
# -------------------------------
assert DA_method in ['EnKF', 'LETKF']

print("=== DFS Estimation Experiment ===")
print(f"DA Method: {DA_method}")
print(f"Ensemble Size: {nens}")
print(f"Sample Size: {Nd}")
print(f"Localization radius: {r} km")
print(f"Obs Distribution: type {distrib}")
print("=================================\n")

# --- Generate model grid points and observation locations
grid = spatial_distribution(args, distrib, plot_obs_distribution)
    
# --- Generate observation and background error covariance matrices
R = corr_func_2d(args, grid, "R")[1]
B = corr_func_2d(args, grid, "B")[1]

# --- Generate ensemble perturbation matrices
Xb, Yb = XbYb(args, grid, nens, B)

# --- Find which observations are used for each grid point
obs_indices, count = find_local_observations(args, grid, r)

# --- Initialize DFS arrays
nobs = grid["nobs"]
(
DFS_Ya, DFS_Ya_B, DFS_w, DFS_d_theo, DFS_d_theo_B, DFS_d_act, 
 DFS_d_act_B, DFS_d_alt, DFS_d_alt_B, DFS_HO
) = (np.zeros(nobs) for _ in range(10))

# --- Hotta & Ota (2021) approach
if approaches['HO_approach']:
    DFS_HO = dadfs_enkf_HO(args, grid, nens, Xb, Yb, B, R, obs_indices,'B')[2]

# --- Ensemble perturbation-based DFS
if approaches['Ya_approach']:
    if DA_method == "LETKF":
        DFS_Ya = dadfs_etkf(args, grid, nens, Xb, Yb, B, R, obs_indices, 'HK')
    else:
        DFS_Ya = dadfs_enkf(args, grid, nens, Xb, Yb, B, R, obs_indices, 'HK')
        
# --- Ensemble perturbation-based DFS (true B)
if approaches['Ya_approach_B']:
    DFS_Ya_B = dadfs_enkf_B(args, grid, B, R, obs_indices, 'HK')

# --- Loop over avg runs for innovation-based and weighting-vector-based approaches
for i in range(avg):
    rseed_obs = start_seed + i
    np.random.seed(rseed_obs)
    print(f">>> Realization #{i + 1} with seed = {rseed_obs}")

    omb, oma, amb, DFS_w_nd, omb_B, oma_B, amb_B, erra_B, erra = (
        np.zeros((Nd, nobs)) for _ in range(9))
    
    if DA_method == "LETKF":
        if approaches['d_approaches'] or approaches['w_approach']:
            for k in range(Nd):
                DFS_w_nd[k, :], omb[k, :], oma[k, :], amb[k, :], erra[k] = dadfs_etkf(
                    args, grid, nens, Xb, Yb, B, R, obs_indices, 'assimilation')
                
            if approaches['w_approach']:
                DFS_w = np.mean(DFS_w_nd, axis=0)
                
            if approaches['d_approaches']:
                DFS_d_theo, DFS_d_act, DFS_d_alt = d_approach(args.n, nobs, omb, amb, oma, obs_indices, Nd, R)
    else:
        if approaches['d_approaches']:
            for k in range(Nd):
                omb[k, :], oma[k, :], amb[k, :], erra[k] = dadfs_enkf(
                    args, grid, nens, Xb, Yb, B, R, obs_indices, 'assimilation')

            DFS_d_theo, DFS_d_act, DFS_d_alt = d_approach(args.n, nobs, omb, amb, oma, obs_indices, Nd, R)

    if approaches['d_approaches_B']:
        for k in range(Nd):
            omb_B[k, :], oma_B[k, :], amb_B[k, :], erra_B[k] = dadfs_enkf_B(
                args, grid, B, R, obs_indices, 'assimilation')
    
        DFS_d_theo_B, DFS_d_act_B, DFS_d_alt_B = d_approach(
            args.n, nobs, omb_B, amb_B, oma_B, obs_indices, Nd, R)
        
    # -------------------------------
    # Print and optionally save the DFS estimates
    # -------------------------------
    output(
        DFS_Ya, DFS_d_act, DFS_d_theo, DFS_w, DFS_d_alt,
        DFS_Ya_B, DFS_d_theo_B, DFS_d_act_B, DFS_d_alt_B, DFS_HO,
        count, erra, save, distrib, nens, Nd, r, DA_method,
        rseed_obs, args.corr_o, args.l_o, grid["ylon"], grid["ylat"]
    )