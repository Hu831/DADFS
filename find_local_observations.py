import numpy as np
import great_circle_calculator.great_circle_calculator as gcc

def find_local_observations(args, grid, r):
    """
    Identify observations within a given radius for each model grid point.

    For each model grid point, this function checks which observations fall within
    the specified localization radius `r`, and records the relationships in a binary
    mask matrix. It also counts how many times each observation is used.

    Parameters
    ----------
    args : ExperimentConfig
        A configuration object that provides:
            - n    : number of model grid points
            - nobs : number of observation
            
    grid : dict 
        - xlon : Longitudes of model grid points.
        - xlat : Latitudes of model grid points.
        - ylon : Longitudes of observation points.
        - ylat : Latitudes of observation points.
            
    r : float
        Localization radius (in kilometers).

    Returns
    -------
    obs_indices : ndarray of shape (n, nobs)
        Binary matrix indicating which observations are within range of each grid point.
        Entry (l, i) is 1 if the ith observation is used for the lth grid point, 0 otherwise.
    count : ndarray of shape (nobs,)
        Number of times each observation is used across all grid points.
    """
    
    nobs = grid["nobs"]
    xlon = grid["xlon"]
    ylon = grid["ylon"]
    xlat = grid["xlat"]
    ylat = grid["ylat"]
    
    # Initialize binary mask matrix
    obs_indices = np.zeros((args.n, nobs))

    # Loop over all model grid points and observations
    for l in range(args.n):
        for i in range(nobs):
            dist = gcc.distance_between_points((xlon[l], xlat[l]), (ylon[i], ylat[i]), unit='kilometers')
            if dist <= r:
                obs_indices[l, i] = 1

    # Count how many grid points each observation contributes to
    count = np.sum(obs_indices, axis=0)
      
    return(obs_indices, count)