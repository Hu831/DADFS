import numpy as np
import great_circle_calculator.great_circle_calculator as gcc
from math import sin
from numpy import linalg as LA

def corr_func_2d(args, grid, option):
    """
    Construct a spatial error covariance matrix using specified correlation 
    functions for either background (B) or observation (R) errors.

    Parameters
    ----------
    args : ExperimentConfig
        A configuration object containing parameters:
            - corr_b, l_b, sigma_b: for background error
            - corr_o, l_o, sigma_o: for observation error
            - dist_type: 'great circle' or 'chordal'
            - earth_radius: radius of Earth (used for chordal distance)

    grid : dict
        A dictionary containing coordinates:
            - For option == 'B': {'xlon', 'xlat'}
            - For option == 'R': {'ylon', 'ylat'}

    option : str
        'B' for background error covariance;
        'R' for observation error covariance.

    Returns
    -------
    dist_matrix : ndarray of shape (n, n)
        Matrix of pairwise distances between points.

    cov_matrix : ndarray of shape (n, n)
        Spatial error covariance matrix.

    cond : float
        Condition number of the covariance matrix (used to assess numerical stability).
    """
    
    if option == "B":
        lon = grid["xlon"]
        lat = grid["xlat"]
        n = len(grid["xlon"])
        func = args.corr_b
        l = args.l_b
        sigma = args.sigma_b
        
    if option == "R":
        lon = grid["ylon"]
        lat = grid["ylat"]
        n = len(grid["ylon"])
        func = args.corr_o
        l = args.l_o
        sigma = args.sigma_o
        
    # Compute distance matrix between all pairs of points
    dist_matrix = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            dist_km = gcc.distance_between_points((lon[i], lat[i]), (lon[j], lat[j]), unit='kilometers')
            if args.dist_type == 'great circle':
                dist_matrix[i, j] = dist_km
            elif args.dist_type == 'chordal':
                dist_matrix[i, j] = 2 * args.earth_radius * sin(dist_km / (2 * args.earth_radius))
            else:
                raise ValueError("dist_type must be 'great circle' or 'chordal'")
    
    # Initialize correlation matrix
    corr_matrix = np.zeros_like(dist_matrix)

    # Compute correlation matrix based on selected function
    if func == "SOAR":
        corr_matrix = (1 + dist_matrix / l) * np.exp(-dist_matrix / l)
    elif func == "FOAR":
        corr_matrix = np.exp(-dist_matrix / l)
    elif func == "diag":
        corr_matrix = np.eye(n)
    else:
        raise ValueError("func must be 'SOAR', 'FOAR', or 'diag'")
        
    # Scale by variance to get covariance matrix
    cov_matrix = sigma**2 * corr_matrix
    
    # Compute condition number of the covariance matrix
    cond = LA.cond(cov_matrix)
    
    return (dist_matrix, cov_matrix, cond)