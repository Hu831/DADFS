import numpy as np
from numpy.linalg import eigh

from etkf import etkf
from innovation import innovation
from H_operator_linear import H_operator_linear
from generate_ensemble_members import generate_ensemble_members
from get_local_matrices import get_local_matrices

def dadfs_etkf(args, grid, nens, Xb, Yb, B, R, obs_indices, option):
    """
    Perform Local Ensemble Transform Kalman Filter (LETKF) update and 
    estimate the Degrees of Freedom for Signal (DFS).

    This function estimates either:
    - Theoretical DFS using analysis ensemble perturbations in observation space (`option='HK'`), or
    - Actual DFS using the weighting-vector-based approach, also returning innovation, increment and residual vectors (`option='assimilation'`).

    Parameters
    ----------
    args : ExperimentConfig
        Configuration object containing global experiment settings, including:
        - n : int
            Number of model grid points.
        - interpolator : str
            Observation operator interpolation method ('linear' or 'nearest').

    grid : dict
        Dictionary containing spatial coordinates and observation count:
        - 'xlon', 'xlat' : model grid point longitudes and latitudes.
        - 'ylon', 'ylat' : observation point longitudes and latitudes.
        - 'nobs' : number of observations.

    nens : int
        Ensemble size.

    Xb : ndarray of shape (n, nens)
        Background ensemble perturbation matrix in model space.

    Yb : ndarray of shape (nobs, nens)
        Background ensemble perturbation matrix in observation space.

    B : ndarray of shape (n, n)
        Background error covariance matrix.

    R : ndarray of shape (nobs, nobs)
        Observation error covariance matrix.

    obs_indices : ndarray of shape (n, nobs)
        Binary matrix indicating which observations are used at each model grid point.

    option : str
        Mode of operation:
        - 'HK' : Estimate DFS using analysis ensemble perturbations.
        - 'assimilation' : Perform assimilation and return innovation/residual/increment vectors

    Returns
    -------
    If option == 'HK':
        DFS_theo : ndarray of shape (nobs,)
            Theoretical DFS estimated using analysis ensemble perturbations.

    If option == 'assimilation':
        DFS_act : ndarray of shape (nobs,)
            Actual DFS estimated using the weighting-vector-based approach.
        omb : ndarray of shape (nobs,)
            Innovation vector (observation - background).
        oma : ndarray of shape (nobs,)
            Residual vector (observation - analysis).
        amb : ndarray of shape (nobs,)
            Analysis increment in observation space (analysis - background).
        erra_std : float
            Standard deviation of analysis error in model space.
    """
    
    nobs = grid["nobs"]
    xlon, xlat, ylon, ylat = grid["xlon"], grid["xlat"], grid["ylon"], grid["ylat"]
    n = args.n
    interpolator = args.interpolator
    
    DFS_act = np.zeros(nobs)
    DFS_theo = np.zeros(nobs)
    
    # Generate background and observation errors and innovation
    errb, erro, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)
    erra = errb.copy()

    # Generate perturbed observations and background ensemble 
    # Only used for EnKF, but also run for LETKF to obtain the same set of random numbers
    obs_pert, d_ens, errb_ens = generate_ensemble_members(
        n, nobs, nens, xlon, xlat, ylon, ylat, Xb, R, d, errb, interpolator
    )
    
    # Loop over each model grid point for local ETKF update
    for l in range(n):
        if obs_indices[l,:].any():
            # Extract local obs info for grid point l
            Yb_l, d_l, R_l, index = get_local_matrices(Yb, d, R, obs_indices[l, :])

            # Perform ETKF locally: get the weighting vector and transform matrix
            w, W = etkf(nens, Yb_l, R_l, d_l)

            # Eigendecomposition of R_l for inverse computation
            val, vec = eigh(R_l)
            
            if option == 'HK':
                # Theoretical DFS
                invR_l = vec @ np.diag(1.0 / val) @ vec.T
                Ya_l = Yb_l @ W
                work_Ya = (1 / (nens - 1)) * Ya_l @ Ya_l.T @ invR_l
                for i in range(len(index)):
                    DFS_theo[index[i]] += work_Ya[i, i]
                    
            if option == 'assimilation':
                # Actual DFS
                sqrt_invR_l = vec @ np.diag(np.sqrt(1.0 / val)) @ vec.T
                work = sqrt_invR_l @ Yb[index, :] @ w
                left = sqrt_invR_l @ d[index] - work

                for i in range(len(index)):
                    DFS_act[index[i]] += left[i] * work[i]

                # Update analysis error at this grid point
                erra[l] = errb[l] + Xb[l, :] @ w

    if option == 'HK':
        return(DFS_theo)
    
    if option == 'assimilation':
        # The innovation, residual and increment vectors  
        ya = H_operator_linear(xlon, xlat, erra, interpolator)
        omb = d                        # O - B
        oma = erro - ya(ylon, ylat)    # O - A
        amb = d - oma                  # A - B
        return DFS_act, omb, oma, amb, np.std(erra)
    
    nobs = grid["nobs"]
    xlon, xlat, ylon, ylat = grid["xlon"], grid["xlat"], grid["ylon"], grid["ylat"]
    n = args.n
    interpolator = args.interpolator
    

    DFS_act = np.zeros(nobs)
    DFS_theo = np.zeros(nobs)
    
    # Generate background and observation errors and innovation
    errb, erro, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)
    erra = errb.copy()

    # Generate perturbed observations and background ensemble 
    # Only used for EnKF, but also run for LETKF to obtain the same set of random numbers
    obs_pert, d_ens, errb_ens = generate_ensemble_members(
        n, nobs, nens, xlon, xlat, ylon, ylat, Xb, R, d, errb, interpolator
    )
    
    # Loop over each model grid point for local ETKF update
    for l in range(n):
        if obs_indices[l,:].any():
            # Extract local obs info for grid point l
            Yb_l, d_l, R_l, index = get_local_matrices(Yb, d, R, obs_indices[l, :])

            # Perform ETKF locally: get the weighting vector and transform matrix
            w, W = etkf(nens, Yb_l, R_l, d_l)

            # Eigendecomposition of R_l for inverse computation
            val, vec = eigh(R_l)
            
            if option == 'HK':
                # Theoretical DFS
                invR_l = vec @ np.diag(1.0 / val) @ vec.T
                Ya_l = Yb_l @ W
                work_Ya = (1 / (nens - 1)) * Ya_l @ Ya_l.T @ invR_l
                for i in range(len(index)):
                    DFS_theo[index[i]] += work_Ya[i, i]
                    
            if option == 'assimilation':
                # Actual DFS
                sqrt_invR_l = vec @ np.diag(np.sqrt(1.0 / val)) @ vec.T
                work = sqrt_invR_l @ Yb[index, :] @ w
                left = sqrt_invR_l @ d[index] - work

                for i in range(len(index)):
                    DFS_act[index[i]] += left[i] * work[i]

                # Update analysis error at this grid point
                erra[l] = errb[l] + Xb[l, :] @ w

    if option == 'HK':
        return(DFS_theo)
    
    if option == 'assimilation':
        # The innovation, residual and increment vectors  
        ya = H_operator_linear(xlon, xlat, erra, interpolator)
        omb = d                        # O - B
        oma = erro - ya(ylon, ylat)    # O - A
        amb = d - oma                  # A - B
        return DFS_act, omb, oma, amb, np.std(erra)