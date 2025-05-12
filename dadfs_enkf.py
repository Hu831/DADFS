import numpy as np
from numpy.linalg import inv

from enkf import enkf
from innovation import innovation
from H_operator_linear import H_operator_linear
from generate_ensemble_members import generate_ensemble_members
from get_local_matrices import get_local_matrices

def dadfs_enkf(args, grid, nens, Xb, Yb, B, R, obs_indices, option):
    """
    Perform EnKF update and estimate Degrees of Freedom for Signal (DFS).

    Depending on the selected option, this function either:
    - Estimates the theoretical DFS using background ensemble perturbations,
    - Or returns innovation, residual and increment vectors.

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
    Xb : ndarray of shape (n, N)
        Background ensemble perturbation matrix.
    Yb : ndarray of shape (nobs, N)
        Background ensemble perturbation matrix in observation space.
    B : ndarray of shape (n, n)
        Background error covariance matrix.
    R : ndarray of shape (nobs, nobs)
        Observation error covariance matrix.
    obs_indices : ndarray of shape (n, nobs)
        Localization mask: 1 if obs i is used for grid point l, else 0.
    option : str
        'HK' to estimate theoretical DFS;
        'assimilation' to returns innovation, residual and increment vectors.

    Returns
    -------
    If option == 'HK':
        DFS : ndarray of shape (nobs,)
            Theoretical DFS estimate from trace(HK).

    If option == 'assimilation':
        omb : ndarray of shape (nobs,)
            Innovation vector (observation - background).
        oma : ndarray of shape (nobs,)
            Residual vector (observation - analysis).
        amb : ndarray of shape (nobs,)
            Observation-space increment (analysis - background).
        erra_std : float
            Standard deviation of analysis error at model grid points.
    """

    nobs = grid["nobs"]
    xlon, xlat, ylon, ylat = grid["xlon"], grid["xlat"], grid["ylon"], grid["ylat"]
    n = args.n
    interpolator = args.interpolator

    match option:
    
        case "HK":
            _, _, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)

            DFS  = np.zeros(nobs)  
        
            # Loop over model grid points
            for l in range(n):
                if obs_indices[l, :].any():
                    Yb_l, d_l, R_l, index = get_local_matrices(Yb, d, R, obs_indices[l, :])
                    YbYb = Yb_l @ Yb_l.T
                    work_Yb = YbYb @ inv(YbYb + (nens - 1) * R_l)
                    for i in range(len(index)):
                        DFS[index[i]] += work_Yb[i, i]
            return(DFS)
    
        case "assimilation":
            errb, erro, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)
            erra = errb.copy()

            # Generate ensemble members
            obs_pert, d_ens, errb_ens = generate_ensemble_members(
                n, nobs, nens, xlon, xlat, ylon, ylat, Xb, R, d, errb, interpolator
            )

            for l in range(n):
                if obs_indices[l, :].any():
                    Yb_l, d_l, R_l, index = get_local_matrices(Yb, d, R, obs_indices[l, :])
                    d_ens_l = d_ens[index, :]
                    _, erra[l], _ = enkf(
                        nens, Xb[l, :], Yb_l, R_l, d_l, errb[l], d_ens_l, errb_ens[l, :]
                    )
                    
            # Compute innovation, residual, and increment in observation space
            ya = H_operator_linear(xlon, xlat, erra, interpolator)
            omb = d + np.mean(obs_pert, axis=1)              # O - B
            oma = erro - ya(ylon, ylat) + np.mean(obs_pert, axis=1)  # O - A
            amb = d - oma                                    #  A - B
            erra_std = np.std(erra)
                
            return omb, oma, amb, erra_std