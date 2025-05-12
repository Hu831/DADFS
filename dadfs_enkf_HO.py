import numpy as np
from numpy.linalg import eigh

from H_operator_linear import H_operator_linear
from get_local_matrices import get_local_matrices

def dadfs_enkf_HO(args, grid, nens, Xb, Yb, B, R, obs_indices, option):
    """
    Perform Ensemble Kalman Filter update and estimate DFS using the 
    Hotta & Ota (2021) approach.

    This function computes the global Kalman gain matrix `gK`, projects it 
    through the observation operator to obtain `HK`, and extracts the 
    diagonal elements to compute DFS (Degrees of Freedom for Signal).

    Two modes are supported:
    - 'Pb': Uses ensemble-based background error covariances, `Xb` and `Yb`.
    - 'B' : Uses the static background error covariance matrix `B`.

    Parameters
    ----------
    args : ExperimentConfig
        Configuration object containing general experiment settings, including:
        - n : int
            Number of model grid points.
        - interpolator : str
            Type of interpolation for the observation operator ('linear' or 'nearest').

    grid : dict
        Dictionary containing spatial coordinate arrays:
        - 'xlon', 'xlat' : model grid point coordinates (length n)
        - 'ylon', 'ylat' : observation point coordinates (length nobs)
        - 'nobs' : number of observations

    nens : int
        Number of ensemble members.

    Xb : ndarray, shape (n, nens)
        Background ensemble perturbation matrix in model space.

    Yb : ndarray, shape (nobs, nens)
        Background ensemble perturbation matrix in observation space.

    B : ndarray, shape (n, n)
        Static background error covariance matrix.

    R : ndarray, shape (nobs, nobs)
        Observation error covariance matrix.

    obs_indices : ndarray, shape (n, nobs)
        Binary matrix indicating which observations are used at each model grid point.

    option : str
        Either 'Pb' (for ensemble-based background covariance) or 'B' (for static).

    Returns
    -------
    gK : ndarray, shape (n, nobs)
        Global Kalman gain matrix (assembled from localized gains).

    HK : ndarray, shape (nobs, nobs)
        Product of the observation operator and the Kalman gain matrix.

    DFS_HO : ndarray, shape (nobs,)
        Degrees of Freedom for Signal estimated as diag(HK).
    """
    
    nobs = grid["nobs"]
    xlon, xlat, ylon, ylat = grid["xlon"], grid["xlat"], grid["ylon"], grid["ylat"]
    n = args.n
    interpolator = args.interpolator
    
    gK = np.zeros((n, nobs))  # Global Kalman gain

    match option:

        case "Pb":
            # Loop over model grid points to compute local Kalman gain
            for l in range(n):
                if obs_indices[l, :].any():
                    Yb_l, _, R_l, index = get_local_matrices(Yb, np.zeros(nobs), R, obs_indices[l, :])
                    
                    # Eigen-decomposition of innovation covariance matrix
                    val, vec = eigh(Yb_l @ Yb_l.T + (nens - 1) * R_l)

                    # Local Kalman gain for this grid point
                    gK[l, index] = Xb[l, :] @ Yb_l.T @ (vec @ np.diag(1.0 / val) @ vec.T)

        case "B":
            # Compute BH and HBH globally using observation operator
            BH = np.zeros((n, nobs))
            HBH = np.zeros((nobs, nobs))

            for i in range(n):
                h = H_operator_linear(xlon, xlat, B[i, :], interpolator)
                BH[i, :] = h(ylon, ylat)

            for i in range(nobs):
                h = H_operator_linear(xlon, xlat, BH[:, i], interpolator)
                HBH[:, i] = h(ylon, ylat)

            # Loop over model grid points to compute local Kalman gain using B
            for l in range(n):
                if obs_indices[l, :].any():
                    index = np.nonzero(obs_indices[l, :])[0]
                    HBH_l = HBH[np.ix_(index, index)]
                    R_l = R[np.ix_(index, index)]

                    val, vec = eigh(HBH_l + R_l)
                    gK[l, index] = BH[l, index] @ (vec @ np.diag(1.0 / val) @ vec.T)

        case _:
            raise ValueError("Invalid option. Choose 'Pb' or 'B'.")

    # Apply observation operator to Kalman gain columns to get HK
    HK = np.zeros((nobs, nobs))
    for i in range(nobs):
        h = H_operator_linear(xlon, xlat, gK[:, i], interpolator)
        HK[i, :] = h(ylon, ylat)

    DFS_HO = np.diag(HK)  # Hotta & Ota DFS estimate (diagonal of H*K)

    return gK, HK, DFS_HO