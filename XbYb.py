import numpy as np
from H_operator_linear import H_operator_linear

def XbYb(args, grid, nens, B,):
    """
    Generate ensemble perturbation matrices in model and observation space.

    This function generates an ensemble of background perturbations `Xb` using 
    a specified background error covariance matrix `B`. It also applies an 
    observation operator to each ensemble member to generate `Yb`, the 
    perturbations projected into observation space.

    Parameters
    ----------
    args : ExperimentConfig
        A configuration object that provides:
            - n : number of model grid points
            - interpolator : type of interpolation (e.g., 'linear', 'nearest')
            - rseed_ctrl   : random seed for reproducibility
            
    grid : dict
        - xlon : longitudes for all model grid points.
        - xlat : latitudes for all model grid points.
        - ylon : longitudes for observation locations.
        - ylat : latitudes for observation locations.
        - nobs : number of observation points 

    nens : int
        Ensemble size.
    B : ndarray of shape (n, n)
        Background error covariance matrix.

    Returns
    -------
    Xb : ndarray of shape (n, N)
        Ensemble perturbation matrix in model space.
    Yb : ndarray of shape (nobs, N)
        Ensemble perturbation matrix in observation space.
    """
    
    np.random.seed(args.rseed_ctrl)
    
    # Generate ensemble samples from multivariate normal distribution ~ N(0, B)
    Xb = np.random.multivariate_normal(mean=np.zeros(args.n), cov=B, size=nens).T

    # Compute the sample mean to remove sampling bias
    sample_mean = np.mean(Xb, axis=1)

    # Allocate memory for the perturbation matrix in observation space
    Yb = np.zeros((grid["nobs"], nens))
    
    for k in range(nens):
        # Subtract mean from each member to center ensemble
        Xb[:, k] -= sample_mean

        # Apply the linear observation operator to obtain observation space perturbations
        h = H_operator_linear(grid["xlon"], grid["xlat"], Xb[:, k], args.interpolator)
        Yb[:, k] = h(grid["ylon"], grid["ylat"])
        
    return Xb, Yb