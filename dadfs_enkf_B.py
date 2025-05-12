import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigh

from innovation import innovation
from H_operator_linear import H_operator_linear

def dadfs_enkf_B(args, grid, B, R, obs_indices, option):
    """
    Perform EnKF update and estimate Degrees of Freedom for Signal (DFS) using true B.

    This function computes either:
    - The theoretical DFS, or
    - Innovation, residual, and increment vectors.

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
        
    B : ndarray of shape (n, n)
        True background error covariance matrix.
        
    R : ndarray of shape (nobs, nobs)
        Observation error covariance matrix.
        
    obs_indices : ndarray of shape (n, nobs)
        Localization mask; 1 if obs i is used at grid point l, else 0.
        
    option : str
        - 'HK': Estimate theoretical DFS;
        - 'assimilation': return nnovation, residual, and increment vectors.

    Returns
    -------
    If option == 'HK':
        DFS : ndarray of shape (nobs,)
            Theoretical DFS estimate for each observation.

    If option == 'assimilation':
        omb : ndarray of shape (nobs,)
            Innovation vector (O - B).
        oma : ndarray of shape (nobs,)
            Residual vector (O - A).
        amb : ndarray of shape (nobs,)
            Analysis increment in obs space (A - B).
        erra_std : float
            Standard deviation of analysis error in model space.
    """
    
    nobs = grid["nobs"]
    xlon, xlat, ylon, ylat = grid["xlon"], grid["xlat"], grid["ylon"], grid["ylat"]
    n = args.n
    interpolator = args.interpolator
    
    DFS  = np.zeros(nobs)      

    # Generate synthetic innovation and background/obs error vectors
    errb, erro, _, d = innovation(n, nobs, B, R, xlon, xlat, ylon, ylat, interpolator)
    erra = errb.copy()
    
    # Calculate BH and HBH
    BH = np.zeros((n, nobs))
    HBH = np.zeros((nobs, nobs))
    
    for i in range(n):
        h = H_operator_linear(xlon, xlat, B[i, :], interpolator)
        BH[i, :] = h(ylon, ylat)
       
    for i in range(nobs):
        h = H_operator_linear(xlon, xlat, BH[:, i], interpolator)
        HBH[:, i] = h(ylon, ylat) 

    # Loop over model grid points for local updates
    for l in range(n):
        if obs_indices[l,:].any():
            index = np.nonzero(obs_indices[l, :])[0]
            HBH_l = HBH[np.ix_(index, index)]
            R_l = R[np.ix_(index, index)]
            
            if option == "HK":
                work_Yb = HBH_l @ inv(HBH_l + R_l)
                DFS[index] += np.diag(work_Yb)
                  
            if option == 'assimilation':
                val, vec = eigh(HBH_l + R_l)
                K_l = BH[l, index] @ (vec @ np.diag(1.0 / val) @ vec.T)
                # Calculate the analysis error
                erra[l] = errb[l] + K_l @ d[index]
        
    if option == "HK":
        return(DFS)
    
    if option == "assimilation":
        # The innovation, residual and increment vectors  
        ya = H_operator_linear(xlon, xlat, erra, interpolator)
        omb = d
        oma = erro - ya(ylon, ylat) 
        amb = d - oma 
            
        return omb, oma, amb, np.std(erra)