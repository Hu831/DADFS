import numpy as np
from numpy.linalg import eigh, inv

def d_approach(n, nobs, omb, amb, oma, obs_indices, Nd, R):
    """
    Estimate DFS using innovation-based approaches.

    This function computes theoretical and actual DFS using the innovation (O-B), 
    residual (O-A), and observation-space increment (A-B) vectors. 
    
    Parameters
    ----------
    n : int
        Number of model grid points.
    nobs : int
        Number of observations.
    omb : ndarray of shape (Nd, nobs)
        Sample of innovation vectors (observation - background).
    amb : ndarray of shape (Nd, nobs)
        Sample of observation-space increments (analysis - background).
    oma : ndarray of shape (Nd, nobs)
        Sample of residuals (observation - analysis).
    obs_indices : ndarray of shape (n, nobs)
        Binary localization mask; 1 if obs i is used at grid point l.
    Nd : int
        Sample size of omb/amb/oma.
    R : ndarray of shape (nobs, nobs)
        Observation error covariance matrix.

    Returns
    -------
    DFS_d_theo : ndarray of shape (nobs,)
        Theoretical DFS (original innovation-based approach).
    DFS_d_act : ndarray of shape (nobs,)
        Actual DFS 
    DFS_d_alt : ndarray of shape (nobs,)
        Theoretical DFS (alternative innovation-based approach).
    """
        
    DFS_d_theo = np.zeros(nobs)      
    DFS_d_act  = np.zeros(nobs)
    DFS_d_alt  = np.zeros(nobs)

    for l in range(n):
        if obs_indices[l, :].any():
            # Extract relevant observation indices for this grid point
            index = np.nonzero(obs_indices[l,:])[0]
            k = len(index)
                        
            # Initialize expectation matrices for numerator/denominator terms
            exp_up = np.zeros((k, k))
            exp_down = np.zeros((k, k))
            exp_up_alt = np.zeros((k, k))
            exp_down_alt = np.zeros((k, k))
            
            # Square-root inverse of local R
            R_l = R[np.ix_(index, index)]
            val, vec = eigh(R_l)
            sqrt_invR_l = vec @ np.diag(np.sqrt(1.0 / val)) @ vec.T
                    
            for j in range(Nd):
                # Normalize obs-space vectors
                omb_l = sqrt_invR_l @ omb[j, index]
                amb_l = sqrt_invR_l @ amb[j, index]
                oma_l = sqrt_invR_l @ oma[j, index]
                
                # Compute expected matrices
                exp_up += np.outer(amb_l, oma_l)
                exp_down += np.outer(omb_l, oma_l)
                
                exp_up_alt += np.outer(amb_l, omb_l)
                exp_down_alt += np.outer(omb_l, omb_l)
        
            # Average over Nd samples
            exp_up /= Nd
            exp_down /= Nd
            exp_up_alt /= Nd
            exp_down_alt /= Nd
            
            # Invert and compute DFS estimates
            work = exp_up @ inv(exp_down)
            work_alt = exp_up_alt @ inv(exp_down_alt)
            
            for i in range(k):
                DFS_d_act[index[i]] += exp_up[i, i]     # actual
                DFS_d_theo[index[i]] += work[i, i]      # theoretical original
                DFS_d_alt[index[i]] += work_alt[i, i]   # theoretical alternative

    return DFS_d_theo, DFS_d_act, DFS_d_alt