import numpy as np

def get_local_matrices(Yb, d, R, obs_indices_l):
    """
    Extracts local observations and local matrices for a specific grid point.
    
    Parameters:
        
    Yb (ndarray): Background ensemble perturbations in observation space.
    d (ndarray): Innovation vector (observations minus background mean).
    R (ndarray): Observation error covariance matrix.
    obs_indices_l (ndarray): binary array indicating relevant observations for the local analysis.
    
    Returns:
        
    Yb_l (ndarray): Local background ensemble perturbations.
    d_l (ndarray): Local innovation vector.
    R_l (ndarray): Local observation error covariance matrix.
    index (ndarray): Indices of local observations.
    """

    # Indices of observations selected for the n-th grid point
    index = np.nonzero(obs_indices_l)[0]
              
    Yb_l = Yb[index,:]
    d_l = d[index]
    R_l = R[index,:][:,index]
    
    return(Yb_l, d_l, R_l, index)