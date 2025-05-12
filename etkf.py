import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigh

def etkf(N, Yb, R, d):
    """Pefform the ETKF algortihm.
 
    Parameters
    ----------
    N : int
        Ensemble size.
    Yb : numpy.ndarray
        The background ensemble perturbation matrix in observation space.
    R : numpy.ndarray
        The observation error covariance matrix.
    d : numpy.ndarray
        The innovation vector.

    Returns
    -------
    w : N x 1 array
        The ETKF weighting vector.
    W : N x N matrix 
        The ETKF transformation matrix.
    """
    
    # The covariance inflation factor
    rho = 1.0
    
    # An intermediate matrix
    C = Yb.T @ inv(R)
    
    # Eigenvalue decomposition
    val, vec = eigh((N - 1) * np.identity(N) / rho + C @ Yb)
    
    # The weighting vector
    w = vec @ np.diag(1.0/val) @ vec.T @ (C @ d)
    
    assert (val >= 0).all()
    
    # The transformation matrix
    W = np.sqrt(N - 1) * vec @ np.diag(np.sqrt(1.0/val)) @ vec.T

    return(w, W)