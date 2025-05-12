import numpy as np

def thinning(nlon, nlat, ylon, ylat, lon_intervals, lat_intervals):
    """
    Perform spatial thinning to ensure at most one observation per grid box.

    This function assigns each observation to a grid cell defined by 
    `lon_intervals` and `lat_intervals`, then selects at most one observation 
    per cell. It is useful for reducing observation redundancy in dense regions.

    Parameters
    ----------
    nlon : int
        Number of grid divisions in the longitude direction.
    nlat : int
        Number of grid divisions in the latitude direction.
    ylon : ndarray
        Array of longitudes for all candidate observations.
    ylat : ndarray
        Array of latitudes for all candidate observations.
    lon_intervals : ndarray
        Array of longitude bin edges.
    lat_intervals : ndarray
        Array of latitude bin edges.

    Returns
    -------
    ylon : ndarray
        Thinned observation longitudes.
    ylat : ndarray
        Thinned observation latitudes.
    nobs : int
        Number of observations after thinning.
    """
    
    # Assign each observation to a grid box (bin index starts from 0)
    ilon = np.digitize(ylon, lon_intervals) - 1
    ilat = np.digitize(ylat, lat_intervals) - 1

    # Mask observations that fall outside the valid grid
    valid = (ilon >= 0) & (ilon < nlon - 1) & (ilat >= 0) & (ilat < nlat - 1)

    # Identify unique grid boxes and keep only the first observation in each
    unique_boxes, idx = np.unique(ilon[valid] + ilat[valid] * nlon, return_index=True)
    
    # Apply thinning based on selected indices
    ylon, ylat = ylon[valid][idx], ylat[valid][idx]

    return ylon, ylat, len(ylon)