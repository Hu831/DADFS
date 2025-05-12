import numpy as np
import matplotlib.pyplot as plt
import great_circle_calculator.great_circle_calculator as gcc
from thinning import thinning

def spatial_distribution(args, distrib, ifplot):
    """
    Generate spatial locations of observations in longitude and latitude.
      
    This function creates observation locations based on different spatial 
    distribution schemes over a structured latitude-longitude grid. It supports
    generating observations directly on model grid points, or at randomly 
    distributed locations. Optionally, it can visualize both the 
    model grid and the generated observation locations.
      
    Parameters
    ----------
    args : ExperimentConfig
    A configuration object containing model setup parameters:
        - nlon, nlat : number of grid points (longitude, latitude)
        - nobs       : desired number of observations
        - lon_range  : (min_lon, max_lon) domain in longitude
        - lat_range  : (min_lat, max_lat) domain in latitude
        - n          : total number of grid points (nlon * nlat)
        
    distrib : int
        Distribution strategy code for placing observations:
            0 - All model grid points are directly observed
            1 - Random subset of model grid points
            2 - Uniform random distribution over domain (with thinning)
            3 - Gaussian distribution centered in domain (with thinning)
            4 - Off-grid locations slightly shifted from grid points
            5 - Regularly spaced subset of grid (every 2nd point)
            
    ifplot : int
        If 1, generate a scatter plot of grid and observation locations.
        If 0, skip plotting.
      
    Returns
    -------
    grid : A dictionary containing 
        - xlon : longitudes for all model grid points.
        - xlat : latitudes for all model grid points.
        - ylon : longitudes for observation locations.
        - ylat : latitudes for observation locations.
        - nobs : Final number of observations (may be reduced due to thinning).
    """
    
    np.random.seed(args.rseed_ctrl)

    nobs, nlon, nlat, n = args.nobs, args.nlon, args.nlat, args.n
    lon_range, lat_range = args.lon_range, args.lat_range
  
    # Generate model grid points in longitude and latitude
    lon_intervals = np.linspace(lon_range[0], lon_range[1], nlon)
    lat_intervals = np.linspace(lat_range[0], lat_range[1], nlat)
    xlon_2d, xlat_2d = np.meshgrid(lon_intervals, lat_intervals)
    xlon = xlon_2d.ravel()
    xlat = xlat_2d.ravel()
        
    # Select observation locations based on specified distribution strategy
    if distrib == 0:
        # All model grid points are observation points
        ylon, ylat = xlon.copy(), xlat.copy()
    
    elif distrib == 1:
        # Randomly select observation points from the model grid
        idx = np.random.choice(n, nobs, replace=False)
        ylon, ylat = xlon[idx], xlat[idx]
        
    elif distrib == 2:
        # Uniform random distribution within the domain
        ylon = np.random.uniform(lon_range[0], lon_range[1], nobs)
        ylat = np.random.uniform(lat_range[0], lat_range[1], nobs)
        # Spatial thinning: ensure only one observation per grid box
        ylon, ylat, nobs = thinning(nlon, nlat, ylon, ylat, lon_intervals, lat_intervals)

    elif distrib == 3:
        # Gaussian distribution centered in the domain
        sd = 0.1
        ylon = np.random.normal(np.mean(lon_range), (lon_range[1] - lon_range[0]) * sd, nobs)
        ylat = np.random.normal(np.mean(lat_range), (lat_range[1] - lat_range[0]) * sd, nobs)
        # Spatial thinning: ensure only one observation per grid box
        ylon, ylat, nobs = thinning(nlon, nlat, ylon, ylat, lon_intervals, lat_intervals)
   
    elif distrib == 4:
        # Observations slightly offset from grid, shifted westward
        offset = 0.2
        idx = np.random.choice(n, nobs, replace=False)
        ylon = xlon[idx] - offset
        ylon = np.clip(ylon, lon_range[0], lon_range[1])
        ylat = xlat[idx]
         
    elif distrib == 5:     
        # Regularly spaced observations on a coarser grid (every 2nd grid point)
        ylon_2d, ylat_2d = np.meshgrid(lon_intervals[::2], lat_intervals[::2])
        ylon = ylon_2d.ravel()
        ylat = ylat_2d.ravel()
        
    else:
        raise ValueError("Invalid 'distrib' code: must be 0 to 5.")
            
    # Optionally plot model grid points and observation locations
    if ifplot:
        plt.scatter(xlon, xlat, s=30, marker='.', color='black')
        plt.scatter(ylon, ylat, s=30, marker='x', color='red')
        plt.xlabel("lon")
        plt.ylabel("lat")    
        
        # Calculate grid spacing in km using great-circle distance
        x_grid_length = gcc.distance_between_points((xlon[0], xlat[0]), (xlon[1], xlat[1]), unit='kilometers')
        y_grid_length = gcc.distance_between_points((xlon[0], xlat[0]), (xlon[nlon], xlat[nlon]), unit='kilometers')

        print(f"East-West grid length: {x_grid_length}")
        print(f"North-South grid length: {y_grid_length}")
        
    return {
        "xlon": xlon,
        "xlat": xlat,
        "ylon": ylon,
        "ylat": ylat,
        "nobs": nobs
    }