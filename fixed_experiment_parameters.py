# fixed_experiment_parameters.py

# Domain and grid
nlon = 20                   # Number of grid points in longitude directions
nlat = 10                   # Number of grid points in latitude directions
n = 200                     # Total number of model grid points (nlon * nlat)
lon_range = (-5, 5)         # Western and eastern boundaries of the domain 
lat_range = (-2.5, 2.5)     # Southern and Northern boundaries of the domain 
nobs = 50                   # Number of observations

# Background and observation errors
corr_b = "SOAR"             # Backgroud error correlation function
corr_o = "diag"             # Observation error correlation function
l_b = 80                    # Backgroud error correlation lengthscale
l_o = 20                    # Observation error correlation lengthscale (if corr_o = "FOAR")
sigma_b = 1.0               # Background error standard deviation
sigma_o = 1.0               # Observation error standard deviation
dist_type = "great circle"  # Choose: "great circle" or "chordal"
interpolator = "linear"     # Choose: "linear" or "nearest" 

# Reproducibility 
rseed_ctrl = 1         
       
earth_radius: float = 6371.0  # in km
