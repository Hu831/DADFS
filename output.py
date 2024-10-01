#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print results on screen and save results 

Author: Guannan Hu
"""

import numpy as np
import pandas as pd

def output(DFS_Ya, DFS_d_act, DFS_d_theo, DFS_w_avg, DFS_d_alt, count, erra, save, distrib, N, Nd, r, DA_method, rseed_obs, l_o, ylon, ylat):
    """Print and save results.
    
    Parameters
    ----------
    DFS_Ya : nobs x 1 array
        The theoretical DFS estimated using the ensemble perturbation approach.
    DFS_d_act : nobs x 1 array
        The actual DFS estimated using the innovation-based approach.
    DFS_d_theo : nobs x 1 array
        The theoretical DFS estimated using the innovation-based approach.
    DFS_w_avg : nobs x 1 array
        The actual DFS estimated using the weighting-vector-based approach.
    DFS_d_alt : nobs x 1 array
        The theoretical DFS estimated using the new innovation-based approach.
    count : nobs x 1 array
        The number of grid points each observation has been used for.
    erra : n x 1 array
        The analysis error at each model grid point.
    save : int
        Save results if '1'.
    distrib : int
        One of {'1', '2', '3', '4', '5'}
        Spatial distributions of observations: 1 - On randomly selected model 
        grid points; 2 - Uniform distribution; 3 - Gaussian distribution
        4 - Slightly off model grid; 5 - Regularly on model grid.
    N : int
        Ensemble size.
    Nd : int
        Sample size for the innovation-based and weighting-vector-based 
        approaches.
    r : int
        Localization radius.
    DA_method : str
        'LETKF' or 'EnKF'
    rseed_obs : int
        Random number seed.
    l_o : float
        Observation error correlation lengthscale.
    ylon : nobs x 1 array
        Longitudes of observations.
    ylat : nobs x 1 array
        Latitudes of observations.
    """
    
    print('-----')
    print('The DFS for all observations:')
    print('DFS_Ya:', sum(DFS_Ya / count))
    print('DFS_d_act:', sum(DFS_d_act / count))
    print('DFS_d_theo:', sum(DFS_d_theo / count))
    print('DFS_w:', sum(DFS_w_avg / count))
    print('DFS_d_alt:', sum(DFS_d_alt / count))
    
    print('-----')
    print('Analysis RMSE:', np.mean(erra))

    # Save data
    if save == 1:
        
        # Form the filename for saving data
        filename = 'dist' + str(distrib) + '_N' + str(N) + '_Nd' + str(Nd) + \
            '_r' + str(r) + '_' + DA_method + '_' + str(rseed_obs) + '.csv'
        
        if l_o != 0:
            
           filename = 'dist' + str(distrib) + '_N' + str(N) + '_Nd' + str(Nd) + \
           '_r' + str(r) + '_' + DA_method + '_' + str(l_o) + 'km_' + str(rseed_obs) + '.csv'
        
        # Put data into a pandas data frame
        df = pd.DataFrame({'lon': ylon,'lat': ylat, 'Ya': DFS_Ya,
                            'w': DFS_w_avg,'d_theo': DFS_d_theo, 'd_act': DFS_d_act, 'd_alt': DFS_d_alt,
                            'count': count, "erra": np.mean(erra)})
        
        print('-----')
        print('Saved to:', filename)

        df.to_csv('DFS_' + filename, index=False)