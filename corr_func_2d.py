#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the backgroud and observation error covariance matrices (B and R)

Author: Guannan Hu

"""

import numpy as np
import great_circle_calculator.great_circle_calculator as gcc
from math import sin
from numpy import linalg as LA

def corr_func_2d(lon, lat, dist_type, sigma, l, func):
    """ Generate error covariance matrices using the first-order and 
    second-order auto-regressive correlation function (FOAR and SOAR).
    
    Parameters
    ----------
    lon : numpy.ndarray
        An array of longitudes.
    lat : numpy.ndarray
        An array of latitudes.
    dist_type : str
        'great circle' or 'chordal'
        Distance between two points on a sphere
    sigma : float
        Error standard deviation.
    l : int
        Error correlation lengthscale (km)
    func : str
        One of {'FOAR', 'SOAR', 'diag'}
        Correlation function
        
    Returns
    -------
    dist_matrix : n x n matrix
        Distance between points
    cov_matrix : n x n matrix
        Covariance matrix
    cond : float
        Condition number of the covariance matrix.
    """
    
    earth_radius = 6371.0 
    n = len(lon)
    
    # Distance between points
    dist_matrix = np.empty(shape = (n, n))

    for i in range(n):
        for j in range(n):
            if dist_type == 'great circle':
                dist_matrix[i,j] = gcc.distance_between_points((lon[i], lat[i]), (lon[j], lat[j]), unit='kilometers')
            if dist_type == 'chordal':
                dist_matrix[i,j] = 2 * earth_radius * sin(gcc.distance_between_points((lon[i], lat[i]), (lon[j], lat[j]), unit='kilometers') / earth_radius / 2)
    
    # Correlation matrix
    corr_matrix = np.zeros(np.shape(dist_matrix))
    
    if func == "SOAR":
        for i in range(n):
            for j in range(n):
                corr_matrix[i,j] = (1 + dist_matrix[i,j]/l) * np.exp(-dist_matrix[i,j]/l)  
                
    if func == "FOAR":
        for i in range(n):
            for j in range(n):
                corr_matrix[i,j] = np.exp(-dist_matrix[i,j]/l)  
     
    # Diagonal matrix
    if func == "diag":
        corr_matrix = np.identity(n) 
        
    # Covariance matrix
    cov_matrix = sigma**2 * corr_matrix
    
    # Conditiona number
    cond = LA.cond(cov_matrix)
    
    return (dist_matrix, cov_matrix, cond)