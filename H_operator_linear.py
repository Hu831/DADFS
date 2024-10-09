#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The observation operator

Author: Guannan Hu
"""

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

def H_operator_linear(lon, lat, data, interpolator):
    """ Apply the observation operator which acts as a linear interpolation.
    
    Parameters
    ----------
    lon : numpy.ndarray
        A 1-D array of longitudes.
    lat : numpy.ndarray
        A 1-D array of latitudes.
    data : numpy.ndarray
        Data to interpolate.
    interpolator : str
        'linear' or 'nearest'
        Interpolation type

    Returns
    -------
    scipy.interpolate.interpnd.LinearNDInterpolator

    """

    if len(lon) == 1:
        return NearestNDInterpolator(list(zip(lon, lat)), data)
    
    else:
        if interpolator == 'linear':
            return LinearNDInterpolator(list(zip(lon, lat)), data)
            
        if interpolator == 'nearest':
            return NearestNDInterpolator(list(zip(lon, lat)), data)
