"""Temporal Signal-to-Noise Ratio (tSNR) calculations."""

import numpy as np


def compute_tsnr(timeseries, mask=None):
    """
    Compute temporal Signal-to-Noise Ratio for fMRI data.
    
    Parameters
    ----------
    timeseries : numpy.ndarray
        4D array of shape (x, y, z, time) or 2D array of shape (voxels, time)
    mask : numpy.ndarray, optional
        Binary mask to restrict calculation to specific voxels/regions
        
    Returns
    -------
    tsnr : numpy.ndarray
        tSNR values in the same shape as input (without time dimension)
        
    Notes
    -----
    tSNR = mean(timeseries) / std(timeseries) computed along time axis.
    """
    # Compute along time axis (last axis)
    mean_signal = np.mean(timeseries, axis=-1)
    std_signal = np.std(timeseries, axis=-1)
    
    tsnr = np.divide(mean_signal, std_signal, where=std_signal != 0)
    
    # Apply mask if provided
    if mask is not None:
        tsnr[~mask] = 0
    
    return tsnr
