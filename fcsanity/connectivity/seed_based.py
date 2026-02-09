"""Seed-based functional connectivity analysis."""

import numpy as np
from scipy import stats


def seed_based_correlation(timeseries, seed_mask, method="pearson"):
    """
    Compute seed-based functional connectivity maps.
    
    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array of shape (voxels, time)
    seed_mask : numpy.ndarray
        Binary mask indicating seed region voxels
    method : str, optional
        Correlation method: "pearson" (default) or "spearman"
        
    Returns
    -------
    correlation_map : numpy.ndarray
        1D array of correlations between seed and all voxels
        
    Notes
    -----
    Seed signal is computed as the mean across voxels in seed_mask.
    """
    # Extract seed signal
    seed_signal = np.mean(timeseries[seed_mask], axis=0)
    
    # Standardize seed signal
    seed_signal = (seed_signal - np.mean(seed_signal)) / np.std(seed_signal)
    
    # Compute correlations
    if method == "pearson":
        correlations = np.array([
            np.corrcoef(seed_signal, timeseries[i])[0, 1]
            for i in range(timeseries.shape[0])
        ])
    elif method == "spearman":
        correlations = np.array([
            stats.spearmanr(seed_signal, timeseries[i])[0]
            for i in range(timeseries.shape[0])
        ])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return correlations
