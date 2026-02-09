"""Seed-based functional connectivity analysis."""

import numpy as np
from scipy import stats


def create_spherical_seed(coords, radius_mm, affine, volume_shape):
    """
    Create a spherical seed mask from MNI coordinates.
    
    Parameters
    ----------
    coords : tuple of float
        (x, y, z) coordinates in MNI space (mm)
    radius_mm : float
        Radius of sphere in mm
    affine : numpy.ndarray
        4x4 affine transformation matrix from voxel to world coordinates
    volume_shape : tuple of int
        (x, y, z) shape of the volume
        
    Returns
    -------
    seed_mask : numpy.ndarray
        3D binary mask with sphere at specified coordinates
    """
    # Create coordinate grids
    i, j, k = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing='ij'
    )
    
    # Stack voxel coordinates
    voxel_coords = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    
    # Transform to world coordinates
    world_coords = affine @ voxel_coords
    
    # Calculate distance from seed center
    distances = np.sqrt(
        (world_coords[0] - coords[0])**2 +
        (world_coords[1] - coords[1])**2 +
        (world_coords[2] - coords[2])**2
    )
    
    # Create mask
    seed_mask = (distances <= radius_mm).reshape(volume_shape)
    
    return seed_mask


def compute_seed_correlation_map(timeseries_4d, seed_mask, fisher_z=True):
    """
    Compute seed-based correlation map.
    
    Parameters
    ----------
    timeseries_4d : numpy.ndarray
        4D timeseries (x, y, z, time)
    seed_mask : numpy.ndarray
        3D binary seed mask
    fisher_z : bool
        Whether to apply Fisher Z transformation to correlations
        
    Returns
    -------
    correlation_map : numpy.ndarray
        3D correlation map (or Z-map if fisher_z=True)
    """
    # Reshape to 2D
    shape_3d = timeseries_4d.shape[:3]
    ts_2d = timeseries_4d.reshape(-1, timeseries_4d.shape[3])
    seed_mask_1d = seed_mask.flatten().astype(bool)
    
    # Extract and average seed timeseries
    seed_ts = np.mean(ts_2d[seed_mask_1d], axis=0)
    
    # Standardize seed timeseries
    seed_ts = (seed_ts - np.mean(seed_ts)) / np.std(seed_ts)
    
    # Compute correlations for all voxels
    correlations = np.zeros(ts_2d.shape[0])
    for i in range(ts_2d.shape[0]):
        voxel_ts = ts_2d[i]
        # Standardize voxel timeseries
        voxel_ts_std = (voxel_ts - np.mean(voxel_ts)) / np.std(voxel_ts)
        # Pearson correlation
        correlations[i] = np.mean(seed_ts * voxel_ts_std)
    
    # Apply Fisher Z transformation if requested
    if fisher_z:
        # Clip correlations to avoid inf values
        correlations = np.clip(correlations, -0.9999, 0.9999)
        correlations = np.arctanh(correlations)
    
    # Reshape to 3D
    correlation_map = correlations.reshape(shape_3d)
    
    return correlation_map


def compute_mask_statistics(correlation_map, mask):
    """
    Compute mean values within and outside a mask.
    
    Parameters
    ----------
    correlation_map : numpy.ndarray
        3D map of values (e.g., Z-scores)
    mask : numpy.ndarray
        3D binary mask
        
    Returns
    -------
    stats : dict
        Dictionary with 'mean_within', 'mean_outside', 'std_within', 'std_outside'
    """
    mask_bool = mask.astype(bool).flatten()
    values_flat = correlation_map.flatten()
    
    within_values = values_flat[mask_bool]
    outside_values = values_flat[~mask_bool]
    
    # Remove NaN/inf values
    within_values = within_values[np.isfinite(within_values)]
    outside_values = outside_values[np.isfinite(outside_values)]
    
    stats = {
        'mean_within': float(np.mean(within_values)) if len(within_values) > 0 else np.nan,
        'std_within': float(np.std(within_values)) if len(within_values) > 0 else np.nan,
        'mean_outside': float(np.mean(outside_values)) if len(outside_values) > 0 else np.nan,
        'std_outside': float(np.std(outside_values)) if len(outside_values) > 0 else np.nan,
    }
    
    return stats


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
