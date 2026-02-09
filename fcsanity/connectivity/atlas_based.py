"""Atlas-based functional connectivity analysis."""

import numpy as np


def atlas_fc_matrix(timeseries, atlas_labels, metric="pearson"):
    """
    Compute functional connectivity matrices using atlas ROIs.
    
    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array of shape (voxels, time)
    atlas_labels : numpy.ndarray
        1D array of atlas labels (same length as voxels)
    metric : str, optional
        Correlation metric: "pearson" (default)
        
    Returns
    -------
    fc_matrix : numpy.ndarray
        2D correlation matrix of shape (n_rois, n_rois)
    roi_labels : numpy.ndarray
        Unique ROI labels
        
    Notes
    -----
    ROI signals are computed as mean across voxels in each ROI.
    """
    roi_labels = np.unique(atlas_labels)
    roi_labels = roi_labels[roi_labels != 0]  # Exclude background
    
    n_rois = len(roi_labels)
    roi_signals = np.zeros((n_rois, timeseries.shape[1]))
    
    # Extract mean signal for each ROI
    for i, label in enumerate(roi_labels):
        roi_mask = atlas_labels == label
        roi_signals[i] = np.mean(timeseries[roi_mask], axis=0)
    
    # Standardize signals
    roi_signals = (roi_signals - np.mean(roi_signals, axis=1, keepdims=True)) / \
                  np.std(roi_signals, axis=1, keepdims=True)
    
    # Compute correlation matrix
    if metric == "pearson":
        fc_matrix = np.corrcoef(roi_signals)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return fc_matrix, roi_labels


def split_half_reliability(timeseries, atlas_labels, split_type="temporal"):
    """
    Assess reliability of atlas FC by comparing first/second half or odd/even frames.
    
    Parameters
    ----------
    timeseries : numpy.ndarray
        2D array of shape (voxels, time)
    atlas_labels : numpy.ndarray
        1D array of atlas labels
    split_type : str, optional
        "temporal" for first/second half, "odd_even" for odd/even frames
        
    Returns
    -------
    correlation : float
        Pearson correlation between FC matrices (upper triangle)
    fc_matrix1, fc_matrix2 : tuple of numpy.ndarray
        FC matrices for each split
    """
    n_frames = timeseries.shape[1]
    
    if split_type == "temporal":
        mid = n_frames // 2
        ts1 = timeseries[:, :mid]
        ts2 = timeseries[:, mid:]
    elif split_type == "odd_even":
        ts1 = timeseries[:, ::2]  # Even indices
        ts2 = timeseries[:, 1::2]  # Odd indices
    else:
        raise ValueError(f"Unknown split type: {split_type}")
    
    # Compute FC matrices for each split
    fc_matrix1, _ = atlas_fc_matrix(ts1, atlas_labels)
    fc_matrix2, _ = atlas_fc_matrix(ts2, atlas_labels)
    
    # Get upper triangle indices
    triu_idx = np.triu_indices(fc_matrix1.shape[0], k=1)
    
    # Correlate upper triangles
    correlation = np.corrcoef(
        fc_matrix1[triu_idx],
        fc_matrix2[triu_idx]
    )[0, 1]
    
    return correlation, fc_matrix1, fc_matrix2
