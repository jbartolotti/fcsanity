"""Motion quality assessment for fMRI data."""

import numpy as np


def detect_high_motion(motion_params, threshold=0.5):
    """
    Detect frames with high motion based on motion parameters.
    
    Parameters
    ----------
    motion_params : numpy.ndarray
        Motion parameters array (n_frames, n_params), typically 6 DOF
        (3 translation + 3 rotation)
    threshold : float, optional
        Motion threshold for flagging frames (default: 0.5 mm)
        
    Returns
    -------
    high_motion_frames : numpy.ndarray
        Boolean array indicating frames with excessive motion
    motion_summary : dict
        Summary statistics of motion parameters
    """
    # Compute framewise displacement (FD)
    fd = np.zeros(motion_params.shape[0])
    
    # Calculate derivative (frame-to-frame displacement)
    fd[1:] = np.sqrt(np.sum(np.diff(motion_params, axis=0) ** 2, axis=1))
    fd[0] = 0
    
    high_motion_frames = fd > threshold
    
    motion_summary = {
        "mean_fd": np.mean(fd),
        "max_fd": np.max(fd),
        "percent_high_motion": 100 * np.sum(high_motion_frames) / len(fd),
    }
    
    return high_motion_frames, motion_summary
