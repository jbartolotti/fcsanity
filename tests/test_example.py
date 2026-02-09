"""Example tests for fcsanity modules."""

import numpy as np
import pytest
from fcsanity.metrics import compute_tsnr, detect_high_motion
from fcsanity.connectivity import seed_based_correlation, atlas_fc_matrix


class TestMetrics:
    """Test quality metrics."""
    
    def test_compute_tsnr(self):
        """Test tSNR computation."""
        # Create synthetic data
        timeseries = np.random.randn(100, 50)  # 100 voxels, 50 timepoints
        tsnr = compute_tsnr(timeseries)
        
        assert tsnr.shape == (100,)
        assert np.all(tsnr >= 0)
    
    def test_detect_high_motion(self):
        """Test motion detection."""
        # Create synthetic motion parameters
        motion_params = np.random.randn(100, 6) * 0.1  # 100 frames, 6 DOF
        high_motion, summary = detect_high_motion(motion_params, threshold=0.5)
        
        assert len(high_motion) == 100
        assert isinstance(summary, dict)
        assert "mean_fd" in summary


class TestConnectivity:
    """Test connectivity analysis."""
    
    def test_seed_based_correlation(self):
        """Test seed-based FC computation."""
        timeseries = np.random.randn(100, 50)
        seed_mask = np.zeros(100, dtype=bool)
        seed_mask[:10] = True
        
        corr = seed_based_correlation(timeseries, seed_mask)
        
        assert corr.shape == (100,)
        assert np.all(np.abs(corr) <= 1)
    
    def test_atlas_fc_matrix(self):
        """Test atlas-based FC matrix computation."""
        timeseries = np.random.randn(100, 50)
        atlas_labels = np.repeat(np.arange(1, 11), 10)  # 10 ROIs, 10 voxels each
        
        fc_matrix, roi_labels = atlas_fc_matrix(timeseries, atlas_labels)
        
        assert fc_matrix.shape == (10, 10)
        assert len(roi_labels) == 10
        assert np.allclose(np.diag(fc_matrix), 1.0)
