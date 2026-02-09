# fcsanity

A Python library for sanity-check level analyses of preprocessed fMRI resting state data.

## Overview

`fcsanity` provides tools to assess data quality and basic functional connectivity properties for resting state fMRI datasets. It includes:

- **Quality Metrics**
  - Temporal Signal-to-Noise Ratio (tSNR) in specific regions (e.g., gray matter)
  - Motion detection and high-motion frame identification

- **Seed-Based Connectivity**
  - Seed-based correlation maps (e.g., PCC → Default Mode Network)
  - Support for different correlation methods (Pearson, Spearman)

- **Atlas-Based Connectivity**
  - ROI-to-ROI functional connectivity matrices
  - Split-half reliability assessment (first/second half, odd/even frames)

## Installation

### Development Installation

```bash
cd fcsanity
pip install -e ".[dev]"
```

### Runtime Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from fcsanity.metrics import compute_tsnr, detect_high_motion
from fcsanity.connectivity import seed_based_correlation, atlas_fc_matrix

# Load your preprocessed fMRI data
# timeseries: (n_voxels, n_timepoints)
# atlas_labels: (n_voxels,)
# motion_params: (n_timepoints, 6)

# Compute tSNR in a region
tsnr = compute_tsnr(timeseries, mask=gray_matter_mask)
print(f"Mean tSNR: {np.mean(tsnr):.2f}")

# Detect high motion frames
high_motion, motion_summary = detect_high_motion(motion_params, threshold=0.5)
print(f"High motion frames: {motion_summary['percent_high_motion']:.1f}%")

# Seed-based analysis (e.g., PCC seed)
pcc_corr = seed_based_correlation(timeseries, pcc_mask)

# Atlas-based connectivity
fc_matrix, roi_labels = atlas_fc_matrix(timeseries, atlas_labels)

# Test reliability via split-half correlation
from fcsanity.connectivity import split_half_reliability
corr, fc1, fc2 = split_half_reliability(timeseries, atlas_labels, split_type="temporal")
print(f"Split-half reliability: {corr:.3f}")
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Dependencies

- numpy >= 1.19.0
- scipy >= 1.5.0

## Notes

- Designed for Linux server deployment
- Assumes preprocessed fMRI data (motion correction, registration, etc.)
- Handles high-motion participants gracefully
