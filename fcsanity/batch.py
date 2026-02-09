"""Batch processing utilities for multi-subject analyses."""

from pathlib import Path
import pandas as pd
import numpy as np
from .metrics import compute_tsnr
from .io import load_afni_as_nifti, load_nifti, load_censor_file


def compute_tsnr_with_censoring(errts_data, censor_vector, mask):
    """
    Compute tSNR while handling censored volumes.
    
    Since censored volumes are set to 0 in errts, they artificially reduce
    tSNR. This function handles that by:
    1. Computing tSNR using all timepoints (standard)
    2. Computing tSNR using only uncensored timepoints (more representative)
    
    Parameters
    ----------
    errts_data : numpy.ndarray
        4D errts timeseries (x, y, z, time)
    censor_vector : numpy.ndarray or None
        Boolean vector (time,) where True = uncensored
    mask : numpy.ndarray
        3D binary mask (x, y, z)
        
    Returns
    -------
    results : dict
        Dictionary with tSNR metrics
    """
    # Reshape to 2D (voxels, time)
    errts_2d = errts_data.reshape(-1, errts_data.shape[3])
    mask_1d = mask.astype(bool).flatten()
    
    # Extract brain voxels
    brain_voxels = errts_2d[mask_1d]
    
    results = {}
    
    # Standard tSNR (all timepoints)
    tsnr_all = compute_tsnr(brain_voxels)
    results["tsnr_all_mean"] = float(np.mean(tsnr_all))
    results["tsnr_all_std"] = float(np.std(tsnr_all))
    results["tsnr_all_median"] = float(np.median(tsnr_all))
    
    # tSNR with censoring excluded
    if censor_vector is not None:
        brain_voxels_uncensored = brain_voxels[:, censor_vector]
        tsnr_uncensored = compute_tsnr(brain_voxels_uncensored)
        results["tsnr_uncensored_mean"] = float(np.mean(tsnr_uncensored))
        results["tsnr_uncensored_std"] = float(np.std(tsnr_uncensored))
        results["tsnr_uncensored_median"] = float(np.median(tsnr_uncensored))
        
        # Censoring stats
        n_censored = np.sum(~censor_vector)
        n_total = len(censor_vector)
        results["n_volumes_total"] = int(n_total)
        results["n_volumes_censored"] = int(n_censored)
        results["percent_censored"] = float(100 * n_censored / n_total)
    else:
        results["n_volumes_total"] = errts_data.shape[3]
        results["n_volumes_censored"] = 0
        results["percent_censored"] = 0.0
    
    return results


def find_subject_dirs(base_path, subject_ids=None, timepoints=None):
    """
    Find preprocessed data directories for subjects and timepoints.
    
    Assumes directory structure: base_path/{SUBJECT_ID}_{TIMEPOINT}/...
    
    Parameters
    ----------
    base_path : Path or str
        Base path containing subject directories
    subject_ids : list of str, optional
        Subject IDs to process. If None, finds all subjects.
    timepoints : list of str, optional
        Timepoints to process. If None, processes all available.
        
    Returns
    -------
    subject_dirs : list of tuple
        List of (subject_id, timepoint, full_subject_path) tuples
    """
    base_path = Path(base_path)
    
    if timepoints is None:
        timepoints = ["T1", "T2", "T12"]
    
    subject_dirs = []
    
    # If subject_ids not provided, find all
    if subject_ids is None:
        if not base_path.exists():
            return []
        
        # Scan for all subject directories
        all_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        subject_ids = []
        for d in all_dirs:
            # Extract subject ID from folder name (e.g., "2003_T1" -> "2003")
            parts = d.name.split("_")
            if len(parts) >= 2 and parts[-1] in timepoints:
                subject_ids.append(parts[0])
        subject_ids = sorted(list(set(subject_ids)))
    
    # Find directories for each subject/timepoint combination
    for subject_id in subject_ids:
        for timepoint in timepoints:
            subject_dir = base_path / f"{subject_id}_{timepoint}"
            if subject_dir.exists():
                subject_dirs.append((subject_id, timepoint, subject_dir))
    
    return subject_dirs


# =============================================================================
# tSNR Pipeline
# =============================================================================

class TSnrConfig:
    """Configuration for tSNR pipeline."""
    
    def __init__(self, 
                 timeseries_basename="pb06.{subject_id}_{timepoint}.scale+tlrc",
                 global_mask_path=None,
                 censor_basename="censor_{subject_id}_{timepoint}_combined_2.1D",
                 resting_subfolder="Resting",
                 output_individual_json=True):
        """
        Parameters
        ----------
        timeseries_basename : str
            AFNI timeseries file basename with {subject_id} and {timepoint} placeholders
            (default: pb06 scaled pre-regression timeseries)
        global_mask_path : Path or str, optional
            Path to global GM mask file (applied to all subjects)
        censor_basename : str
            Censoring file basename with {subject_id} and {timepoint} placeholders
        resting_subfolder : str
            Name of resting state subfolder within subject_timepoint directory
        output_individual_json : bool
            Whether to save individual subject JSON results
        """
        self.timeseries_basename = timeseries_basename
        self.global_mask_path = global_mask_path
        self.censor_basename = censor_basename
        self.resting_subfolder = resting_subfolder
        self.output_individual_json = output_individual_json


def run_tsnr_pipeline(base_path, output_path, subject_ids=None, timepoints=None, config=None):
    """
    Run tSNR analysis pipeline on multiple subjects.
    
    Assumes standard AFNI directory structure:
    base_path/{SUBJECT_ID}_{TIMEPOINT}/{resting_subfolder}/{SUBJECT_ID}_{TIMEPOINT}.rest.results/
    
    Parameters
    ----------
    base_path : Path or str
        Base path containing subject directories
    output_path : Path or str
        Path where to save results
    subject_ids : list of str, optional
        Subject IDs to process. If None, processes all found.
    timepoints : list of str, optional
        Timepoints to process (e.g., ['T1', 'T2']). If None, finds all available.
    config : TSnrConfig, optional
        Configuration object. If None, uses defaults.
        
    Returns
    -------
    results_df : pandas.DataFrame
        Summary results dataframe
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    if config is None:
        config = TSnrConfig()
    
    if timepoints is None:
        timepoints = ["T1", "T2", "T12"]
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load global mask if provided
    global_mask = None
    if config.global_mask_path:
        print(f"Loading global mask: {config.global_mask_path}")
        global_mask = load_nifti(".", config.global_mask_path)
    
    # Find subjects to process
    subject_dirs = find_subject_dirs(base_path, subject_ids, timepoints)
    
    if not subject_dirs:
        raise ValueError(f"No subjects found in {base_path}")
    
    print(f"Found {len(subject_dirs)} subject/timepoint combinations to process")
    
    # Process each subject
    results_list = []
    n_processed = 0
    n_errors = 0
    errors_log = []
    
    output_tsv = output_path / "tsnr_summary.tsv"
    
    for i, (subject_id, timepoint, subject_dir) in enumerate(subject_dirs, 1):
        print(f"[{i}/{len(subject_dirs)}] {subject_id}_{timepoint}...", end=" ")
        
        # Full path to results directory
        results_dir = subject_dir / config.resting_subfolder / f"{subject_id}_{timepoint}.rest.results"
        
        try:
            # Load timeseries
            ts_img, ts_data = load_afni_as_nifti(
                results_dir,
                config.timeseries_basename.format(subject_id=subject_id, timepoint=timepoint)
            )
            
            # Use global mask or load per-subject mask
            if global_mask is not None:
                mask = global_mask
            else:
                mask = load_nifti(results_dir, config.mask_filename)
            
            # Load censoring vector
            censor = load_censor_file(
                results_dir,
                config.censor_basename.format(subject_id=subject_id, timepoint=timepoint)
            )
            
            # Compute tSNR
            results = compute_tsnr_with_censoring(ts_data, censor, mask)
            
            # Add subject/timepoint info
            results["subject_id"] = subject_id
            results["timepoint"] = timepoint
            results_list.append(results)
            
            # Print summary
            tsnr_val = results.get('tsnr_uncensored_mean', results.get('tsnr_all_mean'))
            censor_pct = results.get('percent_censored', 0)
            print(f"✓ tSNR={tsnr_val:.2f} censor={censor_pct:.1f}%")
            
            n_processed += 1
            
            # Save incremental results to TSV
            df_current = pd.DataFrame(results_list)
            cols = ["subject_id", "timepoint"] + [c for c in df_current.columns if c not in ["subject_id", "timepoint"]]
            df_current = df_current[cols]
            df_current.to_csv(output_tsv, sep="\t", index=False)
            
        except Exception as e:
            print(f"✗ {str(e)}")
            errors_log.append(f"{subject_id}_{timepoint}: {str(e)}")
            n_errors += 1
    
    # Save final results and errors log
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Reorder columns: subject_id, timepoint first
        cols = ["subject_id", "timepoint"] + [c for c in df.columns if c not in ["subject_id", "timepoint"]]
        df = df[cols]
        
        print(f"\n✓ Summary saved to: {output_tsv}")
        
        # Save errors log if any
        if errors_log:
            error_file = output_path / "tsnr_errors.log"
            with open(error_file, 'w') as f:
                f.write("\n".join(errors_log))
            print(f"⚠ Errors logged to: {error_file}")
        
        print(f"\nProcessed: {n_processed} | Errors: {n_errors}")
        
        return df
    else:
        raise ValueError("No subjects processed successfully")
