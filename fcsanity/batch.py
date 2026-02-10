"""Batch processing utilities for multi-subject analyses."""

from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from .metrics import compute_tsnr
from .io import load_afni_as_nifti, load_nifti, load_censor_file
from .connectivity.seed_based import (
    create_spherical_seed, 
    compute_seed_correlation_map,
    compute_mask_statistics
)


def find_highest_pb_scaled(subject_dir, subject_id, timepoint):
    """
    Find the highest-numbered pb*X*.scale+tlrc file in a subject directory.
    
    AFNI's preprocessing creates pbNN files with incrementing numbers. The
    highest number is the final output before nuisance regression.
    
    Parameters
    ----------
    subject_dir : Path or str
        Directory to search for pb files
    subject_id : str
        Subject ID (for matching filenames)
    timepoint : str
        Timepoint (for matching filenames)
        
    Returns
    -------
    basename : str
        Basename of the highest pb*X*.scale+tlrc file (without +tlrc extension)
        
    Raises
    ------
    FileNotFoundError
        If no pb*.scale+tlrc files found
    """
    subject_dir = Path(subject_dir)
    
    # Search for all pb*scale+tlrc.BRIK files (matches any task name)
    pattern = f"pb*{subject_id}_{timepoint}*.scale+tlrc.BRIK"
    pb_files = list(subject_dir.glob(pattern))
    
    if not pb_files:
        raise FileNotFoundError(
            f"No pb*scale+tlrc files found in {subject_dir} "
            f"matching {subject_id}_{timepoint}"
        )
    
    # Extract pb number from filename (e.g., "pb06" -> 6)
    def get_pb_number(filepath):
        filename = filepath.name
        # Extract the pb number (digits after 'pb')
        import re
        match = re.search(r'pb(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    # Get the file with highest pb number
    highest_pb_file = max(pb_files, key=get_pb_number)
    
    # Return basename without extension
    basename = highest_pb_file.name.replace("+tlrc.BRIK", "")
    return basename


def find_censor_file(subject_dir, subject_id, timepoint):
    """
    Find censoring file matching the pattern censor_{subject_id}_{timepoint}*combined_2.1D.
    
    Handles any task name between timepoint and 'combined'.
    
    Parameters
    ----------
    subject_dir : Path or str
        Directory to search for censor file
    subject_id : str
        Subject ID
    timepoint : str
        Timepoint
        
    Returns
    -------
    filename : str or None
        Filename of the censor file, or None if not found
    """
    subject_dir = Path(subject_dir)
    
    # Search for censor files with any task name
    pattern = f"censor_{subject_id}_{timepoint}*combined_2.1D"
    censor_files = list(subject_dir.glob(pattern))
    
    if not censor_files:
        return None
    
    # Return the first match (should be only one)
    return censor_files[0].name


def compute_tsnr_with_censoring(errts_data, censor_vector, mask):
    """
    Compute tSNR on uncensored voxels only.
    
    Parameters
    ----------
    errts_data : numpy.ndarray
        4D timeseries (x, y, z, time)
    censor_vector : numpy.ndarray or None
        Boolean vector (time,) where True = uncensored. If None, uses all timepoints.
    mask : numpy.ndarray
        3D binary mask (x, y, z)
        
    Returns
    -------
    results : dict
        Dictionary with tSNR metrics and censoring info
    """
    # Reshape to 2D (voxels, time)
    ts_2d = errts_data.reshape(-1, errts_data.shape[3])
    mask_1d = mask.astype(bool).flatten()
    
    # Extract brain voxels
    brain_voxels = ts_2d[mask_1d]
    
    results = {}
    
    # Select uncensored frames only
    if censor_vector is not None:
        brain_voxels_clean = brain_voxels[:, censor_vector]
        n_censored = np.sum(~censor_vector)
        n_total = len(censor_vector)
        results["n_volumes_total"] = int(n_total)
        results["n_volumes_censored"] = int(n_censored)
        results["percent_censored"] = float(100 * n_censored / n_total)
    else:
        brain_voxels_clean = brain_voxels
        results["n_volumes_total"] = brain_voxels.shape[1]
        results["n_volumes_censored"] = 0
        results["percent_censored"] = 0.0
    
    # Compute tSNR on clean data
    tsnr_clean = compute_tsnr(brain_voxels_clean)
    results["tsnr_mean"] = float(np.mean(tsnr_clean))
    results["tsnr_std"] = float(np.std(tsnr_clean))
    results["tsnr_median"] = float(np.median(tsnr_clean))
    
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
                 use_highest_pb=True,
                 mask_filename=None,
                 global_mask_path=None,
                 censor_basename="censor_{subject_id}_{timepoint}_combined_2.1D",
                 resting_subfolder="Resting",
                 output_individual_json=True):
        """
        Parameters
        ----------
        use_highest_pb : bool
            If True, automatically find and use the highest-numbered pb*scale+tlrc file.
            If False, use mask_filename to specify the file basename.
        mask_filename : str, optional
            Mask filename (not used if use_highest_pb=True)
        global_mask_path : Path or str, optional
            Path to global GM mask file (applied to all subjects)
        censor_basename : str
            Censoring file basename with {subject_id} and {timepoint} placeholders
        resting_subfolder : str
            Name of resting state subfolder within subject_timepoint directory
        output_individual_json : bool
            Whether to save individual subject JSON results
        """
        self.use_highest_pb = use_highest_pb
        self.mask_filename = mask_filename
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
            if config.use_highest_pb:
                ts_basename = find_highest_pb_scaled(results_dir, subject_id, timepoint)
            else:
                ts_basename = config.mask_filename
            
            ts_img, ts_data = load_afni_as_nifti(results_dir, ts_basename)
            
            # Use global mask or load per-subject mask
            if global_mask is not None:
                mask = global_mask
            else:
                mask = load_nifti(results_dir, config.mask_filename)
            
            # Load censoring vector
            censor_filename = find_censor_file(results_dir, subject_id, timepoint)
            censor = load_censor_file(results_dir, censor_filename) if censor_filename else None
            
            # Compute tSNR
            results = compute_tsnr_with_censoring(ts_data, censor, mask)
            
            # Add subject/timepoint info
            results["subject_id"] = subject_id
            results["timepoint"] = timepoint
            results_list.append(results)
            
            # Print summary
            tsnr_val = results.get('tsnr_mean')
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


# =============================================================================
# Seed-Based Connectivity Pipeline
# =============================================================================

class SCAConfig:
    """Configuration for seed-based connectivity analysis pipeline."""
    
    def __init__(self,
                 seed_coords,
                 seed_radius_mm=6,
                 use_errts=True,
                 network_mask_path=None,
                 gm_mask_path=None,
                 brain_mask_path=None,
                 resting_subfolder="Resting",
                 fisher_z=True):
        """
        Parameters
        ----------
        seed_coords : tuple of float
            (x, y, z) coordinates in MNI space (mm)
        seed_radius_mm : float
            Radius of spherical seed in mm
        use_errts : bool
            If True, use errts (residual after regression). If False, use highest pb file.
        network_mask_path : Path or str, optional
            Path to network mask (e.g., DMN) for computing statistics
        gm_mask_path : Path or str, optional
            Path to gray matter mask
        brain_mask_path : Path or str, optional
            Path to brain mask
        resting_subfolder : str
            Name of resting state subfolder
        fisher_z : bool
            Whether to apply Fisher Z transformation to correlations
        """
        self.seed_coords = seed_coords
        self.seed_radius_mm = seed_radius_mm
        self.use_errts = use_errts
        self.network_mask_path = network_mask_path
        self.gm_mask_path = gm_mask_path
        self.brain_mask_path = brain_mask_path
        self.resting_subfolder = resting_subfolder
        self.fisher_z = fisher_z


def run_sca_pipeline(base_path, output_path, subject_ids=None, timepoints=None, config=None):
    """
    Run seed-based connectivity analysis pipeline on multiple subjects.
    
    Generates correlation maps for each subject and computes network statistics.
    
    Parameters
    ----------
    base_path : Path or str
        Base path containing subject directories
    output_path : Path or str
        Path where to save results and correlation maps
    subject_ids : list of str, optional
        Subject IDs to process. If None, processes all found.
    timepoints : list of str, optional
        Timepoints to process. If None, finds all available.
    config : SCAConfig, optional
        Configuration object. If None, uses defaults.
        
    Returns
    -------
    results_df : pandas.DataFrame
        Summary results dataframe
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    
    if config is None:
        raise ValueError("SCAConfig must be provided with seed coordinates")
    
    if timepoints is None:
        timepoints = ["T1", "T2", "T12"]
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    maps_dir = output_path / "correlation_maps"
    maps_dir.mkdir(exist_ok=True)
    
    # Load network mask if provided
    network_mask = None
    if config.network_mask_path:
        print(f"Loading network mask: {config.network_mask_path}")
        network_mask = load_nifti(".", config.network_mask_path)
    
    # Load GM mask if provided
    gm_mask = None
    if config.gm_mask_path:
        print(f"Loading GM mask: {config.gm_mask_path}")
        gm_mask = load_nifti(".", config.gm_mask_path)
    
    # Load brain mask if provided
    brain_mask = None
    if config.brain_mask_path:
        print(f"Loading brain mask: {config.brain_mask_path}")
        brain_mask = load_nifti(".", config.brain_mask_path)
    
    # Find subjects
    subject_dirs = find_subject_dirs(base_path, subject_ids, timepoints)
    
    if not subject_dirs:
        raise ValueError(f"No subjects found in {base_path}")
    
    print(f"Found {len(subject_dirs)} subject/timepoint combinations to process")
    print(f"Seed: {config.seed_coords} (radius={config.seed_radius_mm}mm)")
    
    # Process each subject
    results_list = []
    n_processed = 0
    n_errors = 0
    errors_log = []
    
    output_tsv = output_path / "sca_summary.tsv"
    
    for i, (subject_id, timepoint, subject_dir) in enumerate(subject_dirs, 1):
        print(f"[{i}/{len(subject_dirs)}] {subject_id}_{timepoint}...", end=" ")
        
        results_dir = subject_dir / config.resting_subfolder / f"{subject_id}_{timepoint}.rest.results"
        
        try:
            # Define output filename
            map_filename = f"{subject_id}_{timepoint}_sca_{'z' if config.fisher_z else 'r'}.nii.gz"
            map_path = maps_dir / map_filename
            
            # Check if output already exists
            if map_path.exists():
                print(f"Loading existing map...", end=" ")
                corr_img = nib.load(str(map_path))
                corr_map = np.asarray(corr_img.dataobj)
            else:
                # Load timeseries
                if config.use_errts:
                    # Find errts file with flexible pattern
                    pattern = f"errts.{subject_id}_{timepoint}*.fanaticor+tlrc.BRIK"
                    errts_files = list(results_dir.glob(pattern))
                    if not errts_files:
                        raise FileNotFoundError(f"No errts file found matching {pattern}")
                    ts_basename = errts_files[0].name.replace("+tlrc.BRIK", "")
                else:
                    ts_basename = find_highest_pb_scaled(results_dir, subject_id, timepoint)
                
                ts_img, ts_data = load_afni_as_nifti(results_dir, ts_basename)
                
                # Create seed mask
                seed_mask = create_spherical_seed(
                    config.seed_coords,
                    config.seed_radius_mm,
                    ts_img.affine,
                    ts_data.shape[:3]
                )
                
                # Compute correlation map
                corr_map = compute_seed_correlation_map(ts_data, seed_mask, fisher_z=config.fisher_z)
                
                # Save correlation map
                corr_img = nib.Nifti1Image(corr_map, ts_img.affine, ts_img.header)
                nib.save(corr_img, str(map_path))
            
            # Compute network statistics if mask provided
            results = {
                "subject_id": subject_id,
                "timepoint": timepoint,
                "map_file": map_filename,
            }
            
            if network_mask is not None:
                stats = compute_mask_statistics(corr_map, network_mask, gm_mask, brain_mask)
                results.update(stats)
                print(f"✓ Z_within={stats['mean_within_network']:.3f} Z_gm_outside={stats['mean_gm_outside_network']:.3f} Z_outside_brain={stats['mean_outside_brain']:.3f}")
            else:
                print("✓ Map saved")
            
            results_list.append(results)
            n_processed += 1
            
            # Save incremental results
            df_current = pd.DataFrame(results_list)
            cols = ["subject_id", "timepoint"] + [c for c in df_current.columns if c not in ["subject_id", "timepoint"]]
            df_current = df_current[cols]
            df_current.to_csv(output_tsv, sep="\t", index=False)
            
        except Exception as e:
            print(f"✗ {str(e)}")
            errors_log.append(f"{subject_id}_{timepoint}: {str(e)}")
            n_errors += 1
    
    # Save final results
    if results_list:
        df = pd.DataFrame(results_list)
        cols = ["subject_id", "timepoint"] + [c for c in df.columns if c not in ["subject_id", "timepoint"]]
        df = df[cols]
        
        print(f"\n✓ Summary saved to: {output_tsv}")
        print(f"✓ Correlation maps saved to: {maps_dir}")
        
        if errors_log:
            error_file = output_path / "sca_errors.log"
            with open(error_file, 'w') as f:
                f.write("\n".join(errors_log))
            print(f"⚠ Errors logged to: {error_file}")
        
        print(f"\nProcessed: {n_processed} | Errors: {n_errors}")
        
        return df
    else:
        raise ValueError("No subjects processed successfully")
