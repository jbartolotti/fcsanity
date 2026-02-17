"""
Command-line interface utilities for seed-based connectivity analysis.

Provides reusable functions for argument parsing, BIDS-aware configuration,
and result reporting across multiple projects.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fcsanity.batch import SCAConfig, validate_resources
from fcsanity.io import get_yeo_network_mask


def _infer_mask_name(mask_path: Optional[Path]) -> Optional[str]:
    if mask_path is None:
        return None
    name = Path(mask_path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def setup_sca_parser(
    program_name: str = "Seed-Based Connectivity Analysis",
    description: str = "Compute seed-based connectivity for fMRI data (BIDS format)",
    default_seed: str = "pcc",
    seed_choices: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    """
    Create a standardized argument parser for SCA pipeline.
    
    This parser provides consistent CLI across all projects with support for:
    - Seed selection
    - Subject filtering
    - Timepoint filtering
    - Listing available seeds
    
    Parameters
    ----------
    program_name : str
        Name of the program for help text
    description : str
        Description of the program
    default_seed : str
        Default seed to use if --seed not provided
    seed_choices : list of str, optional
        List of valid seed names (for validation)
        
    Returns
    -------
    parser : argparse.ArgumentParser
        Configured argument parser
        
    Example
    -------
    >>> parser = setup_sca_parser(
    ...     program_name="CM2228 SCA",
    ...     default_seed="pcc",
    ...     seed_choices=["pcc", "rai", "lips"]
    ... )
    >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        prog=program_name,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--seed",
        type=str,
        default=default_seed,
        choices=seed_choices,
        help=f"Seed to use for connectivity analysis (default: {default_seed})"
    )
    
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=str,
        default=None,
        help="Subject IDs to process (default: all in BIDS dataset)"
    )
    
    parser.add_argument(
        "--timepoints",
        nargs="+",
        type=str,
        default=None,
        help="Timepoints to process (default: all available)"
    )
    
    parser.add_argument(
        "--list-seeds",
        action="store_true",
        help="List available seeds and exit"
    )
    
    return parser


def list_seeds_and_exit(seeds: Dict) -> None:
    """
    Print available seeds to stdout and exit.
    
    Parameters
    ----------
    seeds : dict
        Dictionary mapping seed names to seed information dicts.
        Each dict should have 'name' and 'coords' keys.
        
    Example
    -------
    >>> seeds = {
    ...     "pcc": {"name": "Posterior Cingulate", "coords": (0, -52, 26)},
    ...     "rai": {"name": "Right Anterior Insula", "coords": (38, 22, -2)},
    ... }
    >>> list_seeds_and_exit(seeds)
    """
    print("\nAvailable seeds:")
    for seed_key, seed_info in seeds.items():
        name = seed_info.get("name", "Unknown")
        coords = seed_info.get("coords", "Unknown")
        print(f"  {seed_key:12s} - {name:40s} {coords}")
    print()
    sys.exit(0)


def validate_seed_selection(
    seed_name: str,
    seeds: Dict,
    exit_on_error: bool = True
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Validate that selected seed exists and return seed info.
    
    Parameters
    ----------
    seed_name : str
        Name of the seed to validate
    seeds : dict
        Dictionary of available seeds
    exit_on_error : bool
        If True, print error and exit on validation failure
        If False, return error message
        
    Returns
    -------
    valid : bool
        True if seed is valid
    message : str
        Error message if invalid, empty string if valid
    seed_info : dict or None
        Seed information dict if valid, None otherwise
        
    Example
    -------
    >>> valid, msg, info = validate_seed_selection("pcc", seeds)
    >>> if valid:
    ...     print(f"Selected: {info['name']}")
    """
    if seed_name in seeds:
        return True, "", seeds[seed_name]
    
    error_msg = f"Unknown seed: {seed_name}"
    
    if exit_on_error:
        print(f"✗ {error_msg}", file=sys.stderr)
        sys.exit(1)
    else:
        return False, error_msg, None


def initialize_sca_from_args(
    args: argparse.Namespace,
    seeds: Dict,
    bids_root: Path,
    derivatives_root: Path,
    gm_mask_path: Path,
    brain_mask_path: Path,
    seed_radius_mm: float = 6.0,
    resting_subfolder: str = "Resting",
    fisher_z: bool = True,
    resources_dir: Optional[Path] = None,
    yeo_network_map: Optional[Dict[str, int]] = None,
    yeo_thickness: str = "thick",
    yeo_n_networks: int = 7,
    yeo_overwrite: bool = False,
) -> Tuple[argparse.Namespace, SCAConfig, Path]:
    """
    Initialize SCA configuration from CLI arguments and BIDS structure.
    
    This function handles:
    - Seed validation
    - Output path derivation from BIDS structure
    - Resource validation
    - SCAConfig creation
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from setup_sca_parser()
    seeds : dict
        Available seeds dictionary
    bids_root : Path
        Root BIDS directory
    derivatives_root : Path
        Root derivatives directory (e.g., bids_root/derivatives/fcsanity)
    gm_mask_path : Path
        Path to gray matter mask
    brain_mask_path : Path
        Path to brain mask
    seed_radius_mm : float
        Radius of spherical seed in mm (default: 6.0)
    resting_subfolder : str
        Name of resting-state subfolder (default: "Resting")
    fisher_z : bool
        Apply Fisher Z-transformation (default: True)
    resources_dir : Path, optional
        Resources directory for atlas downloads and generated masks
    yeo_network_map : dict, optional
        Mapping of seed name to Yeo network ID (e.g., {"pcc": 7, "lips": 3})
    yeo_thickness : str
        Atlas thickness variant ("thick" or "thin", default: "thick")
    yeo_n_networks : int
        Number of networks in atlas (7 or 17, default: 7)
    yeo_overwrite : bool
        Overwrite existing Yeo-derived masks (default: False)
        
    Returns
    -------
    args : argparse.Namespace
        Original args (unchanged)
    config : SCAConfig
        Configured SCAConfig object ready for pipeline
    output_path : Path
        BIDS derivatives output path for this seed
        
    Raises
    ------
    ValueError
        If seed selection is invalid
    FileNotFoundError
        If required resources are missing
        
    Example
    -------
    >>> args = parser.parse_args()
    >>> args, config, output_path = initialize_sca_from_args(
    ...     args,
    ...     seeds=SEEDS,
    ...     bids_root=BIDS_ROOT,
    ...     derivatives_root=DERIVATIVES_ROOT,
    ...     gm_mask_path=GM_MASK_PATH,
    ...     brain_mask_path=BRAIN_MASK_PATH,
    ... )
    """
    # Validate seed selection
    valid, msg, seed_info = validate_seed_selection(
        args.seed,
        seeds,
        exit_on_error=True
    )
    
    seed_coords = seed_info["coords"]
    network_mask_path = seed_info.get("network_mask")
    if network_mask_path is not None:
        network_mask_path = Path(network_mask_path)

    mask_name = seed_info.get("mask_name") or _infer_mask_name(network_mask_path)
    
    # Create output path (BIDS-compliant derivatives/fcsanity/SEED_NAME/)
    output_path = derivatives_root / args.seed

    # Optional: fetch Yeo 2011 atlas and generate network mask if missing
    if (network_mask_path is None or not network_mask_path.exists()) and resources_dir:
        if yeo_network_map is None:
            yeo_network_map = {
                "pcc": 7,
                "lips": 3,
            }

        network_id = yeo_network_map.get(args.seed)
        if network_id is not None:
            try:
                network_mask_path = get_yeo_network_mask(
                    network_id=network_id,
                    resources_dir=resources_dir,
                    thickness=yeo_thickness,
                    n_networks=yeo_n_networks,
                    overwrite=yeo_overwrite,
                )
                seed_info["network_mask"] = network_mask_path
                yeo_name_map = {3: "DAN", 7: "DMN"}
                mask_name = seed_info.get("mask_name") or yeo_name_map.get(network_id) or _infer_mask_name(network_mask_path)
                seed_info["mask_name"] = mask_name
            except Exception as exc:
                print(
                    f"⚠ Could not generate Yeo 2011 network mask for seed '{args.seed}': {exc}",
                    file=sys.stderr,
                )
    
    # Validate resources exist
    resources = {
        "GM mask": gm_mask_path,
        "Brain mask": brain_mask_path,
    }
    if network_mask_path:
        resources[f"{args.seed.upper()} network mask"] = network_mask_path
    
    missing = validate_resources(resources, verbose=True)
    
    # Create SCAConfig
    config = SCAConfig(
        seed_coords=seed_coords,
        seed_name=args.seed,
        seed_radius_mm=seed_radius_mm,
        use_errts=True,
        network_mask_path=network_mask_path,
        network_mask_name=mask_name,
        gm_mask_path=gm_mask_path,
        brain_mask_path=brain_mask_path,
        resting_subfolder=resting_subfolder,
        fisher_z=fisher_z
    )
    
    return args, config, output_path


def print_pipeline_header(
    project_name: str,
    seed_name: str,
    seed_info: Dict,
    bids_root: Path,
    output_path: Path,
    seed_radius_mm: float,
    gm_mask_path: Path,
    brain_mask_path: Path,
    network_mask_path: Optional[Path] = None,
) -> None:
    """
    Print standardized pipeline header with configuration details.
    
    Parameters
    ----------
    project_name : str
        Name of the project
    seed_name : str
        Key of selected seed
    seed_info : dict
        Seed information dict with 'name', 'coords', etc.
    bids_root : Path
        BIDS root directory
    output_path : Path
        Output path for this seed
    seed_radius_mm : float
        Seed radius in mm
    gm_mask_path : Path
        Path to GM mask
    brain_mask_path : Path
        Path to brain mask
    network_mask_path : Path, optional
        Path to network mask if available
        
    Example
    -------
    >>> print_pipeline_header(
    ...     project_name="CM2228 NN-RCT",
    ...     seed_name="pcc",
    ...     seed_info=SEEDS["pcc"],
    ...     bids_root=BIDS_ROOT,
    ...     output_path=output_path,
    ...     seed_radius_mm=6.0,
    ...     gm_mask_path=GM_MASK_PATH,
    ...     brain_mask_path=BRAIN_MASK_PATH,
    ... )
    """
    seed_coords = seed_info.get("coords", "Unknown")
    seed_name_human = seed_info.get("name", seed_name)
    
    print("=" * 70)
    print(f"{project_name} - Seed-Based Connectivity Pipeline")
    print("=" * 70)
    print(f"\nBIDS root:    {bids_root}")
    print(f"Seed output:  {output_path}")
    print(f"\nSeed:         {seed_name} - {seed_name_human}")
    print(f"Coordinates:  {seed_coords} (MNI space)")
    print(f"Radius:       {seed_radius_mm}mm")
    if network_mask_path and network_mask_path.exists():
        print(f"Network mask: {network_mask_path}")
    print(f"GM mask:      {gm_mask_path}")
    print(f"Brain mask:   {brain_mask_path}\n")


def print_pipeline_footer(output_path: Path, status: str = "completed") -> None:
    """
    Print standardized pipeline completion message.
    
    Parameters
    ----------
    output_path : Path
        Path where results were saved
    status : str
        Status message ("completed", "failed", etc.)
        
    Example
    -------
    >>> print_pipeline_footer(output_path, status="completed")
    """
    print("=" * 70)
    if status.lower() == "completed":
        print("✓ Pipeline completed successfully")
    else:
        print(f"✓ Pipeline {status}")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}\n")
