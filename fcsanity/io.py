"""I/O utilities for loading and managing fMRI data."""

from pathlib import Path
import json
import numpy as np
import nibabel as nib


def load_afni_as_nifti(subject_dir, base_name):
    """
    Load AFNI BRIK/HEAD files as nibabel image.
    
    Parameters
    ----------
    subject_dir : Path
        Directory containing AFNI files
    base_name : str
        Base name of AFNI files (without +tlrc or extension)
        
    Returns
    -------
    img : nibabel.Nifti1Image
        Loaded image data
    data : numpy.ndarray
        Image data array
    """
    brik_file = Path(subject_dir) / f"{base_name}+tlrc.BRIK"
    if brik_file.exists():
        img = nib.load(str(brik_file))
        data = np.asarray(img.dataobj)
        return img, data
    else:
        raise FileNotFoundError(f"BRIK file not found: {brik_file}")


def load_nifti(subject_dir, filename):
    """
    Load NIfTI file.
    
    Parameters
    ----------
    subject_dir : Path or str
        Directory containing the file
    filename : str
        Filename to load
        
    Returns
    -------
    data : numpy.ndarray
        Image data array
    """
    filepath = Path(subject_dir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    img = nib.load(str(filepath))
    data = np.asarray(img.dataobj)
    return data


def load_censor_file(subject_dir, censor_filename):
    """
    Load censoring vector from 1D file.
    
    Parameters
    ----------
    subject_dir : Path or str
        Directory containing censoring file
    censor_filename : str
        Filename of censoring file
        
    Returns
    -------
    censor : numpy.ndarray or None
        Boolean censoring vector (True = uncensored), or None if file not found
    """
    filepath = Path(subject_dir) / censor_filename
    if not filepath.exists():
        return None
    
    censor = np.loadtxt(str(filepath))
    return censor.astype(bool)

def load_seeds(json_path, resources_dir=None):
    """
    Load seed definitions from JSON file.
    
    Parameters
    ----------
    json_path : Path or str
        Path to seeds.json file
    resources_dir : Path or str, optional
        Path to resources directory. If provided, network_mask paths will be
        resolved relative to this directory.
        
    Returns
    -------
    seeds : dict
        Dictionary mapping seed names to seed information
        Each seed contains: name, coords (list), description, network_mask (path or None)
        
    Examples
    --------
    >>> seeds = load_seeds("seeds.json")
    >>> seeds["pcc"]["coords"]
    (0, -52, 26)
    
    >>> seeds = load_seeds("seeds.json", resources_dir="/path/to/atlases")
    >>> seeds["pcc"]["network_mask"]
    PosixPath('/path/to/atlases/DMN_mask.nii.gz')
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Seeds file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        seeds_dict = json.load(f)
    
    # Convert coords to tuples and resolve network_mask paths
    for seed_name, seed_info in seeds_dict.items():
        # Convert coords list to tuple
        seed_info["coords"] = tuple(seed_info["coords"])
        
        # Resolve network_mask path if provided
        if seed_info.get("network_mask") and resources_dir:
            network_mask = seed_info["network_mask"]
            if network_mask is not None:
                resources_path = Path(resources_dir)
                seed_info["network_mask"] = resources_path / network_mask
    
    return seeds_dict


def fetch_yeo_2011_atlas(
    data_dir,
    thickness: str = "thick",
    n_networks: int = 7,
    verbose: int = 1,
):
    """
    Fetch the Yeo 2011 atlas using nilearn and return the atlas path.
    
    Parameters
    ----------
    data_dir : Path or str
        Directory where atlas files should be downloaded
    thickness : str
        "thick" or "thin" atlas variant (default: "thick")
    n_networks : int
        Number of networks (7 or 17, default: 7)
    verbose : int
        nilearn verbosity level
        
    Returns
    -------
    atlas_path : Path
        Path to the requested atlas NIfTI file
    """
    try:
        from nilearn import datasets
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for fetching Yeo 2011 atlas. "
            "Install it with 'pip install nilearn'."
        ) from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    atlas = datasets.fetch_atlas_yeo_2011(
        data_dir=str(data_dir),
        verbose=verbose,
    )

    atlas_key = f"{thickness}_{n_networks}"
    if not hasattr(atlas, atlas_key):
        raise ValueError(
            f"Requested atlas variant not available: {atlas_key}. "
            "Valid options: thick_7, thin_7, thick_17, thin_17."
        )

    return Path(getattr(atlas, atlas_key))


def get_yeo_network_mask(
    network_id: int,
    resources_dir,
    thickness: str = "thick",
    n_networks: int = 7,
    atlas_subdir: str = "yeo2011",
    overwrite: bool = False,
):
    """
    Create (or return) a binary mask for a single Yeo 2011 network.
    
    Parameters
    ----------
    network_id : int
        Network label ID in the atlas (e.g., 3 = dorsal attention, 7 = default mode)
    resources_dir : Path or str
        Base resources directory to store atlas and masks
    thickness : str
        "thick" or "thin" atlas variant (default: "thick")
    n_networks : int
        Number of networks (7 or 17, default: 7)
    atlas_subdir : str
        Subdirectory under resources_dir for atlas files
    overwrite : bool
        If True, overwrite existing mask file
        
    Returns
    -------
    mask_path : Path
        Path to the created (or existing) network mask NIfTI file
    """
    resources_dir = Path(resources_dir)
    atlas_dir = resources_dir / atlas_subdir
    atlas_path = fetch_yeo_2011_atlas(
        data_dir=atlas_dir,
        thickness=thickness,
        n_networks=n_networks,
    )

    mask_name = f"yeo2011_{n_networks}net_{thickness}_network-{network_id}.nii.gz"
    mask_path = atlas_dir / mask_name

    if mask_path.exists() and not overwrite:
        return mask_path

    atlas_img = nib.load(str(atlas_path))
    atlas_data = np.asarray(atlas_img.dataobj)
    mask_data = (atlas_data == network_id).astype(np.uint8)

    mask_img = nib.Nifti1Image(mask_data, affine=atlas_img.affine, header=atlas_img.header)
    mask_img.header.set_data_dtype(np.uint8)
    nib.save(mask_img, str(mask_path))

    return mask_path