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