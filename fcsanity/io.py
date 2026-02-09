"""I/O utilities for loading and managing fMRI data."""

from pathlib import Path
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
