"""
fcsanity: Sanity checks for preprocessed fMRI resting state data.
"""

__version__ = "0.1.0"

from . import metrics
from . import connectivity
from . import io
from . import batch
from . import interface

__all__ = ["metrics", "connectivity", "io", "batch", "interface"]
