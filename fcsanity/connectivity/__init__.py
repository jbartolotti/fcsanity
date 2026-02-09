"""Functional connectivity analysis."""

from .seed_based import seed_based_correlation
from .atlas_based import atlas_fc_matrix

__all__ = ["seed_based_correlation", "atlas_fc_matrix"]
