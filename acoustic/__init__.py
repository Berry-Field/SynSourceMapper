"""
Acoustic Beamforming Dataset Generator

This package provides tools for generating acoustic beamforming datasets,
including array position management, conventional beamforming algorithms,
and dataset generation utilities.
"""

__version__ = '0.1.0'

# Import main components for easier access
from acoustic.data import ArrayPositions
from acoustic.datasets import AcousticArrayDataset, get_acoustic_array_dataloader
from acoustic.beamforming import (
    ConventionalBeamformingTorch,
    get_steering_vector,
    index_to_position, 
    calculate_noise_scm_formula,
    get_fb_map, 
    create_blurred_image
)

__all__ = [
    'ArrayPositions',
    'AcousticArrayDataset', 
    'get_acoustic_array_dataloader',
    'ConventionalBeamformingTorch',
    'get_steering_vector',
    'index_to_position', 
    'calculate_noise_scm_formula',
    'get_fb_map',
    'create_blurred_image'
]
