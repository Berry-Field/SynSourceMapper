"""
Acoustic Imaging Dataset Module

This package provides tools for acoustic source localization using microphone arrays.

Main Components:
    - LinearImagingDataset: PyTorch dataset for sparse acoustic source imaging
    - LinearForwardOperator: Implicit forward operator for y = Ax
    - build_dataloader: Convenient dataloader factory with configuration support
    - load_yaml_config: Configuration loader with PROJECT_ROOT expansion

Usage:
    from datasets.acoustic_imaging import build_dataloader
    
    dl = build_dataloader(
        cfg_path="configs/dataset_1.yaml",
        batch_size=32,
        need={"source_map", "target_map", "observation_map"}
    )
"""

from .acoustic_imaging import (
    load_yaml_config,
    LinearImagingDataset,
    LinearForwardOperator,
    build_dataloader,
    ScanGrid,
    MicrophoneArray,
    SteeringOperators,
    BlurOperator,
)

__all__ = [
    "load_yaml_config",
    "LinearImagingDataset",
    "LinearForwardOperator",
    "build_dataloader",
    "ScanGrid",
    "MicrophoneArray",
    "SteeringOperators",
    "BlurOperator",
]

__version__ = "1.0.0"

