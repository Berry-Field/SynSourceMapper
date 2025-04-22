import torch
import numpy as np
import os
from acoustic.utils import freq2fbin, fbin2freq
from acoustic.utils.path_utils import get_A_matrix_cache_path

def cal_A_matrix(ws: torch.Tensor, dvecs: torch.Tensor, array_positions: np.ndarray, fs: int = 16000, 
                x_range: tuple = (-50, 50), y_range: tuple = (-50, 50), 
                x_grid: int = 101, y_grid: int = 101, z_scan: int = 50,
                c: int = 343, nfft: int = 1024, 
                ana_freq: int = 100, 
                print_flag: bool = True,
                device: str = 'cpu',
                batch_size: int = 101,
                if_ref_distance_1: bool = False,
                config_params: dict = None) -> torch.Tensor:
    """Calculate A matrix
    
    Args:
        ws: Steering vectors, shape (n_scan_points, n_channels)
        dvecs: Delay vectors, shape (n_scan_points, n_channels)
        array_positions: Array element positions, shape (n_channels, 3)
        fs: Sampling rate, Hz
        x_range: X-axis scanning range (min, max), meters
        y_range: Y-axis scanning range (min, max), meters
        x_grid: Number of grid divisions on X-axis
        y_grid: Number of grid divisions on Y-axis
        z_scan: Scanning plane height, meters
        c: Speed of sound, m/s
        nfft: FFT points for STFT
        ana_freq: Analysis frequency, Hz
        print_flag: Whether to print intermediate process information
        device: Computing device, 'cuda' or 'cpu'
        batch_size: Batch processing size
        if_ref_distance_1: Whether to set reference distance to 1
        config_params: Configuration parameter dictionary, may include cache_dir to specify cache directory
        
    Returns:
        torch.Tensor: A matrix, shape (n_scan_points, n_scan_points)
    """
    if if_ref_distance_1:
        if_ref_distance_1_flag = "ref_1"
    else:
        if_ref_distance_1_flag = "ref_source"

    # Calculate actual frequency bin and frequency
    act_bin = freq2fbin(ana_freq, fs, nfft)
    act_freq = fbin2freq(act_bin, fs, nfft)
    
    # Build configuration parameter dictionary
    if config_params is None:
        config_params = {
            'if_ref_distance_1': if_ref_distance_1,
            'c': c,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'x_range': x_range,
            'y_range': y_range,
            'z_scan': z_scan,
            'fs': fs,
            'act_bin': act_bin,
            'nfft': nfft,
            'act_freq': act_freq
        }
    else:
        # Update with current values if not already present
        if 'act_bin' not in config_params:
            config_params['act_bin'] = act_bin
        if 'act_freq' not in config_params:
            config_params['act_freq'] = act_freq
    
    # Get cache file path
    temp_file = get_A_matrix_cache_path(config_params)
    
    try:
        # Try to load existing tensor
        A = torch.load(temp_file, map_location=device)
        if print_flag:
            print(f"Loaded A matrix from {temp_file}")
        return A.to(device)  # Ensure return to CUDA
    except (FileNotFoundError, RuntimeError) as e:
        if print_flag:
            print(f"Temporary file not found or loading failed: {e}, starting to calculate A matrix...")
    
    n_scan_points = x_grid * y_grid
    n_channels = array_positions.shape[0]
    
    # Convert data to torch complex tensors and move to GPU
    if isinstance(ws, np.ndarray):
        print(f"Converting ws from numpy array to torch tensor, shape: {ws.shape}")
        ws_torch = torch.from_numpy(ws).to(device).type(torch.complex64)  # Use complex type
    else:
        print(f"ws is already torch tensor, shape: {ws.shape}")
        ws_torch = ws.to(device).type(torch.complex64)
        
    if isinstance(dvecs, np.ndarray):
        print(f"Converting dvecs from numpy array to torch tensor, shape: {dvecs.shape}")
        dvecs_torch = torch.from_numpy(dvecs).to(device).type(torch.complex64)  # Use complex type
    else:
        print(f"dvecs is already torch tensor, shape: {dvecs.shape}")
        dvecs_torch = dvecs.to(device).type(torch.complex64)
    
    # Calculate W D^H
    # W: (N, C), D: (N, C)
    # D_H: (C, N)
    D_H = dvecs_torch.conj().transpose(0, 1)  # (C, N)
    WD_H = torch.matmul(ws_torch, D_H)  # (N, N)
    
    # Calculate |WD_H|^2 / C^2
    A = torch.abs(WD_H) ** 2 / (n_channels ** 2)  # (N, N)
    
    # Save CPU version
    try:
        A_cpu = A.cpu()
        torch.save(A_cpu, temp_file)
        if print_flag:
            print(f"A matrix saved to {temp_file}, shape: {A.shape}")
    except Exception as e:
        print(f"Warning: Failed to save A matrix: {e}")
    
    return A  # Return CUDA version directly 