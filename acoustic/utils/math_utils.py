import torch
import numpy as np
import os

def fbin2freq(bin_index: int, fs: int, Nfft: int) -> float:
    """Convert frequency bin index to frequency in Hz.
    
    Args:
        bin_index: Frequency bin index
        fs: Sampling frequency in Hz
        Nfft: FFT size
        
    Returns:
        freq: Frequency in Hz corresponding to the bin index
    """
    freq = round((bin_index * fs) / Nfft, 1)
    return freq


def freq2fbin(analyze_f: float, fs: int, Nfft: int) -> int:
    """Convert frequency in Hz to frequency bin index.
    
    Args:
        analyze_f: Frequency to analyze in Hz
        fs: Sampling frequency in Hz
        Nfft: FFT size
        
    Returns:
        bin_num: Frequency bin index corresponding to the frequency
    """
    bin_num = round(analyze_f * Nfft / fs)
    return bin_num


def normalize_tensor(tensor: torch.Tensor, norm_range: list) -> torch.Tensor:
    """Normalize tensor to norm_range
    
    Args:
        tensor: Input tensor
        norm_range: Normalization range e.g. [0,1]
        
    Returns:
        normalized_tensor: Normalized tensor
    """
    return tensor * (norm_range[1] - norm_range[0]) + norm_range[0] 