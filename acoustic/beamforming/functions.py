import math
import numpy as np
import torch
from acoustic.utils import fbin2freq

def index_to_position(x_idx, y_idx, x_range, y_range, x_grid, y_grid):
    """Convert grid indices to physical coordinates
    
    Args:
        x_idx (int): Grid index in x direction
        y_idx (int): Grid index in y direction
        x_range (tuple): x-axis scan range (min, max), in meters
        y_range (tuple): y-axis scan range (min, max), in meters
        x_grid (int): Number of grid divisions in x direction
        y_grid (int): Number of grid divisions in y direction
        
    Returns:
        tuple: (x, y) physical coordinates
    """
    # Calculate step size in x direction
    x_step = (x_range[1] - x_range[0]) / (x_grid - 1)
    # Calculate step size in y direction
    y_step = (y_range[1] - y_range[0]) / (y_grid - 1)

    x = x_range[0] + x_idx * x_step
    y = y_range[0] + y_idx * y_step
    return x, y

def calculate_noise_scm_formula(ws, device):
    """Calculate noise spatial correlation matrix formula
    
    Args:
        ws (torch.Tensor): Steering vector
        device (torch.device): Computing device
        
    Returns:
        torch.Tensor: Noise spatial correlation matrix formula
    """
    return torch.real(torch.einsum('ij,jk,ik->i', ws, torch.eye(ws.shape[1], dtype=torch.cfloat, device=device), ws.conj())) / (ws.shape[1])

def get_fb_map(scm, ws, nch, fb_order=10):
    """Get Functional Beamforming (FB) results

    Args:
        scm (torch.Tensor): Spatial correlation matrix
        ws (torch.Tensor): Steering vector
        nch (int): Number of channels
        fb_order (int, optional): FB order. Defaults to 10.
        
    Returns:
        torch.Tensor: FB beamforming results
    """
    # Perform eigenvalue decomposition on the spatial correlation matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(scm)  # eigenvalues in ascending order
    eigenvalues_new = torch.sign(eigenvalues) * (torch.abs(eigenvalues) ** (1/fb_order))
    eigenvalues_diag = torch.diag(eigenvalues_new).clone().detach()
    eigenvalues_diag = eigenvalues_diag.to(torch.complex64)
    R_rec = torch.matmul(eigenvectors, torch.matmul(eigenvalues_diag, eigenvectors.conj().T))
    
    # Calculate FB beamforming results
    FB_map = torch.real(torch.einsum('ij,jk,ik->i', ws, R_rec, ws.conj())) / nch
    FB_map = FB_map ** (fb_order)

    return FB_map / nch

def create_blurred_image(image, x_grid, y_grid, x_resolution, y_resolution, blur_level):
    """Create a blurred image
    
    Args:
        image (np.ndarray): Original image
        x_grid (int): Number of grids in x direction
        y_grid (int): Number of grids in y direction
        x_resolution (float): Resolution in x direction
        y_resolution (float): Resolution in y direction
        blur_level (float): Blur level parameter
        
    Returns:
        np.ndarray: Blurred image
    """
    xxx, yyy = np.ogrid[:x_grid, :y_grid]
    blurred_image = np.zeros_like(image)
    
    for j in range(y_grid):
        for i in range(x_grid):
            if image[i, j] > 0:
                mask = 10 ** (-blur_level * (np.sqrt(((xxx - i) * x_resolution)**2 + 
                                              ((yyy - j) * y_resolution)**2)) / 1)
                blurred_image += image[i, j] * mask
                
    return blurred_image 