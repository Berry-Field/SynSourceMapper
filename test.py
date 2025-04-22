import argparse
import yaml
import torch
from acoustic import AcousticArrayDataset

def main():
    parser = argparse.ArgumentParser(description='Acoustic Dataset Generator')
    parser.add_argument('--config', type=str, default='example_config.yaml', 
                        help='Configuration file path')
    parser.add_argument('--device', type=str, default='cuda:7', 
                        help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override seed if specified
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    print("Initializing dataset...")
    dataset = AcousticArrayDataset(config, device)
    dataset.set_seed(config['seed'])
    
    # Generate a sample
    print(f"Generating a sample with seed {config['seed']}...")
    sample = dataset[0]
    
    # Print sample information
    print_sample_info(sample)
    
    print("Done!")

def print_sample_info(sample):
    """Print information about a sample from the dataset"""
    target_map_tensor, nos_tensor, analysis_f_tensor, source_f_tensor, A_matrix, \
    y_tensor, PSF_tensor, source_ac_position_index, source_positions, x_tensor, \
    source_energies, scm_tensor, *remaining = sample
    
    print("Sample information:")
    print(f"Number of sources: {nos_tensor.item()}")
    print(f"Analysis frequency: {analysis_f_tensor.item():.1f} Hz")
    print(f"Source frequency: {source_f_tensor.item():.1f} Hz")
    print(f"Source positions (grid indices):")
    for i in range(nos_tensor.item()):
        x_idx, y_idx = source_ac_position_index[0, i], source_ac_position_index[1, i]
        print(f"  Source {i+1}: ({x_idx}, {y_idx})")
    
    print(f"Source positions (physical coordinates):")
    for i in range(nos_tensor.item()):
        x, y = source_positions[0, i], source_positions[1, i]
        print(f"  Source {i+1}: ({x:.2f}, {y:.2f})")
    
    print(f"Source energies:")
    for i in range(nos_tensor.item()):
        print(f"  Source {i+1}: {source_energies[0, i]:.4f}")
    
    print(f"Tensor shapes:")
    print(f"  Target map: {target_map_tensor.shape}, max: {target_map_tensor.max():.1f}, min: {target_map_tensor.min():.1f}")
    print(f"  Original image: {x_tensor.shape}, max: {x_tensor.max():.1f}, min: {x_tensor.min():.1f}")
    print(f"  DAS result: {y_tensor.shape}, max: {y_tensor.max():.1f}, min: {y_tensor.min():.1f}")
    print(f"  PSF-middle: {PSF_tensor.shape}, max: {PSF_tensor.max():.1f}, min: {PSF_tensor.min():.1f}")
    # print(f"  SCM: {scm_tensor.shape}, max: {torch.abs(scm_tensor).max():.1f}, min: {torch.abs(scm_tensor).min():.1f}")

   
    
    # Call the visualization function
    _visualize_sample(sample)
    
    if len(remaining) > 0:
        FB_map_tensor = remaining[0]
        print(f"  FB map: {FB_map_tensor.shape}, max: {FB_map_tensor.max()}, min: {FB_map_tensor.min()}")

 # Visualize the sample data in a single figure
def _visualize_sample(sample):
    """Visualize sample data in a single figure with multiple subplots"""
    import matplotlib.pyplot as plt
    import numpy as np
    from acoustic.utils.path_utils import get_project_root
    import os
    
    target_map_tensor, _, analysis_f_tensor, _, _, \
    y_tensor, PSF_tensor, _, _, x_tensor, \
    _, _, *remaining = sample
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Extract data from tensors (remove batch dimension and convert to numpy)
    target_map = target_map_tensor[0].cpu().numpy()
    original_image = x_tensor[0].cpu().numpy()
    das_result = y_tensor[0].cpu().numpy()
    psf = PSF_tensor[0].cpu().numpy()
    
    # Plot target map
    im0 = axes[0].imshow(target_map, cmap='viridis')
    axes[0].set_title('Target Map')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot original image
    im1 = axes[1].imshow(original_image, cmap='viridis')
    axes[1].set_title('Original Image')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot DAS result
    im2 = axes[2].imshow(das_result, cmap='viridis')
    axes[2].set_title('DAS Result')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Plot PSF
    im3 = axes[3].imshow(psf, cmap='viridis')
    axes[3].set_title('PSF')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Plot FB map if available
    if len(remaining) > 0:
        FB_map_tensor = remaining[0]
        fb_map = FB_map_tensor[0].cpu().numpy()
        im4 = axes[4].imshow(fb_map, cmap='viridis')
        axes[4].set_title('FB Map')
        plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    else:
        axes[4].set_visible(False)
    
    # Add frequency information to the figure title
    fig.suptitle(f'Acoustic Imaging Results at {analysis_f_tensor.item():.1f} Hz', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig("test.png")
    print("Figure saved to test.png")

if __name__ == "__main__":
    main() 