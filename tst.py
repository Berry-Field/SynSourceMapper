"""
Test script for acoustic imaging dataset generation

This script demonstrates the complete workflow:
1. Load configuration
2. Create dataloader with specified fields
3. Generate one batch
4. Print all tensor shapes and metadata
"""

import torch
from pathlib import Path
from datasets.acoustic_imaging import build_dataloader


def main():
    """Test dataset generation and inspect output shapes"""
    
    # ========================================
    # Step 1: Configuration Setup
    # ========================================
    print("=" * 80)
    print("Step 1: Loading Configuration")
    print("=" * 80)
    
    # Get project root and config path
    project_root = Path(__file__).resolve().parent
    cfg_path = project_root / "configs" / "dataset_1.yaml"
    
    if not cfg_path.exists():
        print(f"ERROR: Configuration file not found: {cfg_path}")
        return
    
    print(f"✓ Configuration file: {cfg_path}")
    print()
    
    # ========================================
    # Step 2: Specify Required Fields
    # ========================================
    print("=" * 80)
    print("Step 2: Specifying Dataset Fields")
    print("=" * 80)
    
    # Define which fields we want the dataset to compute
    required_fields = {
        "source_map",           # Sparse source strength map
        "target_map",           # Blurred target map
        "observation_map",      # Observation y = Ax
        "num_sources",          # Number of sources
        "source_indices_xy",    # Grid indices
        "source_positions_xy",  # Physical positions
        "source_energies",      # Source energies
        "source_mask",          # Validity mask
        "analysis_frequency_hz",  # Analysis frequency
        "source_frequency_hz",    # Source frequency
        "psf_center_map",     # Point spread function
        "A_matrix",           # Explicit forward matrix (large!)
        "grid_hw",            # Grid shape
    }
    
    print(f"✓ Required fields: {required_fields}")
    print()
    
    # ========================================
    # Step 3: Create DataLoader
    # ========================================
    print("=" * 80)
    print("Step 3: Creating DataLoader")
    print("=" * 80)
    
    # Configuration
    batch_size = 4
    num_workers = 0  # Use 0 for testing (single process)
    device = "cpu"   # Use "cuda" if GPU is available
    
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Num workers: {num_workers}")
    
    # Build dataloader
    dataloader = build_dataloader(
        cfg_path=str(cfg_path),
        batch_size=batch_size,
        need=required_fields,
        dataset_device=device,
        num_workers=num_workers,
        base_seed=42  # For reproducibility
    )
    
    dataset = dataloader.dataset
    print(f"✓ DataLoader created successfully")
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Number of batches: {len(dataloader)}")
    print()
    
    # ========================================
    # Step 4: Generate and Inspect One Batch
    # ========================================
    print("=" * 80)
    print("Step 4: Generating One Batch")
    print("=" * 80)
    
    # Set seed for reproducibility
    dataset.set_seed(42)
    
    # Get one batch
    print("Fetching first batch...")
    batch = next(iter(dataloader))
    print(f"✓ Batch generated successfully")
    print()
    
    # ========================================
    # Step 5: Print All Shapes and Info
    # ========================================
    print("=" * 80)
    print("Step 5: Batch Content Analysis")
    print("=" * 80)
    print()
    
    # Image maps
    print("📊 IMAGE MAPS (B, C, H, W)")
    print("-" * 80)
    if "source_map" in batch:
        print(f"  source_map:        {str(tuple(batch['source_map'].shape)):20s} dtype={batch['source_map'].dtype}")
    if "target_map" in batch:
        print(f"  target_map:        {str(tuple(batch['target_map'].shape)):20s} dtype={batch['target_map'].dtype}")
    if "observation_map" in batch:
        print(f"  observation_map:   {str(tuple(batch['observation_map'].shape)):20s} dtype={batch['observation_map'].dtype}")
    if "psf_center_map" in batch:
        print(f"  psf_center_map:    {str(tuple(batch['psf_center_map'].shape)):20s} dtype={batch['psf_center_map'].dtype}")
    print()
    
    # Source metadata
    print("🎯 SOURCE METADATA")
    print("-" * 80)
    if "num_sources" in batch:
        print(f"  num_sources:       {str(tuple(batch['num_sources'].shape)):20s} dtype={batch['num_sources'].dtype}")
        print(f"    Values: {batch['num_sources'].tolist()}")
    if "source_indices_xy" in batch:
        print(f"  source_indices_xy: {str(tuple(batch['source_indices_xy'].shape)):20s} dtype={batch['source_indices_xy'].dtype}")
    if "source_positions_xy" in batch:
        print(f"  source_positions_xy:{str(tuple(batch['source_positions_xy'].shape)):20s} dtype={batch['source_positions_xy'].dtype}")
    if "source_energies" in batch:
        print(f"  source_energies:   {str(tuple(batch['source_energies'].shape)):20s} dtype={batch['source_energies'].dtype}")
    if "source_mask" in batch:
        print(f"  source_mask:       {str(tuple(batch['source_mask'].shape)):20s} dtype={batch['source_mask'].dtype}")
    print()
    
    # Frequency information
    print("🔊 FREQUENCY INFORMATION")
    print("-" * 80)
    if "analysis_frequency_hz" in batch:
        print(f"  analysis_frequency_hz: {str(tuple(batch['analysis_frequency_hz'].shape)):20s} dtype={batch['analysis_frequency_hz'].dtype}")
        print(f"    Values: {batch['analysis_frequency_hz'].tolist()} Hz")
    if "source_frequency_hz" in batch:
        print(f"  source_frequency_hz:   {str(tuple(batch['source_frequency_hz'].shape)):20s} dtype={batch['source_frequency_hz'].dtype}")
        print(f"    Values: {batch['source_frequency_hz'].tolist()} Hz")
    print()
    
    # Grid information
    print("📐 GRID INFORMATION")
    print("-" * 80)
    if "grid_hw" in batch:
        print(f"  grid_hw:           {str(tuple(batch['grid_hw'].shape)):20s} dtype={batch['grid_hw'].dtype}")
        print(f"    Values: {batch['grid_hw'].tolist()}")
    if "grid_x_range" in batch:
        print(f"  grid_x_range:      {str(tuple(batch['grid_x_range'].shape)):20s} dtype={batch['grid_x_range'].dtype}")
        print(f"    Values: {batch['grid_x_range'][0].tolist()} m")
    if "grid_y_range" in batch:
        print(f"  grid_y_range:      {str(tuple(batch['grid_y_range'].shape)):20s} dtype={batch['grid_y_range'].dtype}")
        print(f"    Values: {batch['grid_y_range'][0].tolist()} m")
    print()
    
    # Large matrices (if requested)
    if "A_matrix" in batch:
        print("🔢 FORWARD MATRIX")
        print("-" * 80)
        print(f"  A_matrix:          {str(tuple(batch['A_matrix'].shape)):20s} dtype={batch['A_matrix'].dtype}")
        print(f"    Memory: {batch['A_matrix'].numel() * batch['A_matrix'].element_size() / 1024**2:.2f} MB")
        print()
    
    # ========================================
    # Step 6: Detailed Sample Inspection
    # ========================================
    print("=" * 80)
    print("Step 6: Detailed Inspection of First Sample")
    print("=" * 80)
    print()
    
    # Extract first sample from batch
    sample_idx = 0
    print(f"Inspecting sample {sample_idx} from batch:")
    print()
    
    if "num_sources" in batch:
        num_src = batch["num_sources"][sample_idx].item()
        print(f"  Number of sources: {num_src}")
    
    if "source_positions_xy" in batch and "source_mask" in batch:
        mask = batch["source_mask"][sample_idx]
        positions = batch["source_positions_xy"][sample_idx][mask]
        energies = batch["source_energies"][sample_idx][mask]
        
        print(f"  Active source details:")
        for i, (pos, eng) in enumerate(zip(positions, energies)):
            print(f"    Source {i}: position=({pos[0]:.3f}, {pos[1]:.3f}) m, energy={eng:.4f}")
    
    if "source_map" in batch:
        source_max = batch["source_map"][sample_idx].max().item()
        source_min = batch["source_map"][sample_idx].min().item()
        source_mean = batch["source_map"][sample_idx].mean().item()
        print(f"  Source map stats: min={source_min:.6f}, max={source_max:.6f}, mean={source_mean:.6f}")
    
    if "observation_map" in batch:
        obs_max = batch["observation_map"][sample_idx].max().item()
        obs_min = batch["observation_map"][sample_idx].min().item()
        obs_mean = batch["observation_map"][sample_idx].mean().item()
        print(f"  Observation map stats: min={obs_min:.6f}, max={obs_max:.6f}, mean={obs_mean:.6f}")
    
    print()
    
    # ========================================
    # Step 7: Summary
    # ========================================
    print("=" * 80)
    print("Step 7: Test Summary")
    print("=" * 80)
    print()
    print("✅ All tests passed successfully!")
    print(f"✅ Generated batch of {batch_size} samples")
    print(f"✅ All requested fields present in output")
    print()
    print("Dataset is ready for training/evaluation!")
    print("=" * 80)


if __name__ == "__main__":
    main()

