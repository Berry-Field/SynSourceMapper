# SynSourceMapper

A PyTorch-based synthetic dataset generator for acoustic source localization using microphone arrays.

## Quick Start

### 1. Environment Setup

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.9.0

**Installation:**

```bash
# Clone the repository
git clone https://github.com/yourusername/SynSourceMapper.git
cd SynSourceMapper

# Install dependencies
pip install torch numpy pyyaml

# Optional: for .mat file support
pip install scipy
```

### 2. Run Test

Run the test script to verify installation and generate a sample batch:

```bash
python tst.py
```

**This will:**
1. Load configuration from `configs/dataset_1.yaml`
2. Create a dataloader with specified fields
3. Generate one batch of synthetic acoustic data
4. Print all tensor shapes and sample details
5. Create the `cache_spDataset/` directory for storing computed matrices
6. Display detailed statistics for each field

**Expected output:**
```
Step 1: Loading Configuration
  ✓ Configuration file loaded

Step 2: Specifying Dataset Fields
  ✓ Required fields: source_map, target_map, observation_map, ...

Step 3: Creating DataLoader
  ✓ Dataset size: 25600 samples, Batch size: 4

Step 4: Generating One Batch
  Cache miss, building A ...
  Saved A to cache_spDataset\array_2\A_matrix\...\A.pt
  ✓ Batch generated

Step 5: Batch Content Analysis
  📊 IMAGE MAPS:
    source_map:        (4, 1, 41, 41)  dtype=torch.float32
    target_map:        (4, 1, 41, 41)  dtype=torch.float32
    observation_map:   (4, 1, 41, 41)  dtype=torch.float32
  
  🎯 SOURCE METADATA:
    num_sources:       (4,)            Values: [1, 1, 1, 1]
    source_positions_xy:(4, 1, 2)      dtype=torch.float32
    source_energies:   (4, 1)          dtype=torch.float32
  
  🔢 FORWARD MATRIX:
    A_matrix:          (1681, 1681)    Memory: 10.74 MB

Step 6: Detailed Sample Inspection
  Number of sources: 1
  Source 0: position=(0.123, -1.456) m, energy=0.8234

Step 7: Test Summary
  ✅ All tests passed successfully!
  ✅ Dataset is ready for training/evaluation!
```

### 3. Use in Your Code

**Basic usage:**

```python
from datasets.acoustic_imaging import build_dataloader

# Create dataloader
dl = build_dataloader(
    cfg_path="configs/dataset_1.yaml",
    batch_size=32,
    need={"source_map", "target_map", "observation_map"},
    dataset_device="cpu"  # or "cuda" if GPU available
)

# Set random seed for reproducibility
dl.dataset.set_seed(42)

# Iterate through batches
for batch in dl:
    source = batch["source_map"]        # (B, 1, H, W) - Sparse sources
    target = batch["target_map"]        # (B, 1, H, W) - Blurred targets
    observation = batch["observation_map"]  # (B, 1, H, W) - Observations
    
    # Your training/evaluation code here
    ...
```

### 4. Configuration

Edit `configs/dataset_1.yaml` to customize:

```yaml
scan_set:
  array_id: 2              # Microphone array ID
  x_range: [-2, 2]         # Scan area X range (meters)
  y_range: [-2, 2]         # Scan area Y range (meters)
  z_scan: 2                # Scan plane height (meters)
  x_grid: 41               # Grid resolution X
  y_grid: 41               # Grid resolution Y

source_set:
  freq: [15000]            # Source frequencies (Hz)
  num_samples: 25600       # Dataset size
  nos: [1]                 # Number of sources per sample
  min_source_distance: 3   # Min distance between sources (grid units)
  minEnergy: 0.1          # Minimum source energy

target_map_set:
  blur: true              # Enable/disable blurring
  lowk: 3                 # Blur decay coefficient

cache_dir: cache_spDataset  # Cache directory for matrices
```

### 5. Available Output Fields

Specify which fields to compute using the `need` parameter:

| Field | Shape | Description |
|-------|-------|-------------|
| `source_map` | (B, 1, H, W) | Sparse source strength map |
| `target_map` | (B, 1, H, W) | Blurred target map |
| `observation_map` | (B, 1, H, W) | Observation map y = Ax |
| `psf_center_map` | (B, 1, H, W) | Point spread function |
| `A_matrix` | (N, N) | Forward matrix (cached to disk) |
| `num_sources` | (B,) | Number of active sources |
| `source_indices_xy` | (B, max_nos, 2) | Grid indices |
| `source_positions_xy` | (B, max_nos, 2) | Physical positions (m) |
| `source_energies` | (B, max_nos) | Source energies |
| `source_mask` | (B, max_nos) | Validity mask |
| `analysis_frequency_hz` | (B,) | Analysis frequency |
| `source_frequency_hz` | (B,) | Source frequency |
| `grid_hw` | (B, 2) | Grid shape [H, W] |

**Note:** Only requested fields are computed (lazy evaluation).

## Project Structure

```
SynSourceMapper/
├── arraybox/              # Microphone array geometries
│   ├── __init__.py
│   ├── array_2.npy       # Example array (MIRACLE)
│   └── get_array_positions.py
├── configs/               # Configuration files
│   └── dataset_1.yaml
├── datasets/              # Dataset module
│   ├── __init__.py
│   └── acoustic_imaging.py  # Main module
├── cache_spDataset/       # Auto-created cache directory
├── tst.py                 # Test script
├── .gitignore
└── README.md
```

## Tips

- **First run:** The first time you request `A_matrix`, it will be computed and cached to disk. Subsequent runs will load from cache (~10MB for 41x41 grid).
- **GPU acceleration:** Set `dataset_device="cuda"` for GPU computation (faster for large grids).
- **Memory optimization:** If you don't need the explicit `A_matrix`, exclude it from `need` to save memory.
- **Reproducibility:** Always call `dl.dataset.set_seed(seed)` before iteration for reproducible results.

## Supported Arrays

- **Array 2**: MIRACLE array (default, included in repo)

To add custom arrays, place geometry files (`.npy`, `.mat`, or `.csv`) in the `arraybox/` directory.

## License

MIT License - see LICENSE file

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jia2025dual,
  title={A dual-encoder U-net architecture with prior knowledge embedding for acoustic source mapping},
  author={Jia, Haobo and Yang, Feiran and Hu, Xiaoqing and Yang, Jun},
  journal={The Journal of the Acoustical Society of America},
  volume={158},
  number={3},
  pages={1767--1782},
  year={2025},
  publisher={AIP Publishing}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

