# SynSourceMapper

A Python package for synthetic acoustic source mapping dataset generation and beamforming.

## Project Structure

```
SynSourceMapper/
├── acoustic/                    # Main package directory
│   ├── beamforming/            # Beamforming algorithms
│   │   ├── conventional.py     # Conventional beamforming implementation
│   │   └── functions.py        # Utility functions for beamforming
│   ├── data/                   # Data-related modules
│   │   └── array_positions.py  # Microphone array positions
│   ├── datasets/               # Dataset generation
│   │   └── dataset.py          # AcousticArrayDataset implementation
│   ├── utils/                  # Utility modules
│   │   ├── math_utils.py       # Mathematical utilities
│   │   ├── matrix_utils.py     # Matrix operations
│   │   └── path_utils.py       # Path handling utilities
│   └── __init__.py            # Package initialization
├── test.py                     # Test script
├── pyproject.toml             # Project configuration and dependencies
├── example_config.yaml        # Example configuration file
└── README.md                  # This file
```

## Installation

### Option 1: Using uv

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
uv pip install -e .
```

### Option 2: Using Conda

1. Create and activate conda environment:
```bash
conda create -n synsource python=3.8
conda activate synsource
```

2. Install dependencies:
```bash
pip install -e .
```

## Testing

Run the test script to verify the installation:
```bash
python test.py
```

This will:
1. Load the example configuration
2. Generate a sample dataset
3. Display sample information
4. Save visualization results to `test.png`
