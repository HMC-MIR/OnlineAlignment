Pacakge for online alignment with examples of NOA and OLTW.

# Installation Guide

This guide shows how to install the `online-alignment` package and its dependencies using conda.

## Option 1: Use Existing Environment

If you already have the `online_alignment` conda environment:

```bash
# Activate the environment
conda activate online_alignment

# Install core dependencies
conda install -c conda-forge numpy>=1.20.0 numba>=0.56.0

# Install the package in editable mode
pip install -e .

# (Optional) Install dev dependencies for testing/development
pip install -e ".[dev]"
```

## Option 2: Create New Environment from File

### For basic usage:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate online-alignment
```

### For development (includes test tools):

```bash
# Create environment from environment-dev.yml
conda env create -f environment-dev.yml

# Activate the environment
conda activate online-alignment-dev
```

## Option 3: Manual Installation

### Step 1: Create/Activate Environment

```bash
# Create a new environment (or use existing)
conda create -n online-alignment python=3.10
conda activate online-alignment
```

### Step 2: Install Core Dependencies

```bash
# Install numpy and numba via conda (recommended for better compatibility)
conda install -c conda-forge numpy>=1.20.0 numba>=0.56.0

# Or install via pip
pip install "numpy>=1.20.0" "numba>=0.56.0"
```

### Step 3: Install the Package

```bash
# Navigate to project directory
cd /Users/jeudi/Desktop/OnlineAlignment

# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

### Step 4: (Optional) Install Dev Dependencies

```bash
# Install development dependencies (pytest, black, flake8, mypy)
pip install -e ".[dev]"

# Or install individually
pip install "pytest>=7.0.0" "black>=22.0.0" "flake8>=4.0.0" "mypy>=0.950"
```

## Option 4: Install Pre-built Wheels from GitHub Releases

For users wanting to simply drop the library into their project without cloning the source code, you can install the pre-built wheels directly from GitHub Releases:

```bash
# Navigate to the GitHub Releases page: https://github.com/HMC-MIR/OnlineAlignment/releases
# Copy the link to the latest .whl file and run:
pip install https://github.com/HMC-MIR/OnlineAlignment/releases/download/v0.1.3/online_alignment-0.1.3-py3-none-any.whl

# Or install the source tarball
pip install https://github.com/HMC-MIR/OnlineAlignment/archive/refs/tags/v0.1.3.tar.gz
```

## Verify Installation

```bash
# Activate your environment
conda activate online_alignment  # or your environment name

# Test import
python -c "from online_alignment.cost import CosineDistance, EuclideanDistance; print('Installation successful!')"

# Run tests (if dev dependencies installed)
pytest tests/ -v
```

## Troubleshooting

### If numba installation fails:

```bash
# Try installing from conda-forge
conda install -c conda-forge numba
```

### If you get import errors:

```bash
# Make sure you're in the project directory and environment is activated
conda activate online_alignment
cd /Users/jeudi/Desktop/OnlineAlignment
python -c "import sys; print(sys.executable)"  # Should show conda env path
```

### Update dependencies:

```bash
conda activate online_alignment
pip install --upgrade -e ".[dev]"
```
