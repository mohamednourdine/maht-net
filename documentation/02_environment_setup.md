# Environment Setup for MAHT-Net Implementation

## System Requirements

### Hardware Specifications
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3070/4060 Ti or better)
- **RAM**: Minimum 16GB, recommended 32GB
- **Storage**: At least 50GB free space for datasets and model checkpoints
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
- **OS**: macOS (current), Linux (for production), or Windows with WSL2
- **Python**: 3.8-3.11 (3.10 recommended)
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)

## Python Environment Setup

### Step 1: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n maht-net python=3.10
conda activate maht-net

# Alternative: Using venv
python -m venv maht-net-env
source maht-net-env/bin/activate  # On macOS/Linux
```

### Step 2: Core Dependencies Installation

```bash
# PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Essential scientific computing
pip install numpy scipy scikit-learn pandas matplotlib seaborn

# Image processing
pip install opencv-python Pillow scikit-image albumentations

# Medical imaging
pip install nibabel pydicom SimpleITK

# Visualization and monitoring
pip install tensorboard wandb plotly

# Transformer libraries
pip install transformers timm einops

# Additional utilities
pip install tqdm rich click pyyaml jsonschema
```

### Step 3: Development Tools

```bash
# Code quality and formatting
pip install black isort flake8 mypy pre-commit

# Jupyter environment
pip install jupyter jupyterlab ipywidgets

# Testing framework
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme
```

## Project Structure Setup

### Step 4: Create Project Directory Structure

```bash
cd /var/www/phd-researches/maht-net

# Create main directories
mkdir -p {src,data,experiments,notebooks,tests,configs,checkpoints,logs,results}

# Create source code structure
mkdir -p src/{models,datasets,training,evaluation,utils,visualization}

# Create configuration structure
mkdir -p configs/{models,training,datasets}

# Create data structure
mkdir -p data/{raw,processed,annotations,augmented}
```

### Expected Directory Structure
```
maht-net/
├── documentation/          # Project documentation
├── src/                   # Source code
│   ├── models/           # Model implementations
│   ├── datasets/         # Dataset handling
│   ├── training/         # Training pipeline
│   ├── evaluation/       # Evaluation metrics
│   ├── utils/           # Utility functions
│   └── visualization/   # Plotting and visualization
├── data/                 # Dataset storage
│   ├── raw/             # Original ISBI 2015 data
│   ├── processed/       # Preprocessed images
│   ├── annotations/     # Ground truth labels
│   └── augmented/       # Augmented datasets
├── configs/             # Configuration files
├── experiments/         # Experiment scripts
├── notebooks/          # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── checkpoints/        # Model checkpoints
├── logs/               # Training logs
├── results/            # Experiment results
└── requirements.txt    # Python dependencies
```

## Configuration Management

### Step 5: Create Requirements File

```bash
# Generate requirements.txt
pip freeze > requirements.txt
```

### Step 6: Git Setup

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial project setup"

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# PyTorch
*.pth
*.pt

# Data files
data/raw/
data/processed/
*.dcm
*.dicom

# Logs and checkpoints
logs/
checkpoints/
wandb/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

## Development Environment Configuration

### Step 7: VSCode Setup (Recommended)

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./maht-net-env/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "files.autoSave": "onDelay",
    "files.autoSaveDelay": 1000
}
```

### Step 8: Pre-commit Hooks

```bash
# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
EOF

# Install pre-commit hooks
pre-commit install
```

## Verification Steps

### Step 9: Environment Validation

## Performance Optimization

### Memory Management Settings

```python
# Add to your training scripts
import torch

# Enable memory efficient attention
torch.backends.cuda.enable_flash_sdp(True)

# Set memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.9)

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

### Monitoring Setup

```bash
# Install monitoring tools
pip install gpustat htop

# Monitor GPU usage
gpustat -i 1

# Monitor system resources
htop
```

## Troubleshooting Common Issues

### CUDA Installation Issues
1. **Error**: "CUDA not available"
   - **Solution**: Reinstall PyTorch with correct CUDA version
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Error**: "CUDA out of memory"
   - **Solution**: Reduce batch size or enable gradient checkpointing

### Package Conflicts
1. **Error**: Version incompatibilities
   - **Solution**: Use conda environment with pinned versions
   ```bash
   conda install pytorch=1.13 torchvision=0.14 -c pytorch
   ```

### macOS Specific Issues
1. **MPS Backend**: For Apple Silicon Macs
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

## Next Steps

After completing environment setup:
1. Proceed to [Dataset Preparation](03_dataset_preparation.md)
2. Verify all dependencies with the validation script
3. Familiarize yourself with the project structure
4. Set up version control and backup strategies
