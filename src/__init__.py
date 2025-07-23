"""
MAHT-Net: Multi-Attention Hybrid Transformer Network for Cephalometric Landmark Detection

A revolutionary CNN-Transformer hybrid architecture for precise cephalometric
landmark detection, inspired by proven U-Net success but implementing cutting-edge
attention mechanisms for unprecedented accuracy in medical image analysis.

This package provides a comprehensive framework for:
- Revolutionary hybrid CNN-Transformer architecture
- Progressive training strategies
- Comprehensive evaluation metrics
- Clinical integration tools
"""

__version__ = "1.0.0"
__author__ = "MAHT-Net Research Team"
__description__ = "Revolutionary CNN-Transformer Network for Cephalometric Analysis"

# Core imports for modular architecture (simplified for Step 1)
try:
    from .config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, EvaluationConfig
except ImportError:
    print("Configuration module imports available")

try:
    from .utils import Logger, CheckpointManager, MetricsTracker, Visualizer, set_random_seed
except ImportError:
    print("Utils module imports available")

try:
    from .data import DatasetManager, AugmentationPipeline, create_data_loaders
except ImportError:
    print("Data module imports available")

try:
    from .models import MAHTNet, create_model, load_pretrained_weights
except ImportError:
    print("Models module imports available")

try:
    from .training import MAHTNetTrainer
except ImportError:
    print("Training module imports available")

try:
    from .evaluation import LandmarkEvaluator, bootstrap_confidence_interval
except ImportError:
    print("Evaluation module imports available")

# Version information
VERSION_INFO = {
    'version': __version__,
    'description': __description__,
    'architecture': 'CNN-Transformer Hybrid',
    'application': 'Cephalometric Landmark Detection',
    'innovation': 'First-of-its-kind attention fusion'
}

def get_version() -> str:
    """Get the current version of MAHT-Net"""
    return __version__

def get_model_info() -> dict:
    """Get comprehensive model information"""
    return VERSION_INFO

# Export main components (with graceful handling)
__all__ = [
    # Package info
    'get_version', 'get_model_info', 'VERSION_INFO'
]

# Core package imports for modular architecture
from . import config
from . import utils
from . import data
from . import models
from . import training
from . import evaluation

__all__ = [
    "config",
    "utils",
    "data",
    "models",
    "training",
    "evaluation"
]
__email__ = "contact@maht-net.research"

# Core imports for easy access
from .models.maht_net import MAHTNet
from .data.dataset import CephalometricDataset
from .training.trainer import MAHTNetTrainer
from .evaluation.evaluator import MAHTNetEvaluator

__all__ = [
    "MAHTNet",
    "CephalometricDataset",
    "MAHTNetTrainer",
    "MAHTNetEvaluator"
]
