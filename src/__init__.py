"""
MAHT-Net Package Initialization
Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection

This package implements a state-of-the-art hybrid CNN-Transformer architecture 
for clinical-grade cephalometric landmark detection with comprehensive uncertainty quantification.
"""

__version__ = "1.0.0"
__author__ = "MAHT-Net Research Team"
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
