"""
Utility Functions for MAHT-Net Project

This module provides comprehensive utility functions for logging, visualization,
checkpoint management, and other common operations throughout the project.
"""

import os
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import seaborn as sns
from PIL import Image, ImageDraw
import cv2


class Logger:
    """Comprehensive logging system for MAHT-Net experiments"""

    def __init__(self, name: str, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger with specified configuration

        Args:
            name: Logger name (typically __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)


class CheckpointManager:
    """Manages model checkpoints and training state"""

    def __init__(self, checkpoint_dir: Union[str, Path], max_keep: int = 5):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep

    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       loss: float,
                       metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """
        Save model checkpoint with training state

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state
            epoch: Current epoch number
            loss: Current loss value
            metrics: Dictionary of evaluation metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save regular checkpoint
        filename = f"checkpoint_epoch_{epoch:03d}.pth"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        # Save best model separately
        if is_best:
            best_filepath = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_filepath)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(filepath)

    def load_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            model: PyTorch model to load state into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Specific checkpoint path (defaults to best model)

        Returns:
            Dictionary containing checkpoint metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', float('inf')),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint files, keeping only the most recent ones"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old files beyond max_keep
        for old_file in checkpoint_files[self.max_keep:]:
            old_file.unlink()


class MetricsTracker:
    """Tracks and manages training/validation metrics"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.metrics_history = {
            'train': {},
            'val': {}
        }
        self.current_epoch = 0

    def update(self, metrics: Dict[str, float], phase: str = 'train') -> None:
        """
        Update metrics for current epoch

        Args:
            metrics: Dictionary of metric name -> value
            phase: 'train' or 'val'
        """
        if phase not in self.metrics_history:
            raise ValueError(f"Invalid phase: {phase}")

        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history[phase]:
                self.metrics_history[phase][metric_name] = []
            self.metrics_history[phase][metric_name].append(value)

    def get_best(self, metric_name: str, phase: str = 'val', minimize: bool = True) -> Tuple[float, int]:
        """
        Get best value and epoch for a specific metric

        Args:
            metric_name: Name of the metric
            phase: 'train' or 'val'
            minimize: Whether lower values are better

        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.metrics_history[phase]:
            raise KeyError(f"Metric {metric_name} not found in {phase} history")

        values = self.metrics_history[phase][metric_name]
        if minimize:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return values[best_idx], best_idx

    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics over epochs

        Args:
            save_path: Optional path to save the plot
        """
        # Get all unique metric names
        all_metrics = set()
        for phase_metrics in self.metrics_history.values():
            all_metrics.update(phase_metrics.keys())

        n_metrics = len(all_metrics)
        if n_metrics == 0:
            return

        # Create subplots
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric_name in enumerate(sorted(all_metrics)):
            ax = axes[idx]

            for phase in ['train', 'val']:
                if metric_name in self.metrics_history[phase]:
                    values = self.metrics_history[phase][metric_name]
                    epochs = range(1, len(values) + 1)
                    ax.plot(epochs, values, label=f'{phase.capitalize()}', marker='o')

            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def save_to_json(self, filepath: str) -> None:
        """Save metrics history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def load_from_json(self, filepath: str) -> None:
        """Load metrics history from JSON file"""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)


class Visualizer:
    """Visualization utilities for MAHT-Net results"""

    @staticmethod
    def plot_landmarks_on_image(image: np.ndarray,
                               landmarks: np.ndarray,
                               predictions: Optional[np.ndarray] = None,
                               title: str = "Landmark Visualization") -> plt.Figure:
        """
        Plot landmarks on cephalometric image

        Args:
            image: Input image (H, W) or (H, W, C)
            landmarks: Ground truth landmarks (N, 2)
            predictions: Optional predicted landmarks (N, 2)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Display image
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')

        # Plot ground truth landmarks
        ax.scatter(landmarks[:, 0], landmarks[:, 1],
                  c='red', s=50, marker='o', label='Ground Truth', alpha=0.8)

        # Plot predictions if provided
        if predictions is not None:
            ax.scatter(predictions[:, 0], predictions[:, 1],
                      c='blue', s=50, marker='x', label='Predictions', alpha=0.8)

            # Draw lines connecting GT and predictions
            for i in range(len(landmarks)):
                ax.plot([landmarks[i, 0], predictions[i, 0]],
                       [landmarks[i, 1], predictions[i, 1]],
                       'yellow', alpha=0.5, linewidth=1)

        # Add landmark numbers
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(str(i+1), (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, color='white')

        ax.set_title(title)
        ax.legend()
        ax.axis('off')

        return fig

    @staticmethod
    def plot_attention_maps(attention_maps: List[np.ndarray],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention maps from transformer layers

        Args:
            attention_maps: List of attention maps from different layers
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure
        """
        n_layers = len(attention_maps)
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

        if n_layers == 1:
            axes = [axes]

        for i, attention_map in enumerate(attention_maps):
            ax = axes[i]
            im = ax.imshow(attention_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Layer {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_error_distribution(errors: np.ndarray,
                               landmark_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot distribution of landmark detection errors

        Args:
            errors: Array of errors per landmark (N_samples, N_landmarks)
            landmark_names: Optional names for landmarks

        Returns:
            Matplotlib figure
        """
        n_landmarks = errors.shape[1]

        if landmark_names is None:
            landmark_names = [f'L{i+1}' for i in range(n_landmarks)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot of errors per landmark
        ax1.boxplot(errors, labels=landmark_names)
        ax1.set_title('Error Distribution per Landmark')
        ax1.set_ylabel('Error (mm)')
        ax1.tick_params(axis='x', rotation=45)

        # Overall error histogram
        all_errors = errors.flatten()
        ax2.hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(all_errors), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_errors):.2f}mm')
        ax2.axvline(np.median(all_errors), color='green', linestyle='--',
                   label=f'Median: {np.median(all_errors):.2f}mm')
        ax2.set_title('Overall Error Distribution')
        ax2.set_xlabel('Error (mm)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        plt.tight_layout()
        return fig


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch model

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def create_experiment_dirs(experiment_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Create directory structure for experiment

    Args:
        experiment_dir: Base experiment directory

    Returns:
        Dictionary mapping directory names to paths
    """
    experiment_dir = Path(experiment_dir)

    dirs = {
        'experiment': experiment_dir,
        'checkpoints': experiment_dir / 'checkpoints',
        'logs': experiment_dir / 'logs',
        'visualizations': experiment_dir / 'visualizations',
        'results': experiment_dir / 'results'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


if __name__ == "__main__":
    # Test utilities
    print("Testing MAHT-Net utilities...")

    # Test logger
    logger = Logger("test_logger")
    logger.info("Logger initialized successfully")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({'loss': 0.5, 'accuracy': 0.85}, 'train')
    tracker.update({'loss': 0.6, 'accuracy': 0.82}, 'val')

    print("Utilities tested successfully!")
