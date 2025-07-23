"""
Evaluation Framework for MAHT-Net

This module provides comprehensive evaluation metrics and tools for assessing
the performance of landmark detection models, particularly MAHT-Net.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


class LandmarkEvaluator:
    """
    Comprehensive evaluator for cephalometric landmark detection

    Implements standard metrics including Mean Radial Error (MRE),
    Success Detection Rate (SDR), and statistical analysis.
    """

    def __init__(self, num_landmarks: int = 19, pixels_per_mm: float = 10.0):
        """
        Initialize evaluator with dataset specifications

        Args:
            num_landmarks: Number of landmarks in the dataset
            pixels_per_mm: Pixel to millimeter conversion factor
        """
        self.num_landmarks = num_landmarks
        self.pixels_per_mm = pixels_per_mm
        self.reset()

    def reset(self) -> None:
        """Reset evaluation state for new evaluation session"""
        self.predictions = []
        self.ground_truths = []
        self.errors = []

    def add_batch(self,
                  predictions: np.ndarray,
                  ground_truths: np.ndarray) -> None:
        """
        Add a batch of predictions and ground truths

        Args:
            predictions: Predicted landmarks (batch_size, num_landmarks, 2)
            ground_truths: Ground truth landmarks (batch_size, num_landmarks, 2)
        """
        # Ensure numpy arrays
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(ground_truths, np.ndarray):
            ground_truths = np.array(ground_truths)

        # Store for later analysis
        self.predictions.append(predictions)
        self.ground_truths.append(ground_truths)

        # Calculate errors for this batch
        batch_errors = self._calculate_radial_errors(predictions, ground_truths)
        self.errors.append(batch_errors)

    def _calculate_radial_errors(self,
                                predictions: np.ndarray,
                                ground_truths: np.ndarray) -> np.ndarray:
        """
        Calculate radial (Euclidean) errors between predictions and ground truth

        Args:
            predictions: Predicted landmarks (batch_size, num_landmarks, 2)
            ground_truths: Ground truth landmarks (batch_size, num_landmarks, 2)

        Returns:
            Radial errors in millimeters (batch_size, num_landmarks)
        """
        # Calculate Euclidean distance
        diff = predictions - ground_truths
        radial_errors_pixels = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Convert to millimeters
        radial_errors_mm = radial_errors_pixels / self.pixels_per_mm

        return radial_errors_mm

    def calculate_mre(self) -> float:
        """
        Calculate Mean Radial Error (MRE) across all samples

        Returns:
            Mean radial error in millimeters
        """
        if not self.errors:
            return float('inf')

        all_errors = np.concatenate(self.errors, axis=0)
        return float(np.mean(all_errors))

    def calculate_sdr(self, threshold_mm: float = 2.0) -> float:
        """
        Calculate Success Detection Rate (SDR) for given threshold

        Args:
            threshold_mm: Threshold in millimeters for successful detection

        Returns:
            Success detection rate as percentage (0-100)
        """
        if not self.errors:
            return 0.0

        all_errors = np.concatenate(self.errors, axis=0)
        successful_detections = np.sum(all_errors <= threshold_mm)
        total_detections = all_errors.size

        return (successful_detections / total_detections) * 100.0

    def calculate_sdr_per_landmark(self, threshold_mm: float = 2.0) -> np.ndarray:
        """
        Calculate SDR for each landmark individually

        Args:
            threshold_mm: Threshold in millimeters

        Returns:
            SDR values per landmark (num_landmarks,)
        """
        if not self.errors:
            return np.zeros(self.num_landmarks)

        all_errors = np.concatenate(self.errors, axis=0)  # (total_samples, num_landmarks)

        sdr_per_landmark = []
        for landmark_idx in range(self.num_landmarks):
            landmark_errors = all_errors[:, landmark_idx]
            successful = np.sum(landmark_errors <= threshold_mm)
            total = len(landmark_errors)
            sdr = (successful / total) * 100.0
            sdr_per_landmark.append(sdr)

        return np.array(sdr_per_landmark)

    def calculate_comprehensive_metrics(self,
                                      sdr_thresholds: List[float] = [1.5, 2.0, 2.5, 3.0, 4.0]) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics

        Args:
            sdr_thresholds: List of thresholds for SDR calculation

        Returns:
            Dictionary containing all evaluation metrics
        """
        if not self.errors:
            return {}

        all_errors = np.concatenate(self.errors, axis=0)

        metrics = {
            'mre': self.calculate_mre(),
            'std': float(np.std(all_errors)),
            'median_error': float(np.median(all_errors)),
            'max_error': float(np.max(all_errors)),
            'min_error': float(np.min(all_errors))
        }

        # Add SDR for different thresholds
        for threshold in sdr_thresholds:
            metrics[f'sdr_{threshold}mm'] = self.calculate_sdr(threshold)

        # Add percentile information
        for percentile in [90, 95, 99]:
            metrics[f'p{percentile}_error'] = float(np.percentile(all_errors, percentile))

        return metrics

    def get_per_landmark_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get detailed statistics for each landmark

        Returns:
            Dictionary containing per-landmark statistics
        """
        if not self.errors:
            return {}

        all_errors = np.concatenate(self.errors, axis=0)  # (total_samples, num_landmarks)

        stats = {
            'mean_error': np.mean(all_errors, axis=0),
            'std_error': np.std(all_errors, axis=0),
            'median_error': np.median(all_errors, axis=0),
            'max_error': np.max(all_errors, axis=0),
            'min_error': np.min(all_errors, axis=0)
        }

        # Add SDR for each landmark at different thresholds
        for threshold in [1.5, 2.0, 2.5, 3.0, 4.0]:
            stats[f'sdr_{threshold}mm'] = self.calculate_sdr_per_landmark(threshold)

        return stats

    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save evaluation results to JSON file

        Args:
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            'overall_metrics': self.calculate_comprehensive_metrics(),
            'per_landmark_statistics': {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.get_per_landmark_statistics().items()
            },
            'metadata': {
                'num_landmarks': self.num_landmarks,
                'pixels_per_mm': self.pixels_per_mm,
                'total_samples': sum(len(errors) for errors in self.errors)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def compare_with_baseline(self, baseline_results_path: Union[str, Path]) -> Dict[str, float]:
        """
        Compare current results with baseline model

        Args:
            baseline_results_path: Path to baseline results JSON

        Returns:
            Dictionary with improvement metrics
        """
        baseline_path = Path(baseline_results_path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline results not found: {baseline_path}")

        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)

        current_metrics = self.calculate_comprehensive_metrics()
        baseline_metrics = baseline_data['overall_metrics']

        improvements = {}

        # Calculate improvements for key metrics
        key_metrics = ['mre', 'sdr_2.0mm', 'sdr_4.0mm']
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                current_val = current_metrics[metric]
                baseline_val = baseline_metrics[metric]

                if 'sdr' in metric:  # Higher is better for SDR
                    improvement = current_val - baseline_val
                else:  # Lower is better for error metrics
                    improvement = baseline_val - current_val

                improvements[f'{metric}_improvement'] = improvement
                improvements[f'{metric}_current'] = current_val
                improvements[f'{metric}_baseline'] = baseline_val

        return improvements


def bootstrap_confidence_interval(errors: np.ndarray,
                                 metric_func,
                                 confidence: float = 0.95,
                                 n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a metric using bootstrap sampling

    Args:
        errors: Array of errors
        metric_func: Function to calculate metric (e.g., np.mean)
        confidence: Confidence level (0-1)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (metric_value, lower_ci, upper_ci)
    """
    n_samples = len(errors)
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(errors, size=n_samples, replace=True)
        bootstrap_metric = metric_func(bootstrap_sample)
        bootstrap_metrics.append(bootstrap_metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    metric_value = metric_func(errors)
    lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
    upper_ci = np.percentile(bootstrap_metrics, upper_percentile)

    return metric_value, lower_ci, upper_ci


def statistical_significance_test(errors1: np.ndarray,
                                 errors2: np.ndarray,
                                 alpha: float = 0.05) -> Dict[str, Union[float, bool]]:
    """
    Perform statistical significance test between two sets of errors

    Args:
        errors1: Errors from model 1
        errors2: Errors from model 2
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Note: This is a simplified implementation
    # In practice, you might want to use scipy.stats for more robust tests

    # Basic statistics
    mean1, std1 = np.mean(errors1), np.std(errors1)
    mean2, std2 = np.mean(errors2), np.std(errors2)

    # Simple t-test approximation
    n1, n2 = len(errors1), len(errors2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))

    # Simplified significance check (would need proper t-distribution in practice)
    is_significant = abs(t_stat) > 2.0  # Rough approximation

    return {
        'mean_difference': mean1 - mean2,
        't_statistic': t_stat,
        'is_significant': is_significant,
        'p_value_approx': 2 * (1 - 0.95) if abs(t_stat) > 2.0 else 0.1  # Rough approximation
    }


if __name__ == "__main__":
    # Test the evaluator
    print("Testing MAHT-Net Evaluation Framework...")

    # Create dummy data for testing
    num_samples = 100
    num_landmarks = 19

    # Simulate some predictions and ground truths
    ground_truths = np.random.rand(num_samples, num_landmarks, 2) * 256
    predictions = ground_truths + np.random.normal(0, 5, ground_truths.shape)  # Add some noise

    # Test evaluator
    evaluator = LandmarkEvaluator(num_landmarks=num_landmarks)
    evaluator.add_batch(predictions, ground_truths)

    # Calculate metrics
    mre = evaluator.calculate_mre()
    sdr_2mm = evaluator.calculate_sdr(2.0)

    print(f"Test MRE: {mre:.2f}mm")
    print(f"Test SDR@2mm: {sdr_2mm:.1f}%")
    print("Evaluation framework tested successfully!")
