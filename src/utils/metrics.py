"""
Utility functions for computing landmark detection metrics
Clinical-grade evaluation metrics for cephalometric analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_landmark_metrics(predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           pixel_spacing: float = 0.1) -> Dict[str, float]:
    """
    Compute comprehensive landmark detection metrics
    
    Args:
        predictions: Predicted landmarks [B, N, 2]
        targets: Ground truth landmarks [B, N, 2] 
        pixel_spacing: Pixel spacing in mm (default: 0.1mm)
        
    Returns:
        Dictionary of computed metrics
    """
    
    # Ensure tensors are on CPU
    if predictions.is_cuda:
        predictions = predictions.cpu()
    if targets.is_cuda:
        targets = targets.cpu()
    
    # Convert to numpy
    pred_np = predictions.numpy()
    target_np = targets.numpy()
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum((pred_np - target_np) ** 2, axis=2))  # [B, N]
    
    # Convert pixel distances to mm
    distances_mm = distances * pixel_spacing
    
    # Mean Radial Error (MRE)
    mre = np.mean(distances_mm)
    
    # Standard deviation
    std = np.std(distances_mm)
    
    # Success Detection Rates (SDR)
    sdr_2mm = np.mean(distances_mm <= 2.0) * 100
    sdr_2_5mm = np.mean(distances_mm <= 2.5) * 100
    sdr_3mm = np.mean(distances_mm <= 3.0) * 100
    sdr_4mm = np.mean(distances_mm <= 4.0) * 100
    
    # Per-landmark metrics
    landmark_mre = np.mean(distances_mm, axis=0)
    landmark_std = np.std(distances_mm, axis=0)
    
    # Maximum error
    max_error = np.max(distances_mm)
    
    # 95th percentile error (clinical reliability metric)
    percentile_95 = np.percentile(distances_mm, 95)
    
    return {
        'mre': mre,
        'std': std,
        'max_error': max_error,
        'percentile_95': percentile_95,
        'sdr_2mm': sdr_2mm,
        'sdr_2.5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm,
        'landmark_mre': landmark_mre,
        'landmark_std': landmark_std,
        'total_samples': len(pred_np)
    }


def compute_clinical_accuracy_metrics(predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   uncertainties: Optional[torch.Tensor] = None,
                                   pixel_spacing: float = 0.1) -> Dict[str, float]:
    """
    Compute clinical accuracy metrics following orthodontic standards
    
    Args:
        predictions: Predicted landmarks [B, N, 2]
        targets: Ground truth landmarks [B, N, 2]
        uncertainties: Prediction uncertainties [B, N] (optional)
        pixel_spacing: Pixel spacing in mm
        
    Returns:
        Clinical accuracy metrics
    """
    
    base_metrics = compute_landmark_metrics(predictions, targets, pixel_spacing)
    
    # Clinical acceptability thresholds (based on orthodontic literature)
    clinical_thresholds = {
        'excellent': 1.0,  # mm
        'acceptable': 1.5,  # mm
        'poor': 2.0,       # mm
        'unacceptable': 4.0  # mm
    }
    
    distances_mm = np.sqrt(np.sum((predictions.cpu().numpy() - targets.cpu().numpy()) ** 2, axis=2)) * pixel_spacing
    
    # Clinical accuracy categories
    excellent_rate = np.mean(distances_mm <= clinical_thresholds['excellent']) * 100
    acceptable_rate = np.mean(distances_mm <= clinical_thresholds['acceptable']) * 100
    poor_rate = np.mean((distances_mm > clinical_thresholds['acceptable']) & 
                       (distances_mm <= clinical_thresholds['poor'])) * 100
    unacceptable_rate = np.mean(distances_mm > clinical_thresholds['unacceptable']) * 100
    
    clinical_metrics = {
        'clinical_excellent_rate': excellent_rate,
        'clinical_acceptable_rate': acceptable_rate,
        'clinical_poor_rate': poor_rate,
        'clinical_unacceptable_rate': unacceptable_rate,
        'clinical_overall_acceptable': excellent_rate + acceptable_rate
    }
    
    # Uncertainty-based metrics if available
    if uncertainties is not None:
        uncertainty_np = uncertainties.cpu().numpy()
        
        # Calibration metrics
        high_uncertainty_mask = uncertainty_np > np.median(uncertainty_np)
        low_uncertainty_mask = ~high_uncertainty_mask
        
        high_uncertainty_mre = np.mean(distances_mm[high_uncertainty_mask])
        low_uncertainty_mre = np.mean(distances_mm[low_uncertainty_mask])
        
        clinical_metrics.update({
            'uncertainty_calibration': high_uncertainty_mre - low_uncertainty_mre,
            'high_uncertainty_mre': high_uncertainty_mre,
            'low_uncertainty_mre': low_uncertainty_mre,
            'mean_uncertainty': np.mean(uncertainty_np)
        })
    
    # Combine with base metrics
    base_metrics.update(clinical_metrics)
    
    return base_metrics


def compute_inter_observer_metrics(predictions_a: torch.Tensor,
                                 predictions_b: torch.Tensor,
                                 pixel_spacing: float = 0.1) -> Dict[str, float]:
    """
    Compute inter-observer agreement metrics
    
    Args:
        predictions_a: First observer predictions [B, N, 2]
        predictions_b: Second observer predictions [B, N, 2]
        pixel_spacing: Pixel spacing in mm
        
    Returns:
        Inter-observer agreement metrics
    """
    
    # Compute distances between observers
    distances = np.sqrt(np.sum((predictions_a.cpu().numpy() - predictions_b.cpu().numpy()) ** 2, axis=2))
    distances_mm = distances * pixel_spacing
    
    # Inter-observer agreement
    inter_observer_mre = np.mean(distances_mm)
    inter_observer_std = np.std(distances_mm)
    
    # Agreement within clinical thresholds
    agreement_1mm = np.mean(distances_mm <= 1.0) * 100
    agreement_1_5mm = np.mean(distances_mm <= 1.5) * 100
    agreement_2mm = np.mean(distances_mm <= 2.0) * 100
    
    return {
        'inter_observer_mre': inter_observer_mre,
        'inter_observer_std': inter_observer_std,
        'inter_observer_agreement_1mm': agreement_1mm,
        'inter_observer_agreement_1.5mm': agreement_1_5mm,
        'inter_observer_agreement_2mm': agreement_2mm
    }


def visualize_landmark_errors(predictions: torch.Tensor,
                            targets: torch.Tensor,
                            landmark_names: List[str],
                            save_path: Optional[str] = None,
                            pixel_spacing: float = 0.1) -> plt.Figure:
    """
    Create comprehensive visualization of landmark detection errors
    
    Args:
        predictions: Predicted landmarks [B, N, 2]
        targets: Ground truth landmarks [B, N, 2]
        landmark_names: Names of landmarks
        save_path: Optional path to save figure
        pixel_spacing: Pixel spacing in mm
        
    Returns:
        Matplotlib figure
    """
    
    # Compute errors
    distances = np.sqrt(np.sum((predictions.cpu().numpy() - targets.cpu().numpy()) ** 2, axis=2))
    distances_mm = distances * pixel_spacing
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MAHT-Net Landmark Detection Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot of errors per landmark
    axes[0, 0].boxplot([distances_mm[:, i] for i in range(len(landmark_names))], 
                       labels=landmark_names)
    axes[0, 0].set_title('Error Distribution per Landmark')
    axes[0, 0].set_ylabel('Error (mm)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add clinical threshold lines
    axes[0, 0].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (1mm)')
    axes[0, 0].axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Acceptable (1.5mm)')
    axes[0, 0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Poor (2mm)')
    axes[0, 0].legend()
    
    # 2. Overall error histogram
    axes[0, 1].hist(distances_mm.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(np.mean(distances_mm), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(distances_mm):.2f}mm')
    axes[0, 1].axvline(np.median(distances_mm), color='green', linestyle='-', linewidth=2, label=f'Median: {np.median(distances_mm):.2f}mm')
    axes[0, 1].set_title('Overall Error Distribution')
    axes[0, 1].set_xlabel('Error (mm)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success Detection Rate plot
    thresholds = np.arange(0.5, 5.1, 0.1)
    sdr_values = [np.mean(distances_mm <= t) * 100 for t in thresholds]
    
    axes[1, 0].plot(thresholds, sdr_values, linewidth=2, color='blue')
    axes[1, 0].axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% SDR')
    axes[1, 0].axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% SDR')
    axes[1, 0].axvline(x=2.0, color='red', linestyle='--', alpha=0.7, label='2mm threshold')
    axes[1, 0].set_title('Success Detection Rate (SDR)')
    axes[1, 0].set_xlabel('Error Threshold (mm)')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Per-landmark mean errors
    landmark_means = np.mean(distances_mm, axis=0)
    bars = axes[1, 1].bar(range(len(landmark_names)), landmark_means, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Mean Error per Landmark')
    axes[1, 1].set_xlabel('Landmark')
    axes[1, 1].set_ylabel('Mean Error (mm)')
    axes[1, 1].set_xticks(range(len(landmark_names)))
    axes[1, 1].set_xticklabels(landmark_names, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, landmark_means):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_metrics_report(metrics: Dict[str, float],
                        landmark_names: List[str],
                        save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive metrics report
    
    Args:
        metrics: Computed metrics dictionary
        landmark_names: Names of landmarks
        save_path: Optional path to save report
        
    Returns:
        Formatted report string
    """
    
    report = []
    report.append("=" * 80)
    report.append("MAHT-Net Landmark Detection Performance Report")
    report.append("=" * 80)
    report.append("")
    
    # Overall Performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Mean Radial Error (MRE):      {metrics['mre']:.3f} mm")
    report.append(f"Standard Deviation:           {metrics['std']:.3f} mm")
    report.append(f"Maximum Error:                {metrics['max_error']:.3f} mm")
    report.append(f"95th Percentile Error:        {metrics['percentile_95']:.3f} mm")
    report.append("")
    
    # Success Detection Rates
    report.append("🎯 SUCCESS DETECTION RATES")
    report.append("-" * 30)
    report.append(f"SDR @ 2.0mm:                  {metrics['sdr_2mm']:.1f}%")
    report.append(f"SDR @ 2.5mm:                  {metrics['sdr_2.5mm']:.1f}%")
    report.append(f"SDR @ 3.0mm:                  {metrics['sdr_3mm']:.1f}%")
    report.append(f"SDR @ 4.0mm:                  {metrics['sdr_4mm']:.1f}%")
    report.append("")
    
    # Clinical Assessment
    if 'clinical_excellent_rate' in metrics:
        report.append("🏥 CLINICAL ASSESSMENT")
        report.append("-" * 30)
        report.append(f"Excellent (≤1.0mm):           {metrics['clinical_excellent_rate']:.1f}%")
        report.append(f"Acceptable (≤1.5mm):          {metrics['clinical_acceptable_rate']:.1f}%")
        report.append(f"Poor (1.5-2.0mm):             {metrics['clinical_poor_rate']:.1f}%")
        report.append(f"Unacceptable (>4.0mm):        {metrics['clinical_unacceptable_rate']:.1f}%")
        report.append(f"Overall Acceptable:           {metrics['clinical_overall_acceptable']:.1f}%")
        report.append("")
    
    # Per-landmark Performance
    if 'landmark_mre' in metrics:
        report.append("📍 PER-LANDMARK PERFORMANCE")
        report.append("-" * 30)
        for i, name in enumerate(landmark_names):
            mre = metrics['landmark_mre'][i]
            std = metrics['landmark_std'][i]
            report.append(f"{name:20s}: {mre:.3f} ± {std:.3f} mm")
        report.append("")
    
    # Uncertainty Analysis
    if 'mean_uncertainty' in metrics:
        report.append("🎲 UNCERTAINTY ANALYSIS")
        report.append("-" * 30)
        report.append(f"Mean Uncertainty:             {metrics['mean_uncertainty']:.3f}")
        report.append(f"High Uncertainty MRE:         {metrics['high_uncertainty_mre']:.3f} mm")
        report.append(f"Low Uncertainty MRE:          {metrics['low_uncertainty_mre']:.3f} mm")
        report.append(f"Uncertainty Calibration:      {metrics['uncertainty_calibration']:.3f} mm")
        report.append("")
    
    # Dataset Information
    report.append("📋 DATASET INFORMATION")
    report.append("-" * 30)
    report.append(f"Total Samples:                {metrics['total_samples']}")
    report.append(f"Total Landmarks:              {len(landmark_names)}")
    report.append("")
    
    # Clinical Recommendations
    report.append("💡 CLINICAL RECOMMENDATIONS")
    report.append("-" * 30)
    if metrics['mre'] <= 1.2:
        report.append("Model meets clinical requirements (MRE ≤ 1.2mm)")
    else:
        report.append("⚠️  Model requires improvement for clinical use")
    
    if metrics['sdr_2mm'] >= 95.0:
        report.append("Excellent reliability (SDR@2mm ≥ 95%)")
    elif metrics['sdr_2mm'] >= 90.0:
        report.append("Good reliability (SDR@2mm ≥ 90%)")
    else:
        report.append("⚠️  Reliability needs improvement")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    # Test metrics computation
    print("🧪 Testing landmark metrics computation...")
    
    # Create sample data
    batch_size, num_landmarks = 10, 7
    predictions = torch.randn(batch_size, num_landmarks, 2) * 2 + 100
    targets = torch.randn(batch_size, num_landmarks, 2) * 2 + 100
    
    # Add some noise to predictions
    predictions += torch.randn_like(predictions) * 0.5
    
    # Compute metrics
    metrics = compute_landmark_metrics(predictions, targets)
    
    print("Metrics computed successfully:")
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\n🎉 Metrics module ready for use!")
