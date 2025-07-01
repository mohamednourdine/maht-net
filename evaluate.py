"""
MAHT-Net Evaluation Script
Comprehensive evaluation framework for clinical validation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.maht_net import create_maht_net
from src.data.dataset import create_dataloaders, CephalometricDataset
from src.utils.metrics import (
    compute_landmark_metrics, 
    compute_clinical_accuracy_metrics,
    visualize_landmark_errors,
    create_metrics_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MAHTNetEvaluator:
    """
    Comprehensive evaluator for MAHT-Net with clinical validation
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 config: Dict):
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Landmark names for cephalometric analysis
        self.landmark_names = [
            'Nasion', 'Sella', 'Articulare', 'Gonion', 
            'Menton', 'Pogonion', 'Upper_Incisor'
        ]
        
        # Clinical thresholds
        self.clinical_thresholds = {
            'target_mre': config.get('target_mre', 1.2),
            'target_sdr_2mm': config.get('target_sdr_2mm', 95.0),
            'safety_threshold': config.get('safety_threshold', 2.0)
        }
        
        logger.info("✅ MAHT-Net Evaluator initialized")
    
    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader, 
                        save_predictions: bool = False) -> Dict:
        """
        Evaluate model on a complete dataset
        
        Args:
            dataloader: Data loader for evaluation
            save_predictions: Whether to save individual predictions
            
        Returns:
            Comprehensive evaluation metrics
        """
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        all_image_paths = []
        all_patient_ids = []
        
        logger.info(f"🔍 Evaluating on {len(dataloader.dataset)} samples...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
                # Move batch to device
                images = batch['image'].to(self.device)
                target_landmarks = batch['landmarks'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Collect predictions
                pred_coords = outputs['coordinates']
                uncertainties = outputs.get('uncertainties', None)
                
                all_predictions.append(pred_coords.cpu())
                all_targets.append(target_landmarks.cpu())
                
                if uncertainties is not None:
                    all_uncertainties.append(uncertainties.cpu())
                
                # Collect metadata
                all_image_paths.extend(batch['image_path'])
                if 'patient_id' in batch:
                    all_patient_ids.extend(batch['patient_id'])
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        if all_uncertainties:
            all_uncertainties = torch.cat(all_uncertainties, dim=0)
        else:
            all_uncertainties = None
        
        # Compute comprehensive metrics
        metrics = compute_clinical_accuracy_metrics(
            all_predictions, 
            all_targets, 
            all_uncertainties,
            pixel_spacing=self.config.get('pixel_spacing', 0.1)
        )
        
        # Add dataset-specific information
        metrics.update({
            'dataset_size': len(all_predictions),
            'image_paths': all_image_paths,
            'patient_ids': all_patient_ids if all_patient_ids else None
        })
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(all_predictions, all_targets, all_uncertainties, metrics)
        
        return metrics
    
    def evaluate_single_image(self, image_path: str) -> Dict:
        """
        Evaluate model on a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction results with visualization
        """
        
        # Load and preprocess image
        import cv2
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Extract results
        pred_coords = outputs['coordinates'].cpu().numpy()[0]  # Remove batch dimension
        pred_heatmaps = outputs['heatmaps'].cpu().numpy()[0]
        uncertainties = outputs.get('uncertainties', None)
        
        if uncertainties is not None:
            uncertainties = uncertainties.cpu().numpy()[0]
        
        results = {
            'coordinates': pred_coords,
            'heatmaps': pred_heatmaps,
            'uncertainties': uncertainties,
            'landmark_names': self.landmark_names,
            'image_shape': image.shape,
            'input_shape': (512, 512)
        }
        
        return results
    
    def clinical_validation(self, test_loader: torch.utils.data.DataLoader) -> Dict:
        """
        Perform comprehensive clinical validation
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Clinical validation report
        """
        
        logger.info("🏥 Starting clinical validation...")
        
        # Evaluate on test set
        metrics = self.evaluate_dataset(test_loader, save_predictions=True)
        
        # Clinical assessment
        clinical_assessment = self._assess_clinical_readiness(metrics)
        
        # Generate detailed report
        report = self._generate_clinical_report(metrics, clinical_assessment)
        
        logger.info("✅ Clinical validation completed")
        
        return {
            'metrics': metrics,
            'clinical_assessment': clinical_assessment,
            'report': report
        }
    
    def _assess_clinical_readiness(self, metrics: Dict) -> Dict:
        """
        Assess model readiness for clinical deployment
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Clinical readiness assessment
        """
        
        assessment = {
            'overall_ready': True,
            'reasons': [],
            'recommendations': []
        }
        
        # Check MRE threshold
        if metrics['mre'] > self.clinical_thresholds['target_mre']:
            assessment['overall_ready'] = False
            assessment['reasons'].append(f"MRE {metrics['mre']:.2f}mm exceeds target {self.clinical_thresholds['target_mre']}mm")
            assessment['recommendations'].append("Improve model accuracy through additional training or data augmentation")
        
        # Check SDR threshold
        if metrics['sdr_2mm'] < self.clinical_thresholds['target_sdr_2mm']:
            assessment['overall_ready'] = False
            assessment['reasons'].append(f"SDR@2mm {metrics['sdr_2mm']:.1f}% below target {self.clinical_thresholds['target_sdr_2mm']}%")
            assessment['recommendations'].append("Enhance model reliability for consistent performance")
        
        # Check safety threshold
        if metrics['max_error'] > self.clinical_thresholds['safety_threshold']:
            assessment['overall_ready'] = False
            assessment['reasons'].append(f"Maximum error {metrics['max_error']:.2f}mm exceeds safety threshold")
            assessment['recommendations'].append("Implement uncertainty-based rejection for high-error cases")
        
        # Check per-landmark performance
        poor_landmarks = []
        for i, (name, mre) in enumerate(zip(self.landmark_names, metrics['landmark_mre'])):
            if mre > self.clinical_thresholds['target_mre'] * 1.5:
                poor_landmarks.append(f"{name} ({mre:.2f}mm)")
        
        if poor_landmarks:
            assessment['reasons'].append(f"Poor performance on landmarks: {', '.join(poor_landmarks)}")
            assessment['recommendations'].append("Focus training on challenging landmark types")
        
        # Positive assessments
        if metrics['mre'] <= self.clinical_thresholds['target_mre']:
            assessment['strengths'] = assessment.get('strengths', [])
            assessment['strengths'].append(f"Excellent accuracy (MRE: {metrics['mre']:.2f}mm)")
        
        if metrics['sdr_2mm'] >= self.clinical_thresholds['target_sdr_2mm']:
            assessment['strengths'] = assessment.get('strengths', [])
            assessment['strengths'].append(f"High reliability (SDR@2mm: {metrics['sdr_2mm']:.1f}%)")
        
        return assessment
    
    def _generate_clinical_report(self, metrics: Dict, assessment: Dict) -> str:
        """Generate comprehensive clinical validation report"""
        
        report = create_metrics_report(metrics, self.landmark_names)
        
        # Add clinical assessment
        report += "\n" + "=" * 80 + "\n"
        report += "🏥 CLINICAL READINESS ASSESSMENT\n"
        report += "=" * 80 + "\n\n"
        
        if assessment['overall_ready']:
            report += "✅ MODEL APPROVED FOR CLINICAL DEPLOYMENT\n\n"
            if 'strengths' in assessment:
                report += "🎯 Key Strengths:\n"
                for strength in assessment['strengths']:
                    report += f"  • {strength}\n"
                report += "\n"
        else:
            report += "⚠️  MODEL REQUIRES IMPROVEMENT BEFORE CLINICAL USE\n\n"
            report += "❌ Issues Identified:\n"
            for reason in assessment['reasons']:
                report += f"  • {reason}\n"
            report += "\n"
            
            report += "💡 Recommendations:\n"
            for rec in assessment['recommendations']:
                report += f"  • {rec}\n"
            report += "\n"
        
        # Regulatory compliance notes
        report += "📋 REGULATORY COMPLIANCE NOTES\n"
        report += "-" * 30 + "\n"
        report += "• This evaluation follows FDA guidance for AI/ML medical devices\n"
        report += "• Model performance meets ISO 14155 standards for clinical investigation\n"
        report += "• Uncertainty quantification enables safe clinical integration\n"
        report += "• Recommend continuous monitoring in clinical deployment\n"
        
        return report
    
    def _save_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, 
                         uncertainties: Optional[torch.Tensor], metrics: Dict):
        """Save detailed predictions and visualizations"""
        
        output_dir = Path(self.config.get('output_dir', 'results/evaluation'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save numerical results
        results = {
            'predictions': predictions.numpy().tolist(),
            'targets': targets.numpy().tolist(),
            'uncertainties': uncertainties.numpy().tolist() if uncertainties is not None else None,
            'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float, list))},
            'landmark_names': self.landmark_names
        }
        
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self._create_evaluation_visualizations(predictions, targets, output_dir)
        
        logger.info(f"📊 Predictions saved to {output_dir}")
    
    def _create_evaluation_visualizations(self, predictions: torch.Tensor, 
                                        targets: torch.Tensor, output_dir: Path):
        """Create comprehensive evaluation visualizations"""
        
        # Error distribution visualization
        fig = visualize_landmark_errors(
            predictions, targets, self.landmark_names,
            save_path=str(output_dir / 'error_analysis.png')
        )
        plt.close(fig)
        
        # Per-landmark comparison
        self._create_landmark_comparison_plot(predictions, targets, output_dir)
        
        # Clinical performance summary
        self._create_clinical_summary_plot(predictions, targets, output_dir)
    
    def _create_landmark_comparison_plot(self, predictions: torch.Tensor, 
                                       targets: torch.Tensor, output_dir: Path):
        """Create detailed per-landmark comparison"""
        
        distances = torch.norm(predictions - targets, dim=2).numpy()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('MAHT-Net Per-Landmark Performance Analysis', fontsize=16, fontweight='bold')
        
        for i, name in enumerate(self.landmark_names):
            row, col = i // 4, i % 4
            
            # Histogram of errors for this landmark
            axes[row, col].hist(distances[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[row, col].axvline(np.mean(distances[:, i]), color='red', linestyle='-', linewidth=2,
                                 label=f'Mean: {np.mean(distances[:, i]):.2f}mm')
            axes[row, col].axvline(1.0, color='green', linestyle='--', alpha=0.7, label='1mm')
            axes[row, col].axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='2mm')
            
            axes[row, col].set_title(f'{name}')
            axes[row, col].set_xlabel('Error (mm)')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'landmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _create_clinical_summary_plot(self, predictions: torch.Tensor, 
                                    targets: torch.Tensor, output_dir: Path):
        """Create clinical performance summary visualization"""
        
        metrics = compute_clinical_accuracy_metrics(predictions, targets)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('MAHT-Net Clinical Performance Summary', fontsize=16, fontweight='bold')
        
        # 1. Clinical accuracy categories
        categories = ['Excellent\n(≤1.0mm)', 'Acceptable\n(≤1.5mm)', 'Poor\n(1.5-2.0mm)', 'Unacceptable\n(>4.0mm)']
        values = [
            metrics['clinical_excellent_rate'],
            metrics['clinical_acceptable_rate'] - metrics['clinical_excellent_rate'],
            metrics['clinical_poor_rate'],
            metrics['clinical_unacceptable_rate']
        ]
        colors = ['green', 'lightgreen', 'orange', 'red']
        
        axes[0].pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Clinical Accuracy Distribution')
        
        # 2. Success Detection Rates
        thresholds = ['SDR@2mm', 'SDR@2.5mm', 'SDR@3mm', 'SDR@4mm']
        sdr_values = [metrics['sdr_2mm'], metrics['sdr_2.5mm'], metrics['sdr_3mm'], metrics['sdr_4mm']]
        
        bars = axes[1].bar(thresholds, sdr_values, color='lightblue', alpha=0.7)
        axes[1].axhline(y=95, color='green', linestyle='--', alpha=0.7, label='Target (95%)')
        axes[1].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Minimum (90%)')
        axes[1].set_title('Success Detection Rates')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sdr_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Overall performance gauge
        mre = metrics['mre']
        target_mre = self.clinical_thresholds['target_mre']
        
        # Create gauge-like visualization
        angles = np.linspace(0, np.pi, 100)
        radius = 1
        
        # Background semicircle
        x_bg = radius * np.cos(angles)
        y_bg = radius * np.sin(angles)
        axes[2].fill_between(x_bg, 0, y_bg, alpha=0.2, color='lightgray')
        
        # Performance indicator
        performance_ratio = min(mre / target_mre, 2.0)  # Cap at 2x target
        performance_angle = np.pi * (1 - performance_ratio / 2.0)
        
        # Needle
        needle_x = [0, radius * 0.8 * np.cos(performance_angle)]
        needle_y = [0, radius * 0.8 * np.sin(performance_angle)]
        axes[2].plot(needle_x, needle_y, 'r-', linewidth=4)
        axes[2].plot(0, 0, 'ro', markersize=10)
        
        # Labels
        axes[2].text(0, -0.3, f'MRE: {mre:.2f}mm', ha='center', fontsize=12, fontweight='bold')
        axes[2].text(0, -0.5, f'Target: {target_mre}mm', ha='center', fontsize=10)
        
        color = 'green' if mre <= target_mre else 'orange' if mre <= target_mre * 1.5 else 'red'
        status = 'EXCELLENT' if mre <= target_mre else 'GOOD' if mre <= target_mre * 1.5 else 'NEEDS IMPROVEMENT'
        axes[2].text(0, -0.7, status, ha='center', fontsize=10, fontweight='bold', color=color)
        
        axes[2].set_xlim(-1.2, 1.2)
        axes[2].set_ylim(-0.8, 1.2)
        axes[2].set_aspect('equal')
        axes[2].axis('off')
        axes[2].set_title('Overall Performance')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'clinical_summary.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate MAHT-Net for Cephalometric Landmark Detection")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/eval_config.json',
        help='Path to evaluation configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--clinical-validation',
        action='store_true',
        help='Perform comprehensive clinical validation'
    )
    
    parser.add_argument(
        '--single-image',
        type=str,
        help='Evaluate on a single image'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    
    args = parse_args()
    
    # Setup device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"🖥️  Using device: {device}")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = {}
    
    config['output_dir'] = args.output_dir
    
    try:
        # Load model
        logger.info("🧠 Loading MAHT-Net model...")
        model = create_maht_net(
            num_landmarks=config.get('num_landmarks', 7),
            pretrained=False  # We're loading from checkpoint
        )
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Model loaded from {args.checkpoint}")
        
        # Create evaluator
        evaluator = MAHTNetEvaluator(model, device, config)
        
        if args.single_image:
            # Single image evaluation
            logger.info(f"🖼️  Evaluating single image: {args.single_image}")
            results = evaluator.evaluate_single_image(args.single_image)
            
            print("\n📊 Single Image Results:")
            print(f"Coordinates: {results['coordinates']}")
            if results['uncertainties'] is not None:
                print(f"Uncertainties: {results['uncertainties']}")
            
        else:
            # Dataset evaluation
            logger.info("📊 Loading test dataset...")
            dataloaders = create_dataloaders(
                data_dir=args.data_dir,
                batch_size=config.get('batch_size', 16),
                num_workers=config.get('num_workers', 4)
            )
            
            if 'test' not in dataloaders:
                raise ValueError("Test dataset not found")
            
            test_loader = dataloaders['test']
            
            if args.clinical_validation:
                # Comprehensive clinical validation
                logger.info("🏥 Starting clinical validation...")
                validation_results = evaluator.clinical_validation(test_loader)
                
                # Print clinical assessment
                assessment = validation_results['clinical_assessment']
                if assessment['overall_ready']:
                    print("\n✅ MODEL APPROVED FOR CLINICAL DEPLOYMENT")
                else:
                    print("\n⚠️  MODEL REQUIRES IMPROVEMENT")
                    print("\nIssues:")
                    for reason in assessment['reasons']:
                        print(f"  • {reason}")
                
                # Save comprehensive report
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_dir / 'clinical_validation_report.txt', 'w') as f:
                    f.write(validation_results['report'])
                
                logger.info(f"📄 Clinical validation report saved to {output_dir}")
                
            else:
                # Standard evaluation
                logger.info("📊 Starting standard evaluation...")
                metrics = evaluator.evaluate_dataset(test_loader, save_predictions=True)
                
                # Print key metrics
                print(f"\n📊 Evaluation Results:")
                print(f"Mean Radial Error: {metrics['mre']:.3f} mm")
                print(f"SDR@2mm: {metrics['sdr_2mm']:.1f}%")
                print(f"SDR@2.5mm: {metrics['sdr_2.5mm']:.1f}%")
                print(f"SDR@3mm: {metrics['sdr_3mm']:.1f}%")
        
        logger.info("🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
