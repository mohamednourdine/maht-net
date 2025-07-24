#!/usr/bin/env python3
"""
Production Data Generation Script for MAHT-Net
Processes ISBI 2015 cephalometric dataset for training
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.data.data_generation import EnhancedDataGenerator, DataGenerationConfig
from src.utils.visualization import plot_dataset_statistics

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_generation.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Generate training data for MAHT-Net')
    parser.add_argument('--input-dir', required=True, help='Input dataset directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed data')
    parser.add_argument('--target-size', nargs=2, type=int, default=[512, 512],  help='Target image size (height width)')
    parser.add_argument('--heatmap-size', nargs=2, type=int, default=[128, 128], help='Heatmap size (height width)')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian sigma for heatmaps')
    parser.add_argument('--augmentation', action='store_true', help='Enable data augmentation')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create configuration
    config = DataGenerationConfig(
        target_size=tuple(args.target_size),
        heatmap_size=tuple(args.heatmap_size),
        gaussian_sigma=args.sigma,
        enable_augmentation=args.augmentation,
        save_intermediate_results=args.visualize,
        visualization_samples=5 if args.visualize else 0
    )
    
    logger.info(f"🚀 Starting data generation with configuration:")
    logger.info(f"  • Input: {args.input_dir}")
    logger.info(f"  • Output: {args.output_dir}")
    logger.info(f"  • Target size: {config.target_size}")
    logger.info(f"  • Heatmap size: {config.heatmap_size}")
    logger.info(f"  • Augmentation: {config.enable_augmentation}")
    
    # Initialize data generator
    data_generator = EnhancedDataGenerator(config)
    
    # Process dataset
    split_ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio
    }
    
    stats = data_generator.process_dataset(
        dataset_path=args.input_dir,
        output_path=args.output_dir,
        split_ratios=split_ratios
    )
    
    # Generate statistics visualization
    if args.visualize:
        stats_output = Path(args.output_dir) / 'visualizations' / 'dataset_statistics.png'
        plot_dataset_statistics(stats, save_path=str(stats_output))
        logger.info(f"📊 Dataset statistics saved to: {stats_output}")
    
    logger.info("✅ Data generation completed successfully!")
    logger.info(f"📊 Final statistics: {stats['successful_samples']}/{stats['total_samples']} samples processed")

if __name__ == "__main__":
    main()
