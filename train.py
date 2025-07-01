"""
Training script for MAHT-Net
Main entry point for training the Multi-Stage Attention-enhanced Hybrid Transformer Network
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.maht_net import create_maht_net
from src.data.dataset import create_dataloaders
from src.training.trainer import MAHTNetTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MAHT-Net for Cephalometric Landmark Detection")
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.json',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Override data directory from config'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device ID to use'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced dataset'
    )
    
    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load and validate training configuration"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    # Debug mode adjustments
    if args.debug:
        config['training']['stage_epochs'] = 2
        config['training']['batch_size'] = 2
        config['experiment']['name'] += '_debug'
        logger.info("🐛 Debug mode enabled - using reduced settings")
    
    return config


def setup_device(gpu_id: int = None) -> torch.device:
    """Setup compute device"""
    
    if gpu_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args)
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['experiment']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['random_seed'])
    
    logger.info("🚀 Starting MAHT-Net Training")
    logger.info(f"📋 Experiment: {config['experiment']['name']}")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Create model
        logger.info("🧠 Creating MAHT-Net model...")
        model = create_maht_net(
            num_landmarks=config['model']['num_landmarks'],
            pretrained=config['model']['pretrained'],
            heatmap_size=config['model']['heatmap_size']
        )
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        dataloaders = create_dataloaders(
            data_dir=config['data']['data_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            image_size=tuple(config['data']['image_size']),
            heatmap_size=tuple(config['data']['heatmap_size'])
        )
        
        # Verify data loaders
        required_splits = ['train', 'val']
        for split in required_splits:
            if split not in dataloaders:
                raise ValueError(f"Missing required data split: {split}")
        
        logger.info(f"Data loaders created:")
        for split, loader in dataloaders.items():
            logger.info(f"  {split}: {len(loader.dataset)} samples")
        
        # Create trainer
        logger.info("🏋️ Creating trainer...")
        trainer = MAHTNetTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            config=config['training'],
            device=device,
            experiment_dir=config['experiment']['output_dir']
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"🔄 Resuming training from {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Progressive 3-stage training
        stage_epochs = config['training']['stage_epochs']
        
        logger.info("🎯 Starting Progressive 3-Stage Training Strategy")
        logger.info(f"📅 Each stage: {stage_epochs} epochs")
        
        for stage in range(1, 4):
            stage_name = {
                1: "CNN Baseline Training",
                2: "Transformer Integration", 
                3: "Full Multi-Task Learning"
            }[stage]
            
            logger.info(f"\n🏆 Stage {stage}: {stage_name}")
            logger.info("=" * 50)
            
            trainer.train_stage(stage, stage_epochs)
            
            # Clinical validation check
            if stage == 3:
                logger.info("🏥 Performing final clinical validation...")
                trainer.clinical_validation()
        
        logger.info("\n🎉 MAHT-Net training completed successfully!")
        
        # Save final training summary
        trainer.save_training_summary()
        
        # Generate training report
        logger.info("Generating training report...")
        trainer.generate_training_report()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
        if 'trainer' in locals():
            trainer.save_training_summary()
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise
    
    finally:
        logger.info("🧹 Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
