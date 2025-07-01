"""
MAHT-Net Training Implementation
Comprehensive training pipeline with progressive 3-stage strategy,
clinical validation, and production-ready monitoring
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..models.maht_net import MAHTNet
from ..data.dataset import CephalometricDataset
from ..utils.metrics import compute_landmark_metrics
from ..utils.losses import CombinedLandmarkLoss

logger = logging.getLogger(__name__)


class MAHTNetTrainer:
    """
    Comprehensive trainer for MAHT-Net with progressive 3-stage training strategy
    
    Stage 1: CNN Baseline Training (Weeks 9-12)
    Stage 2: Transformer Integration (Weeks 13-16) 
    Stage 3: Full Multi-Task Learning (Weeks 17-20)
    """
    
    def __init__(self,
                 model: MAHTNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device,
                 experiment_dir: str):
        """
        Initialize the MAHT-Net trainer
        
        Args:
            model: MAHT-Net model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Computing device (CPU/GPU)
            experiment_dir: Directory to save experiment results
        """
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self._setup_training_components()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_mre = float('inf')
        self.training_stage = 1
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info("✅ MAHT-Net Trainer initialized successfully")
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def _setup_training_components(self):
        """Setup optimizers, schedulers, loss functions, and monitoring"""
        
        # Loss function with multi-task weighting
        self.criterion = CombinedLandmarkLoss(
            heatmap_weight=self.config.get('heatmap_weight', 1.0),
            coord_weight=self.config.get('coord_weight', 1.0),
            uncertainty_weight=self.config.get('uncertainty_weight', 0.1)
        )
        
        # Progressive optimizer setup
        self.optimizer = self._setup_progressive_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=self.config.get('scheduler_tmult', 2),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # TensorBoard logging
        self.writer = SummaryWriter(
            log_dir=self.experiment_dir / 'tensorboard'
        )
        
        # Model checkpointing
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def _setup_progressive_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with progressive parameter groups"""
        
        # Define parameter groups for progressive training
        cnn_params = []
        transformer_params = []
        decoder_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'feature_extractor' in name:
                cnn_params.append(param)
            elif 'transformer_bottleneck' in name:
                transformer_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            elif 'detection_head' in name:
                head_params.append(param)
        
        # Progressive learning rates
        base_lr = self.config.get('base_lr', 1e-4)
        
        param_groups = [
            {'params': cnn_params, 'lr': base_lr * 0.1, 'name': 'cnn'},
            {'params': transformer_params, 'lr': base_lr, 'name': 'transformer'},
            {'params': decoder_params, 'lr': base_lr, 'name': 'decoder'},
            {'params': head_params, 'lr': base_lr * 2, 'name': 'head'}
        ]
        
        return optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
    
    def _setup_logging(self):
        """Setup comprehensive logging for training monitoring"""
        
        # File logging
        log_file = self.experiment_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with comprehensive monitoring"""
        
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'heatmap_loss': 0.0,
            'coord_loss': 0.0,
            'uncertainty_loss': 0.0,
            'mre': 0.0
        }
        
        num_batches = len(self.train_loader)
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Stage {self.training_stage} - Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            target_landmarks = batch['landmarks'].to(self.device)
            target_heatmaps = batch['heatmaps'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute losses
            loss_dict = self.criterion(
                outputs,
                {
                    'landmarks': target_landmarks,
                    'heatmaps': target_heatmaps
                }
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value.item() * batch_size
            
            # Compute MRE for monitoring
            with torch.no_grad():
                pred_coords = outputs['coordinates']
                mre = compute_landmark_metrics(pred_coords, target_landmarks)['mre']
                epoch_metrics['mre'] += mre * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'MRE': f"{mre:.2f}mm"
            })
            
            # Log batch metrics to TensorBoard
            if batch_idx % self.config.get('log_interval', 100) == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self._log_batch_metrics(loss_dict, global_step)
        
        # Average metrics over epoch
        total_samples = len(self.train_loader.dataset)
        for key in epoch_metrics:
            epoch_metrics[key] /= total_samples
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate model performance with clinical metrics"""
        
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'heatmap_loss': 0.0,
            'coord_loss': 0.0,
            'uncertainty_loss': 0.0,
            'mre': 0.0,
            'sdr_2mm': 0.0,
            'sdr_2.5mm': 0.0,
            'sdr_3mm': 0.0,
            'sdr_4mm': 0.0
        }
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                images = batch['image'].to(self.device)
                target_landmarks = batch['landmarks'].to(self.device)
                target_heatmaps = batch['heatmaps'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute losses
                loss_dict = self.criterion(
                    outputs,
                    {
                        'landmarks': target_landmarks,
                        'heatmaps': target_heatmaps
                    }
                )
                
                # Accumulate losses
                batch_size = images.size(0)
                for key, value in loss_dict.items():
                    if key in val_metrics:
                        val_metrics[key] += value.item() * batch_size
                
                # Store predictions for comprehensive metrics
                pred_coords = outputs['coordinates']
                all_predictions.append(pred_coords.cpu())
                all_targets.append(target_landmarks.cpu())
        
        # Compute comprehensive metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        detailed_metrics = compute_landmark_metrics(all_predictions, all_targets)
        val_metrics.update(detailed_metrics)
        
        # Average metrics
        total_samples = len(self.val_loader.dataset)
        for key in ['total_loss', 'heatmap_loss', 'coord_loss', 'uncertainty_loss']:
            val_metrics[key] /= total_samples
        
        return val_metrics
    
    def train_stage(self, stage: int, num_epochs: int):
        """
        Train a specific stage of the progressive training strategy
        
        Stage 1: CNN Baseline Training
        Stage 2: Transformer Integration  
        Stage 3: Full Multi-Task Learning
        """
        
        self.training_stage = stage
        logger.info(f"🚀 Starting Training Stage {stage}")
        
        # Configure stage-specific settings
        self._configure_training_stage(stage)
        
        for epoch in range(num_epochs):
            self.current_epoch = (stage - 1) * num_epochs + epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_metrics = self.validate_epoch()
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Model checkpointing
            self._save_checkpoint(val_metrics)
            
            # Early stopping check
            if self._should_early_stop(val_metrics):
                logger.info(f"⏰ Early stopping triggered at epoch {self.current_epoch + 1}")
                break
            
            # Clinical validation milestone
            if (epoch + 1) % self.config.get('clinical_eval_interval', 10) == 0:
                self._clinical_validation_check(val_metrics)
        
        logger.info(f"✅ Stage {stage} training completed")
    
    def _configure_training_stage(self, stage: int):
        """Configure model and training parameters for specific stage"""
        
        if stage == 1:
            # Stage 1: CNN Baseline - Freeze transformer
            logger.info("🔒 Stage 1: Freezing transformer components")
            for name, param in self.model.named_parameters():
                if 'transformer_bottleneck' in name:
                    param.requires_grad = False
                    
        elif stage == 2:
            # Stage 2: Transformer Integration - Unfreeze transformer
            logger.info("🔓 Stage 2: Unfreezing transformer components")
            for param in self.model.parameters():
                param.requires_grad = True
                
            # Reduce learning rates for stability
            for group in self.optimizer.param_groups:
                group['lr'] *= 0.5
                
        elif stage == 3:
            # Stage 3: Full Multi-Task Learning - All components active
            logger.info("🎯 Stage 3: Full multi-task training")
            for param in self.model.parameters():
                param.requires_grad = True
            
            # Fine-tune loss weights
            self.criterion.update_weights(
                heatmap_weight=0.8,
                coord_weight=1.0,
                uncertainty_weight=0.2
            )
    
    def _log_batch_metrics(self, loss_dict: Dict[str, torch.Tensor], global_step: int):
        """Log batch-level metrics to TensorBoard"""
        
        for key, value in loss_dict.items():
            self.writer.add_scalar(f'batch/{key}', value.item(), global_step)
        
        # Log learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'lr/group_{i}', group['lr'], global_step)
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch-level metrics"""
        
        epoch = self.current_epoch
        
        # Console logging
        logger.info(
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Val MRE: {val_metrics['mre']:.2f}mm | "
            f"SDR@2mm: {val_metrics['sdr_2mm']:.1f}%"
        )
        
        # TensorBoard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        for key, value in val_metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
    
    def _save_checkpoint(self, val_metrics: Dict):
        """Save model checkpoint with comprehensive metadata"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'stage': self.training_stage,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config,
            'model_info': self.model.get_model_info()
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model based on validation MRE
        if val_metrics['mre'] < self.best_val_mre:
            self.best_val_mre = val_metrics['mre']
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"💾 New best model saved (MRE: {self.best_val_mre:.2f}mm)")
        
        # Save stage checkpoints
        if self.current_epoch % self.config.get('save_interval', 20) == 0:
            stage_path = self.checkpoint_dir / f'stage_{self.training_stage}_epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, stage_path)
    
    def _should_early_stop(self, val_metrics: Dict) -> bool:
        """Check early stopping criteria"""
        
        patience = self.config.get('early_stop_patience', 20)
        min_delta = self.config.get('early_stop_min_delta', 0.01)
        
        # Implementation of early stopping logic
        # This is a simplified version - full implementation would track improvement history
        
        return False  # Disable early stopping for progressive training
    
    def _clinical_validation_check(self, val_metrics: Dict):
        """Perform clinical validation milestone checks"""
        
        mre = val_metrics['mre']
        sdr_2mm = val_metrics['sdr_2mm']
        
        # Clinical thresholds
        clinical_mre_threshold = 1.5  # mm
        clinical_sdr_threshold = 90.0  # %
        
        if mre <= clinical_mre_threshold and sdr_2mm >= clinical_sdr_threshold:
            logger.info(f"🏥 Clinical milestone achieved! MRE: {mre:.2f}mm, SDR@2mm: {sdr_2mm:.1f}%")
        else:
            logger.info(f"🏥 Clinical progress: MRE: {mre:.2f}mm (target: ≤{clinical_mre_threshold}mm), "
                       f"SDR@2mm: {sdr_2mm:.1f}% (target: ≥{clinical_sdr_threshold}%)")
    
    def save_training_summary(self):
        """Save comprehensive training summary"""
        
        summary = {
            'experiment_dir': str(self.experiment_dir),
            'total_epochs': self.current_epoch + 1,
            'training_stages': self.training_stage,
            'best_val_mre': self.best_val_mre,
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'model_info': self.model.get_model_info()
        }
        
        summary_path = self.experiment_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📊 Training summary saved to {summary_path}")


def train_maht_net(config_path: str):
    """
    Main training function for MAHT-Net with progressive 3-stage strategy
    
    Args:
        config_path: Path to training configuration file
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Create model
    model = MAHTNet(
        num_landmarks=config['model']['num_landmarks'],
        pretrained=config['model']['pretrained']
    )
    
    # Create datasets and dataloaders
    from ..data.dataset import create_dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create trainer
    trainer = MAHTNetTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config['training'],
        device=device,
        experiment_dir=config['experiment']['output_dir']
    )
    
    # Progressive 3-stage training
    stage_epochs = config['training']['stage_epochs']
    
    try:
        for stage in range(1, 4):
            trainer.train_stage(stage, stage_epochs)
        
        logger.info("🎉 MAHT-Net training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        
    finally:
        # Save training summary
        trainer.save_training_summary()
        trainer.writer.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/train_config.json"
    
    train_maht_net(config_path)
