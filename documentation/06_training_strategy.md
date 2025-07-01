# Training Strategy: MAHT-Net Optimization Framework

## Training Philosophy

The training strategy for MAHT-Net addresses the unique challenges of:
- **Limited Dataset Size** (400 images)
- **Complex Multi-Component Architecture**
- **Multi-Task Learning** (heatmaps + coordinates)
- **Medical Image Precision Requirements**

## Progressive Training Strategy

### Stage 1: Baseline Establishment (Epochs 1-50)

**Objective**: Establish stable baseline performance with simplified architecture

```python
# src/training/progressive_trainer.py
class ProgressiveTrainer:
    """Progressive training strategy for MAHT-Net"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_stage = 1
        
    def stage_1_baseline(self):
        """Stage 1: Train with frozen transformer, simple loss"""
        
        # Freeze transformer components
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        
        # Simple MSE loss only
        criterion = nn.MSELoss()
        
        # Conservative learning rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-5,
            weight_decay=1e-4
        )
        
        return criterion, optimizer
    
    def stage_2_transformer(self):
        """Stage 2: Unfreeze transformer, add SSIM loss"""
        
        # Unfreeze transformer
        for param in self.model.transformer.parameters():
            param.requires_grad = True
        
        # Multi-component loss
        criterion = MAHTNetLoss(
            heatmap_weight=1.0,
            coordinate_weight=0.0,  # No coordinate loss yet
            ssim_weight=0.3
        )
        
        # Lower learning rate for pretrained components
        optimizer = self._setup_differential_optimizer()
        
        return criterion, optimizer
    
    def stage_3_full_model(self):
        """Stage 3: Full model with all loss components"""
        
        # Full multi-task loss
        criterion = MAHTNetLoss(
            heatmap_weight=1.0,
            coordinate_weight=0.1,
            ssim_weight=0.5
        )
        
        optimizer = self._setup_differential_optimizer()
        
        return criterion, optimizer
```

### Stage 2: Transformer Integration (Epochs 51-100)

**Objective**: Gradually introduce transformer complexity and SSIM loss

- Unfreeze transformer components
- Add structural similarity loss
- Implement attention visualization
- Monitor gradient flow

### Stage 3: Full Multi-Task Training (Epochs 101-200)

**Objective**: Full model training with all loss components

- Complete loss function activation
- Fine-tune loss weight balance
- Implement uncertainty quantification
- Clinical validation preparation

## Advanced Training Techniques

### 1. Curriculum Learning

```python
# src/training/curriculum.py
class CurriculumLearning:
    """Implement curriculum learning for medical images"""
    
    def __init__(self, dataset, difficulty_metric='image_quality'):
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric
        self.current_difficulty = 0.3  # Start with easier samples
        
    def get_curriculum_subset(self, epoch):
        """Get training subset based on curriculum"""
        
        # Gradually increase difficulty
        if epoch < 50:
            self.current_difficulty = 0.3 + (epoch / 50) * 0.4  # 0.3 -> 0.7
        else:
            self.current_difficulty = 0.7 + ((epoch - 50) / 150) * 0.3  # 0.7 -> 1.0
        
        # Select samples based on difficulty
        if self.difficulty_metric == 'image_quality':
            subset_indices = self._select_by_image_quality()
        elif self.difficulty_metric == 'landmark_visibility':
            subset_indices = self._select_by_landmark_visibility()
        
        return torch.utils.data.Subset(self.dataset, subset_indices)
    
    def _select_by_image_quality(self):
        """Select samples based on image quality metrics"""
        # Implementation based on contrast, noise, blur metrics
        qualities = self._compute_image_qualities()
        threshold = np.percentile(qualities, self.current_difficulty * 100)
        return np.where(qualities <= threshold)[0]
    
    def _compute_image_qualities(self):
        """Compute image quality metrics for curriculum"""
        qualities = []
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            image = sample['image']
            
            # Compute quality metrics
            contrast = self._compute_contrast(image)
            noise_level = self._compute_noise(image)
            blur_metric = self._compute_blur(image)
            
            # Combined quality score (lower = easier)
            quality = 0.4 * (1 - contrast) + 0.3 * noise_level + 0.3 * blur_metric
            qualities.append(quality)
        
        return np.array(qualities)
```

### 2. Data Augmentation Strategy

```python
# src/training/augmentation_strategy.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AdaptiveAugmentation:
    """Adaptive augmentation strategy for training stages"""
    
    def __init__(self, stage='baseline'):
        self.stage = stage
        self.augmentations = self._get_stage_augmentations()
    
    def _get_stage_augmentations(self):
        """Get augmentation pipeline for current training stage"""
        
        if self.stage == 'baseline':
            # Conservative augmentation for baseline
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.3
                ),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        elif self.stage == 'transformer':
            # Moderate augmentation for transformer training
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.05, 
                    rotate_limit=10, 
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, 
                    contrast_limit=0.15, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 30), p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        elif self.stage == 'full':
            # Aggressive augmentation for full model
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.8),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.6
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.4),
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    alpha_affine=50, 
                    p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(
                    distort_limit=0.1, 
                    shift_limit=0.1, 
                    p=0.3
                ),
                # Medical-specific augmentations
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.3),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def update_stage(self, new_stage):
        """Update augmentation strategy for new training stage"""
        self.stage = new_stage
        self.augmentations = self._get_stage_augmentations()
```

### 3. Loss Function Design

```python
# src/training/advanced_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM
import numpy as np

class AdaptiveLoss(nn.Module):
    """Adaptive loss function that adjusts weights during training"""
    
    def __init__(self, 
                 initial_weights={'heatmap': 1.0, 'coord': 0.1, 'ssim': 0.5},
                 adaptation_strategy='uncertainty'):
        super().__init__()
        
        self.weights = nn.Parameter(torch.tensor(list(initial_weights.values())))
        self.adaptation_strategy = adaptation_strategy
        
        # Individual loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
        
        # For uncertainty weighting
        self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, predictions, targets, epoch=None):
        """Compute adaptive loss"""
        
        # Individual losses
        heatmap_loss = self.mse_loss(predictions['heatmaps'], targets['heatmaps'])
        
        coord_loss = self.l1_loss(
            predictions['coordinates'], 
            targets['landmarks']
        ) if 'coordinates' in predictions else torch.tensor(0.0)
        
        ssim_loss = 1 - self.ssim_loss(
            predictions['heatmaps'].unsqueeze(2),
            targets['heatmaps'].unsqueeze(2)
        )
        
        losses = [heatmap_loss, coord_loss, ssim_loss]
        
        if self.adaptation_strategy == 'uncertainty':
            # Uncertainty-based weighting
            weighted_losses = []
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                weighted_losses.append(weighted_loss)
            
            total_loss = sum(weighted_losses)
        
        elif self.adaptation_strategy == 'gradnorm':
            # GradNorm-based adaptation
            total_loss = self._gradnorm_weighting(losses, predictions, targets)
        
        else:
            # Fixed weighting
            total_loss = sum(w * loss for w, loss in zip(self.weights, losses))
        
        return {
            'total': total_loss,
            'heatmap_mse': heatmap_loss,
            'coordinate_l1': coord_loss,
            'ssim': ssim_loss,
            'weights': self.weights.detach().cpu().numpy()
        }

class FocalLoss(nn.Module):
    """Focal loss for addressing hard sample mining"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """Compute focal loss for heatmap regression"""
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-mse_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        focal_loss = focal_weight * mse_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WingLoss(nn.Module):
    """Wing loss for robust landmark detection"""
    
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)
    
    def forward(self, predictions, targets):
        """Compute wing loss"""
        diff = torch.abs(predictions - targets)
        
        # Wing loss formulation
        wing_loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        
        return wing_loss.mean()
```

### 4. Regularization Strategies

```python
# src/training/regularization.py
import torch
import torch.nn as nn

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for transformers"""
    
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output

class SpatialDropout2d(nn.Module):
    """Spatial dropout for preserving spatial structure"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Create spatial mask
        batch_size, channels = x.shape[:2]
        mask = torch.bernoulli(
            torch.full((batch_size, channels, 1, 1), 1 - self.p, device=x.device)
        )
        
        return x * mask / (1 - self.p)

class LandmarkConsistencyLoss(nn.Module):
    """Consistency loss for anatomically plausible landmarks"""
    
    def __init__(self, anatomical_constraints):
        super().__init__()
        self.constraints = anatomical_constraints
    
    def forward(self, coordinates):
        """Enforce anatomical constraints"""
        consistency_loss = 0
        
        for constraint in self.constraints:
            if constraint['type'] == 'distance':
                # Distance constraint between landmarks
                p1, p2 = constraint['landmarks']
                expected_dist = constraint['distance']
                tolerance = constraint['tolerance']
                
                actual_dist = torch.norm(
                    coordinates[:, p1] - coordinates[:, p2], 
                    dim=1
                )
                
                dist_error = torch.abs(actual_dist - expected_dist)
                consistency_loss += torch.clamp(dist_error - tolerance, min=0).mean()
            
            elif constraint['type'] == 'angle':
                # Angle constraint between three landmarks
                p1, p2, p3 = constraint['landmarks']
                expected_angle = constraint['angle']
                tolerance = constraint['tolerance']
                
                # Compute angle
                v1 = coordinates[:, p1] - coordinates[:, p2]
                v2 = coordinates[:, p3] - coordinates[:, p2]
                
                actual_angle = torch.acos(
                    torch.clamp(
                        torch.sum(v1 * v2, dim=1) / 
                        (torch.norm(v1, dim=1) * torch.norm(v2, dim=1)),
                        -1, 1
                    )
                )
                
                angle_error = torch.abs(actual_angle - expected_angle)
                consistency_loss += torch.clamp(angle_error - tolerance, min=0).mean()
        
        return consistency_loss
```

## Optimization Strategies

### 1. Learning Rate Scheduling

```python
# src/training/schedulers.py
import torch
import math

class WarmupCosineScheduler:
    """Warmup + Cosine annealing scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.base_lrs[i] * epoch / self.warmup_epochs
                param_group['lr'] = lr
        else:
            # Cosine annealing
            for i, param_group in enumerate(self.optimizer.param_groups):
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr = self.min_lr + (self.base_lrs[i] - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
                param_group['lr'] = lr

class CyclicalLRScheduler:
    """Cyclical learning rate for finding optimal LR"""
    
    def __init__(self, optimizer, base_lr, max_lr, step_size_up):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_num = 0
    
    def step(self):
        cycle = math.floor(1 + self.step_num / (2 * self.step_size_up))
        x = abs(self.step_num / self.step_size_up - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.step_num += 1
```

### 2. Gradient Management

```python
# src/training/gradient_utils.py
import torch
import torch.nn as nn

class GradientClipper:
    """Advanced gradient clipping strategies"""
    
    def __init__(self, clip_type='norm', clip_value=1.0):
        self.clip_type = clip_type
        self.clip_value = clip_value
    
    def clip_gradients(self, model):
        if self.clip_type == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        elif self.clip_type == 'adaptive':
            self._adaptive_clip(model)
    
    def _adaptive_clip(self, model):
        """Adaptive gradient clipping based on gradient norms"""
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            median_norm = torch.median(torch.tensor(grad_norms))
            clip_value = min(self.clip_value, median_norm * 2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

class GradientAccumulator:
    """Gradient accumulation for large effective batch sizes"""
    
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def step(self, loss, model, optimizer, scaler=None):
        """Accumulate gradients and step when ready"""
        loss = loss / self.accumulation_steps
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            return True
        
        return False
```

## Training Monitoring and Validation

### 1. Comprehensive Metrics Tracking

```python
# src/training/monitoring.py
import torch
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class TrainingMonitor:
    """Comprehensive training monitoring"""
    
    def __init__(self, experiment_name, use_wandb=True):
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Setup logging
        self.writer = SummaryWriter(f"logs/{experiment_name}")
        
        if use_wandb:
            wandb.init(project="maht-net", name=experiment_name)
        
        # Metric history
        self.train_history = {'loss': [], 'mre': [], 'sdr_2mm': []}
        self.val_history = {'loss': [], 'mre': [], 'sdr_2mm': []}
        
    def log_metrics(self, metrics, step, phase='train'):
        """Log metrics to all monitoring systems"""
        
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, step)
        
        # Wandb
        if self.use_wandb:
            wandb_metrics = {f"{phase}_{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=step)
        
        # Update history
        if phase == 'train':
            history = self.train_history
        else:
            history = self.val_history
        
        for key in ['loss', 'mre', 'sdr_2mm']:
            if key in metrics:
                history[key].append(metrics[key])
    
    def log_images(self, images, predictions, targets, step, num_samples=4):
        """Log image predictions for visual inspection"""
        
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
        
        for i in range(num_samples):
            # Original image
            img = images[i, 0].cpu().numpy()
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Predicted heatmap
            pred_hm = predictions['heatmaps'][i].sum(0).cpu().numpy()
            axes[1, i].imshow(pred_hm, cmap='hot')
            axes[1, i].set_title('Predicted')
            axes[1, i].axis('off')
            
            # Ground truth heatmap
            gt_hm = targets['heatmaps'][i].sum(0).cpu().numpy()
            axes[2, i].imshow(gt_hm, cmap='hot')
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.writer.add_figure('Predictions', fig, step)
        
        # Log to Wandb
        if self.use_wandb:
            wandb.log({"predictions": wandb.Image(fig)}, step=step)
        
        plt.close(fig)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.train_history['loss'], label='Train')
        axes[0].plot(self.val_history['loss'], label='Validation')
        axes[0].set_title('Loss')
        axes[0].legend()
        
        # MRE curves
        axes[1].plot(self.train_history['mre'], label='Train')
        axes[1].plot(self.val_history['mre'], label='Validation')
        axes[1].set_title('Mean Radial Error (mm)')
        axes[1].legend()
        
        # SDR curves
        axes[2].plot(self.train_history['sdr_2mm'], label='Train')
        axes[2].plot(self.val_history['sdr_2mm'], label='Validation')
        axes[2].set_title('Success Detection Rate @2mm')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{self.experiment_name}_training_curves.png')
        
        if self.use_wandb:
            wandb.log({"training_curves": wandb.Image(fig)})
        
        return fig
```

### 2. Cross-Validation Strategy

```python
# src/training/cross_validation.py
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data import Subset

class CrossValidator:
    """K-fold cross-validation for small datasets"""
    
    def __init__(self, dataset, k_folds=5, stratify_by=None):
        self.dataset = dataset
        self.k_folds = k_folds
        self.stratify_by = stratify_by
        
        if stratify_by:
            # Stratified K-fold based on metadata
            labels = self._extract_stratification_labels()
            self.kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            self.splits = list(self.kf.split(range(len(dataset)), labels))
        else:
            # Standard K-fold
            self.kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            self.splits = list(self.kf.split(range(len(dataset))))
    
    def get_fold_datasets(self, fold_idx):
        """Get train and validation datasets for a specific fold"""
        train_indices, val_indices = self.splits[fold_idx]
        
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        
        return train_dataset, val_dataset
    
    def _extract_stratification_labels(self):
        """Extract labels for stratification"""
        if self.stratify_by == 'age_group':
            # Stratify by age group (pediatric vs adult)
            return [self._get_age_group(i) for i in range(len(self.dataset))]
        elif self.stratify_by == 'image_quality':
            # Stratify by image quality
            return [self._get_quality_group(i) for i in range(len(self.dataset))]
        else:
            return [0] * len(self.dataset)  # No stratification
    
    def run_cross_validation(self, model_class, config):
        """Run complete cross-validation"""
        fold_results = []
        
        for fold in range(self.k_folds):
            print(f"Training fold {fold + 1}/{self.k_folds}")
            
            # Get fold datasets
            train_dataset, val_dataset = self.get_fold_datasets(fold)
            
            # Create fresh model
            model = model_class(config.model)
            
            # Train model
            trainer = MAHTNetTrainer(model, train_dataset, val_dataset, config)
            results = trainer.train()
            
            fold_results.append(results)
        
        # Aggregate results
        return self._aggregate_cv_results(fold_results)
```

## Hyperparameter Optimization

### 1. Bayesian Optimization

```python
# src/training/hyperopt.py
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameters"""
    
    def __init__(self, model_class, dataset, n_trials=50):
        self.model_class = model_class
        self.dataset = dataset
        self.n_trials = n_trials
    
    def objective(self, trial):
        """Optuna objective function"""
        
        # Suggest hyperparameters
        config = self._suggest_hyperparameters(trial)
        
        # Cross-validation
        cv = CrossValidator(self.dataset, k_folds=3)  # Reduced folds for speed
        fold_results = []
        
        for fold in range(3):
            train_dataset, val_dataset = cv.get_fold_datasets(fold)
            
            # Create model and train
            model = self.model_class(config.model)
            trainer = MAHTNetTrainer(model, train_dataset, val_dataset, config)
            
            # Early training for hyperparameter search
            result = trainer.train_limited(max_epochs=50)
            fold_results.append(result['best_mre'])
            
            # Pruning
            trial.report(result['best_mre'], fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return average MRE across folds
        return np.mean(fold_results)
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for trial"""
        
        # Learning rate
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        
        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        
        # Transformer parameters
        transformer_layers = trial.suggest_int('transformer_layers', 4, 12)
        transformer_heads = trial.suggest_categorical('transformer_heads', [6, 8, 12, 16])
        
        # Loss weights
        heatmap_weight = trial.suggest_uniform('heatmap_weight', 0.5, 2.0)
        ssim_weight = trial.suggest_uniform('ssim_weight', 0.1, 1.0)
        
        # Create config
        config = ExperimentConfig(
            model=ModelConfig(
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads
            ),
            data=DataConfig(
                batch_size=batch_size
            ),
            training=TrainingConfig(
                learning_rate=lr,
                heatmap_loss_weight=heatmap_weight,
                ssim_loss_weight=ssim_weight
            )
        )
        
        return config
    
    def optimize(self):
        """Run hyperparameter optimization"""
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        
        return study.best_params, study.best_value
```

## Training Pipeline Integration

### Complete Training Script

```python
# scripts/train_maht_net.py
import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train MAHT-Net')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--cross_val', action='store_true', help='Use cross-validation')
    parser.add_argument('--hyperopt', action='store_true', help='Run hyperparameter optimization')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(42)
    logger = setup_logging('logs', config.experiment_name)
    
    # Load dataset
    dataset = CephalometricDataset(config.data.data_dir)
    
    if args.hyperopt:
        # Hyperparameter optimization
        optimizer = HyperparameterOptimizer(MAHTNet, dataset)
        best_params, best_score = optimizer.optimize()
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best MRE: {best_score:.4f}")
        
    elif args.cross_val:
        # Cross-validation
        cv = CrossValidator(dataset, k_folds=5)
        results = cv.run_cross_validation(MAHTNet, config)
        logger.info(f"CV Results: {results}")
        
    else:
        # Standard training
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create model
        model = MAHTNet(
            encoder_type=config.model.encoder_type,
            transformer_type=config.model.transformer_type,
            num_landmarks=config.model.num_landmarks
        )
        
        # Train
        trainer = MAHTNetTrainer(model, train_dataset, val_dataset, config)
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        trainer.train()

if __name__ == "__main__":
    main()
```

This comprehensive training strategy addresses the unique challenges of training MAHT-Net on the limited ISBI 2015 dataset while maximizing performance through advanced techniques and careful monitoring.
