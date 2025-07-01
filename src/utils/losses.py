"""
Combined Loss Functions for MAHT-Net
Multi-task loss combining heatmap regression, coordinate prediction, and uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class HeatmapLoss(nn.Module):
    """
    Loss function for heatmap regression with focal loss modification
    Handles class imbalance in landmark detection
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha  # Focal loss alpha
        self.beta = beta    # Focal loss beta
        
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for heatmap regression
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, L, H, W]
            target_heatmaps: Target heatmaps [B, L, H, W]
            
        Returns:
            Heatmap loss value
        """
        
        # Ensure predictions are in valid range [0, 1]
        pred_heatmaps = torch.clamp(pred_heatmaps, 1e-7, 1 - 1e-7)
        
        # Positive and negative pixel masks
        pos_mask = target_heatmaps.eq(1).float()
        neg_mask = target_heatmaps.lt(1).float()
        
        # Number of positive pixels per heatmap
        num_pos = pos_mask.sum(dim=[2, 3], keepdim=True)
        
        # Positive loss (focal loss modification)
        pos_loss = torch.log(pred_heatmaps) * torch.pow(1 - pred_heatmaps, self.alpha) * pos_mask
        
        # Negative loss (focal loss modification)
        neg_weights = torch.pow(1 - target_heatmaps, self.beta)
        neg_loss = torch.log(1 - pred_heatmaps) * torch.pow(pred_heatmaps, self.alpha) * neg_weights * neg_mask
        
        # Normalize by number of positive pixels
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        num_pos_total = num_pos.sum()
        if num_pos_total == 0:
            return -neg_loss
        else:
            return -(pos_loss + neg_loss) / num_pos_total


class CoordinateLoss(nn.Module):
    """
    Loss function for direct coordinate regression
    Uses smooth L1 loss for robustness to outliers
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute smooth L1 loss for coordinate regression
        
        Args:
            pred_coords: Predicted coordinates [B, L, 2]
            target_coords: Target coordinates [B, L, 2]
            
        Returns:
            Coordinate loss value
        """
        
        loss = F.smooth_l1_loss(pred_coords, target_coords, reduction=self.reduction)
        return loss


class UncertaintyLoss(nn.Module):
    """
    Loss function for uncertainty estimation
    Encourages higher uncertainty for more difficult predictions
    """
    
    def __init__(self, lambda_var: float = 1.0):
        super().__init__()
        self.lambda_var = lambda_var
        
    def forward(self, 
                pred_coords: torch.Tensor, 
                target_coords: torch.Tensor,
                uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware loss
        
        Args:
            pred_coords: Predicted coordinates [B, L, 2]
            target_coords: Target coordinates [B, L, 2]
            uncertainties: Predicted uncertainties [B, L]
            
        Returns:
            Uncertainty loss value
        """
        
        # Compute coordinate errors
        coord_errors = torch.norm(pred_coords - target_coords, dim=2)  # [B, L]
        
        # Uncertainty-weighted loss
        # Higher uncertainty should correlate with higher errors
        weighted_errors = coord_errors / (uncertainties + 1e-8)
        regularization = torch.log(uncertainties + 1e-8)
        
        uncertainty_loss = weighted_errors + self.lambda_var * regularization
        
        return uncertainty_loss.mean()


class CombinedLandmarkLoss(nn.Module):
    """
    Combined multi-task loss for MAHT-Net
    Balances heatmap regression, coordinate prediction, and uncertainty estimation
    """
    
    def __init__(self,
                 heatmap_weight: float = 1.0,
                 coord_weight: float = 1.0,
                 uncertainty_weight: float = 0.1,
                 adaptive_weighting: bool = True):
        super().__init__()
        
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.uncertainty_weight = uncertainty_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Individual loss functions
        self.heatmap_loss = HeatmapLoss()
        self.coord_loss = CoordinateLoss()
        self.uncertainty_loss = UncertaintyLoss()
        
        # For adaptive weighting
        if adaptive_weighting:
            self.register_buffer('heatmap_running_loss', torch.tensor(1.0))
            self.register_buffer('coord_running_loss', torch.tensor(1.0))
            self.register_buffer('uncertainty_running_loss', torch.tensor(1.0))
            self.momentum = 0.9
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            predictions: Dictionary with 'heatmaps', 'coordinates', 'uncertainties'
            targets: Dictionary with 'heatmaps', 'landmarks'
            
        Returns:
            Dictionary of individual and total losses
        """
        
        # Extract predictions and targets
        pred_heatmaps = predictions['heatmaps']
        pred_coords = predictions['coordinates']
        pred_uncertainties = predictions.get('uncertainties', None)
        
        target_heatmaps = targets['heatmaps']
        target_coords = targets['landmarks']
        
        # Compute individual losses
        heatmap_loss_val = self.heatmap_loss(pred_heatmaps, target_heatmaps)
        coord_loss_val = self.coord_loss(pred_coords, target_coords)
        
        losses = {
            'heatmap_loss': heatmap_loss_val,
            'coord_loss': coord_loss_val
        }
        
        # Uncertainty loss (if available)
        if pred_uncertainties is not None:
            uncertainty_loss_val = self.uncertainty_loss(pred_coords, target_coords, pred_uncertainties)
            losses['uncertainty_loss'] = uncertainty_loss_val
        else:
            losses['uncertainty_loss'] = torch.tensor(0.0, device=pred_coords.device)
        
        # Adaptive weighting
        if self.adaptive_weighting and self.training:
            self._update_adaptive_weights(losses)
        
        # Compute total loss
        total_loss = (
            self.heatmap_weight * losses['heatmap_loss'] +
            self.coord_weight * losses['coord_loss'] +
            self.uncertainty_weight * losses['uncertainty_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _update_adaptive_weights(self, losses: Dict[str, torch.Tensor]):
        """Update adaptive weights based on loss magnitudes"""
        
        # Update running averages
        with torch.no_grad():
            self.heatmap_running_loss = (
                self.momentum * self.heatmap_running_loss + 
                (1 - self.momentum) * losses['heatmap_loss'].detach()
            )
            self.coord_running_loss = (
                self.momentum * self.coord_running_loss + 
                (1 - self.momentum) * losses['coord_loss'].detach()
            )
            self.uncertainty_running_loss = (
                self.momentum * self.uncertainty_running_loss + 
                (1 - self.momentum) * losses['uncertainty_loss'].detach()
            )
            
            # Compute relative weights (inverse of running loss magnitude)
            total_loss_magnitude = (
                self.heatmap_running_loss + 
                self.coord_running_loss + 
                self.uncertainty_running_loss + 1e-8
            )
            
            self.heatmap_weight = total_loss_magnitude / (self.heatmap_running_loss + 1e-8)
            self.coord_weight = total_loss_magnitude / (self.coord_running_loss + 1e-8)
            self.uncertainty_weight = 0.1 * total_loss_magnitude / (self.uncertainty_running_loss + 1e-8)
    
    def update_weights(self, heatmap_weight: float, coord_weight: float, uncertainty_weight: float):
        """Manually update loss weights"""
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.uncertainty_weight = uncertainty_weight


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features
    Helps with spatial consistency in heatmaps
    """
    
    def __init__(self, layers: list = ['relu1_2', 'relu2_2', 'relu3_3']):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.layers = layers
        self.layer_weights = [1.0, 1.0, 1.0]  # Equal weighting
        
    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between heatmaps
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, L, H, W]
            target_heatmaps: Target heatmaps [B, L, H, W]
            
        Returns:
            Perceptual loss value
        """
        
        # Sum heatmaps across landmarks to get single-channel maps
        pred_sum = pred_heatmaps.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        target_sum = target_heatmaps.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Convert to 3-channel for VGG (repeat channels)
        pred_rgb = pred_sum.repeat(1, 3, 1, 1)
        target_rgb = target_sum.repeat(1, 3, 1, 1)
        
        # Resize to minimum VGG input size if needed
        if pred_rgb.size(-1) < 224:
            pred_rgb = F.interpolate(pred_rgb, size=(224, 224), mode='bilinear', align_corners=False)
            target_rgb = F.interpolate(target_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features and compute loss
        pred_features = self._extract_features(pred_rgb)
        target_features = self._extract_features(target_rgb)
        
        perceptual_loss = 0.0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            perceptual_loss += self.layer_weights[i] * F.mse_loss(pred_feat, target_feat)
        
        return perceptual_loss
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified VGG layers"""
        
        features = []
        layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        
        x = self.features[0](x)  # First conv
        
        layer_idx = 0
        for i, layer in enumerate(self.features[1:], 1):
            x = layer(x)
            
            if isinstance(layer, nn.ReLU) and layer_idx < len(layer_names):
                if layer_names[layer_idx] in self.layers:
                    features.append(x)
                layer_idx += 1
                
                if len(features) == len(self.layers):
                    break
        
        return features


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between heatmap and coordinate predictions
    Ensures spatial consistency between different output heads
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                pred_heatmaps: torch.Tensor, 
                pred_coords: torch.Tensor,
                heatmap_size: tuple = (64, 64)) -> torch.Tensor:
        """
        Compute consistency loss between heatmaps and coordinates
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, L, H, W]
            pred_coords: Predicted coordinates [B, L, 2]
            heatmap_size: Size of heatmaps
            
        Returns:
            Consistency loss value
        """
        
        B, L, H, W = pred_heatmaps.shape
        
        # Extract coordinates from heatmaps using spatial expectation
        heatmap_coords = self._extract_coords_from_heatmaps(pred_heatmaps)
        
        # Scale coordinate predictions to heatmap space
        scaled_coords = pred_coords.clone()
        scaled_coords[:, :, 0] = scaled_coords[:, :, 0] * W / 512  # Assuming 512x512 input
        scaled_coords[:, :, 1] = scaled_coords[:, :, 1] * H / 512
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(heatmap_coords, scaled_coords)
        
        return consistency_loss
    
    def _extract_coords_from_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract coordinates from heatmaps using spatial expectation"""
        
        B, L, H, W = heatmaps.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32, device=heatmaps.device).view(1, 1, H, 1)
        x_coords = torch.arange(W, dtype=torch.float32, device=heatmaps.device).view(1, 1, 1, W)
        
        # Normalize heatmaps
        heatmaps_norm = heatmaps / (heatmaps.sum(dim=[2, 3], keepdim=True) + 1e-8)
        
        # Compute spatial expectation
        x_exp = (heatmaps_norm * x_coords).sum(dim=[2, 3])  # [B, L]
        y_exp = (heatmaps_norm * y_coords).sum(dim=[2, 3])  # [B, L]
        
        coords = torch.stack([x_exp, y_exp], dim=2)  # [B, L, 2]
        
        return coords


def create_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function based on configuration
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Configured loss function
    """
    
    loss_type = config.get('type', 'combined')
    
    if loss_type == 'combined':
        return CombinedLandmarkLoss(
            heatmap_weight=config.get('heatmap_weight', 1.0),
            coord_weight=config.get('coord_weight', 1.0),
            uncertainty_weight=config.get('uncertainty_weight', 0.1),
            adaptive_weighting=config.get('adaptive_weighting', True)
        )
    
    elif loss_type == 'heatmap':
        return HeatmapLoss(
            alpha=config.get('alpha', 2.0),
            beta=config.get('beta', 4.0)
        )
    
    elif loss_type == 'coordinate':
        return CoordinateLoss(
            reduction=config.get('reduction', 'mean')
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("🧪 Testing MAHT-Net loss functions...")
    
    # Create sample data
    batch_size, num_landmarks, heatmap_size = 4, 7, 64
    
    predictions = {
        'heatmaps': torch.rand(batch_size, num_landmarks, heatmap_size, heatmap_size),
        'coordinates': torch.rand(batch_size, num_landmarks, 2) * 512,
        'uncertainties': torch.rand(batch_size, num_landmarks) * 0.5
    }
    
    targets = {
        'heatmaps': torch.rand(batch_size, num_landmarks, heatmap_size, heatmap_size),
        'landmarks': torch.rand(batch_size, num_landmarks, 2) * 512
    }
    
    # Test combined loss
    criterion = CombinedLandmarkLoss()
    losses = criterion(predictions, targets)
    
    print("Loss computation successful:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n🎉 Loss functions ready for training!")
