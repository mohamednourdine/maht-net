"""
MAHT-Net: Multi-Stage Attention-enhanced Hybrid Transformer Network
Main model architecture implementation for cephalometric landmark detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import timm
from einops import rearrange
import numpy as np


class MultiScaleFeatureExtractor(nn.Module):
    """
    CNN-based multi-scale feature extractor using EfficientNet-B3 backbone
    Extracts hierarchical features at 5 different scales for comprehensive analysis
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # EfficientNet-B3 backbone for robust feature extraction
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4, 5)  # Multi-scale outputs
        )
        
        # Feature dimension mapping
        self.feature_dims = [24, 32, 48, 136, 384]  # EfficientNet-B3 dimensions
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from input image"""
        features = self.backbone(x)
        return features


class SpatialAttentionModule(nn.Module):
    """
    Spatial attention mechanism for enhanced feature focusing
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to input features"""
        att_weights = self.attention(x)
        return x * att_weights


class VisionTransformerBottleneck(nn.Module):
    """
    Vision Transformer bottleneck for global context modeling
    Processes the deepest CNN features with transformer attention
    """
    
    def __init__(self, 
                 feature_dim: int = 384,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 mlp_ratio: int = 4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Positional embeddings for spatial awareness
        self.pos_embed = nn.Parameter(torch.randn(1, 256, feature_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * mlp_ratio,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through Vision Transformer"""
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions and add positional embeddings
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add positional embeddings (interpolate if needed)
        if x_flat.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2).unsqueeze(0),
                size=x_flat.size(1),
                mode='linear'
            ).squeeze(0).transpose(0, 1).unsqueeze(0)
        else:
            pos_embed = self.pos_embed
            
        x_flat = x_flat + pos_embed
        
        # Apply transformer
        x_transformed = self.transformer(x_flat)
        x_transformed = self.norm(x_transformed)
        
        # Reshape back to spatial format
        x_out = rearrange(x_transformed, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x_out


class AttentionGatedDecoder(nn.Module):
    """
    Attention-gated decoder with FPN-style multi-scale feature fusion
    Combines features from different scales with attention mechanisms
    """
    
    def __init__(self, feature_dims: List[int], output_dim: int = 256):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Lateral connections for FPN
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        
        # Attention gates for each scale
        self.attention_gates = nn.ModuleList([
            SpatialAttentionModule(output_dim) for _ in feature_dims
        ])
        
        # Fusion convolutions
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(output_dim, output_dim, 3, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True)
            ) for _ in feature_dims[:-1]
        ])
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features with attention gating"""
        
        # Apply lateral connections
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway with attention
        output = laterals[-1]  # Start from deepest features
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            output = F.interpolate(
                output, 
                size=laterals[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # Apply attention to current level features
            attended_lateral = self.attention_gates[i](laterals[i])
            output = output + attended_lateral
            
            # Fusion convolution
            if i > 0:
                output = self.fusion_convs[i-1](output)
                
        return output


class LandmarkDetectionHead(nn.Module):
    """
    Dual-output detection head for heatmap regression and coordinate prediction
    Includes uncertainty quantification for clinical safety
    """
    
    def __init__(self, 
                 input_dim: int = 256, 
                 num_landmarks: int = 7,
                 heatmap_size: int = 64):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Heatmap regression branch
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, 1),
            nn.Sigmoid()  # Normalized heatmaps
        )
        
        # Coordinate regression branch
        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_landmarks * 2)  # x, y coordinates
        )
        
        # Uncertainty estimation branch
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_landmarks),  # Uncertainty per landmark
            nn.Softplus()  # Positive uncertainty values
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions with uncertainty quantification"""
        
        # Generate heatmaps
        heatmaps = self.heatmap_head(x)
        
        # Resize heatmaps to target size
        if heatmaps.size(-1) != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps, 
                size=(self.heatmap_size, self.heatmap_size),
                mode='bilinear', 
                align_corners=False
            )
        
        # Direct coordinate regression
        coordinates = self.coord_head(x)
        coordinates = coordinates.view(-1, self.num_landmarks, 2)
        
        # Uncertainty estimation
        uncertainties = self.uncertainty_head(x)
        
        return {
            'heatmaps': heatmaps,
            'coordinates': coordinates,
            'uncertainties': uncertainties
        }


class MAHTNet(nn.Module):
    """
    MAHT-Net: Multi-Stage Attention-enhanced Hybrid Transformer Network
    
    Main architecture combining CNN feature extraction, Transformer bottleneck,
    and attention-gated decoding for precise cephalometric landmark detection.
    """
    
    def __init__(self, 
                 num_landmarks: int = 7,
                 pretrained: bool = True,
                 heatmap_size: int = 64):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Multi-scale CNN feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(pretrained=pretrained)
        
        # Vision Transformer bottleneck for global context
        self.transformer_bottleneck = VisionTransformerBottleneck(
            feature_dim=384,  # EfficientNet-B3 final feature dimension
            num_heads=8,
            num_layers=4
        )
        
        # Attention-gated decoder
        self.decoder = AttentionGatedDecoder(
            feature_dims=self.feature_extractor.feature_dims,
            output_dim=256
        )
        
        # Landmark detection head
        self.detection_head = LandmarkDetectionHead(
            input_dim=256,
            num_landmarks=num_landmarks,
            heatmap_size=heatmap_size
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MAHT-Net
        
        Args:
            x: Input cephalometric X-ray images [B, C, H, W]
            
        Returns:
            Dictionary containing:
            - heatmaps: Probability heatmaps for each landmark [B, L, H, W]
            - coordinates: Direct coordinate predictions [B, L, 2]
            - uncertainties: Uncertainty estimates per landmark [B, L]
        """
        
        # Extract multi-scale CNN features
        cnn_features = self.feature_extractor(x)
        
        # Apply transformer to deepest features for global context
        enhanced_features = list(cnn_features)
        enhanced_features[-1] = self.transformer_bottleneck(enhanced_features[-1])
        
        # Decode with attention-gated fusion
        decoded_features = self.decoder(enhanced_features)
        
        # Generate final predictions
        predictions = self.detection_head(decoded_features)
        
        return predictions
    
    def extract_coordinates_from_heatmaps(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract coordinate predictions from heatmaps using spatial expectation
        
        Args:
            heatmaps: Predicted heatmaps [B, L, H, W]
            
        Returns:
            coordinates: Extracted coordinates [B, L, 2]
        """
        B, L, H, W = heatmaps.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, dtype=torch.float32, device=heatmaps.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=heatmaps.device)
        
        # Compute spatial expectation
        coordinates = []
        for b in range(B):
            batch_coords = []
            for l in range(L):
                hm = heatmaps[b, l]
                
                # Normalize heatmap
                hm_norm = hm / (hm.sum() + 1e-8)
                
                # Compute expected coordinates
                y_exp = (hm_norm.sum(dim=1) * y_coords).sum()
                x_exp = (hm_norm.sum(dim=0) * x_coords).sum()
                
                batch_coords.append([x_exp, y_exp])
            
            coordinates.append(batch_coords)
        
        return torch.tensor(coordinates, device=heatmaps.device)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get comprehensive model information for monitoring and debugging"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MAHT-Net',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_landmarks': self.num_landmarks,
            'backbone': 'EfficientNet-B3',
            'transformer_layers': 4,
            'transformer_heads': 8,
            'feature_dims': self.feature_extractor.feature_dims
        }


def create_maht_net(num_landmarks: int = 7, 
                   pretrained: bool = True,
                   heatmap_size: int = 64) -> MAHTNet:
    """
    Factory function to create MAHT-Net model
    
    Args:
        num_landmarks: Number of landmarks to detect (default: 7 for cephalometric)
        pretrained: Use pretrained CNN backbone (recommended)
        heatmap_size: Output heatmap resolution
        
    Returns:
        Initialized MAHT-Net model
    """
    
    model = MAHTNet(
        num_landmarks=num_landmarks,
        pretrained=pretrained,
        heatmap_size=heatmap_size
    )
    
    return model


if __name__ == "__main__":
    # Test model instantiation
    model = create_maht_net(num_landmarks=7)
    print("✅ MAHT-Net model created successfully")
    print(f"📊 Model info: {model.get_model_info()}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)  # Batch of 2 images
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"🔍 Output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
