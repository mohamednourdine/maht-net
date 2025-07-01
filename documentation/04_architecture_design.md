# Architecture Design: MAHT-Net Implementation Details

## Overall Architecture Overview

MAHT-Net integrates four key components in a hybrid encoder-decoder architecture:

1. **Pretrained CNN Encoder** (EfficientNet-B3 or ResNet-34)
2. **Vision Transformer Bottleneck** (ViT or Swin Transformer)
3. **Attention-Gated Decoder** with multi-scale feature fusion
4. **Heatmap Regression Head** for landmark prediction

## Component 1: Encoder Architecture

### EfficientNet-B3 Encoder Implementation

```python
# src/models/encoders.py
import torch
import torch.nn as nn
import timm
from typing import List, Dict

class EfficientNetEncoder(nn.Module):
    """EfficientNet-B3 encoder with feature extraction at multiple scales"""
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        
        # Load pretrained EfficientNet-B3
        self.backbone = timm.create_model(
            'efficientnet_b3', 
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4, 5]  # Extract features at 5 levels
        )
        
        # Feature dimensions for EfficientNet-B3
        self.feature_dims = [24, 32, 48, 136, 384]  # Channels at each level
        self.feature_strides = [2, 4, 8, 16, 32]    # Downsampling factors
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors at different scales
        """
        features = self.backbone(x)
        return features

class ResNetEncoder(nn.Module):
    """ResNet-34 encoder alternative"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-34
        resnet = timm.create_model('resnet34', pretrained=pretrained)
        
        # Extract layers for feature extraction
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels, stride 4
        self.layer2 = resnet.layer2  # 128 channels, stride 8
        self.layer3 = resnet.layer3  # 256 channels, stride 16
        self.layer4 = resnet.layer4  # 512 channels, stride 32
        
        self.feature_dims = [64, 64, 128, 256, 512]
        self.feature_strides = [2, 4, 8, 16, 32]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass with multi-scale feature extraction"""
        features = []
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 1/2 resolution
        
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution
        
        x = self.layer2(x)
        features.append(x)  # 1/8 resolution
        
        x = self.layer3(x)
        features.append(x)  # 1/16 resolution
        
        x = self.layer4(x)
        features.append(x)  # 1/32 resolution
        
        return features
```

## Component 2: Vision Transformer Bottleneck

### Transformer Bottleneck Implementation

```python
# src/models/transformer_bottleneck.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import math

class TransformerBottleneck(nn.Module):
    """Vision Transformer bottleneck for global context modeling"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 6,
                 patch_size: int = 8,
                 transformer_type: str = 'vit'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.transformer_type = transformer_type
        
        # Input projection to match transformer dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        if transformer_type == 'vit':
            self.transformer = self._create_vit_model(num_heads, num_layers)
        elif transformer_type == 'swin':
            self.transformer = self._create_swin_model()
        else:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
        
        # Output projection back to original dimensions
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(hidden_dim)
    
    def _create_vit_model(self, num_heads: int, num_layers: int):
        """Create Vision Transformer model"""
        config = ViTConfig(
            hidden_size=self.hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=self.hidden_dim * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            patch_size=self.patch_size,
            num_channels=self.input_dim,
            image_size=224  # Will be adjusted dynamically
        )
        
        return ViTModel(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer bottleneck
        
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Transformed features [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions and transpose for transformer
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        
        # Project to transformer dimensions
        x_proj = self.input_projection(x_flat)  # [B, H*W, hidden_dim]
        
        # Add positional encoding
        x_pos = self.pos_encoding(x_proj, H, W)
        
        # Pass through transformer
        if self.transformer_type == 'vit':
            # Prepare for ViT (requires specific format)
            transformer_output = self.transformer(
                pixel_values=x_pos.unsqueeze(-1).unsqueeze(-1)
            )
            x_transformed = transformer_output.last_hidden_state
        else:
            x_transformed = self.transformer(x_pos)
        
        # Project back to original dimensions
        x_out = self.output_projection(x_transformed)  # [B, H*W, C]
        
        # Reshape back to spatial format
        x_out = x_out.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return x + x_out

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial features"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Add 2D positional encoding"""
        B, N, D = x.shape
        
        # Create position indices
        pos_h = torch.arange(H, dtype=torch.float32, device=x.device)
        pos_w = torch.arange(W, dtype=torch.float32, device=x.device)
        
        # Create encoding dimensions
        div_term = torch.exp(torch.arange(0, D//2, 2, dtype=torch.float32, device=x.device) * 
                           -(math.log(10000.0) / (D//2)))
        
        # Height encoding
        pos_h = pos_h.unsqueeze(1) * div_term
        pos_h_sin = torch.sin(pos_h)
        pos_h_cos = torch.cos(pos_h)
        
        # Width encoding
        pos_w = pos_w.unsqueeze(1) * div_term
        pos_w_sin = torch.sin(pos_w)
        pos_w_cos = torch.cos(pos_w)
        
        # Combine encodings
        pos_encoding = torch.zeros(H, W, D, device=x.device)
        pos_encoding[:, :, 0::4] = pos_h_sin.unsqueeze(1).repeat(1, W, 1)
        pos_encoding[:, :, 1::4] = pos_h_cos.unsqueeze(1).repeat(1, W, 1)
        pos_encoding[:, :, 2::4] = pos_w_sin.unsqueeze(0).repeat(H, 1, 1)
        pos_encoding[:, :, 3::4] = pos_w_cos.unsqueeze(0).repeat(H, 1, 1)
        
        # Flatten and expand for batch
        pos_encoding = pos_encoding.view(H*W, D).unsqueeze(0).repeat(B, 1, 1)
        
        return x + pos_encoding
```

## Component 3: Attention-Gated Decoder

### Attention Gate Implementation

```python
# src/models/attention_gates.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention gate for filtering relevant features"""
    
    def __init__(self, 
                 gate_channels: int,
                 feature_channels: int, 
                 intermediate_channels: int = None):
        super().__init__()
        
        if intermediate_channels is None:
            intermediate_channels = feature_channels // 2
        
        # Gate signal processing
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Feature signal processing
        self.feature_conv = nn.Sequential(
            nn.Conv2d(feature_channels, intermediate_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Attention computation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, 1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-gated features
        
        Args:
            gate: Gate signal from deeper layer [B, Cg, Hg, Wg]
            features: Features to be gated [B, Cf, Hf, Wf]
            
        Returns:
            Attention-weighted features [B, Cf, Hf, Wf]
        """
        # Resize gate to match feature dimensions if needed
        if gate.shape[2:] != features.shape[2:]:
            gate = F.interpolate(gate, size=features.shape[2:], 
                               mode='bilinear', align_corners=False)
        
        # Process gate and feature signals
        gate_processed = self.gate_conv(gate)
        feature_processed = self.feature_conv(features)
        
        # Combine and compute attention
        combined = self.relu(gate_processed + feature_processed)
        attention_weights = self.attention_conv(combined)
        
        # Apply attention to features
        gated_features = features * attention_weights
        
        return gated_features

class MultiScaleFeatureFusion(nn.Module):
    """Feature Pyramid Network-style multi-scale fusion"""
    
    def __init__(self, feature_dims: List[int], output_dim: int = 256):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Lateral connections (1x1 convolutions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(output_dim, output_dim, 3, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True)
            ) for _ in feature_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Multi-scale feature fusion
        
        Args:
            features: List of feature tensors from encoder
            
        Returns:
            List of fused feature tensors
        """
        # Apply lateral connections
        lateral_features = [
            lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway
        fused_features = [lateral_features[-1]]  # Start with deepest features
        
        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsample deeper features
            upsampled = F.interpolate(
                fused_features[0], 
                size=lateral_features[i].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # Add lateral connection
            fused = lateral_features[i] + upsampled
            fused_features.insert(0, fused)
        
        # Apply output convolutions
        output_features = [
            output_conv(feat) for output_conv, feat in zip(self.output_convs, fused_features)
        ]
        
        return output_features
```

### Decoder Implementation

```python
# src/models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGatedDecoder(nn.Module):
    """Attention-gated decoder with multi-scale feature fusion"""
    
    def __init__(self, 
                 encoder_dims: List[int],
                 decoder_dims: List[int] = [256, 128, 64, 32, 16],
                 num_landmarks: int = 7):
        super().__init__()
        
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.num_landmarks = num_landmarks
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(encoder_dims, decoder_dims[0])
        
        # Attention gates for each decoder level
        self.attention_gates = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            gate_channels = decoder_dims[i]
            feature_channels = decoder_dims[0]  # Fused features have same channels
            self.attention_gates.append(
                AttentionGate(gate_channels, feature_channels)
            )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            in_channels = decoder_dims[i] * 2  # Skip connection doubles channels
            out_channels = decoder_dims[i + 1]
            
            self.decoder_blocks.append(
                DecoderBlock(in_channels, out_channels)
            )
        
        # Final heatmap generation
        self.heatmap_head = HeatmapHead(decoder_dims[-1], num_landmarks)
    
    def forward(self, 
                encoder_features: List[torch.Tensor],
                transformer_output: torch.Tensor) -> torch.Tensor:
        """
        Decoder forward pass
        
        Args:
            encoder_features: Multi-scale features from encoder
            transformer_output: Features from transformer bottleneck
            
        Returns:
            Heatmaps for each landmark [B, num_landmarks, H, W]
        """
        # Multi-scale feature fusion
        fused_features = self.feature_fusion(encoder_features)
        
        # Start decoding from transformer output
        x = transformer_output
        
        # Decoder pathway with attention gates
        for i, (decoder_block, attention_gate) in enumerate(
            zip(self.decoder_blocks, self.attention_gates)
        ):
            # Skip connection with attention gating
            skip_features = fused_features[-(i+2)]  # Corresponding encoder features
            gated_skip = attention_gate(x, skip_features)
            
            # Upsample current features
            x = F.interpolate(x, size=gated_skip.shape[2:], 
                            mode='bilinear', align_corners=False)
            
            # Concatenate with gated skip connection
            x = torch.cat([x, gated_skip], dim=1)
            
            # Apply decoder block
            x = decoder_block(x)
        
        # Generate final heatmaps
        heatmaps = self.heatmap_head(x)
        
        return heatmaps

class DecoderBlock(nn.Module):
    """Individual decoder block with residual connections"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Add residual connection if dimensions match
        if out.shape == residual.shape:
            out = out + residual
        
        return out
```

## Component 4: Heatmap Regression Head

```python
# src/models/heatmap_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapHead(nn.Module):
    """Heatmap regression head for landmark prediction"""
    
    def __init__(self, 
                 in_channels: int,
                 num_landmarks: int,
                 heatmap_size: tuple = (128, 128)):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Individual heatmap prediction for each landmark
        self.heatmap_convs = nn.ModuleList([
            nn.Conv2d(32, 1, 1) for _ in range(num_landmarks)
        ])
        
        # Coordinate regression (auxiliary task)
        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_landmarks * 2)  # x, y for each landmark
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Generate heatmaps and coordinate predictions
        
        Args:
            x: Feature tensor [B, C, H, W]
            
        Returns:
            Dictionary containing heatmaps and coordinates
        """
        # Refine features
        features = self.feature_refine(x)
        
        # Generate heatmaps for each landmark
        heatmaps = []
        for heatmap_conv in self.heatmap_convs:
            heatmap = heatmap_conv(features)
            heatmap = torch.sigmoid(heatmap)  # Ensure positive values
            heatmaps.append(heatmap)
        
        heatmaps = torch.cat(heatmaps, dim=1)  # [B, num_landmarks, H, W]
        
        # Resize to target heatmap size if needed
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps, 
                size=self.heatmap_size,
                mode='bilinear', 
                align_corners=False
            )
        
        # Auxiliary coordinate prediction
        coordinates = self.coord_head(features)
        coordinates = coordinates.view(-1, self.num_landmarks, 2)
        
        return {
            'heatmaps': heatmaps,
            'coordinates': coordinates
        }

class SoftArgmax2D(nn.Module):
    """Soft argmax for differentiable coordinate extraction from heatmaps"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract coordinates using soft argmax
        
        Args:
            heatmaps: [B, N, H, W] heatmaps
            
        Returns:
            coordinates: [B, N, 2] (x, y) coordinates
        """
        B, N, H, W = heatmaps.shape
        
        # Apply temperature
        heatmaps = heatmaps / self.temperature
        
        # Softmax to ensure sum to 1
        heatmaps_flat = heatmaps.view(B, N, H * W)
        heatmaps_soft = F.softmax(heatmaps_flat, dim=2)
        heatmaps_soft = heatmaps_soft.view(B, N, H, W)
        
        # Create coordinate grids
        x_coords = torch.linspace(0, W-1, W, device=heatmaps.device)
        y_coords = torch.linspace(0, H-1, H, device=heatmaps.device)
        
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        
        # Compute expected coordinates
        x_pred = (heatmaps_soft * x_grid).sum(dim=[2, 3])
        y_pred = (heatmaps_soft * y_grid).sum(dim=[2, 3])
        
        coordinates = torch.stack([x_pred, y_pred], dim=2)
        
        return coordinates
```

## Complete MAHT-Net Architecture

```python
# src/models/maht_net.py
import torch
import torch.nn as nn
from typing import Dict, List

class MAHTNet(nn.Module):
    """Complete MAHT-Net architecture"""
    
    def __init__(self,
                 encoder_type: str = 'efficientnet',
                 transformer_type: str = 'vit',
                 num_landmarks: int = 7,
                 pretrained: bool = True):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Encoder
        if encoder_type == 'efficientnet':
            self.encoder = EfficientNetEncoder(pretrained=pretrained)
        elif encoder_type == 'resnet':
            self.encoder = ResNetEncoder(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported encoder: {encoder_type}")
        
        # Transformer bottleneck
        bottleneck_dim = self.encoder.feature_dims[-1]
        self.transformer = TransformerBottleneck(
            input_dim=bottleneck_dim,
            transformer_type=transformer_type
        )
        
        # Decoder
        self.decoder = AttentionGatedDecoder(
            encoder_dims=self.encoder.feature_dims,
            num_landmarks=num_landmarks
        )
        
        # Coordinate extraction
        self.soft_argmax = SoftArgmax2D()
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MAHT-Net
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with heatmaps and coordinates
        """
        # Encoder
        encoder_features = self.encoder(x)
        
        # Transformer bottleneck on deepest features
        transformer_output = self.transformer(encoder_features[-1])
        
        # Decoder
        decoder_output = self.decoder(encoder_features, transformer_output)
        
        # Extract coordinates from heatmaps
        coordinates_soft = self.soft_argmax(decoder_output['heatmaps'])
        
        return {
            'heatmaps': decoder_output['heatmaps'],
            'coordinates': coordinates_soft,
            'coordinates_aux': decoder_output.get('coordinates', None)
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for each component"""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'encoder': encoder_params,
            'transformer': transformer_params,
            'decoder': decoder_params,
            'total': encoder_params + transformer_params + decoder_params
        }
```

## Architecture Validation

```python
# src/models/model_validation.py
def validate_maht_net():
    """Validate MAHT-Net architecture"""
    
    # Create model
    model = MAHTNet(
        encoder_type='efficientnet',
        transformer_type='vit',
        num_landmarks=7
    )
    
    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Validate outputs
    print("MAHT-Net Architecture Validation:")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Heatmap shape: {output['heatmaps'].shape}")
    print(f"Coordinates shape: {output['coordinates'].shape}")
    
    # Parameter count
    param_count = model.get_parameter_count()
    print(f"\nParameter Count:")
    for component, count in param_count.items():
        print(f"  {component}: {count:,}")
    
    # Memory usage estimation
    total_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024**2  # MB
    print(f"\nEstimated Memory Usage: {total_memory:.1f} MB")
    
    return model

if __name__ == "__main__":
    model = validate_maht_net()
```

## Design Considerations and Optimizations

### Memory Optimization Strategies

1. **Gradient Checkpointing**: Trade computation for memory
2. **Mixed Precision Training**: Use FP16 where possible
3. **Progressive Resizing**: Start with smaller images, increase gradually
4. **Efficient Attention**: Use memory-efficient attention implementations

### Model Variants

1. **MAHT-Net-Small**: Reduced transformer layers and hidden dimensions
2. **MAHT-Net-Large**: Increased capacity for better performance
3. **MAHT-Net-Efficient**: Optimized for inference speed

### Integration Points

- **Loss Function Integration**: Multi-task learning with weighted losses
- **Training Strategy**: Progressive training and component-wise fine-tuning
- **Inference Optimization**: TensorRT/ONNX conversion for deployment

## Next Steps

1. **Implementation Validation**: Test each component individually
2. **Integration Testing**: Verify end-to-end functionality
3. **Training Pipeline**: Proceed to [Training Strategy](06_training_strategy.md)
4. **Performance Optimization**: Memory and speed optimization
