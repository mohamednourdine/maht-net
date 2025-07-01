# Troubleshooting Guide: Systematic Problem-Solving for MAHT-Net

## Problem-Solving Philosophy

This troubleshooting guide provides systematic approaches to identify, diagnose, and resolve common issues in MAHT-Net development and deployment. Rather than exhaustive code examples, we focus on methodological frameworks for problem identification and resolution strategies.

## Systematic Debugging Methodology

### 1. Problem Identification Framework
- **Symptom Documentation**: Systematic recording of error messages, unexpected behaviors, and performance issues
- **Context Analysis**: Environment, data, and configuration state assessment
- **Reproducibility Testing**: Establishing consistent reproduction steps
- **Isolation Strategy**: Component-by-component problem localization

### 2. Root Cause Analysis Process
- **Error Pattern Recognition**: Classification of issues by type and origin
- **Dependency Chain Analysis**: Systematic examination of component interactions
- **Configuration Validation**: Verification of environment and parameter settings
- **Data Flow Tracing**: Following data transformation and processing paths

## Development Environment Issues

### 1. Installation and Dependency Management
**Common Issues:**
- CUDA compatibility and memory allocation problems
- Package version conflicts and dependency resolution
- Environment inconsistencies across development and deployment

**Resolution Strategy:**
- **Environment Isolation**: Use of containerization and virtual environments
- **Version Management**: Systematic pinning of critical dependencies
- **Hardware Compatibility**: Verification of CUDA versions and device capabilities
- **Progressive Installation**: Step-by-step dependency resolution with validation

**Diagnostic Commands:**
- Hardware and CUDA verification: `nvidia-smi`, `nvcc --version`
- PyTorch compatibility checks: `torch.cuda.is_available()`
- Environment validation: `pip list`, `conda list`

### 2. Memory and Resource Management
**Issue Categories:**
- GPU memory overflow during training or inference
- CPU memory exhaustion with large datasets
- Storage space limitations for data and model checkpoints

**Management Strategies:**
- **Memory Profiling**: Systematic monitoring of resource utilization
- **Batch Size Optimization**: Dynamic adjustment based on available resources
- **Gradient Accumulation**: Effective training with limited memory
- **Data Loading Optimization**: Efficient pipeline design for large datasets

## Data Processing and Loading Issues

### 1. Dataset Integrity and Validation
**Common Problems:**
- Missing or corrupted image files
- Inconsistent annotation formats and coordinate systems
- Data shape mismatches and preprocessing errors

**Validation Framework:**
- **Data Integrity Checks**: Systematic verification of file existence and format
- **Annotation Consistency**: Validation of landmark coordinates and labeling standards
- **Shape Consistency**: Verification of expected dimensions and data types
- **Statistical Validation**: Distribution analysis and outlier detection

### 2. Data Loading Pipeline Optimization
**Performance Issues:**
- Slow data loading affecting training efficiency
- Memory bottlenecks in data preprocessing
- Inconsistent batch composition and sampling

**Optimization Strategies:**
- **Pipeline Profiling**: Identification of bottlenecks in data loading workflow
- **Parallel Processing**: Multi-worker data loading configuration
- **Caching Strategies**: Intelligent data caching for frequently accessed samples
- **Preprocessing Optimization**: Efficient transformation and augmentation pipelines

## Model Architecture and Training Issues

### 1. Architecture Configuration Problems
**Design Issues:**
- Dimension mismatches in network layers
- Incompatible transformer and CNN component integration
- Attention mechanism configuration errors

**Resolution Approach:**
- **Architecture Validation**: Systematic verification of layer compatibility
- **Dimension Tracking**: Forward pass shape analysis and debugging
- **Component Testing**: Individual module validation before integration
- **Gradual Integration**: Progressive architecture building with validation

### 2. Training Convergence and Stability
**Training Problems:**
- Loss function oscillation and instability
- Gradient explosion or vanishing gradients
- Slow convergence or training plateaus

**Diagnostic Framework:**
- **Loss Analysis**: Pattern recognition in training and validation curves
- **Gradient Monitoring**: Tracking gradient norms and flow through layers
- **Learning Rate Optimization**: Systematic hyperparameter tuning
- **Regularization Assessment**: Evaluation of dropout, weight decay effectiveness

### 3. Performance Optimization
**Efficiency Issues:**
- Slow training speed and computational bottlenecks
- Memory inefficient operations and unnecessary computations
- Poor GPU utilization and resource management

**Optimization Methodology:**
- **Profiling Analysis**: Detailed computational bottleneck identification
- **Memory Optimization**: Efficient tensor operations and memory reuse
- **Batch Size Tuning**: Optimal batch size for hardware utilization
- **Mixed Precision Training**: Performance improvements through precision optimization
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        print("\\nGradient flow analysis:")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm < 1e-7:
                    print(f"    WARNING: Very small gradient in {name}")
                elif grad_norm > 100:
                    print(f"    WARNING: Very large gradient in {name}")
            else:
                print(f"  {name}: No gradient")

# Usage
debugger = ModelDebugger(model)
debugger.debug_forward_pass(torch.randn(1, 3, 512, 512))
```

**Common Fixes:**

```python
# Fix 1: Add adaptive pooling for dimension matching
class AdaptiveConnector(nn.Module):
    def __init__(self, input_dim, output_dim, target_size=None):
        super().__init__()
        self.target_size = target_size
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        if self.target_size and x.shape[2:] != self.target_size:
            x = F.adaptive_avg_pool2d(x, self.target_size)
        
        if len(x.shape) == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.view(B, C, H*W).transpose(1, 2)  # [B, H*W, C]
        
        x = self.projection(x)
        return x

# Fix 2: Dynamic shape handling
def safe_concatenate(tensor1, tensor2, dim=1):
    """Safely concatenate tensors with potential size mismatches"""
    
    if tensor1.shape[2:] != tensor2.shape[2:]:
        # Resize tensor2 to match tensor1
        target_size = tensor1.shape[2:]
        tensor2 = F.interpolate(tensor2, size=target_size, mode='bilinear', align_corners=False)
    
    return torch.cat([tensor1, tensor2], dim=dim)
```

## Training Issues

### 1. Training Instability

#### Problem: Loss Explosion or NaN Values
**Symptoms:**
- Loss becomes NaN after few iterations
- Loss increases dramatically
- Gradients become very large

**Debugging and Solutions:**

```python
# src/training/stability_checker.py
import torch
import torch.nn as nn
import numpy as np

class TrainingStabilityChecker:
    """Monitor and fix training stability issues"""
    
    def __init__(self, model, threshold_grad_norm=10.0):
        self.model = model
        self.threshold_grad_norm = threshold_grad_norm
        self.loss_history = []
        self.grad_norm_history = []
    
    def check_model_health(self, loss_value):
        """Check model health during training"""
        
        issues = []
        
        # Check for NaN loss
        if torch.isnan(loss_value):
            issues.append("NaN loss detected")
        
        # Check for infinite loss
        if torch.isinf(loss_value):
            issues.append("Infinite loss detected")
        
        # Check for exploding loss
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            if loss_value > 10 * np.mean(recent_losses):
                issues.append("Loss explosion detected")
        
        self.loss_history.append(loss_value.item())
        
        return issues
    
    def check_gradient_health(self):
        """Check gradient health"""
        
        total_norm = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for NaN gradients
                if torch.isnan(param.grad).any():
                    return [f"NaN gradient in {name}"]
                
                # Check for very large gradients
                if param_norm > 100:
                    return [f"Large gradient in {name}: {param_norm:.2f}"]
        
        total_norm = total_norm ** (1. / 2)
        self.grad_norm_history.append(total_norm)
        
        if total_norm > self.threshold_grad_norm:
            return [f"Gradient norm too large: {total_norm:.2f}"]
        
        return []
    
    def apply_emergency_fixes(self, optimizer):
        """Apply emergency fixes for training instability"""
        
        # 1. Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 2. Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            print(f"Reduced learning rate to {param_group['lr']}")
        
        # 3. Reset problematic parameters
        self._reset_problematic_parameters()
    
    def _reset_problematic_parameters(self):
        """Reset parameters that have become problematic"""
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Resetting parameter {name} due to NaN gradients")
                # Reinitialize with small random values
                with torch.no_grad():
                    param.normal_(0, 0.01)

# Integration in training loop
stability_checker = TrainingStabilityChecker(model)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... forward pass ...
        loss = criterion(predictions, targets)
        
        # Check stability before backward pass
        loss_issues = stability_checker.check_model_health(loss)
        if loss_issues:
            print(f"Loss issues detected: {loss_issues}")
            # Skip this batch or apply fixes
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient health
        grad_issues = stability_checker.check_gradient_health()
        if grad_issues:
            print(f"Gradient issues detected: {grad_issues}")
            stability_checker.apply_emergency_fixes(optimizer)
        
        optimizer.step()
```

#### Problem: Slow or No Convergence
**Symptoms:**
- Loss plateaus early
- Very slow training progress
- Validation performance doesn't improve

**Solutions:**

```python
# src/training/convergence_accelerator.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ConvergenceAccelerator:
    """Accelerate convergence with adaptive strategies"""
    
    def __init__(self, model, optimizer, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        # Adaptive strategies
        self.strategies = {
            'learning_rate_cycling': False,
            'gradient_accumulation': 1,
            'warmup_epochs': 0
        }
    
    def step(self, loss, epoch):
        """Adaptive step based on training progress"""
        
        # Update scheduler
        self.scheduler.step(loss)
        
        # Check for improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Apply adaptive strategies
        if self.patience_counter > self.patience // 2:
            self._apply_convergence_strategies(epoch)
    
    def _apply_convergence_strategies(self, epoch):
       

### 2. Memory Issues

#### Problem: CUDA Out of Memory
**Symptoms:**
- `RuntimeError: CUDA out of memory`
- Training crashes during forward or backward pass

**Memory Optimization Solutions:**

### 3. Performance Issues

#### Problem: Slow Training Speed
**Symptoms:**
- Very slow epochs
- Low GPU utilization
- Long data loading times

**Performance Optimization:**

This troubleshooting guide provides comprehensive solutions for the most common issues encountered during MAHT-Net development and deployment. Regular use of the debugging tools and health checks can prevent many issues before they become critical problems.
