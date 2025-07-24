"""
Enhanced Visualization Utilities for MAHT-Net
Provides comprehensive plotting and visualization capabilities for cephalometric data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
import cv2
import torch
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CephalometricVisualizer:
    """
    Comprehensive visualization utility for cephalometric landmark detection
    """
    
    def __init__(self, 
                 landmark_names: Optional[List[str]] = None,
                 colors: Optional[List[str]] = None,
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer
        
        Args:
            landmark_names: Names of the landmarks
            colors: Colors for each landmark
            figsize: Default figure size
        """
        
        # Default ISBI 2015 19 landmarks
        self.landmark_names = landmark_names or [
            'S', 'N', 'ANS', 'PNS', 'A', 'SPPN', 'SPIN', 'Me', 'Gn', 'B',
            'Pog', 'Id', 'Is', 'U1', 'L1', 'Ar', 'Go', 'Gon', 'Or'
        ]
        
        # Generate distinct colors for landmarks
        if colors is None:
            self.colors = plt.cm.tab20(np.linspace(0, 1, len(self.landmark_names)))
        else:
            self.colors = colors
            
        self.figsize = figsize
        
        # Landmark groups for medical interpretation
        self.landmark_groups = {
            'cranial_base': ['S', 'N'],
            'maxilla': ['ANS', 'PNS', 'A', 'SPPN', 'SPIN'],
            'mandible': ['Me', 'Gn', 'B', 'Pog', 'Go', 'Gon', 'Ar'],
            'dental': ['Id', 'Is', 'U1', 'L1'],
            'soft_tissue': ['Or']
        }
    
    def plot_image_with_landmarks(self,
                                 image: Union[np.ndarray, torch.Tensor],
                                 landmarks: np.ndarray,
                                 title: str = "Cephalometric X-ray with Landmarks",
                                 save_path: Optional[str] = None,
                                 show_labels: bool = True,
                                 show_connections: bool = False,
                                 original_size: Optional[Tuple[int, int]] = None,
                                 target_size: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot image with landmark annotations
        
        Args:
            image: Input image (H, W) or (H, W, C)
            landmarks: Landmark coordinates (N, 2)
            title: Plot title
            save_path: Path to save the plot
            show_labels: Whether to show landmark labels
            show_connections: Whether to show anatomical connections
            original_size: Original image size for scaling display
            target_size: Target size information
            
        Returns:
            matplotlib Figure
        """
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            if image.dim() == 3 and image.shape[0] <= 3:  # CHW format
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # Ensure image is 2D for grayscale
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image.squeeze(-1)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Display image
        ax.imshow(image, cmap='gray', aspect='equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add size information to title if provided
        if original_size and target_size:
            size_info = f" (Original: {original_size[1]}×{original_size[0]} → Target: {target_size[1]}×{target_size[0]})"
            ax.set_title(title + size_info, fontsize=12)
        
        # Plot landmarks
        for i, (x, y) in enumerate(landmarks):
            color = self.colors[i % len(self.colors)]
            
            # Plot landmark point
            ax.scatter(x, y, c=[color], s=100, marker='o', 
                      edgecolors='white', linewidth=2, alpha=0.8, zorder=5)
            
            # Add label if requested
            if show_labels and i < len(self.landmark_names):
                self._add_text_with_outline(ax, (x, y), self.landmark_names[i], 
                                          fontsize=10, color='white')
        
        # Add anatomical connections if requested
        if show_connections:
            self._draw_anatomical_connections(ax, landmarks)
        
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend for landmark groups
        if show_labels:
            self._add_landmark_legend(ax)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_size_comparison(self,
                           original_image: np.ndarray,
                           resized_image: np.ndarray,
                           original_landmarks: np.ndarray,
                           resized_landmarks: np.ndarray,
                           title: str = "Size Comparison",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot original and resized images side by side with landmarks
        
        Args:
            original_image: Original image
            resized_image: Resized image
            original_landmarks: Original landmark coordinates
            resized_landmarks: Resized landmark coordinates
            title: Main title
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title(f'Original Size: {original_image.shape[1]}×{original_image.shape[0]}', 
                     fontsize=12, fontweight='bold')
        
        # Plot original landmarks
        for i, (x, y) in enumerate(original_landmarks):
            color = self.colors[i % len(self.colors)]
            ax1.scatter(x, y, c=[color], s=60, marker='o', 
                       edgecolors='white', linewidth=1, alpha=0.8)
            if i < len(self.landmark_names):
                self._add_text_with_outline(ax1, (x, y), self.landmark_names[i], 
                                          fontsize=8, color='white')
        
        # Resized image
        ax2.imshow(resized_image, cmap='gray')
        ax2.set_title(f'Resized: {resized_image.shape[1]}×{resized_image.shape[0]}', 
                     fontsize=12, fontweight='bold')
        
        # Plot resized landmarks
        for i, (x, y) in enumerate(resized_landmarks):
            color = self.colors[i % len(self.colors)]
            ax2.scatter(x, y, c=[color], s=60, marker='o', 
                       edgecolors='white', linewidth=1, alpha=0.8)
            if i < len(self.landmark_names):
                self._add_text_with_outline(ax2, (x, y), self.landmark_names[i], 
                                          fontsize=8, color='white')
        
        # Remove ticks
        for ax in [ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmaps_with_image(self,
                                image: np.ndarray,
                                heatmaps: np.ndarray,
                                landmarks: np.ndarray,
                                title: str = "Generated Heatmaps",
                                save_path: Optional[str] = None,
                                max_heatmaps: int = 12) -> plt.Figure:
        """
        Plot image with generated heatmaps
        
        Args:
            image: Input image
            heatmaps: Generated heatmaps (N, H, W)
            landmarks: Original landmark coordinates
            title: Plot title
            save_path: Path to save the plot
            max_heatmaps: Maximum number of heatmaps to display
            
        Returns:
            matplotlib Figure
        """
        
        n_landmarks = min(len(heatmaps), max_heatmaps)
        
        # Calculate grid size
        cols = 4
        rows = (n_landmarks + cols - 1) // cols + 1  # +1 for original image
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()
        
        # Plot original image with landmarks
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image with Landmarks', fontweight='bold')
        
        for i, (x, y) in enumerate(landmarks[:n_landmarks]):
            color = self.colors[i % len(self.colors)]
            axes[0].scatter(x, y, c=[color], s=60, marker='o', 
                          edgecolors='white', linewidth=1, alpha=0.8)
            if i < len(self.landmark_names):
                self._add_text_with_outline(axes[0], (x, y), self.landmark_names[i], 
                                          fontsize=8, color='white')
        
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot heatmaps
        for i in range(n_landmarks):
            ax_idx = i + 1
            axes[ax_idx].imshow(heatmaps[i], cmap='jet', alpha=0.8)
            
            landmark_name = self.landmark_names[i] if i < len(self.landmark_names) else f"L{i+1}"
            axes[ax_idx].set_title(f'{landmark_name} Heatmap', fontweight='bold')
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])
        
        # Hide unused subplots
        for i in range(n_landmarks + 1, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_data_pipeline_overview(self,
                                   samples: List[Dict],
                                   max_samples: int = 6,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot overview of the complete data pipeline
        
        Args:
            samples: List of processed samples
            max_samples: Maximum number of samples to show
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure
        """
        
        n_samples = min(len(samples), max_samples)
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples[:n_samples]):
            # Original image with landmarks
            original_img = sample['original_image']
            original_landmarks = sample['original_landmarks']
            
            axes[i, 0].imshow(original_img, cmap='gray')
            axes[i, 0].set_title(f'Sample {i+1}: Original\n{original_img.shape[1]}×{original_img.shape[0]}')
            
            for j, (x, y) in enumerate(original_landmarks):
                color = self.colors[j % len(self.colors)]
                axes[i, 0].scatter(x, y, c=[color], s=40, marker='o', 
                                 edgecolors='white', linewidth=1, alpha=0.8)
            
            # Resized image with landmarks
            resized_img = sample['resized_image']
            resized_landmarks = sample['resized_landmarks']
            
            axes[i, 1].imshow(resized_img, cmap='gray')
            axes[i, 1].set_title(f'Resized\n{resized_img.shape[1]}×{resized_img.shape[0]}')
            
            for j, (x, y) in enumerate(resized_landmarks):
                color = self.colors[j % len(self.colors)]
                axes[i, 1].scatter(x, y, c=[color], s=40, marker='o', 
                                 edgecolors='white', linewidth=1, alpha=0.8)
            
            # Heatmap overlay
            if 'heatmaps' in sample:
                heatmaps = sample['heatmaps']
                combined_heatmap = np.max(heatmaps, axis=0)
                
                axes[i, 2].imshow(resized_img, cmap='gray', alpha=0.7)
                axes[i, 2].imshow(combined_heatmap, cmap='jet', alpha=0.5)
                axes[i, 2].set_title('Combined Heatmaps')
            
            # Remove ticks
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        
        fig.suptitle('Data Pipeline Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_text_with_outline(self, ax, pos, text, fontsize=10, color='white', outline_width=2):
        """Add text with black outline for better visibility"""
        text_obj = ax.text(pos[0], pos[1] - 10, text, 
                          fontsize=fontsize, color=color, 
                          ha='center', va='bottom', fontweight='bold')
        text_obj.set_path_effects([
            patheffects.Stroke(linewidth=outline_width, foreground='black'),
            patheffects.Normal()
        ])
    
    def _draw_anatomical_connections(self, ax, landmarks):
        """Draw anatomical connections between related landmarks"""
        # Define connections based on anatomical relationships
        connections = [
            (0, 1),   # S-N (cranial base)
            (2, 3),   # ANS-PNS (maxilla)
            (7, 8),   # Me-Gn (mandible)
            (10, 11), # Pog-Id (anterior mandible)
            (12, 13), # Is-U1 (upper teeth)
            (14, 15), # L1-Ar (dental-TMJ)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                ax.plot([start_point[0], end_point[0]], 
                       [start_point[1], end_point[1]], 
                       'b-', alpha=0.3, linewidth=1)
    
    def _add_landmark_legend(self, ax):
        """Add legend for landmark groups"""
        legend_elements = []
        for group_name, group_landmarks in self.landmark_groups.items():
            # Use first landmark color for group
            if group_landmarks and group_landmarks[0] in self.landmark_names:
                idx = self.landmark_names.index(group_landmarks[0])
                color = self.colors[idx % len(self.colors)]
                legend_elements.append(patches.Patch(color=color, label=group_name.replace('_', ' ').title()))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                     bbox_to_anchor=(1, 1), fontsize=8)


def plot_dataset_statistics(stats: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comprehensive dataset statistics
    
    Args:
        stats: Dataset statistics dictionary
        save_path: Path to save the plot
        
    Returns:
        matplotlib Figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Image size distribution
    if 'image_sizes' in stats:
        widths, heights = zip(*stats['image_sizes'])
        axes[0, 0].scatter(widths, heights, alpha=0.6)
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        axes[0, 0].set_title('Image Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Landmark distribution (first landmark as example)
    if 'landmark_coords' in stats and len(stats['landmark_coords']) > 0:
        first_landmark_coords = np.array([coords[0] for coords in stats['landmark_coords']])
        axes[0, 1].scatter(first_landmark_coords[:, 0], first_landmark_coords[:, 1], alpha=0.6)
        axes[0, 1].set_xlabel('X coordinate')
        axes[0, 1].set_ylabel('Y coordinate')
        axes[0, 1].set_title('First Landmark Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Sample count by split
    if 'split_counts' in stats:
        splits = list(stats['split_counts'].keys())
        counts = list(stats['split_counts'].values())
        axes[1, 0].bar(splits, counts)
        axes[1, 0].set_xlabel('Dataset Split')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Samples per Split')
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            axes[1, 0].text(i, count + max(counts) * 0.01, str(count), 
                           ha='center', va='bottom')
    
    # Processing quality metrics
    if 'quality_metrics' in stats:
        metrics = stats['quality_metrics']
        
        # Filter metrics to only include numeric values
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                numeric_metrics[key] = value
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                # For tuples like (width, height), take the first value or average
                if isinstance(value[0], (int, float)):
                    numeric_metrics[f"{key}_avg"] = sum(value) / len(value)
        
        if numeric_metrics:
            metric_names = list(numeric_metrics.keys())
            metric_values = list(numeric_metrics.values())
            
            axes[1, 1].barh(metric_names, metric_values)
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_title('Data Quality Metrics')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        else:
            # If no numeric metrics, show a text message
            axes[1, 1].text(0.5, 0.5, 'No numeric metrics available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Data Quality Metrics')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
    
    fig.suptitle('Dataset Statistics Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
