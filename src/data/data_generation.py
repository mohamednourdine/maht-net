"""
Enhanced Data Generation Pipeline for MAHT-Net
Handles complete data preprocessing, resizing, augmentation, and heatmap generation
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ..utils.visualization import CephalometricVisualizer

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for data generation pipeline"""
    # Image processing
    original_size: Optional[Tuple[int, int]] = None  # Auto-detected from data
    target_size: Tuple[int, int] = (512, 512)
    heatmap_size: Tuple[int, int] = (128, 128)
    
    # Heatmap generation
    gaussian_sigma: float = 2.0
    heatmap_amplitude: float = 1.0
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.7
    
    # Quality control
    min_image_size: Tuple[int, int] = (256, 256)
    max_landmark_displacement: float = 0.1  # As fraction of image size
    
    # Output
    save_intermediate_results: bool = True
    visualization_samples: int = 5
    

class EnhancedGaussianHeatmapGenerator:
    """
    Enhanced Gaussian heatmap generator with multiple modes and quality control
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float = 2.0,
                 amplitude: float = 1.0,
                 normalize: bool = True):
        """
        Initialize the heatmap generator
        
        Args:
            target_size: Size of input images (H, W)
            heatmap_size: Size of output heatmaps (H, W)
            sigma: Gaussian standard deviation
            amplitude: Peak amplitude of Gaussian
            normalize: Whether to normalize heatmaps to [0, 1]
        """
        
        self.target_size = target_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.amplitude = amplitude
        self.normalize = normalize
        
        # Calculate scaling factors
        self.scale_x = heatmap_size[1] / target_size[1]
        self.scale_y = heatmap_size[0] / target_size[0]
        
        logger.info(f"Heatmap generator initialized: {target_size} -> {heatmap_size}")
        logger.info(f"Scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}")
    
    def generate_heatmap(self, 
                        landmarks: np.ndarray,
                        return_individual: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate Gaussian heatmaps for landmarks
        
        Args:
            landmarks: Landmark coordinates (N, 2) in target image coordinates
            return_individual: Whether to return individual heatmaps
            
        Returns:
            Combined heatmap or tuple of (combined, individual) heatmaps
        """
        
        num_landmarks = len(landmarks)
        individual_heatmaps = np.zeros((num_landmarks, *self.heatmap_size), dtype=np.float32)
        
        for i, (x, y) in enumerate(landmarks):
            # Scale coordinates to heatmap size
            hm_x = x * self.scale_x
            hm_y = y * self.scale_y
            
            # Generate individual heatmap
            heatmap = self._create_gaussian(self.heatmap_size, (hm_x, hm_y), self.sigma)
            individual_heatmaps[i] = heatmap
        
        # Create combined heatmap (max operation)
        combined_heatmap = np.max(individual_heatmaps, axis=0)
        
        if return_individual:
            return combined_heatmap, individual_heatmaps
        else:
            return individual_heatmaps
    
    def _create_gaussian(self, size: Tuple[int, int], center: Tuple[float, float], sigma: float) -> np.ndarray:
        """Create a 2D Gaussian heatmap"""
        
        h, w = size
        cx, cy = center
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[0:h, 0:w]
        
        # Calculate distances
        distance_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        
        # Generate Gaussian
        gaussian = self.amplitude * np.exp(-distance_sq / (2 * sigma ** 2))
        
        # Normalize if requested
        if self.normalize:
            gaussian = gaussian / (self.amplitude if self.amplitude > 0 else 1.0)
        
        return gaussian.astype(np.float32)
    
    def visualize_heatmap_generation(self, 
                                   image: np.ndarray,
                                   landmarks: np.ndarray,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the heatmap generation process
        
        Args:
            image: Input image
            landmarks: Landmark coordinates
            save_path: Path to save visualization
            
        Returns:
            matplotlib Figure
        """
        
        combined_heatmap, individual_heatmaps = self.generate_heatmap(landmarks, return_individual=True)
        
        # Use visualizer for plotting
        visualizer = CephalometricVisualizer()
        return visualizer.plot_heatmaps_with_image(
            image, individual_heatmaps, landmarks, 
            title="Heatmap Generation Visualization",
            save_path=save_path
        )


class EnhancedDataGenerator:
    """
    Complete data generation pipeline for MAHT-Net training
    """
    
    def __init__(self, config: DataGenerationConfig):
        """
        Initialize the data generator
        
        Args:
            config: Data generation configuration
        """
        
        self.config = config
        self.visualizer = CephalometricVisualizer()
        
        # Initialize heatmap generator
        self.heatmap_generator = EnhancedGaussianHeatmapGenerator(
            target_size=config.target_size,
            heatmap_size=config.heatmap_size,
            sigma=config.gaussian_sigma,
            amplitude=config.heatmap_amplitude
        )
        
        # Setup augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        logger.info("Enhanced data generator initialized")
    
    def process_dataset(self,
                       dataset_path: str,
                       output_path: str,
                       split_ratios: Dict[str, float] = None) -> Dict:
        """
        Process complete dataset with resizing, augmentation, and heatmap generation
        
        Args:
            dataset_path: Path to input dataset
            output_path: Path for processed output
            split_ratios: Train/val/test split ratios
            
        Returns:
            Processing statistics
        """
        
        if split_ratios is None:
            split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        logger.info(f"🔄 Processing dataset: {dataset_path}")
        logger.info(f"📁 Output directory: {output_path}")
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'heatmaps').mkdir(exist_ok=True)
        (output_path / 'visualizations').mkdir(exist_ok=True)
        
        # Load raw data
        raw_samples = self._load_raw_dataset(dataset_path)
        logger.info(f"📊 Loaded {len(raw_samples)} raw samples")
        
        # Process samples
        processed_samples = []
        processing_stats = {
            'total_samples': len(raw_samples),
            'successful_samples': 0,
            'failed_samples': 0,
            'original_sizes': [],
            'landmark_distributions': [],
            'quality_metrics': {}
        }
        
        for i, sample in enumerate(tqdm(raw_samples, desc="Processing samples")):
            try:
                processed_sample = self._process_single_sample(sample, output_path, i)
                if processed_sample:
                    processed_samples.append(processed_sample)
                    processing_stats['successful_samples'] += 1
                    
                    # Collect statistics
                    processing_stats['original_sizes'].append(sample['original_size'])
                    processing_stats['landmark_distributions'].append(processed_sample['landmarks'])
                else:
                    processing_stats['failed_samples'] += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process sample {i}: {str(e)}")
                processing_stats['failed_samples'] += 1
        
        # Create dataset splits
        split_samples = self._create_dataset_splits(processed_samples, split_ratios)
        
        # Save processed dataset
        self._save_processed_dataset(split_samples, output_path)
        
        # Generate visualizations
        if self.config.save_intermediate_results:
            self._generate_pipeline_visualizations(processed_samples, output_path)
        
        # Calculate final statistics
        processing_stats['quality_metrics'] = self._calculate_quality_metrics(processed_samples)
        
        logger.info(f"✅ Processing complete: {processing_stats['successful_samples']}/{processing_stats['total_samples']} samples processed")
        
        return processing_stats
    
    def _process_single_sample(self, sample: Dict, output_path: Path, sample_id: int) -> Optional[Dict]:
        """
        Process a single sample through the complete pipeline
        
        Args:
            sample: Raw sample data
            output_path: Output directory
            sample_id: Sample identifier
            
        Returns:
            Processed sample or None if failed
        """
        
        # Load and validate image
        image_path = sample['image_path']
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None
        
        original_size = image.shape[:2]  # (H, W)
        landmarks = np.array(sample['landmarks'])  # (N, 2)
        
        # Quality checks
        if not self._validate_sample_quality(image, landmarks):
            return None
        
        # Resize image and scale landmarks
        resized_image, scaled_landmarks = self._resize_image_and_landmarks(
            image, landmarks, self.config.target_size
        )
        
        # Apply augmentation if enabled and training
        augmented_image, augmented_landmarks = self._apply_augmentation(
            resized_image, scaled_landmarks
        )
        
        # Generate heatmaps
        heatmaps = self.heatmap_generator.generate_heatmap(augmented_landmarks)
        
        # Generate descriptive filename based on original image and augmentation
        original_name = Path(image_path).stem  # Get filename without extension
        
        # Determine augmentation suffix
        aug_suffix = ""
        augmentation_detected = False
        landmark_diff = 0.0
        
        if self.config.enable_augmentation:
            # Check if augmentation was actually applied by comparing landmarks
            # Only compare if shapes match
            if augmented_landmarks.shape == scaled_landmarks.shape:
                landmark_diff = float(np.mean(np.abs(augmented_landmarks - scaled_landmarks)))
                if landmark_diff > 0.1:  # Threshold to detect if augmentation occurred
                    aug_suffix = "_aug"
                    augmentation_detected = True
                else:
                    aug_suffix = "_noaug"
            else:
                # If shapes don't match, something went wrong with augmentation
                logger.warning(f"Landmark shape mismatch: {augmented_landmarks.shape} vs {scaled_landmarks.shape}")
                aug_suffix = "_noaug"
        else:
            aug_suffix = "_orig"
        
        # Create descriptive filename: originalname_aug_count_targetsize
        target_size_str = f"{self.config.target_size[0]}x{self.config.target_size[1]}"
        descriptive_name = f"{original_name}{aug_suffix}_{sample_id:03d}_{target_size_str}"
        
        # Create processed sample
        processed_sample = {
            'sample_id': sample_id,
            'original_image_path': str(image_path),
            'original_name': original_name,
            'descriptive_name': descriptive_name,
            'original_size': original_size,
            'target_size': self.config.target_size,
            'landmarks': augmented_landmarks.tolist(),
            'image_path': f'images/{descriptive_name}.png',
            'heatmap_path': f'heatmaps/{descriptive_name}.npy',
            'metadata': {
                'original_landmarks': landmarks.tolist(),
                'scaled_landmarks': scaled_landmarks.tolist(),
                'augmentation_applied': bool(self.config.enable_augmentation),
                'augmentation_detected': bool(augmentation_detected),
                'augmentation_strength': float(landmark_diff)
            }
        }
        
        # Save processed image and heatmaps
        cv2.imwrite(str(output_path / processed_sample['image_path']), augmented_image)
        np.save(str(output_path / processed_sample['heatmap_path']), heatmaps)
        
        # Save visualization if requested
        if self.config.save_intermediate_results and sample_id < self.config.visualization_samples:
            self._save_sample_visualization(
                image, resized_image, landmarks, scaled_landmarks, 
                augmented_image, augmented_landmarks, heatmaps,
                output_path / 'visualizations' / f'{descriptive_name}_pipeline.png',
                descriptive_name
            )
        
        return processed_sample
    
    def _resize_image_and_landmarks(self, 
                                   image: np.ndarray, 
                                   landmarks: np.ndarray, 
                                   target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize image and scale landmarks accordingly
        
        Args:
            image: Input image (H, W)
            landmarks: Landmark coordinates (N, 2)
            target_size: Target size (H, W)
            
        Returns:
            Resized image and scaled landmarks
        """
        
        original_height, original_width = image.shape
        target_height, target_width = target_size
        
        # Calculate scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Resize image
        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # Scale landmarks
        scaled_landmarks = landmarks.copy()
        scaled_landmarks[:, 0] *= scale_x  # X coordinates
        scaled_landmarks[:, 1] *= scale_y  # Y coordinates
        
        return resized_image, scaled_landmarks
    
    def _apply_augmentation(self, 
                          image: np.ndarray, 
                          landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation pipeline to image and landmarks
        
        Args:
            image: Input image
            landmarks: Landmark coordinates
            
        Returns:
            Augmented image and landmarks
        """
        
        if not self.config.enable_augmentation:
            return image, landmarks
        
        # Convert landmarks to albumentations keypoint format
        keypoints = [(x, y) for x, y in landmarks]
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image, keypoints=keypoints)
        
        # Extract results
        augmented_image = augmented['image']
        augmented_keypoints = np.array(augmented['keypoints'])
        
        return augmented_image, augmented_keypoints
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create albumentations augmentation pipeline"""
        
        transforms = []
        
        if self.config.enable_augmentation:
            transforms = [
                # Geometric transformations
                A.Affine(
                    translate_percent=0.1,
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    p=self.config.augmentation_probability * 0.8
                ),
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    p=self.config.augmentation_probability * 0.3
                ),
                
                # Intensity transformations
                # Intensity transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=self.config.augmentation_probability * 0.5
                ),
                A.GaussNoise(
                    noise_scale_factor=0.1,
                    p=self.config.augmentation_probability * 0.3
                ),
                A.Blur(blur_limit=3, p=self.config.augmentation_probability * 0.2),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=self.config.augmentation_probability * 0.3
                )
            ]
        
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False
            )
        )
    
    def _validate_sample_quality(self, image: np.ndarray, landmarks: np.ndarray) -> bool:
        """Validate sample quality"""
        
        h, w = image.shape
        
        # Check minimum size
        if h < self.config.min_image_size[0] or w < self.config.min_image_size[1]:
            logger.debug(f"Image too small: {h}x{w} < {self.config.min_image_size}")
            return False
        
        # Check landmark bounds
        if np.any(landmarks < 0) or np.any(landmarks[:, 0] >= w) or np.any(landmarks[:, 1] >= h):
            logger.debug(f"Landmarks out of bounds")
            return False
        
        # Check expected number of landmarks (ISBI should have 19)
        expected_landmarks = 19
        if len(landmarks) != expected_landmarks:
            logger.debug(f"Unexpected landmark count: {len(landmarks)} != {expected_landmarks}")
            return False
        
        # Check image quality (not completely black/white)
        if np.std(image) < 10:
            logger.debug(f"Low image variance: {np.std(image)}")
            return False
        
        return True
    
    def _load_raw_dataset(self, dataset_path: Path) -> List[Dict]:
        """Load raw dataset samples"""
        
        # Check if this looks like an ISBI dataset structure
        if (dataset_path / 'RawImage').exists() and (dataset_path / 'AnnotationsByMD').exists():
            logger.info("Detected ISBI dataset structure, using ISBI loader")
            from .isbi_loader import ISBIDatasetLoader
            
            loader = ISBIDatasetLoader(str(dataset_path))
            samples = loader.load_dataset(
                image_split='train',  # Use training images
                annotation_type='senior',  # Use senior annotations
                max_samples=None  # Load all samples
            )
            return samples
        
        # Original JSON-based loading
        samples = []
        
        # Look for annotation files
        annotation_files = list(dataset_path.glob("*.json"))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {dataset_path}")
        
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                samples.extend(data)
            else:
                samples.append(data)
        
        return samples
    
    def _create_dataset_splits(self, samples: List[Dict], ratios: Dict[str, float]) -> Dict[str, List[Dict]]:
        """Create train/val/test splits"""
        
        np.random.shuffle(samples)
        n_samples = len(samples)
        
        # Calculate split indices
        train_end = int(n_samples * ratios['train'])
        val_end = train_end + int(n_samples * ratios['val'])
        
        splits = {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }
        
        logger.info(f"Dataset splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def _save_processed_dataset(self, splits: Dict[str, List[Dict]], output_path: Path):
        """Save processed dataset splits"""
        
        for split_name, samples in splits.items():
            split_file = output_path / f'{split_name}_annotations.json'
            with open(split_file, 'w') as f:
                json.dump(samples, f, indent=2)
            
            logger.info(f"Saved {split_name} split: {len(samples)} samples -> {split_file}")
    
    def _save_sample_visualization(self, 
                                  original_image: np.ndarray,
                                  resized_image: np.ndarray,
                                  original_landmarks: np.ndarray,
                                  resized_landmarks: np.ndarray,
                                  augmented_image: np.ndarray,
                                  augmented_landmarks: np.ndarray,
                                  heatmaps: np.ndarray,
                                  save_path: Path,
                                  descriptive_name: str = None):
        """Save comprehensive sample visualization with descriptive naming"""
        
        # Extract augmentation info from filename if provided
        aug_info = ""
        if descriptive_name:
            if "_aug_" in descriptive_name:
                aug_info = " (Augmented)"
            elif "_noaug_" in descriptive_name:
                aug_info = " (No Augmentation Applied)"
            elif "_orig_" in descriptive_name:
                aug_info = " (Original)"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Add main title with descriptive information
        main_title = f'Data Pipeline: {descriptive_name if descriptive_name else "Sample"}{aug_info}'
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        # Original
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title(f'Original ({original_image.shape[1]}×{original_image.shape[0]})')
        for i, (x, y) in enumerate(original_landmarks):
            axes[0, 0].scatter(x, y, c='red', s=30, marker='o')
        
        # Resized
        axes[0, 1].imshow(resized_image, cmap='gray')
        axes[0, 1].set_title(f'Resized ({resized_image.shape[1]}×{resized_image.shape[0]})')
        for i, (x, y) in enumerate(resized_landmarks):
            axes[0, 1].scatter(x, y, c='red', s=30, marker='o')
        
        # Augmented
        axes[0, 2].imshow(augmented_image, cmap='gray')
        axes[0, 2].set_title(f'Processed{aug_info}')
        for i, (x, y) in enumerate(augmented_landmarks):
            axes[0, 2].scatter(x, y, c='red', s=30, marker='o')
        
        # Heatmaps
        combined_heatmap = np.max(heatmaps, axis=0)
        axes[1, 0].imshow(combined_heatmap, cmap='jet')
        axes[1, 0].set_title('Combined Heatmap')
        
        # Overlay
        axes[1, 1].imshow(augmented_image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(combined_heatmap, cmap='jet', alpha=0.5)
        axes[1, 1].set_title('Heatmap Overlay')
        
        # Individual heatmaps (first few)
        n_show = min(6, len(heatmaps))
        if n_show > 0:
            # Create a subplot for individual heatmaps
            axes[1, 2].set_title('Individual Heatmaps')
            axes[1, 2].axis('off')
            
            # Create small subplots within the main subplot
            for i in range(n_show):
                sub_ax = plt.subplot2grid((4, 3), (2 + i//3, 2), colspan=1, rowspan=1)
                sub_ax.imshow(heatmaps[i], cmap='jet')
                sub_ax.set_title(f'L{i+1}', fontsize=8)
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
        
        # Remove ticks
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_pipeline_visualizations(self, samples: List[Dict], output_path: Path):
        """Generate overview visualizations of the pipeline"""
        
        # Create sample data for visualization
        viz_samples = []
        for i, sample in enumerate(samples[:self.config.visualization_samples]):
            # Load images and create visualization data
            image_path = output_path / sample['image_path']
            if image_path.exists():
                processed_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                landmarks = np.array(sample['landmarks'])
                
                # Load heatmaps
                heatmap_path = output_path / sample['heatmap_path']
                if heatmap_path.exists():
                    heatmaps = np.load(str(heatmap_path))
                    
                    viz_samples.append({
                        'original_image': processed_image,  # Placeholder for now
                        'resized_image': processed_image,
                        'original_landmarks': landmarks,
                        'resized_landmarks': landmarks,
                        'heatmaps': heatmaps
                    })
        
        if viz_samples:
            # Generate pipeline overview
            self.visualizer.plot_data_pipeline_overview(
                viz_samples,
                save_path=str(output_path / 'visualizations' / 'pipeline_overview.png')
            )
    
    def _calculate_quality_metrics(self, samples: List[Dict]) -> Dict:
        """Calculate quality metrics for processed dataset"""
        
        if not samples:
            return {}
        
        # Extract landmark data
        all_landmarks = [np.array(sample['landmarks']) for sample in samples]
        
        # Calculate statistics
        landmark_means = np.mean([np.mean(landmarks, axis=0) for landmarks in all_landmarks], axis=0)
        landmark_stds = np.std([np.std(landmarks, axis=0) for landmarks in all_landmarks], axis=0)
        
        metrics = {
            'num_samples': len(samples),
            'avg_landmark_x': float(landmark_means[0]),
            'avg_landmark_y': float(landmark_means[1]),
            'landmark_std_x': float(landmark_stds[0]),
            'landmark_std_y': float(landmark_stds[1]),
            'target_size': self.config.target_size,
            'heatmap_size': self.config.heatmap_size
        }
        
        return metrics
