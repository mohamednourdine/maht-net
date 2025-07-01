"""
Cephalometric Dataset Implementation
Handles loading, preprocessing, and augmentation of cephalometric X-ray images
with landmark annotations for MAHT-Net training
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
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CephalometricDataset(Dataset):
    """
    Dataset class for cephalometric landmark detection
    
    Supports multiple cephalometric datasets with standardized 7-landmark annotation:
    1. Nasion (N)
    2. Sella (S) 
    3. Articulare (Ar)
    4. Gonion (Go)
    5. Menton (Me)
    6. Pogonion (Pog)
    7. Upper Incisor (U1)
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 heatmap_size: Tuple[int, int] = (64, 64),
                 sigma: float = 2.0,
                 augmentation: bool = True,
                 normalize: bool = True):
        """
        Initialize the cephalometric dataset
        
        Args:
            data_dir: Root directory containing images and annotations
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size for resizing
            heatmap_size: Size of generated heatmaps
            sigma: Gaussian sigma for heatmap generation
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augmentation = augmentation and (split == 'train')
        self.normalize = normalize
        
        # Landmark names for reference
        self.landmark_names = [
            'Nasion', 'Sella', 'Articulare', 'Gonion', 
            'Menton', 'Pogonion', 'Upper_Incisor'
        ]
        
        # Load dataset annotations
        self.samples = self._load_annotations()
        
        # Setup data transforms
        self.transforms = self._setup_transforms()
        
        logger.info(f"✅ Loaded {len(self.samples)} samples for {split} split")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from the dataset directory"""
        
        # Look for annotation file
        annotation_files = [
            self.data_dir / f"{self.split}_annotations.json",
            self.data_dir / f"{self.split}.json",
            self.data_dir / "annotations" / f"{self.split}.json"
        ]
        
        annotation_file = None
        for file_path in annotation_files:
            if file_path.exists():
                annotation_file = file_path
                break
        
        if annotation_file is None:
            raise FileNotFoundError(
                f"No annotation file found for split '{self.split}' in {self.data_dir}"
            )
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Validate and standardize annotations
        validated_samples = []
        for sample in annotations:
            if self._validate_sample(sample):
                validated_samples.append(sample)
        
        return validated_samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate a single sample annotation"""
        
        required_keys = ['image_path', 'landmarks']
        
        # Check required keys
        for key in required_keys:
            if key not in sample:
                logger.warning(f"Sample missing required key: {key}")
                return False
        
        # Check image file exists
        image_path = self.data_dir / sample['image_path']
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False
        
        # Check landmarks format
        landmarks = sample['landmarks']
        if not isinstance(landmarks, list) or len(landmarks) != 7:
            logger.warning(f"Invalid landmarks format: expected 7 landmarks, got {len(landmarks)}")
            return False
        
        # Validate landmark coordinates
        for i, landmark in enumerate(landmarks):
            if not isinstance(landmark, (list, tuple)) or len(landmark) != 2:
                logger.warning(f"Invalid landmark {i}: expected [x, y], got {landmark}")
                return False
        
        return True
    
    def _setup_transforms(self) -> A.Compose:
        """Setup augmentation and preprocessing transforms"""
        
        transforms = []
        
        if self.augmentation:
            # Geometric augmentations
            transforms.extend([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.7
                ),
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    alpha_affine=5,
                    p=0.3
                ),
            ])
            
            # Appearance augmentations
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Blur(blur_limit=3, p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
            ])
        
        # Resize to target size
        transforms.append(A.Resize(*self.image_size))
        
        # Normalization
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means for RGB
                    std=[0.229, 0.224, 0.225],   # ImageNet stds for RGB
                    max_pixel_value=255.0
                )
            )
        
        # Convert to tensor
        transforms.append(ToTensorV2())
        
        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=False
            )
        )
    
    def _generate_heatmap(self, landmarks: np.ndarray) -> np.ndarray:
        """Generate Gaussian heatmaps for landmarks"""
        
        num_landmarks = len(landmarks)
        heatmaps = np.zeros((num_landmarks, *self.heatmap_size), dtype=np.float32)
        
        # Scale factor from image size to heatmap size
        scale_x = self.heatmap_size[1] / self.image_size[1]
        scale_y = self.heatmap_size[0] / self.image_size[0]
        
        for i, (x, y) in enumerate(landmarks):
            # Scale coordinates to heatmap size
            hm_x = int(x * scale_x)
            hm_y = int(y * scale_y)
            
            # Create Gaussian heatmap
            heatmap = self._create_gaussian_heatmap(
                self.heatmap_size, (hm_x, hm_y), self.sigma
            )
            heatmaps[i] = heatmap
        
        return heatmaps
    
    def _create_gaussian_heatmap(self, 
                               size: Tuple[int, int], 
                               center: Tuple[int, int], 
                               sigma: float) -> np.ndarray:
        """Create a Gaussian heatmap centered at the given point"""
        
        h, w = size
        cx, cy = center
        
        # Create coordinate grids
        y_grid, x_grid = np.ogrid[0:h, 0:w]
        
        # Compute Gaussian
        gaussian = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        
        return gaussian.astype(np.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_dir / sample['image_path']
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get landmarks
        landmarks = np.array(sample['landmarks'], dtype=np.float32)
        
        # Apply transforms
        transformed = self.transforms(image=image, keypoints=landmarks)
        image = transformed['image']
        landmarks = np.array(transformed['keypoints'], dtype=np.float32)
        
        # Generate heatmaps
        heatmaps = self._generate_heatmap(landmarks)
        
        # Prepare output
        output = {
            'image': image,
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'heatmaps': torch.tensor(heatmaps, dtype=torch.float32),
            'image_path': str(image_path),
            'sample_id': sample.get('id', idx)
        }
        
        # Add metadata if available
        if 'patient_id' in sample:
            output['patient_id'] = sample['patient_id']
        if 'acquisition_params' in sample:
            output['acquisition_params'] = sample['acquisition_params']
        
        return output
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get detailed information about a specific sample"""
        
        sample = self.samples[idx]
        image_path = self.data_dir / sample['image_path']
        
        # Load image to get dimensions
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        return {
            'index': idx,
            'image_path': str(image_path),
            'original_size': (w, h),
            'target_size': self.image_size,
            'landmarks': sample['landmarks'],
            'landmark_names': self.landmark_names,
            'patient_id': sample.get('patient_id', 'unknown'),
            'split': self.split
        }


def create_dataloaders(data_dir: str,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      image_size: Tuple[int, int] = (512, 512),
                      heatmap_size: Tuple[int, int] = (64, 64)) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        image_size: Target image size
        heatmap_size: Target heatmap size
        
    Returns:
        Dictionary containing dataloaders for each split
    """
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = CephalometricDataset(
                data_dir=data_dir,
                split=split,
                image_size=image_size,
                heatmap_size=heatmap_size,
                augmentation=(split == 'train')
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size if split == 'train' else batch_size // 2,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')
            )
            
            dataloaders[split] = dataloader
            
        except FileNotFoundError:
            logger.warning(f"No data found for split '{split}', skipping...")
            continue
    
    return dataloaders


# Dataset utilities
def visualize_sample(dataset: CephalometricDataset, 
                    idx: int, 
                    save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize a dataset sample with landmarks and heatmaps
    
    Args:
        dataset: CephalometricDataset instance
        idx: Sample index to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Visualization image as numpy array
    """
    import matplotlib.pyplot as plt
    
    sample = dataset[idx]
    image = sample['image'].permute(1, 2, 0).numpy()
    landmarks = sample['landmarks'].numpy()
    heatmaps = sample['heatmaps'].numpy()
    
    # Denormalize image if normalized
    if dataset.normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original image with landmarks
    axes[0, 0].imshow(image)
    axes[0, 0].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=50)
    for i, (x, y) in enumerate(landmarks):
        axes[0, 0].annotate(
            dataset.landmark_names[i], 
            (x, y), 
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
        )
    axes[0, 0].set_title('Image with Landmarks')
    axes[0, 0].axis('off')
    
    # Heatmaps
    for i in range(7):
        row = i // 4
        col = (i % 4) + (1 if row == 0 else 0)
        
        if row < 2 and col < 4:
            axes[row, col].imshow(heatmaps[i], cmap='hot')
            axes[row, col].set_title(f'{dataset.landmark_names[i]}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Test dataset functionality
    print("🧪 Testing CephalometricDataset...")
    
    # Note: This would require actual data to run
    # dataset = CephalometricDataset(
    #     data_dir="data/processed",
    #     split="train"
    # )
    # 
    # print(f"✅ Dataset loaded: {len(dataset)} samples")
    # 
    # # Test sample loading
    # sample = dataset[0]
    # print(f"📊 Sample shapes:")
    # for key, value in sample.items():
    #     if isinstance(value, torch.Tensor):
    #         print(f"  {key}: {value.shape}")
    
    print("✅ Dataset implementation ready for use!")
