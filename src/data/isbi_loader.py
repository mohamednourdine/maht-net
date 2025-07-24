"""
ISBI 2015 Dataset Loader for MAHT-Net
Handles the specific format of the ISBI 2015 Cephalometric X-ray dataset
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ISBIDatasetLoader:
    """
    Loader for ISBI 2015 Cephalometric X-ray dataset
    Handles the specific directory structure and annotation format
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize ISBI dataset loader
        
        Args:
            dataset_root: Root directory containing the ISBI dataset
        """
        self.dataset_root = Path(dataset_root)
        
        # ISBI dataset structure
        self.image_dirs = {
            'train': self.dataset_root / 'RawImage' / 'TrainingData',
            'test1': self.dataset_root / 'RawImage' / 'Test1Data', 
            'test2': self.dataset_root / 'RawImage' / 'Test2Data'
        }
        
        self.annotation_dirs = {
            'senior': self.dataset_root / 'AnnotationsByMD' / '400_senior',
            'junior': self.dataset_root / 'AnnotationsByMD' / '400_junior'
        }
        
        # Validate dataset structure
        self._validate_dataset_structure()
        
        logger.info(f"ISBI dataset loader initialized for: {dataset_root}")
    
    def _validate_dataset_structure(self):
        """Validate that all required directories exist"""
        
        # Check image directories
        for split, img_dir in self.image_dirs.items():
            if not img_dir.exists():
                logger.warning(f"Image directory not found: {img_dir}")
        
        # Check annotation directories  
        for annotator, ann_dir in self.annotation_dirs.items():
            if not ann_dir.exists():
                logger.warning(f"Annotation directory not found: {ann_dir}")
    
    def load_dataset(self, 
                    image_split: str = 'train',
                    annotation_type: str = 'senior',
                    max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load dataset samples with images and annotations
        
        Args:
            image_split: Which image split to use ('train', 'test1', 'test2')
            annotation_type: Which annotations to use ('senior', 'junior')
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of sample dictionaries
        """
        
        img_dir = self.image_dirs[image_split]
        ann_dir = self.annotation_dirs[annotation_type]
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not ann_dir.exists():
            raise ValueError(f"Annotation directory not found: {ann_dir}")
        
        # Get list of image files
        image_files = sorted([f for f in img_dir.glob('*.bmp')])
        if max_samples:
            image_files = image_files[:max_samples]
        
        samples = []
        
        for img_file in image_files:
            # Get corresponding annotation file
            base_name = img_file.stem
            ann_file = ann_dir / f"{base_name}.txt"
            
            if not ann_file.exists():
                logger.warning(f"Annotation file not found: {ann_file}")
                continue
            
            # Load landmarks
            landmarks = self._load_landmarks(ann_file)
            if landmarks is None:
                continue
            
            # Get image dimensions
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not load image: {img_file}")
                    continue
                original_size = (img.shape[0], img.shape[1])  # (height, width)
            except Exception as e:
                logger.warning(f"Error reading image {img_file}: {e}")
                continue
            
            # Create sample
            sample = {
                'image_path': str(img_file),
                'landmarks': landmarks,
                'original_size': original_size,
                'sample_id': base_name,
                'split': image_split,
                'annotator': annotation_type
            }
            
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from ISBI dataset")
        return samples
    
    def _load_landmarks(self, annotation_file: Path) -> Optional[List[List[float]]]:
        """
        Load landmarks from ISBI annotation file
        
        Args:
            annotation_file: Path to annotation file
            
        Returns:
            List of [x, y] landmark coordinates or None if error
        """
        
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            # ISBI format: first 19 lines are landmark coordinates (x,y)
            landmarks = []
            for i in range(min(19, len(lines))):
                line = lines[i].strip()
                if ',' in line:
                    x_str, y_str = line.split(',')
                    x = float(x_str.strip())
                    y = float(y_str.strip())
                    landmarks.append([x, y])
                else:
                    logger.warning(f"Invalid landmark format in {annotation_file}: {line}")
                    return None
            
            if len(landmarks) != 19:
                logger.warning(f"Expected 19 landmarks, got {len(landmarks)} in {annotation_file}")
                return None
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error loading landmarks from {annotation_file}: {str(e)}")
            return None
    
    def save_as_json(self, 
                    samples: List[Dict], 
                    output_file: str,
                    make_paths_relative: bool = True):
        """
        Save samples as JSON annotation file
        
        Args:
            samples: List of sample dictionaries
            output_file: Output JSON file path
            make_paths_relative: Whether to make image paths relative
        """
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process samples for JSON serialization
        processed_samples = []
        for sample in samples:
            processed_sample = sample.copy()
            
            # Make image paths relative if requested
            if make_paths_relative:
                img_path = Path(sample['image_path'])
                if img_path.is_absolute():
                    # Make relative to output directory
                    try:
                        processed_sample['image_path'] = str(img_path.relative_to(output_path.parent))
                    except ValueError:
                        # If can't make relative, keep as is
                        processed_sample['image_path'] = str(img_path)
            
            processed_samples.append(processed_sample)
        
        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(processed_samples, f, indent=2)
        
        logger.info(f"Saved {len(processed_samples)} samples to {output_file}")
    
    def get_dataset_statistics(self, samples: List[Dict]) -> Dict:
        """
        Calculate dataset statistics
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Dictionary with statistics
        """
        
        if not samples:
            return {}
        
        # Calculate image size statistics
        image_sizes = []
        landmark_coords = []
        
        for sample in samples[:10]:  # Sample first 10 for speed
            try:
                img = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    image_sizes.append((img.shape[0], img.shape[1]))  # (H, W)
                    landmark_coords.append(sample['landmarks'])
            except Exception as e:
                logger.warning(f"Could not load image {sample['image_path']}: {e}")
        
        stats = {
            'total_samples': len(samples),
            'image_sizes': image_sizes,
            'landmark_coords': landmark_coords,
            'splits': {},
            'annotators': {}
        }
        
        # Count by split and annotator
        for sample in samples:
            split = sample.get('split', 'unknown')
            annotator = sample.get('annotator', 'unknown')
            
            stats['splits'][split] = stats['splits'].get(split, 0) + 1
            stats['annotators'][annotator] = stats['annotators'].get(annotator, 0) + 1
        
        return stats
    
    def create_train_val_test_splits(self, 
                                   samples: List[Dict],
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Dict[str, List[Dict]]:
        """
        Create train/validation/test splits from samples
        
        Args:
            samples: List of sample dictionaries
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Shuffle samples
        np.random.seed(random_seed)
        shuffled_samples = samples.copy()
        np.random.shuffle(shuffled_samples)
        
        n_samples = len(shuffled_samples)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        splits = {
            'train': shuffled_samples[:train_end],
            'val': shuffled_samples[train_end:val_end],
            'test': shuffled_samples[val_end:]
        }
        
        logger.info(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits


def convert_isbi_to_maht_format(isbi_root: str, 
                               output_dir: str,
                               image_split: str = 'train',
                               annotation_type: str = 'senior',
                               max_samples: Optional[int] = None) -> str:
    """
    Convert ISBI dataset to MAHT-Net format
    
    Args:
        isbi_root: Root directory of ISBI dataset
        output_dir: Output directory for converted dataset
        image_split: Which image split to use
        annotation_type: Which annotations to use
        max_samples: Maximum number of samples
        
    Returns:
        Path to the converted dataset JSON file
    """
    
    # Load ISBI dataset
    loader = ISBIDatasetLoader(isbi_root)
    samples = loader.load_dataset(image_split, annotation_type, max_samples)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_file = output_path / 'annotations.json'
    loader.save_as_json(samples, str(json_file))
    
    # Generate statistics
    stats = loader.get_dataset_statistics(samples)
    stats_file = output_path / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Converted ISBI dataset: {len(samples)} samples -> {json_file}")
    
    return str(json_file)
