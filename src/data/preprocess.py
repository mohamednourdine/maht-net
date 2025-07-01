"""
Data preprocessing utilities for MAHT-Net
Handles cephalometric X-ray image preprocessing and annotation standardization
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CephalometricDataProcessor:
    """
    Comprehensive data processor for cephalometric datasets
    Handles multiple dataset formats and standardizes to MAHT-Net format
    """
    
    def __init__(self, 
                 output_dir: str,
                 target_size: Tuple[int, int] = (512, 512),
                 quality_check: bool = True):
        """
        Initialize the data processor
        
        Args:
            output_dir: Output directory for processed data
            target_size: Target image size for resizing
            quality_check: Whether to perform quality checks
        """
        
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.quality_check = quality_check
        
        # Standard 7 landmarks for cephalometric analysis
        self.landmark_names = [
            'Nasion', 'Sella', 'Articulare', 'Gonion', 
            'Menton', 'Pogonion', 'Upper_Incisor'
        ]
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        
        logger.info(f"Data processor initialized with output: {self.output_dir}")
    
    def process_dataset(self, 
                       input_dir: str,
                       dataset_name: str,
                       annotation_format: str = 'auto') -> Dict:
        """
        Process a complete cephalometric dataset
        
        Args:
            input_dir: Input directory containing images and annotations
            dataset_name: Name of the dataset
            annotation_format: Format of annotations ('json', 'csv', 'xml', 'auto')
            
        Returns:
            Processing statistics
        """
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        logger.info(f"🔄 Processing dataset: {dataset_name}")
        logger.info(f"📂 Input directory: {input_dir}")
        
        # Detect annotation format if auto
        if annotation_format == 'auto':
            annotation_format = self._detect_annotation_format(input_path)
        
        # Load annotations
        annotations = self._load_annotations(input_path, annotation_format)
        
        # Process images and annotations
        processed_samples = []
        failed_samples = []
        
        for annotation in tqdm(annotations, desc="Processing samples"):
            try:
                processed_sample = self._process_single_sample(
                    annotation, input_path, dataset_name
                )
                if processed_sample:
                    processed_samples.append(processed_sample)
            except Exception as e:
                logger.warning(f"⚠️  Failed to process sample {annotation.get('id', 'unknown')}: {str(e)}")
                failed_samples.append(annotation)
        
        # Save processed annotations
        output_file = self.output_dir / 'annotations' / f'{dataset_name}_processed.json'
        self._save_annotations(processed_samples, output_file)
        
        # Generate processing report
        stats = {
            'dataset_name': dataset_name,
            'total_samples': len(annotations),
            'successful_samples': len(processed_samples),
            'failed_samples': len(failed_samples),
            'success_rate': len(processed_samples) / len(annotations) * 100,
            'output_directory': str(self.output_dir),
            'target_size': self.target_size,
            'landmark_names': self.landmark_names
        }
        
        logger.info(f"Processing completed: {len(processed_samples)}/{len(annotations)} samples")
        
        return stats
    
    def _detect_annotation_format(self, input_path: Path) -> str:
        """Auto-detect annotation format"""
        
        # Look for common annotation files
        if list(input_path.glob('*.json')):
            return 'json'
        elif list(input_path.glob('*.csv')):
            return 'csv'
        elif list(input_path.glob('*.xml')):
            return 'xml'
        else:
            raise ValueError("Could not detect annotation format")
    
    def _load_annotations(self, input_path: Path, format_type: str) -> List[Dict]:
        """Load annotations based on format"""
        
        if format_type == 'json':
            return self._load_json_annotations(input_path)
        elif format_type == 'csv':
            return self._load_csv_annotations(input_path)
        elif format_type == 'xml':
            return self._load_xml_annotations(input_path)
        else:
            raise ValueError(f"Unsupported annotation format: {format_type}")
    
    def _load_json_annotations(self, input_path: Path) -> List[Dict]:
        """Load JSON format annotations"""
        
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            raise ValueError("No JSON annotation files found")
        
        all_annotations = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                all_annotations.extend(data)
            elif isinstance(data, dict):
                if 'annotations' in data:
                    all_annotations.extend(data['annotations'])
                elif 'samples' in data:
                    all_annotations.extend(data['samples'])
                else:
                    all_annotations.append(data)
        
        return all_annotations
    
    def _load_csv_annotations(self, input_path: Path) -> List[Dict]:
        """Load CSV format annotations"""
        
        csv_files = list(input_path.glob('*.csv'))
        if not csv_files:
            raise ValueError("No CSV annotation files found")
        
        all_annotations = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            # Convert DataFrame to list of dictionaries
            for _, row in df.iterrows():
                annotation = row.to_dict()
                all_annotations.append(annotation)
        
        return all_annotations
    
    def _load_xml_annotations(self, input_path: Path) -> List[Dict]:
        """Load XML format annotations (placeholder)"""
        
        # XML parsing would be implemented here based on specific format
        # This is a placeholder for XML support
        raise NotImplementedError("XML annotation format not yet implemented")
    
    def _process_single_sample(self, 
                              annotation: Dict, 
                              input_path: Path, 
                              dataset_name: str) -> Optional[Dict]:
        """Process a single image-annotation pair"""
        
        # Extract image path
        image_path = self._get_image_path(annotation, input_path)
        if not image_path.exists():
            raise ValueError(f"Image not found: {image_path}")
        
        # Load and validate image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Quality check
        if self.quality_check and not self._quality_check_image(image):
            raise ValueError("Image failed quality check")
        
        # Extract and validate landmarks
        landmarks = self._extract_landmarks(annotation)
        if not self._validate_landmarks(landmarks, image.shape):
            raise ValueError("Invalid landmarks")
        
        # Process image
        processed_image, scale_factors = self._process_image(image)
        
        # Scale landmarks
        scaled_landmarks = self._scale_landmarks(landmarks, scale_factors)
        
        # Generate output filename
        sample_id = annotation.get('id', Path(image_path).stem)
        output_filename = f"{dataset_name}_{sample_id}.jpg"
        output_path = self.output_dir / 'images' / output_filename
        
        # Save processed image
        cv2.imwrite(str(output_path), processed_image)
        
        # Create processed annotation
        processed_annotation = {
            'id': sample_id,
            'image_path': f"images/{output_filename}",
            'landmarks': scaled_landmarks,
            'landmark_names': self.landmark_names,
            'original_size': list(image.shape[:2]),
            'processed_size': list(processed_image.shape[:2]),
            'scale_factors': scale_factors,
            'dataset': dataset_name,
            'patient_id': annotation.get('patient_id', sample_id),
            'acquisition_params': annotation.get('acquisition_params', {})
        }
        
        return processed_annotation
    
    def _get_image_path(self, annotation: Dict, input_path: Path) -> Path:
        """Extract image path from annotation"""
        
        # Try different common field names
        image_fields = ['image_path', 'image', 'filename', 'file', 'path']
        
        for field in image_fields:
            if field in annotation:
                image_file = annotation[field]
                if isinstance(image_file, str):
                    # Try relative to input directory first
                    rel_path = input_path / image_file
                    if rel_path.exists():
                        return rel_path
                    
                    # Try absolute path
                    abs_path = Path(image_file)
                    if abs_path.exists():
                        return abs_path
        
        raise ValueError("Could not find image path in annotation")
    
    def _extract_landmarks(self, annotation: Dict) -> List[List[float]]:
        """Extract landmarks from annotation"""
        
        # Try different common field names
        landmark_fields = ['landmarks', 'points', 'coordinates', 'keypoints']
        
        for field in landmark_fields:
            if field in annotation:
                landmarks = annotation[field]
                
                # Ensure we have exactly 7 landmarks
                if isinstance(landmarks, list) and len(landmarks) == 7:
                    # Validate each landmark has x, y coordinates
                    processed_landmarks = []
                    for landmark in landmarks:
                        if isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
                            processed_landmarks.append([float(landmark[0]), float(landmark[1])])
                        else:
                            raise ValueError(f"Invalid landmark format: {landmark}")
                    
                    return processed_landmarks
        
        raise ValueError("Could not find valid landmarks in annotation")
    
    def _validate_landmarks(self, landmarks: List[List[float]], image_shape: Tuple) -> bool:
        """Validate landmark coordinates"""
        
        h, w = image_shape[:2]
        
        for x, y in landmarks:
            # Check if coordinates are within image bounds
            if not (0 <= x <= w and 0 <= y <= h):
                return False
            
            # Check for valid numeric values
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return False
        
        return True
    
    def _quality_check_image(self, image: np.ndarray) -> bool:
        """Perform basic quality checks on image"""
        
        h, w = image.shape[:2]
        
        # Check minimum size
        if h < 256 or w < 256:
            return False
        
        # Check if image is too dark or too bright
        mean_intensity = np.mean(image)
        if mean_intensity < 20 or mean_intensity > 235:
            return False
        
        # Check contrast
        std_intensity = np.std(image)
        if std_intensity < 10:  # Very low contrast
            return False
        
        return True
    
    def _process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Process image to target size and format"""
        
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scale factors
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        # Resize image
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Optional: Apply histogram equalization for better contrast
        if len(resized_image.shape) == 3:
            # Convert to LAB color space for better histogram equalization
            lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
            resized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return resized_image, [scale_x, scale_y]
    
    def _scale_landmarks(self, landmarks: List[List[float]], scale_factors: List[float]) -> List[List[float]]:
        """Scale landmarks according to image resize factors"""
        
        scale_x, scale_y = scale_factors
        scaled_landmarks = []
        
        for x, y in landmarks:
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            scaled_landmarks.append([scaled_x, scaled_y])
        
        return scaled_landmarks
    
    def _save_annotations(self, annotations: List[Dict], output_file: Path):
        """Save processed annotations to JSON file"""
        
        output_data = {
            'dataset_info': {
                'num_samples': len(annotations),
                'num_landmarks': 7,
                'landmark_names': self.landmark_names,
                'image_size': self.target_size,
                'processing_date': pd.Timestamp.now().isoformat()
            },
            'samples': annotations
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Saved annotations: {output_file}")
    
    def create_train_val_test_splits(self, 
                                   annotation_file: str,
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   random_seed: int = 42) -> Dict:
        """
        Create train/validation/test splits from processed annotations
        
        Args:
            annotation_file: Path to processed annotation file
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Split statistics
        """
        
        # Validate ratios
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        np.random.seed(random_seed)
        
        # Shuffle samples
        indices = np.random.permutation(len(samples))
        
        # Calculate split sizes
        n_total = len(samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create splits
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Extract samples for each split
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        # Save split files
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            split_data = {
                'dataset_info': data['dataset_info'].copy(),
                'split': split_name,
                'samples': split_samples
            }
            split_data['dataset_info']['num_samples'] = len(split_samples)
            
            split_file = self.output_dir / 'annotations' / f'{split_name}_annotations.json'
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
        
        # Generate split statistics
        split_stats = {
            'total_samples': n_total,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
            'train_ratio': len(train_samples) / n_total,
            'val_ratio': len(val_samples) / n_total,
            'test_ratio': len(test_samples) / n_total,
            'random_seed': random_seed
        }
        
        logger.info(f"Created data splits:")
        logger.info(f"  Train: {len(train_samples)} samples ({len(train_samples)/n_total*100:.1f}%)")
        logger.info(f"  Val: {len(val_samples)} samples ({len(val_samples)/n_total*100:.1f}%)")
        logger.info(f"  Test: {len(test_samples)} samples ({len(test_samples)/n_total*100:.1f}%)")
        
        return split_stats


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description="Preprocess cephalometric datasets for MAHT-Net")
    
    parser.add_argument('--input-dir', required=True, help='Input directory containing raw data')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed data')
    parser.add_argument('--dataset-name', required=True, help='Name of the dataset')
    parser.add_argument('--annotation-format', default='auto', choices=['auto', 'json', 'csv', 'xml'],
                       help='Format of annotation files')
    parser.add_argument('--target-size', nargs=2, type=int, default=[512, 512],
                       help='Target image size (width height)')
    parser.add_argument('--no-quality-check', action='store_true',
                       help='Skip image quality checks')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = CephalometricDataProcessor(
            output_dir=args.output_dir,
            target_size=tuple(args.target_size),
            quality_check=not args.no_quality_check
        )
        
        # Process dataset
        stats = processor.process_dataset(
            input_dir=args.input_dir,
            dataset_name=args.dataset_name,
            annotation_format=args.annotation_format
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Successful: {stats['successful_samples']}")
        print(f"  Failed: {stats['failed_samples']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        # Create splits if requested
        if args.create_splits:
            annotation_file = Path(args.output_dir) / 'annotations' / f'{args.dataset_name}_processed.json'
            split_stats = processor.create_train_val_test_splits(
                annotation_file=str(annotation_file),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )
            
            print(f"\nData splits created:")
            print(f"  Train: {split_stats['train_samples']} samples")
            print(f"  Val: {split_stats['val_samples']} samples")
            print(f"  Test: {split_stats['test_samples']} samples")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
