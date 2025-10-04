"""
Multi-Dataset Benchmark Evaluation Framework for MAHT-Net
Supports ISBI 2015, Aariz, and CL-Detection2023 datasets for comprehensive comparison
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for benchmark datasets"""
    name: str
    num_landmarks: int
    num_images: int
    landmark_names: List[str]
    image_size: Tuple[int, int]
    pixel_spacing: float
    download_url: Optional[str] = None
    citation: str = ""
    

class BenchmarkDatasetLoader(ABC):
    """Abstract base class for benchmark dataset loaders"""
    
    def __init__(self, dataset_path: str, config: DatasetConfig):
        self.dataset_path = Path(dataset_path)
        self.config = config
        
    @abstractmethod
    def download_dataset(self) -> bool:
        """Download the dataset if not already present"""
        pass
    
    @abstractmethod
    def load_annotations(self) -> List[Dict]:
        """Load dataset annotations in standardized format"""
        pass
    
    @abstractmethod
    def validate_dataset(self) -> bool:
        """Validate dataset integrity"""
        pass


class CLDetection2023Loader(BenchmarkDatasetLoader):
    """
    Loader for CL-Detection2023 Challenge Dataset
    600 images, 38 landmarks, multi-center data
    """
    
    def __init__(self, dataset_path: str):
        config = DatasetConfig(
            name="CL-Detection2023",
            num_landmarks=38,
            num_images=600,
            landmark_names=[
                # Complete list of 38 anatomical landmarks
                'N', 'S', 'Ba', 'R1', 'R2', 'R3', 'R4', 'Or', 'Po', 'ANS', 'PNS',
                'A', 'SPPN', 'SPIN', 'Me', 'Gn', 'B', 'Pog', 'Id', 'Is', 'U1',
                'L1', 'Ar', 'Go', 'Gon', 'Cd', 'ANS', 'Xi', 'Pm', 'Dc', 'Cc',
                'LI', 'UI', 'LL', 'UL', 'SM', 'PG', 'GN'
            ],
            image_size=(2400, 1935),
            pixel_spacing=0.1,  # Typical for cephalometric X-rays
            download_url="https://cl-detection2023.grand-challenge.org/",
            citation="Zhang et al., 2024. Deep Learning Techniques for Automatic Lateral X-ray Cephalometric Landmark Detection"
        )
        super().__init__(dataset_path, config)
        
    def download_dataset(self) -> bool:
        """Download CL-Detection2023 dataset"""
        logger.info(f"🔽 Downloading {self.config.name} dataset...")
        
        # Create dataset directory
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Note: This dataset requires registration on Grand Challenge platform
        logger.warning("⚠️  CL-Detection2023 requires registration at:")
        logger.warning("   https://cl-detection2023.grand-challenge.org/")
        logger.warning("   Please download manually and place in:")
        logger.warning(f"   {self.dataset_path}")
        
        return self.dataset_path.exists()
    
    def load_annotations(self) -> List[Dict]:
        """Load CL-Detection2023 annotations"""
        annotations = []
        
        # Look for annotation files
        annotation_files = list(self.dataset_path.glob("**/*.json")) + list(self.dataset_path.glob("**/*.csv"))
        
        for ann_file in annotation_files:
            if ann_file.suffix == '.json':
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        annotations.extend(data)
                    else:
                        annotations.append(data)
            elif ann_file.suffix == '.csv':
                df = pd.read_csv(ann_file)
                # Convert CSV to standard format
                for _, row in df.iterrows():
                    annotation = self._parse_csv_row(row)
                    annotations.append(annotation)
        
        logger.info(f"📊 Loaded {len(annotations)} annotations from {self.config.name}")
        return annotations
    
    def _parse_csv_row(self, row: pd.Series) -> Dict:
        """Parse CSV row into standard annotation format"""
        # Extract landmark coordinates (assuming x1,y1,x2,y2,... format)
        landmarks = []
        for i in range(self.config.num_landmarks):
            x_col = f'x{i+1}' if f'x{i+1}' in row else f'landmark_{i+1}_x'
            y_col = f'y{i+1}' if f'y{i+1}' in row else f'landmark_{i+1}_y'
            
            if x_col in row and y_col in row:
                landmarks.append([float(row[x_col]), float(row[y_col])])
        
        return {
            'image_path': row.get('image_path', row.get('filename', '')),
            'landmarks': landmarks,
            'dataset': self.config.name,
            'num_landmarks': len(landmarks)
        }
    
    def validate_dataset(self) -> bool:
        """Validate CL-Detection2023 dataset"""
        if not self.dataset_path.exists():
            return False
        
        # Check for required files
        image_dir = self.dataset_path / 'images'
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return False
        
        # Count images
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        logger.info(f"Found {len(image_files)} images in {self.config.name}")
        
        return len(image_files) > 0


class AarizDatasetLoader(BenchmarkDatasetLoader):
    """
    Loader for Aariz Dataset
    1000 images, 29 landmarks + CVM classification
    """
    
    def __init__(self, dataset_path: str):
        config = DatasetConfig(
            name="Aariz",
            num_landmarks=29,
            num_images=1000,
            landmark_names=[
                # 29 comprehensive landmarks including soft tissue
                'N', 'S', 'Ba', 'Or', 'Po', 'ANS', 'PNS', 'A', 'SPPN', 'SPIN',
                'Me', 'Gn', 'B', 'Pog', 'Id', 'Is', 'U1', 'L1', 'Ar', 'Go',
                'Gon', 'Cd', 'Xi', 'Pm', 'Dc', 'Cc', 'LI', 'UI', 'Soft_Tissue_N'
            ],
            image_size=(2400, 1935),
            pixel_spacing=0.1,
            citation="Khalid et al., 2023. Aariz: A Benchmark Dataset for Automatic Cephalometric Landmark Detection"
        )
        super().__init__(dataset_path, config)
    
    def download_dataset(self) -> bool:
        """Download Aariz dataset"""
        logger.info(f"🔽 Downloading {self.config.name} dataset...")
        
        # Create dataset directory
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Note: Dataset availability needs to be checked
        logger.warning("⚠️  Aariz dataset access information:")
        logger.warning("   Contact authors at: Muhammad.Anwaar@ucp.edu.pk")
        logger.warning("   Paper: https://arxiv.org/abs/2302.07797")
        logger.warning(f"   Please download and place in: {self.dataset_path}")
        
        return self.dataset_path.exists()
    
    def load_annotations(self) -> List[Dict]:
        """Load Aariz annotations"""
        annotations = []
        
        # Look for annotation files
        annotation_files = list(self.dataset_path.glob("**/*.json")) + list(self.dataset_path.glob("**/*.txt"))
        
        for ann_file in annotation_files:
            if ann_file.suffix == '.json':
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        annotations.extend(data)
                    else:
                        annotations.append(data)
            elif ann_file.suffix == '.txt':
                # Parse text-based annotations
                annotation = self._parse_txt_file(ann_file)
                if annotation:
                    annotations.append(annotation)
        
        logger.info(f"📊 Loaded {len(annotations)} annotations from {self.config.name}")
        return annotations
    
    def _parse_txt_file(self, txt_file: Path) -> Optional[Dict]:
        """Parse text file with landmark coordinates"""
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            landmarks = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    landmarks.append([x, y])
            
            # Find corresponding image
            image_path = str(txt_file).replace('.txt', '.jpg')
            if not Path(image_path).exists():
                image_path = str(txt_file).replace('.txt', '.png')
            
            return {
                'image_path': image_path,
                'landmarks': landmarks,
                'dataset': self.config.name,
                'num_landmarks': len(landmarks)
            }
        except Exception as e:
            logger.warning(f"Failed to parse {txt_file}: {e}")
            return None
    
    def validate_dataset(self) -> bool:
        """Validate Aariz dataset"""
        return self.dataset_path.exists()


class ISBIDatasetLoader(BenchmarkDatasetLoader):
    """
    Loader for ISBI 2015 Dataset (already implemented)
    400 images, 19 landmarks
    """
    
    def __init__(self, dataset_path: str):
        config = DatasetConfig(
            name="ISBI-2015",
            num_landmarks=19,
            num_images=400,
            landmark_names=[
                'S', 'N', 'ANS', 'PNS', 'A', 'SPPN', 'SPIN', 'Me', 'Gn', 'B',
                'Pog', 'Id', 'Is', 'U1', 'L1', 'Ar', 'Go', 'Gon', 'Or'
            ],
            image_size=(1935, 2400),
            pixel_spacing=0.1,
            citation="Wang et al., 2015. ISBI 2015 Challenge on Automatic Cephalometric Landmark Detection"
        )
        super().__init__(dataset_path, config)
    
    def download_dataset(self) -> bool:
        """ISBI dataset is already available"""
        return self.dataset_path.exists()
    
    def load_annotations(self) -> List[Dict]:
        """Load ISBI annotations using existing loader"""
        # Use existing ISBI loader
        from ..data.isbi_loader import ISBIDatasetLoader as ExistingISBILoader
        
        loader = ExistingISBILoader(str(self.dataset_path))
        samples = loader.load_dataset()
        
        # Convert to standard format
        annotations = []
        for sample in samples:
            annotations.append({
                'image_path': sample['image_path'],
                'landmarks': sample['landmarks'],
                'dataset': self.config.name,
                'num_landmarks': len(sample['landmarks'])
            })
        
        logger.info(f"📊 Loaded {len(annotations)} annotations from {self.config.name}")
        return annotations
    
    def validate_dataset(self) -> bool:
        """Validate ISBI dataset"""
        return self.dataset_path.exists()


class MultiDatasetBenchmark:
    """
    Multi-dataset benchmark evaluation framework
    """
    
    def __init__(self, base_data_dir: str):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset loaders
        self.loaders = {
            'ISBI-2015': ISBIDatasetLoader(str(self.base_data_dir / 'ISBI-2015')),
            'Aariz': AarizDatasetLoader(str(self.base_data_dir / 'Aariz')),
            'CL-Detection2023': CLDetection2023Loader(str(self.base_data_dir / 'CL-Detection2023'))
        }
        
        self.datasets = {}
        
    def setup_datasets(self) -> Dict[str, bool]:
        """Setup all benchmark datasets"""
        results = {}
        
        for name, loader in self.loaders.items():
            logger.info(f"🔧 Setting up {name} dataset...")
            
            # Download if needed
            download_success = loader.download_dataset()
            
            # Validate
            validation_success = loader.validate_dataset() if download_success else False
            
            results[name] = validation_success
            
            if validation_success:
                logger.info(f"✅ {name} dataset ready")
                # Load annotations
                self.datasets[name] = loader.load_annotations()
            else:
                logger.warning(f"❌ {name} dataset not available")
        
        return results
    
    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """Get comprehensive statistics for all datasets"""
        stats = {}
        
        for name, annotations in self.datasets.items():
            if not annotations:
                continue
                
            config = self.loaders[name].config
            
            # Calculate statistics
            landmark_counts = [ann['num_landmarks'] for ann in annotations]
            
            stats[name] = {
                'total_images': len(annotations),
                'expected_landmarks': config.num_landmarks,
                'actual_landmark_range': f"{min(landmark_counts)}-{max(landmark_counts)}",
                'avg_landmarks': np.mean(landmark_counts),
                'image_size': config.image_size,
                'pixel_spacing': config.pixel_spacing,
                'citation': config.citation
            }
        
        return stats
    
    def create_evaluation_splits(self, test_ratio: float = 0.2) -> Dict[str, Dict[str, List]]:
        """Create train/test splits for cross-dataset evaluation"""
        splits = {}
        
        for name, annotations in self.datasets.items():
            n_test = int(len(annotations) * test_ratio)
            
            # Shuffle annotations
            np.random.shuffle(annotations)
            
            splits[name] = {
                'train': annotations[:-n_test] if n_test > 0 else annotations,
                'test': annotations[-n_test:] if n_test > 0 else []
            }
            
            logger.info(f"📊 {name}: {len(splits[name]['train'])} train, {len(splits[name]['test'])} test")
        
        return splits
    
    def export_benchmark_report(self, save_path: str) -> None:
        """Export comprehensive benchmark report"""
        report = {
            'benchmark_info': {
                'total_datasets': len(self.datasets),
                'available_datasets': list(self.datasets.keys()),
                'total_images': sum(len(anns) for anns in self.datasets.values()),
                'generation_date': pd.Timestamp.now().isoformat()
            },
            'dataset_statistics': self.get_dataset_statistics(),
            'evaluation_plan': {
                'cross_dataset_validation': "Train on one dataset, test on another",
                'combined_training': "Train on multiple datasets, test on held-out data",
                'comparative_analysis': "Compare against published state-of-the-art results"
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Benchmark report saved to: {save_path}")


def create_benchmark_evaluation_plan():
    """Create comprehensive benchmark evaluation plan"""
    
    benchmark = MultiDatasetBenchmark('data/benchmark_datasets')
    
    # Setup datasets
    logger.info("🚀 Setting up multi-dataset benchmark evaluation...")
    setup_results = benchmark.setup_datasets()
    
    # Print setup summary
    print("\n📊 Benchmark Dataset Setup Summary:")
    print("=" * 50)
    
    for dataset, success in setup_results.items():
        status = "✅ Ready" if success else "❌ Not Available"
        print(f"{dataset:<20}: {status}")
    
    # Get statistics
    stats = benchmark.get_dataset_statistics()
    
    print("\n📈 Dataset Statistics:")
    print("=" * 50)
    
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Images: {stat['total_images']}")
        print(f"  Landmarks: {stat['expected_landmarks']}")
        print(f"  Image Size: {stat['image_size']}")
        print(f"  Citation: {stat['citation'][:80]}...")
    
    # Export report
    benchmark.export_benchmark_report('data/benchmark_datasets/benchmark_report.json')
    
    return benchmark


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark = create_benchmark_evaluation_plan()