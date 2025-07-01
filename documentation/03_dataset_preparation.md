# Dataset Preparation: ISBI 2015 Cephalometric Challenge

## Dataset Overview

### ISBI 2015 Challenge Dataset
- **Source**: IEEE International Symposium on Biomedical Imaging 2015
- **Total Images**: 400 lateral cephalometric radiographs
- **Landmarks**: 7 anatomical points per image
- **Image Format**: Various (DICOM, PNG, JPEG)
- **Resolution**: Typically 2048x2048 or similar high resolution
- **Age Range**: Mixed pediatric and adult subjects

### Landmark Points (7 Critical Points)
1. **Sella (S)** - Center of sella turcica
2. **Nasion (N)** - Most anterior point on frontonasal suture
3. **A-point (A)** - Deepest point on maxillary anterior alveolar process
4. **B-point (B)** - Deepest point on mandibular anterior alveolar process
5. **Pogonion (Pog)** - Most anterior point on chin
6. **Menton (Me)** - Most inferior point on mandibular symphysis
7. **Gnathion (Gn)** - Most anterior-inferior point on mandibular symphysis

## Data Acquisition and Setup

### Step 1: Download and Organize ISBI 2015 Dataset

**What We'll Do**:
```bash
# Create comprehensive data directory structure
mkdir -p data/{raw,processed,annotations,splits,augmented,cache,backup}
mkdir -p data/raw/{images,landmarks,metadata}
mkdir -p data/processed/{images,heatmaps,coordinates}
mkdir -p data/splits/{train,val,test}

# Download ISBI 2015 dataset (adjust URL based on actual source)
# Note: You may need to request access from ISBI organizers
echo "Downloading ISBI 2015 Cephalometric Dataset..."
wget -O data/raw/isbi2015_cephalometric.zip "DATASET_URL"
unzip data/raw/isbi2015_cephalometric.zip -d data/raw/

# Organize files into proper structure
python scripts/organize_raw_data.py --input data/raw --output data/raw/organized
```

**Expected Outcomes**:
- 400 lateral cephalometric radiographs properly organized
- Landmark annotations in standardized format
- Metadata files for patient demographics (if available)
- Quality assessment reports for each image

### Step 2: Comprehensive Data Quality Assessment

**What We'll Analyze**:
```bash
# Run comprehensive data inspection
python scripts/inspect_dataset.py --data-dir data/raw/organized --output data/quality_report.json

# Generate visual quality assessment
python scripts/visualize_dataset.py --data-dir data/raw/organized --output data/quality_plots/
```

**Quality Checks We'll Perform**:
- **Image Resolution Analysis**: Check for consistent dimensions and DPI
- **Contrast and Brightness Assessment**: Identify images needing enhancement
- **Landmark Annotation Validation**: Verify all 7 landmarks are present
- **Anatomical Consistency Check**: Detect potential annotation errors
- **File Format Standardization**: Convert all images to consistent format

**Success Criteria**:
- All 400 images successfully loaded and processed
- Complete landmark annotations (7 points × 400 images = 2,800 points)
- Image quality scores above acceptable threshold (>6/10)
- No missing or corrupted files

## Advanced Data Preprocessing Pipeline

### Step 3: Image Standardization and Enhancement

**What We'll Implement**:
1. **Image Format Standardization**
   - Convert all images to PNG format with 16-bit depth
   - Standardize image orientation and coordinate systems
   - Remove DICOM metadata while preserving image quality
   - Validate image integrity after conversion

2. **Resolution and Sizing Strategy**
   - Analyze optimal input resolution for model performance
   - Implement aspect-ratio preserving resize to 512×512
   - Create multi-scale versions (256×256, 512×512, 1024×1024)
   - Maintain landmark coordinate accuracy during resizing

3. **Contrast and Quality Enhancement**
   - Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
   - Implement noise reduction using bilateral filtering
   - Enhance bone structure visibility while preserving soft tissue
   - Create quality-enhanced versions for training

4. **Preprocessing Validation**
   - Visual quality assessment before/after preprocessing
   - Landmark coordinate validation after transformations
   - Statistical analysis of image properties post-processing
   - Generate preprocessing quality reports

**Implementation Commands**:
```bash
# Run complete preprocessing pipeline
python scripts/preprocess_images.py \
    --input data/raw/organized \
    --output data/processed \
    --target-size 512 \
    --enhance-contrast \
    --validate-landmarks

# Generate preprocessing reports
python scripts/preprocessing_report.py \
    --original data/raw/organized \
    --processed data/processed \
    --output reports/preprocessing_analysis.html
```

### Step 4: Heatmap Generation for Ground Truth

**What We'll Create**:
1. **Gaussian Heatmap Generation**
   - Generate 2D Gaussian heatmaps for each landmark
   - Optimize sigma values for different landmark types
   - Create multi-scale heatmaps for FPN training
   - Validate heatmap-to-coordinate conversion accuracy

2. **Heatmap Quality Optimization**
   - Test different Gaussian sigma values (1.0, 1.5, 2.0, 2.5)
   - Implement adaptive sigma based on image resolution
   - Create normalized heatmaps with proper scaling
   - Generate confidence masks for uncertain landmarks

3. **Multi-Resolution Heatmap Creation**
   - Create heatmaps at 128×128, 256×256, 512×512 resolutions
   - Ensure consistency across different scales
   - Implement downsampling strategies that preserve peak locations
   - Validate coordinate extraction from multi-scale heatmaps

**Heatmap Generation Process**:
```bash
# Generate ground truth heatmaps
python scripts/generate_heatmaps.py \
    --landmarks data/processed/coordinates \
    --output data/processed/heatmaps \
    --sigma 1.5 \
    --multi-scale \
    --validate

# Validate heatmap quality
python scripts/validate_heatmaps.py \
    --heatmaps data/processed/heatmaps \
    --coordinates data/processed/coordinates \
    --output reports/heatmap_validation.json
```

### Step 5: Data Splitting and Cross-Validation Setup

**What We'll Organize**:
1. **Stratified Splitting Strategy**
   - Ensure balanced representation across age groups
   - Maintain equal landmark difficulty distribution
   - Preserve anatomical variation in each split
   - Account for potential patient-level clustering

2. **Cross-Validation Framework**
   - Implement 5-fold stratified cross-validation
   - Create patient-level splits to avoid data leakage
   - Generate reproducible splits with fixed random seeds
   - Validate split quality and balance

3. **Test Set Isolation**
   - Reserve 20% of data for final testing (80 images)
   - Ensure test set represents full data distribution
   - Keep test set completely separate until final evaluation
   - Create multiple test scenarios (easy, medium, hard cases)

**Splitting Implementation**:
```bash
# Create stratified data splits
python scripts/create_data_splits.py \
    --data-dir data/processed \
    --output data/splits \
    --cv-folds 5 \
    --test-ratio 0.2 \
    --stratify-by age_group,landmark_difficulty \
    --seed 42

# Validate split quality
python scripts/validate_splits.py \
    --splits-dir data/splits \
    --output reports/split_analysis.json
```

### Step 6: Advanced Data Augmentation Pipeline

**What We'll Implement**:
1. **Medical Image Specific Augmentations**
   - Realistic geometric transformations (rotation ±15°, scaling 0.8-1.2)
   - X-ray specific intensity modifications (brightness, contrast, gamma)
   - Anatomically plausible deformations
   - Equipment simulation (different X-ray machines, exposures)

2. **Progressive Augmentation Strategy**
   - Light augmentation for Stage 1 training
   - Medium augmentation for Stage 2 training
   - Heavy augmentation for Stage 3 robust training
   - Curriculum-based augmentation difficulty

3. **Quality-Preserving Augmentation**
   - Maintain landmark annotation accuracy
   - Preserve anatomical relationships
   - Avoid unrealistic image distortions
   - Validate augmented samples for clinical plausibility

**Augmentation Pipeline**:
```bash
# Generate augmented training data
python scripts/generate_augmentations.py \
    --input data/splits/train \
    --output data/augmented \
    --augmentation-factor 3 \
    --strategy progressive \
    --validate-landmarks

# Create augmentation quality report
python scripts/augmentation_report.py \
    --original data/splits/train \
    --augmented data/augmented \
    --output reports/augmentation_analysis.html
```
import numpy as np
import torch
from typing import Tuple, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CephalometricPreprocessor:
    """Preprocessing pipeline for cephalometric radiographs"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.mean = 0.485  # ImageNet mean for grayscale
        self.std = 0.229   # ImageNet std for grayscale
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        image = image.astype(np.float32)
        
        # Handle different bit depths
        if image.max() > 1.0:
            if image.max() <= 255:
                image = image / 255.0
            elif image.max() <= 4095:  # 12-bit DICOM
                image = image / 4095.0
            else:  # 16-bit
                image = image / 65535.0
                
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        # Convert to uint8 for CLAHE
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8)
        
        return enhanced.astype(np.float32) / 255.0
    
    def resize_with_aspect_ratio(self, image: np.ndarray, 
                                landmarks: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Calculate padding offsets
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Adjust landmarks if provided
        if landmarks is not None:
            landmarks = landmarks.copy()
            landmarks[:, 0] = landmarks[:, 0] * scale + x_offset
            landmarks[:, 1] = landmarks[:, 1] * scale + y_offset
            
        return padded, landmarks
    
    def preprocess(self, image: np.ndarray, 
                  landmarks: np.ndarray = None) -> Dict:
        """Complete preprocessing pipeline"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Enhance contrast
        image = self.enhance_contrast(image)
        
        # Resize with aspect ratio preservation
        image, landmarks = self.resize_with_aspect_ratio(image, landmarks)
        
        # Convert to 3-channel for pretrained models
        image = np.stack([image, image, image], axis=0)
        
        # Normalize for ImageNet pretrained models
        image = (image - self.mean) / self.std
        
        result = {
            'image': torch.FloatTensor(image),
            'landmarks': torch.FloatTensor(landmarks) if landmarks is not None else None
        }
        
        return result
```

### Step 4: Heatmap Generation

```python
# src/datasets/heatmap_generator.py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from typing import Tuple

class HeatmapGenerator:
    """Generate Gaussian heatmaps for landmark annotations"""
    
    def __init__(self, output_size: Tuple[int, int] = (128, 128), sigma: float = 2.0):
        self.output_size = output_size
        self.sigma = sigma
    
    def generate_heatmap(self, landmarks: np.ndarray, 
                        input_size: Tuple[int, int]) -> np.ndarray:
        """Generate heatmaps for all landmarks"""
        h_out, w_out = self.output_size
        h_in, w_in = input_size
        
        # Calculate scaling factors
        scale_x = w_out / w_in
        scale_y = h_out / h_in
        
        # Initialize heatmaps
        heatmaps = np.zeros((len(landmarks), h_out, w_out), dtype=np.float32)
        
        for i, (x, y) in enumerate(landmarks):
            # Scale coordinates
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            
            # Create Gaussian heatmap
            heatmap = self._create_gaussian_heatmap(
                (int(x_scaled), int(y_scaled)), 
                (h_out, w_out)
            )
            
            heatmaps[i] = heatmap
        
        return heatmaps
    
    def _create_gaussian_heatmap(self, center: Tuple[int, int], 
                                size: Tuple[int, int]) -> np.ndarray:
        """Create single Gaussian heatmap"""
        x_center, y_center = center
        h, w = size
        
        # Create coordinate grids
        x = np.arange(0, w, 1, np.float32)
        y = np.arange(0, h, 1, np.float32)
        y = y[:, np.newaxis]
        
        # Calculate Gaussian
        heatmap = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        
        return heatmap
    
    def extract_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract coordinates from heatmaps using soft argmax"""
        batch_size, num_landmarks, h, w = heatmaps.shape
        
        # Create coordinate grids
        x_coords = torch.arange(w, dtype=torch.float32, device=heatmaps.device)
        y_coords = torch.arange(h, dtype=torch.float32, device=heatmaps.device)
        
        # Reshape for broadcasting
        x_coords = x_coords.view(1, 1, 1, w)
        y_coords = y_coords.view(1, 1, h, 1)
        
        # Normalize heatmaps
        heatmaps_norm = heatmaps / (heatmaps.sum(dim=[2, 3], keepdim=True) + 1e-8)
        
        # Calculate weighted coordinates
        x_pred = (heatmaps_norm * x_coords).sum(dim=[2, 3])
        y_pred = (heatmaps_norm * y_coords).sum(dim=[2, 3])
        
        # Stack coordinates
        coordinates = torch.stack([x_pred, y_pred], dim=-1)
        
        return coordinates
```

### Step 5: Data Augmentation Strategy

```python
# src/datasets/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CephalometricAugmentation:
    """Augmentation pipeline specifically designed for cephalometric images"""
    
    def __init__(self, training: bool = True):
        self.training = training
        
        if training:
            self.transform = A.Compose([
                # Geometric transformations
                A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=10, 
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                
                # Intensity transformations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.6
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                
                # Elastic deformation (crucial for medical images)
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50, 
                    alpha_affine=50, 
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                
                # Grid distortion
                A.GridDistortion(p=0.3),
                
                # Optical distortion
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
                
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            # Validation/test: minimal augmentation
            self.transform = A.Compose([
                # Only normalization for validation
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __call__(self, image: np.ndarray, landmarks: np.ndarray):
        """Apply augmentation"""
        # Convert landmarks to albumentations format
        keypoints = [(x, y) for x, y in landmarks]
        
        # Apply transformations
        transformed = self.transform(image=image, keypoints=keypoints)
        
        # Extract results
        aug_image = transformed['image']
        aug_landmarks = np.array(transformed['keypoints'])
        
        return aug_image, aug_landmarks
```

### Step 6: Dataset Class Implementation

```python
# src/datasets/cephalometric_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class CephalometricDataset(Dataset):
    """PyTorch Dataset for ISBI 2015 Cephalometric data"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 target_size: Tuple[int, int] = (512, 512),
                 heatmap_size: Tuple[int, int] = (128, 128)):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.heatmap_size = heatmap_size
        
        # Initialize processors
        self.preprocessor = CephalometricPreprocessor(target_size)
        self.heatmap_generator = HeatmapGenerator(heatmap_size)
        
        # Load metadata
        self.samples = self._load_dataset_info()
        
    def _load_dataset_info(self) -> pd.DataFrame:
        """Load dataset information and create train/val/test splits"""
        
        # Load image files
        image_dir = self.data_dir / "images"
        image_files = sorted(list(image_dir.glob("*")))
        
        # Load annotations
        annotation_file = self.data_dir / "landmarks.csv"  # Adjust based on actual format
        
        if annotation_file.exists():
            annotations = pd.read_csv(annotation_file)
        else:
            # Create annotations from individual files if needed
            annotations = self._create_annotations_dataframe(image_files)
        
        # Create splits (80/10/10 for train/val/test)
        np.random.seed(42)
        indices = np.random.permutation(len(annotations))
        
        train_idx = indices[:int(0.8 * len(indices))]
        val_idx = indices[int(0.8 * len(indices)):int(0.9 * len(indices))]
        test_idx = indices[int(0.9 * len(indices)):]
        
        split_mapping = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        return annotations.iloc[split_mapping[self.split]].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        
        # Load image
        image_path = self.samples.iloc[idx]['image_path']
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Load landmarks
        landmarks = self._extract_landmarks(idx)
        
        # Apply augmentation if in training mode
        if self.transform and self.split == 'train':
            image, landmarks = self.transform(image, landmarks)
        
        # Preprocess
        processed = self.preprocessor.preprocess(image, landmarks)
        
        # Generate heatmaps
        heatmaps = self.heatmap_generator.generate_heatmap(
            landmarks, self.target_size
        )
        
        return {
            'image': processed['image'],
            'landmarks': processed['landmarks'],
            'heatmaps': torch.FloatTensor(heatmaps),
            'image_id': self.samples.iloc[idx]['image_id']
        }
    
    def _extract_landmarks(self, idx: int) -> np.ndarray:
        """Extract landmark coordinates for a sample"""
        row = self.samples.iloc[idx]
        
        landmarks = []
        for i in range(7):  # 7 landmarks
            x = row[f'landmark_{i}_x']
            y = row[f'landmark_{i}_y']
            landmarks.append([x, y])
        
        return np.array(landmarks, dtype=np.float32)
```

### Step 7: Data Validation and Quality Checks

```python
# src/datasets/validation.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

def validate_dataset(dataset: CephalometricDataset, num_samples: int = 5):
    """Validate dataset integrity and visualize samples"""
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        image = batch['image'][0]
        heatmaps = batch['heatmaps'][0]
        landmarks = batch['landmarks'][0]
        
        # Display original image
        img_display = image[0].cpu().numpy()  # Take first channel
        axes[0, i].imshow(img_display, cmap='gray')
        axes[0, i].scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20)
        axes[0, i].set_title(f'Sample {i+1}')
        axes[0, i].axis('off')
        
        # Display heatmap overlay
        heatmap_sum = heatmaps.sum(dim=0).cpu().numpy()
        axes[1, i].imshow(img_display, cmap='gray', alpha=0.7)
        axes[1, i].imshow(heatmap_sum, cmap='hot', alpha=0.3)
        axes[1, i].set_title('Heatmap Overlay')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/processed/dataset_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print dataset statistics
    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Heatmap shape: {batch['heatmaps'].shape}")
    print(f"Landmarks shape: {batch['landmarks'].shape}")

def check_data_distribution(train_dataset, val_dataset, test_dataset):
    """Analyze data distribution across splits"""
    
    splits = {
        'Train': train_dataset,
        'Validation': val_dataset,
        'Test': test_dataset
    }
    
    print("Dataset Split Distribution:")
    print("-" * 30)
    for name, dataset in splits.items():
        print(f"{name}: {len(dataset)} samples")
    
    # Landmark distribution analysis
    all_landmarks = []
    for dataset in splits.values():
        for i in range(len(dataset)):
            sample = dataset[i]
            all_landmarks.append(sample['landmarks'].numpy())
    
    all_landmarks = np.array(all_landmarks)
    
    print(f"\nLandmark Statistics:")
    print(f"Mean X: {all_landmarks[:, :, 0].mean():.2f}")
    print(f"Mean Y: {all_landmarks[:, :, 1].mean():.2f}")
    print(f"Std X: {all_landmarks[:, :, 0].std():.2f}")
    print(f"Std Y: {all_landmarks[:, :, 1].std():.2f}")
```

## Data Quality Assurance

### Step 8: Quality Control Script

```python
# quality_control.py
def run_quality_control():
    """Comprehensive quality control for the dataset"""
    
    # Load datasets
    train_dataset = CephalometricDataset('data/raw', split='train')
    val_dataset = CephalometricDataset('data/raw', split='val')
    test_dataset = CephalometricDataset('data/raw', split='test')
    
    # Validate datasets
    print("Running dataset validation...")
    validate_dataset(train_dataset)
    
    # Check distributions
    check_data_distribution(train_dataset, val_dataset, test_dataset)
    
    # Check for data leakage
    train_ids = set([train_dataset.samples.iloc[i]['image_id'] for i in range(len(train_dataset))])
    val_ids = set([val_dataset.samples.iloc[i]['image_id'] for i in range(len(val_dataset))])
    test_ids = set([test_dataset.samples.iloc[i]['image_id'] for i in range(len(test_dataset))])
    
    assert len(train_ids & val_ids) == 0, "Data leakage between train and validation"
    assert len(train_ids & test_ids) == 0, "Data leakage between train and test"
    assert len(val_ids & test_ids) == 0, "Data leakage between validation and test"
    
    print("✓ No data leakage detected")
    print("✓ Dataset preparation completed successfully")

if __name__ == "__main__":
    run_quality_control()
```

## Next Steps

After completing dataset preparation:

1. **Verify Data Integrity**: Run the quality control script
2. **Baseline Performance**: Test with simple models to establish baseline
3. **Proceed to Architecture Design**: Move to [Architecture Design](04_architecture_design.md)
4. **Monitor Data Pipeline**: Ensure efficient loading during training

## Key Considerations

- **Memory Management**: Large images may require batch size reduction
- **Augmentation Balance**: Too much augmentation can hurt landmark precision
- **Heatmap Resolution**: Balance between accuracy and computational efficiency
- **Cross-validation**: Consider k-fold validation due to limited dataset size
