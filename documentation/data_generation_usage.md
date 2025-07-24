# MAHT-Net Data Generation Usage Examples

## Basic Usage
```bash
python scripts/generate_training_data.py \
    --input-dir data/raw/ISBI_2015 \
    --output-dir data/processed/maht_net_512 \
    --augmentation \
    --visualize
```

## Custom Configuration
```bash
python scripts/generate_training_data.py \
    --input-dir data/raw/ISBI_2015 \
    --output-dir data/processed/maht_net_1024 \
    --target-size 1024 1024 \
    --heatmap-size 256 256 \
    --sigma 3.0 \
    --augmentation \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

## High Resolution Processing
```bash
python scripts/generate_training_data.py \
    --input-dir data/raw/ISBI_2015 \
    --output-dir data/processed/maht_net_hires \
    --target-size 768 768 \
    --heatmap-size 192 192 \
    --sigma 2.5 \
    --augmentation \
    --visualize \
    --verbose
```

## Expected Output Structure
```
data/processed/maht_net_512/
├── images/                 # Processed images
│   ├── sample_000001.png
│   ├── sample_000002.png
│   └── ...
├── heatmaps/              # Generated heatmaps
│   ├── sample_000001.npy
│   ├── sample_000002.npy
│   └── ...
├── visualizations/        # Quality visualizations
│   ├── pipeline_overview.png
│   ├── dataset_statistics.png
│   └── sample_*.png
├── train_annotations.json # Training split
├── val_annotations.json   # Validation split
└── test_annotations.json  # Test split
```

## Integration with Training
```python
from src.data.dataset import CephalometricDataset
from torch.utils.data import DataLoader

# Load processed dataset
train_dataset = CephalometricDataset(
    data_dir='data/processed/maht_net_512',
    split='train',
    image_size=(512, 512),
    heatmap_size=(128, 128)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)
```
