"""
Configuration Management System for MAHT-Net

This module provides comprehensive configuration management using dataclasses
and YAML files for all aspects of the MAHT-Net project.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import yaml
import os

@dataclass
class DataConfig:
    """Configuration for data processing and augmentation"""

    # Dataset paths
    dataset_path: str = "data"  # Root dataset directory
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    dataset_name: str = "isbi_2015"    # Image specifications
    image_size: Tuple[int, int] = (256, 256)
    original_size: Tuple[int, int] = (1935, 2400)
    num_landmarks: int = 19
    pixels_per_mm: float = 10.0

    # Heatmap generation (inspired by proven success)
    heatmap_size: Tuple[int, int] = (256, 256)
    heatmap_sigma: float = 5.0  # Alias for gaussian_sigma
    gaussian_sigma: float = 5.0
    gaussian_amplitude: float = 1000.0
    heatmap_amplitude: float = 1000.0  # Alias for gaussian_amplitude

    # Data splits
    train_split: float = 0.85
    val_split: float = 0.15
    random_seed: int = 42

    # Augmentation parameters (inspired by legacy success)
    use_augmentation: bool = True
    elastic_sigma: float = 10.0
    elastic_alpha: float = 15.0
    affine_rotation: float = 5.0
    affine_scale_range: Tuple[float, float] = (0.95, 1.05)
    affine_translation: float = 0.03
    horizontal_flip: bool = False  # Avoid for medical symmetry

    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.train_split + self.val_split == 1.0, "Train and validation splits must sum to 1.0"
        assert self.num_landmarks > 0, "Number of landmarks must be positive"
        assert self.gaussian_sigma > 0, "Gaussian sigma must be positive"


@dataclass
class ModelConfig:
    """Configuration for MAHT-Net model architecture"""

    # Input configuration
    input_channels: int = 1  # Grayscale X-ray images
    
    # Model type
    model_name: str = "maht_net"

    # CNN Encoder configuration
    encoder_type: str = "unet_encoder"  # Start with U-Net inspired encoder
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    encoder_dropout: float = 0.4

    # Vision Transformer configuration
    use_transformer: bool = True
    transformer_dim: int = 1024
    transformer_heads: int = 12
    transformer_layers: int = 6
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.1

    # Decoder configuration
    decoder_type: str = "attention_fpn"
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    use_attention_gates: bool = True

    # Output configuration
    num_classes: int = 19  # Number of landmarks (alias for output_channels)
    output_channels: int = 19  # Number of landmarks
    use_coordinate_regression: bool = True
    use_uncertainty: bool = True

    def __post_init__(self):
        """Validate model configuration"""
        assert len(self.encoder_channels) > 0, "Encoder channels cannot be empty"
        assert self.transformer_heads > 0, "Transformer heads must be positive"
        assert self.output_channels > 0, "Output channels must be positive"


@dataclass
class TrainingConfig:
    """Configuration for training process"""

    # Basic training parameters (inspired by proven success)
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 200

    # Optimizer configuration
    optimizer: str = "adam"
    weight_decay: float = 0.0

    # Learning rate scheduling (inspired by legacy success)
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Loss function configuration
    primary_loss: str = "mse"
    use_auxiliary_losses: bool = True
    heatmap_loss_weight: float = 1.0
    coordinate_loss_weight: float = 0.5
    attention_loss_weight: float = 0.1

    # Progressive training strategy
    use_progressive_training: bool = True
    stage_1_epochs: int = 50  # Foundation training
    stage_2_epochs: int = 75  # Transformer integration
    stage_3_epochs: int = 75  # Full architecture training

    # Regularization
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 25

    # Training monitoring
    log_interval: int = 10
    save_interval: int = 5
    validate_interval: int = 1

    def __post_init__(self):
        """Validate training configuration"""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""

    # Evaluation metrics
    primary_metrics: List[str] = field(default_factory=lambda: ["mre", "sdr_2mm", "sdr_4mm"])
    sdr_thresholds: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 4.0])

    # Clinical thresholds (based on legacy insights)
    clinical_threshold_mm: float = 2.0
    excellent_threshold_mm: float = 1.5

    # Visualization settings
    save_attention_maps: bool = True
    save_prediction_images: bool = True
    max_visualization_samples: int = 20

    # Statistical analysis
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000

    def __post_init__(self):
        """Validate evaluation configuration"""
        assert all(t > 0 for t in self.sdr_thresholds), "SDR thresholds must be positive"
        assert 0 < self.confidence_level < 1, "Confidence level must be between 0 and 1"


@dataclass
class ExperimentConfig:
    """Main configuration class combining all components"""

    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: str = "maht_net_experiment"
    project_root: str = "/var/www/phd-researches/maht-net"
    device: str = "cuda"
    mixed_precision: bool = True
    random_seed: int = 42

    # Logging and output
    log_level: str = "INFO"
    output_dir: str = "experiments"
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "results"

    def __post_init__(self):
        """Validate and setup experiment configuration"""
        # Ensure output directories exist
        for dir_path in [self.output_dir, self.checkpoint_dir, self.results_dir]:
            Path(self.project_root) / dir_path

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert lists back to tuples where needed
        def convert_lists_to_tuples(obj, target_keys):
            """Convert specific list values back to tuples"""
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k in target_keys and isinstance(v, list):
                        result[k] = tuple(v)
                    elif isinstance(v, dict):
                        result[k] = convert_lists_to_tuples(v, target_keys)
                    else:
                        result[k] = v
                return result
            return obj

        # Define which keys should be converted back to tuples
        tuple_keys = {'image_size', 'original_size', 'heatmap_size', 'affine_scale_range'}
        config_dict = convert_lists_to_tuples(config_dict, tuple_keys)

        return cls(**config_dict)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict for YAML serialization with tuple handling
        def convert_tuples(obj):
            """Convert tuples to lists for YAML compatibility"""
            if isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tuples(item) for item in obj]
            else:
                return obj

        config_dict = {
            'data': convert_tuples(self.data.__dict__),
            'model': convert_tuples(self.model.__dict__),
            'training': convert_tuples(self.training.__dict__),
            'evaluation': convert_tuples(self.evaluation.__dict__),
            'experiment_name': self.experiment_name,
            'project_root': self.project_root,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'results_dir': self.results_dir
        }

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_experiment_dir(self) -> Path:
        """Get the full path to experiment directory"""
        return Path(self.project_root) / self.output_dir / self.experiment_name

    def get_checkpoint_dir(self) -> Path:
        """Get the full path to checkpoint directory"""
        return Path(self.project_root) / self.checkpoint_dir

    def get_results_dir(self) -> Path:
        """Get the full path to results directory"""
        return Path(self.project_root) / self.results_dir / self.experiment_name


def create_default_configs() -> None:
    """Create default configuration files for different experiment types"""

    base_dir = Path("configs")
    base_dir.mkdir(exist_ok=True)

    # Create different experiment configurations
    configs = {
        "baseline_unet.yaml": ExperimentConfig(
            experiment_name="baseline_unet",
            model=ModelConfig(
                model_name="unet_baseline",
                use_transformer=False,
                use_attention_gates=False
            ),
            training=TrainingConfig(
                use_progressive_training=False,
                num_epochs=100
            )
        ),
        "maht_net_stage1.yaml": ExperimentConfig(
            experiment_name="maht_net_stage1",
            model=ModelConfig(
                model_name="maht_net",
                use_transformer=False,  # Start without transformer
                use_attention_gates=True
            ),
            training=TrainingConfig(
                use_progressive_training=True,
                num_epochs=50
            )
        ),
        "maht_net_full.yaml": ExperimentConfig(
            experiment_name="maht_net_full",
            model=ModelConfig(
                model_name="maht_net",
                use_transformer=True,
                use_attention_gates=True,
                use_coordinate_regression=True,
                use_uncertainty=True
            ),
            training=TrainingConfig(
                use_progressive_training=True,
                num_epochs=200
            )
        )
    }

    # Save configuration files
    for filename, config in configs.items():
        config.to_yaml(base_dir / filename)
        print(f"Created configuration: {base_dir / filename}")


def load_experiment_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        ExperimentConfig object with loaded configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    print(f"Loading configuration from: {config_path}")

    # Create configuration object from YAML using the proper method
    config = ExperimentConfig.from_yaml(config_path)

    print(f"✅ Configuration loaded successfully!")
    return config


if __name__ == "__main__":
    # Create default configuration files
    create_default_configs()

    # Test configuration loading
    config = ExperimentConfig()
    print(f"Default configuration created successfully!")
    print(f"Experiment name: {config.experiment_name}")
    print(f"Model type: {config.model.model_name}")
    print(f"Batch size: {config.training.batch_size}")
