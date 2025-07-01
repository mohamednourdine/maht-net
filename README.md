# MAHT-Net: Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection

## Overview

This project implements MAHT-Net for cephalometric landmark detection on the ISBI 2015 dataset. The architecture combines the strengths of CNNs and Transformers in a hybrid approach to achieve state-of-the-art performance on 7 landmark points.

## Architecture

MAHT-Net combines the strengths of CNNs and Transformers in a hybrid architecture:
- **Encoder**: Multi-scale CNN feature extraction (ResNet/EfficientNet backbone)
- **Bottleneck**: Vision Transformer with self-attention mechanisms
- **Decoder**: Attention-gated upsampling with feature pyramid fusion
- **Output**: Heatmap regression for 7 landmark points

## Documentation

This project includes comprehensive documentation covering all aspects of implementation:

1. **[Project Overview](documentation/01_project_overview.md)** - Research context, objectives, and expected outcomes
2. **[Environment Setup](documentation/02_environment_setup.md)** - Python environment, dependencies, and development tools
3. **[Dataset Preparation](documentation/03_dataset_preparation.md)** - ISBI 2015 dataset handling and preprocessing
4. **[Architecture Design](documentation/04_architecture_design.md)** - Detailed MAHT-Net component specifications
5. **[Implementation Plan](documentation/05_implementation_plan.md)** - Step-by-step development roadmap
6. **[Training Strategy](documentation/06_training_strategy.md)** - Training pipeline, loss functions, and optimization
7. **[Evaluation Framework](documentation/07_evaluation_framework.md)** - Metrics, validation, and benchmarking
8. **[Ablation Studies](documentation/08_ablation_studies.md)** - Component-wise analysis and optimization
9. **[Troubleshooting Guide](documentation/09_troubleshooting_guide.md)** - Common issues and solutions
10. **[Clinical Integration](documentation/10_clinical_integration.md)** - Deployment and practical applications

## Quick Start

1. Review the [Project Overview](documentation/01_project_overview.md) to understand the research context
2. Set up your environment following [Environment Setup](documentation/02_environment_setup.md)
3. Prepare your dataset using [Dataset Preparation](documentation/03_dataset_preparation.md)
4. Follow the [Implementation Plan](documentation/05_implementation_plan.md) for systematic development

## Key Implementation Considerations

- **Progressive Development**: Start with baseline U-Net, then add components incrementally
- **Modular Design**: Each component (encoder, transformer, decoder) should be independently testable
- **Memory Management**: Transformer components can be memory-intensive; implement gradient checkpointing
- **Validation Strategy**: Use proper cross-validation due to limited dataset size (400 images)

## Research Goals

- Achieve **15-25% improvement** in Mean Radial Error (MRE) over baseline U-Net
- Improve **Success Detection Rate (SDR) @2mm** through attention mechanisms
- Demonstrate clinical viability with **sub-2mm accuracy** for critical landmarks

## Project Structure

```
maht-net/
├── README.md
├── documentation/          # Comprehensive implementation guide
├── src/                   # Source code (to be created)
│   ├── models/           # Model architectures
│   ├── datasets/         # Data loading and preprocessing
│   ├── training/         # Training loops and utilities
│   ├── evaluation/       # Metrics and evaluation
│   └── utils/            # Common utilities
├── configs/              # Configuration files
├── data/                 # Dataset storage
├── experiments/          # Experiment results
├── checkpoints/          # Model checkpoints
└── logs/                 # Training logs
```

## Contributors

**Mohamed Nourdine** - *First Contributor & Project Contact*
- PhD Candidate
- Cloud Expert
- Co-Founder & CTO @ Henddu
- Contact: contact@mnourdine.com

## License

This project is part of ongoing PhD research. Please contact the project maintainer for usage permissions and collaboration opportunities.
