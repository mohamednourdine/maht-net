#!/usr/bin/env fish

# MAHT-Net Interactive Project Setup Script
# Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection

function show_header
    echo "🏥 =================================="
    echo "🏥  MAHT-Net Project Setup"
    echo "🏥  Clinical-Grade AI for"
    echo "🏥  Cephalometric Analysis"
    echo "🏥 =================================="
    echo ""
end

function check_requirements
    echo "Checking system requirements..."
    
    # Check Python version
    if not command -q python3
        echo "❌ Python 3 is required but not installed"
        return 1
    end
    
    set python_version (python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Python $python_version detected"
    
    # Check if we're on a suitable platform
    set os_name (uname -s)
    echo "Operating System: $os_name"
    
    # Check GPU availability
    if command -q nvidia-smi
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    else
        echo "⚠️  No NVIDIA GPU detected - CPU training will be slower"
    end
    
    echo ""
end

function setup_virtual_environment
    echo "🐍 Setting up Python virtual environment..."
    
    if test -d venv
        echo "📁 Virtual environment already exists"
        read -P "🗑️  Remove existing environment? (y/N): " response
        if test "$response" = "y" -o "$response" = "Y"
            rm -rf venv
            echo "🗑️  Removed existing virtual environment"
        else
            echo "⏭️  Skipping environment creation"
            return 0
        end
    end
    
    python3 -m venv venv
    echo "Virtual environment created"
    
    # Activate environment
    source venv/bin/activate.fish
    echo "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    echo "Package managers updated"
    
    echo ""
end

function install_dependencies
    echo "📦 Installing MAHT-Net dependencies..."
    
    # Check if requirements.txt exists
    if not test -f requirements.txt
        echo "❌ requirements.txt not found"
        return 1
    end
    
    # Install main dependencies
    echo "📚 Installing main dependencies..."
    pip install -r requirements.txt
    
    # Install development dependencies if available
    if test -f requirements-dev.txt
        read -P "🛠️  Install development dependencies? (Y/n): " response
        if test "$response" != "n" -a "$response" != "N"
            echo "🛠️  Installing development dependencies..."
            pip install -r requirements-dev.txt
        end
    end
    
    echo "Dependencies installed successfully"
    echo ""
end

function create_project_structure
    echo "📁 Creating MAHT-Net project structure..."
    
    # Data directories
    mkdir -p data/{raw,processed,augmented,splits}
    mkdir -p data/annotations/{train,val,test}
    
    # Model directories  
    mkdir -p models/{checkpoints,pretrained,configs,exports}
    
    # Results directories
    mkdir -p results/{experiments,ablation,clinical_validation,reports}
    
    # Logs directories
    mkdir -p logs/{training,evaluation,deployment,monitoring}
    
    # Scripts directories
    mkdir -p scripts/{preprocessing,evaluation,deployment,monitoring}
    
    # Notebooks directory
    mkdir -p notebooks/{exploration,visualization,analysis}
    
    # Tests directories
    mkdir -p tests/{unit,integration,clinical}
    
    # Documentation assets
    mkdir -p docs/{images,diagrams,reports}
    
    echo "Project structure created"
    
    # Create placeholder files
    echo "📝 Creating placeholder files..."
    
    # Data README files
    echo "# Raw cephalometric X-ray images" > data/raw/README.md
    echo "# Processed and normalized images" > data/processed/README.md
    echo "# Augmented training data" > data/augmented/README.md
    echo "# Train/validation/test splits" > data/splits/README.md
    
    # Model README files
    echo "# Model checkpoints during training" > models/checkpoints/README.md
    echo "# Pre-trained model weights" > models/pretrained/README.md
    echo "# Model configuration files" > models/configs/README.md
    echo "# Exported models for deployment" > models/exports/README.md
    
    # Results README files
    echo "# Training and evaluation results" > results/experiments/README.md
    echo "# Ablation study results" > results/ablation/README.md
    echo "# Clinical validation results" > results/clinical_validation/README.md
    echo "# Generated reports and visualizations" > results/reports/README.md
    
    echo "Placeholder files created"
    echo ""
end

function setup_git_hooks
    echo "🔧 Setting up Git hooks..."
    
    if not test -d .git
        echo "⚠️  Not a Git repository - skipping hook setup"
        return 0
    end
    
    # Check if pre-commit is available
    if command -q pre-commit
        pre-commit install
        echo "Pre-commit hooks installed"
    else
        echo "⚠️  Pre-commit not available - install with: pip install pre-commit"
    end
    
    echo ""
end

function download_sample_data
    echo "📥 Setting up sample data..."
    
    read -P "Download sample cephalometric dataset for testing? (Y/n): " response
    if test "$response" = "n" -o "$response" = "N"
        echo "⏭️  Skipping sample data download"
        echo ""
        return 0
    end
    
    # Create sample data structure
    mkdir -p data/raw/sample
    mkdir -p data/processed/sample
    
    # Note: In a real implementation, this would download actual sample data
    echo "📝 Creating sample annotation files..."
    
    # Create sample annotation JSON
    cat > data/processed/sample/train_annotations.json << 'EOF'
{
  "dataset": "sample_cephalometric",
  "num_landmarks": 7,
  "landmark_names": ["Nasion", "Sella", "Articulare", "Gonion", "Menton", "Pogonion", "Upper_Incisor"],
  "samples": [
    {
      "id": "sample_001",
      "image_path": "sample/sample_001.jpg",
      "landmarks": [[256, 100], [200, 120], [180, 200], [190, 300], [250, 400], [280, 380], [270, 250]],
      "patient_id": "SAMPLE_001",
      "acquisition_params": {"kvp": 80, "mas": 10, "pixel_spacing": [0.1, 0.1]}
    }
  ]
}
EOF
    
    echo "Sample annotation files created"
    echo "📝 Note: Add your actual cephalometric images to data/raw/ directory"
    echo ""
end

function create_environment_file
    echo "🌐 Creating environment configuration..."
    
    cat > .env << EOF
# MAHT-Net Environment Configuration

# Project Settings
PROJECT_NAME="MAHT-Net"
PROJECT_VERSION="1.0.0"
ENVIRONMENT="development"

# Paths
DATA_DIR="data/processed"
MODEL_DIR="models"
RESULTS_DIR="results"
LOGS_DIR="logs"

# Training Configuration
BATCH_SIZE=8
NUM_WORKERS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=36

# AWS Configuration (if using cloud)
AWS_REGION="us-east-1"
S3_BUCKET=""
EC2_INSTANCE_TYPE="g5.xlarge"

# API Configuration
API_HOST="0.0.0.0"
API_PORT=8000

# Monitoring
TENSORBOARD_PORT=6006
WANDB_PROJECT="maht-net"

# Clinical Settings
TARGET_MRE=1.2
TARGET_SDR_2MM=95.0
SAFETY_THRESHOLD=2.0

EOF
    
    echo "Environment file created (.env)"
    echo ""
end

function show_next_steps
    echo "🎉 MAHT-Net setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "=============="
    echo ""
    echo "1. 🐍 Activate virtual environment:"
    echo "   source venv/bin/activate.fish"
    echo ""
    echo "2. Prepare your dataset:"
    echo "   - Place cephalometric X-ray images in data/raw/"
    echo "   - Create annotation files in data/processed/"
    echo "   - Run: python scripts/preprocessing/prepare_data.py"
    echo ""
    echo "3. 🧠 Start training:"
    echo "   make train"
    echo "   # or directly: python train.py --config configs/train_config.json"
    echo ""
    echo "4. Monitor training:"
    echo "   make monitor"
    echo "   # or: tensorboard --logdir=logs/training --port=6006"
    echo ""
    echo "5. 🏥 Evaluate model:"
    echo "   make eval"
    echo ""
    echo "6. ☁️  Deploy to AWS (optional):"
    echo "   make aws-setup"
    echo "   make deploy"
    echo ""
    echo "📚 Documentation:"
    echo "================="
    echo "• Project Overview: documentation/01_project_overview.md"
    echo "• Environment Setup: documentation/02_environment_setup.md"
    echo "• Training Guide: documentation/06_training_strategy.md"
    echo "• Troubleshooting: documentation/09_troubleshooting_guide.md"
    echo ""
    echo "🤝 Contributing:"
    echo "==============="
    echo "• See CONTRIBUTING.md for development guidelines"
    echo "• Join our research community discussions"
    echo "• Report issues on our GitHub repository"
    echo ""
    echo "💡 Tips:"
    echo "======="
    echo "• Use 'make help' to see all available commands"
    echo "• Check logs/ directory for training progress"
    echo "• Monitor GPU usage with: watch -n 1 nvidia-smi"
    echo "• For clinical deployment, see documentation/10_clinical_integration.md"
    echo ""
end

function main
    show_header
    
    echo "This script will set up the complete MAHT-Net development environment"
    echo "⏱️  Estimated time: 5-10 minutes (depending on internet speed)"
    echo ""
    
    read -P "🚀 Continue with setup? (Y/n): " response
    if test "$response" = "n" -o "$response" = "N"
        echo "⏹️  Setup cancelled"
        exit 0
    end
    
    echo ""
    
    # Run setup steps
    check_requirements; or exit 1
    setup_virtual_environment; or exit 1
    install_dependencies; or exit 1
    create_project_structure
    setup_git_hooks
    download_sample_data
    create_environment_file
    
    echo ""
    show_next_steps
end

# Run main function
main
