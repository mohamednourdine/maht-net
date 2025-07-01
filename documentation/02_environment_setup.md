# Environment Setup for MAHT-Net Implementation

## AWS EC2 Infrastructure Setup

### EC2 Instance Specifications

#### Recommended Instance Types
- **Primary**: `g5.xlarge` (1x NVIDIA A10G, 16GB GPU memory, 4 vCPUs, 16GB RAM)
- **High Performance**: `g5.2xlarge` (1x NVIDIA A10G, 24GB GPU memory, 8 vCPUs, 32GB RAM)
- **Budget Option**: `g4dn.xlarge` (1x NVIDIA T4, 16GB GPU memory, 4 vCPUs, 16GB RAM)
- **Development**: `g4dn.large` (1x NVIDIA T4, 16GB GPU memory, 2 vCPUs, 8GB RAM)

#### Storage Requirements
- **Root Volume**: 100GB gp3 SSD (minimum)
- **Additional EBS**: 200GB gp3 for datasets and checkpoints
- **Backup Storage**: S3 bucket for model artifacts and results

#### Network Configuration
- **Security Group**: SSH (22), Jupyter (8888), TensorBoard (6006), Custom ports for monitoring
- **Elastic IP**: Recommended for consistent access
- **VPC**: Default VPC with internet gateway access

### AMI Selection and Initial Setup

#### Recommended AMI
- **Primary Choice**: Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 20.04)
- **Alternative**: Ubuntu 22.04 LTS with manual CUDA installation
- **Region**: us-east-1 or us-west-2 (for lower costs and better availability)

#### Instance Launch Configuration
```bash
# Launch instance with AWS CLI (optional)
aws ec2 run-instances \
    --image-id ami-0c94855ba95b798c7 \
    --instance-type g5.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 100,
            "VolumeType": "gp3",
            "DeleteOnTermination": true
        }
    }]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MAHT-Net-Training}]'
```

## System Requirements

### Hardware Specifications (EC2)
- **GPU**: NVIDIA A10G (24GB VRAM) or T4 (16GB VRAM)
- **RAM**: 16-32GB (instance dependent)
- **Storage**: 100GB root + 200GB additional EBS
- **vCPUs**: 4-8 cores (instance dependent)
- **Network**: Enhanced networking enabled

### Software Requirements
- **OS**: Ubuntu 20.04/22.04 LTS (recommended for EC2)
- **Python**: 3.10 (recommended)
- **CUDA**: 12.1 with cuDNN 8.9
- **Docker**: Optional for containerized deployment

## Initial EC2 Setup and Configuration

### Step 1: Connect to EC2 Instance

```bash
# Connect via SSH (replace with your details)
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-public-ip

# Optional: Create SSH config for easier access
echo "Host maht-net
    HostName your-ec2-public-ip
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3" >> ~/.ssh/config

# Connect using alias
ssh maht-net
```

### Step 2: System Updates and Essential Tools

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    htop \
    tree \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install fish shell (optional, for better CLI experience)
sudo apt-add-repository ppa:fish-shell/release-3
sudo apt update
sudo apt install -y fish

# Set fish as default shell (optional)
chsh -s /usr/bin/fish
```

### Step 3: NVIDIA Drivers and CUDA Setup

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# If using Deep Learning AMI, CUDA should be pre-installed
# Verify CUDA installation
nvcc --version

# If manual installation needed:
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Install CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add CUDA to PATH (add to ~/.bashrc or ~/.config/fish/config.fish)
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

### Step 4: Conda Installation and Environment Setup

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
$HOME/miniconda3/bin/conda init bash
$HOME/miniconda3/bin/conda init fish  # if using fish shell

# Reload shell
source ~/.bashrc
# or for fish: source ~/.config/fish/config.fish

# Update conda
conda update -n base -c defaults conda

# Create MAHT-Net environment
conda create -n maht-net python=3.10 -y
conda activate maht-net

# Verify environment
which python
python --version
```

### Step 5: PyTorch and Deep Learning Dependencies

```bash
# Ensure conda environment is activated
conda activate maht-net

# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Install additional deep learning libraries
pip install timm transformers einops

# Install scientific computing stack
pip install numpy scipy scikit-learn pandas matplotlib seaborn plotly

# Install image processing libraries
pip install opencv-python Pillow scikit-image albumentations

# Install medical imaging libraries
pip install nibabel pydicom SimpleITK

# Install experiment tracking and visualization
pip install tensorboard wandb

# Install utility libraries
pip install tqdm rich click pyyaml jsonschema

# Install development tools
pip install black isort flake8 mypy pre-commit pytest pytest-cov

# Install Jupyter for interactive development
pip install jupyter jupyterlab ipywidgets

# Install monitoring tools
pip install gpustat psutil
```

### Step 6: Additional System Tools and Monitoring

```bash
# Install system monitoring tools
sudo apt install -y nvidia-ml-py3

# Install Docker (optional, for containerized deployment)
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER

# Install AWS CLI for S3 integration
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI (optional, for S3 backup)
aws configure
```

## Project Setup and Configuration

### Step 7: Clone and Setup Project Structure

```bash
# Create project directory
sudo mkdir -p /var/www/phd-researches
sudo chown $USER:$USER /var/www/phd-researches
cd /var/www/phd-researches

# Clone the project (replace with your repository)
git clone https://github.com/your-username/maht-net.git
cd maht-net

# Alternative: Create project from scratch
mkdir -p maht-net && cd maht-net

# Create comprehensive directory structure
mkdir -p {src,data,experiments,notebooks,tests,configs,checkpoints,logs,results,scripts}
mkdir -p src/{models,datasets,training,evaluation,utils,visualization}
mkdir -p configs/{models,training,datasets,experiments}
mkdir -p data/{raw,processed,annotations,augmented,external}
mkdir -p experiments/{baseline,ablation,optimization}
mkdir -p scripts/{preprocessing,training,evaluation,deployment}

# Create additional AWS-specific directories
mkdir -p {aws,deployment,monitoring}
mkdir -p aws/{s3_sync,cloudwatch,lambda}
mkdir -p deployment/{docker,terraform,kubernetes}
mkdir -p monitoring/{grafana,prometheus,custom}
```

### Expected Enhanced Directory Structure
```
maht-net/
├── documentation/           # Project documentation
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── datasets/          # Dataset handling
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Evaluation metrics
│   ├── utils/            # Utility functions
│   └── visualization/    # Plotting and visualization
├── data/                  # Dataset storage
│   ├── raw/              # Original ISBI 2015 data
│   ├── processed/        # Preprocessed images
│   ├── annotations/      # Ground truth labels
│   ├── augmented/        # Augmented datasets
│   └── external/         # External datasets
├── configs/               # Configuration files
│   ├── models/           # Model configurations
│   ├── training/         # Training configurations
│   ├── datasets/         # Dataset configurations
│   └── experiments/      # Experiment configurations
├── experiments/           # Experiment organization
│   ├── baseline/         # Baseline experiments
│   ├── ablation/         # Ablation studies
│   └── optimization/     # Hyperparameter optimization
├── scripts/               # Utility scripts
│   ├── preprocessing/    # Data preprocessing scripts
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation scripts
│   └── deployment/       # Deployment scripts
├── aws/                   # AWS-specific files
│   ├── s3_sync/          # S3 synchronization scripts
│   ├── cloudwatch/       # CloudWatch configuration
│   └── lambda/           # Lambda functions
├── deployment/            # Deployment configurations
│   ├── docker/           # Docker configurations
│   ├── terraform/        # Infrastructure as code
│   └── kubernetes/       # K8s manifests
├── monitoring/            # Monitoring setup
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── results/               # Experiment results
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Development environment
└── Makefile              # Build automation
```

### Step 8: Environment Configuration Files

```bash
# Create comprehensive requirements.txt
cat > requirements.txt << 'EOF'
# Deep Learning Core
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
timm>=0.9.0
transformers>=4.35.0
einops>=0.7.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.0.0

# Image Processing
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-image>=0.22.0
albumentations>=1.3.0

# Medical Imaging
nibabel>=5.0.0
pydicom>=2.4.0
SimpleITK>=2.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Experiment Tracking
tensorboard>=2.15.0
wandb>=0.16.0

# Development Tools
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.6.0
pre-commit>=3.5.0
pytest>=7.4.0
pytest-cov>=4.1.0

# Jupyter
jupyter>=1.0.0
jupyterlab>=4.0.0
ipywidgets>=8.1.0

# Utilities
tqdm>=4.66.0
rich>=13.6.0
click>=8.1.0
pyyaml>=6.0.0
jsonschema>=4.19.0

# Monitoring
gpustat>=1.1.0
psutil>=5.9.0

# AWS Integration
boto3>=1.29.0
botocore>=1.32.0
EOF

# Install from requirements
pip install -r requirements.txt

# Create conda environment export
conda env export > environment.yml
```

### Step 9: Git Configuration and Repository Setup

```bash
# Initialize git repository if not cloned
git init

# Configure git (replace with your details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
env/
venv/
ENV/
env.bak/
venv.bak/
.conda/

# PyTorch
*.pth
*.pt
*.ckpt

# Data files (large files should be in S3)
data/raw/
data/processed/
data/external/
*.dcm
*.dicom
*.nii
*.nii.gz

# Logs and checkpoints
logs/
checkpoints/
wandb/
tensorboard_logs/
mlruns/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# AWS
.aws/
*.pem
*.key

# Temporary files
tmp/
temp/
.tmp/

# Results and artifacts (should be backed up to S3)
results/experiments/
results/plots/
results/reports/

# Environment files
.env
.env.local
.env.*.local

# Cache
.cache/
.pytest_cache/
.mypy_cache/
.ruff_cache/
EOF

# Create initial commit
git add .
git commit -m "Initial MAHT-Net project setup for AWS EC2"
```

## AWS Integration and Storage Setup

### Step 10: S3 Bucket Configuration

```bash
# Create S3 bucket for project data and artifacts
aws s3 mb s3://maht-net-research-$(date +%s) --region us-east-1

# Set bucket name as environment variable
echo 'export MAHT_NET_S3_BUCKET=s3://maht-net-research-your-bucket-name' >> ~/.bashrc
source ~/.bashrc

# Create S3 folder structure
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key data/raw/
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key data/processed/
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key checkpoints/
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key results/
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key experiments/
aws s3api put-object --bucket $MAHT_NET_S3_BUCKET --key logs/

# Create S3 sync script
cat > scripts/s3_sync.sh << 'EOF'
#!/bin/bash
# S3 synchronization script for MAHT-Net

BUCKET_NAME="${MAHT_NET_S3_BUCKET:-s3://maht-net-research-default}"
PROJECT_DIR="/var/www/phd-researches/maht-net"

# Function to sync results to S3
sync_results_to_s3() {
    echo "Syncing results to S3..."
    aws s3 sync $PROJECT_DIR/results/ $BUCKET_NAME/results/ --exclude "*.tmp" --exclude "__pycache__/*"
    aws s3 sync $PROJECT_DIR/logs/ $BUCKET_NAME/logs/ --exclude "*.tmp"
    echo "Results sync completed."
}

# Function to sync checkpoints to S3
sync_checkpoints_to_s3() {
    echo "Syncing checkpoints to S3..."
    aws s3 sync $PROJECT_DIR/checkpoints/ $BUCKET_NAME/checkpoints/ --exclude "*.tmp"
    echo "Checkpoints sync completed."
}

# Function to download data from S3
sync_data_from_s3() {
    echo "Syncing data from S3..."
    aws s3 sync $BUCKET_NAME/data/ $PROJECT_DIR/data/ --exclude "*.tmp"
    echo "Data sync completed."
}

# Execute based on argument
case "$1" in
    "upload-results")
        sync_results_to_s3
        ;;
    "upload-checkpoints")
        sync_checkpoints_to_s3
        ;;
    "download-data")
        sync_data_from_s3
        ;;
    "full-backup")
        sync_results_to_s3
        sync_checkpoints_to_s3
        ;;
    *)
        echo "Usage: $0 {upload-results|upload-checkpoints|download-data|full-backup}"
        exit 1
        ;;
esac
EOF

chmod +x scripts/s3_sync.sh
```

### Step 11: Development Environment Configuration

```bash
# Create development environment script
cat > scripts/setup_dev_env.sh << 'EOF'
#!/bin/bash
# Development environment setup script

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate maht-net

# Set environment variables
export PYTHONPATH="/var/www/phd-researches/maht-net/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Create symbolic links for easy access
ln -sf /var/www/phd-researches/maht-net ~/maht-net

# Start Jupyter Lab (optional)
if [ "$1" = "jupyter" ]; then
    cd /var/www/phd-researches/maht-net
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
fi

echo "Development environment ready!"
echo "Project path: /var/www/phd-researches/maht-net"
echo "To start Jupyter: ./scripts/setup_dev_env.sh jupyter"
EOF

chmod +x scripts/setup_dev_env.sh

# Make script executable on login
echo 'source /var/www/phd-researches/maht-net/scripts/setup_dev_env.sh' >> ~/.bashrc
```

### Step 12: Pre-commit Hooks and Code Quality

```bash
# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: debug-statements
EOF

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

## System Verification and Validation

### Step 13: Comprehensive Environment Validation

```bash
# Create validation script
cat > scripts/validate_environment.sh << 'EOF'
#!/bin/bash
# Comprehensive environment validation script

echo "=== MAHT-Net Environment Validation ==="
echo "Timestamp: $(date)"
echo

# Check Python environment
echo "1. Python Environment:"
conda info --envs | grep maht-net
python --version
which python
echo

# Check GPU and CUDA
echo "2. GPU and CUDA:"
nvidia-smi
nvcc --version
echo

# Check PyTorch installation
echo "3. PyTorch Installation:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo

# Check key libraries
echo "4. Key Libraries:"
python -c "
try:
    import transformers; print(f'Transformers: {transformers.__version__}')
except: print('Transformers: NOT FOUND')

try:
    import timm; print(f'TIMM: {timm.__version__}')
except: print('TIMM: NOT FOUND')

try:
    import cv2; print(f'OpenCV: {cv2.__version__}')
except: print('OpenCV: NOT FOUND')

try:
    import wandb; print(f'WandB: {wandb.__version__}')
except: print('WandB: NOT FOUND')
"
echo

# Check disk space
echo "5. Storage:"
df -h /var/www/phd-researches/maht-net
echo

# Check AWS CLI
echo "6. AWS Configuration:"
aws --version
aws s3 ls $MAHT_NET_S3_BUCKET 2>/dev/null && echo "S3 bucket accessible" || echo "S3 bucket not configured"
echo

# Test basic model creation
echo "7. Model Creation Test:"
python -c "
import torch
import torch.nn as nn

# Test basic model creation
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# Test GPU transfer
if torch.cuda.is_available():
    model = model.cuda()
    print('Model GPU transfer: SUCCESS')
else:
    print('Model GPU transfer: SKIPPED (no GPU)')

# Test forward pass
x = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    x = x.cuda()

with torch.no_grad():
    y = model(x)
    print(f'Forward pass: SUCCESS, output shape: {y.shape}')
"
echo

echo "=== Validation Complete ==="
EOF

chmod +x scripts/validate_environment.sh

# Run validation
./scripts/validate_environment.sh
```

### Step 14: Performance Monitoring Setup

```bash
# Create monitoring script
cat > scripts/monitor_system.sh << 'EOF'
#!/bin/bash
# System monitoring script for training

# Create monitoring function
monitor_training() {
    echo "Starting system monitoring..."
    
    # Create monitoring log directory
    mkdir -p logs/monitoring
    
    # GPU monitoring
    nvidia-smi --query-gpu=timestamp,gpu_name,gpu_usage.gpu,memory.used,memory.total,temperature.gpu --format=csv --loop=60 > logs/monitoring/gpu_usage.csv &
    GPU_PID=$!
    
    # System monitoring
    vmstat 60 > logs/monitoring/system_stats.log &
    VMSTAT_PID=$!
    
    # Disk usage monitoring
    while true; do
        df -h /var/www/phd-researches/maht-net >> logs/monitoring/disk_usage.log
        sleep 300  # Every 5 minutes
    done &
    DISK_PID=$!
    
    echo "Monitoring started. PIDs: GPU=$GPU_PID, VMSTAT=$VMSTAT_PID, DISK=$DISK_PID"
    echo "To stop monitoring: kill $GPU_PID $VMSTAT_PID $DISK_PID"
    
    # Save PIDs for later cleanup
    echo "$GPU_PID $VMSTAT_PID $DISK_PID" > logs/monitoring/monitor_pids.txt
}

# Stop monitoring function
stop_monitoring() {
    if [ -f logs/monitoring/monitor_pids.txt ]; then
        PIDS=$(cat logs/monitoring/monitor_pids.txt)
        kill $PIDS 2>/dev/null
        rm logs/monitoring/monitor_pids.txt
        echo "Monitoring stopped."
    else
        echo "No active monitoring found."
    fi
}

case "$1" in
    "start")
        monitor_training
        ;;
    "stop")
        stop_monitoring
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
EOF

chmod +x scripts/monitor_system.sh
```

## Security and Cost Optimization

### Step 15: Security Configuration

```bash
# Configure EC2 security best practices
cat > scripts/security_setup.sh << 'EOF'
#!/bin/bash
# Security configuration for EC2 instance

# Update system packages
sudo apt update && sudo apt upgrade -y

# Configure fail2ban for SSH protection
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Configure automatic security updates
sudo apt install -y unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' | sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades

# Set up log rotation for training logs
sudo tee /etc/logrotate.d/maht-net << EOL
/var/www/phd-researches/maht-net/logs/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOL

# Secure SSH configuration
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
sudo tee -a /etc/ssh/sshd_config << EOL

# Additional security settings
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
EOL

sudo systemctl reload sshd

echo "Security configuration completed."
EOF

chmod +x scripts/security_setup.sh
```

### Step 16: Cost Optimization Scripts

```bash
# Create cost optimization script
cat > scripts/cost_optimization.sh << 'EOF'
#!/bin/bash
# Cost optimization utilities for AWS EC2

# Function to stop instance safely
stop_instance() {
    echo "Preparing to stop EC2 instance..."
    
    # Sync important data to S3
    ./scripts/s3_sync.sh full-backup
    
    # Stop monitoring
    ./scripts/monitor_system.sh stop
    
    # Save conda environment
    conda env export > environment_backup.yml
    
    # Create shutdown log
    echo "Instance stopped at $(date)" >> logs/instance_lifecycle.log
    
    # Stop the instance (requires AWS CLI configured with appropriate permissions)
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 stop-instances --instance-ids $INSTANCE_ID
    
    echo "Instance stop initiated. Data backed up to S3."
}

# Function to prepare for hibernation
prepare_hibernation() {
    echo "Preparing instance for hibernation..."
    
    # Stop all training processes
    pkill -f "python.*train"
    
    # Sync data
    ./scripts/s3_sync.sh full-backup
    
    # Clear temporary files
    find /tmp -type f -name "*.tmp" -delete
    find . -type f -name "*.pyc" -delete
    
    echo "Instance prepared for hibernation."
}

# Function to setup auto-shutdown
setup_auto_shutdown() {
    local HOURS=${1:-8}
    echo "Setting up auto-shutdown in $HOURS hours..."
    
    # Create shutdown script
    echo "#!/bin/bash" > /tmp/auto_shutdown.sh
    echo "./scripts/cost_optimization.sh stop" >> /tmp/auto_shutdown.sh
    chmod +x /tmp/auto_shutdown.sh
    
    # Schedule shutdown
    echo "/tmp/auto_shutdown.sh" | at now + $HOURS hours
    
    echo "Auto-shutdown scheduled for $HOURS hours from now."
}

case "$1" in
    "stop")
        stop_instance
        ;;
    "hibernate")
        prepare_hibernation
        ;;
    "auto-shutdown")
        setup_auto_shutdown $2
        ;;
    *)
        echo "Usage: $0 {stop|hibernate|auto-shutdown [hours]}"
        echo "  stop: Safely stop the instance with full backup"
        echo "  hibernate: Prepare for hibernation"
        echo "  auto-shutdown: Schedule automatic shutdown (default 8 hours)"
        exit 1
        ;;
esac
EOF

chmod +x scripts/cost_optimization.sh
```

### Step 17: Remote Development Setup

```bash
# Setup remote development access
cat > scripts/remote_access.sh << 'EOF'
#!/bin/bash
# Remote development access configuration

# Start Jupyter Lab for remote access
start_jupyter() {
    echo "Starting Jupyter Lab..."
    cd /var/www/phd-researches/maht-net
    
    # Generate Jupyter config if not exists
    if [ ! -f ~/.jupyter/jupyter_lab_config.py ]; then
        jupyter lab --generate-config
        
        # Set password (you'll be prompted)
        jupyter lab password
    fi
    
    # Start Jupyter Lab
    nohup jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --notebook-dir=/var/www/phd-researches/maht-net > logs/jupyter.log 2>&1 &
    
    echo "Jupyter Lab started. Access at: http://$(curl -s ifconfig.me):8888"
}

# Start TensorBoard
start_tensorboard() {
    echo "Starting TensorBoard..."
    
    nohup tensorboard \
        --logdir=logs/tensorboard \
        --host=0.0.0.0 \
        --port=6006 > logs/tensorboard_server.log 2>&1 &
    
    echo "TensorBoard started. Access at: http://$(curl -s ifconfig.me):6006"
}

# Setup SSH tunnel helper
setup_ssh_tunnel() {
    local LOCAL_PORT=${1:-8888}
    local REMOTE_PORT=${2:-8888}
    local PUBLIC_IP=$(curl -s ifconfig.me)
    
    echo "SSH tunnel command for local access:"
    echo "ssh -L $LOCAL_PORT:localhost:$REMOTE_PORT -N -f ubuntu@$PUBLIC_IP"
    echo "Then access http://localhost:$LOCAL_PORT"
}

case "$1" in
    "jupyter")
        start_jupyter
        ;;
    "tensorboard")
        start_tensorboard
        ;;
    "tunnel")
        setup_ssh_tunnel $2 $3
        ;;
    "all")
        start_jupyter
        start_tensorboard
        echo "All services started."
        ;;
    *)
        echo "Usage: $0 {jupyter|tensorboard|tunnel|all}"
        echo "  jupyter: Start Jupyter Lab server"
        echo "  tensorboard: Start TensorBoard server"
        echo "  tunnel: Show SSH tunnel command"
        echo "  all: Start all services"
        exit 1
        ;;
esac
EOF

chmod +x scripts/remote_access.sh
```

## Troubleshooting and Common Issues

### CUDA and GPU Issues

#### Issue: CUDA Not Available
**Symptoms**: `torch.cuda.is_available()` returns `False`
**Solutions**:
```bash
# Check NVIDIA driver installation
nvidia-smi

# Reinstall CUDA-compatible PyTorch
conda remove pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Issue: CUDA Out of Memory
**Symptoms**: `RuntimeError: CUDA out of memory`
**Solutions**:
```bash
# Monitor GPU memory usage
watch -n 1 nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in training configuration
# Enable gradient checkpointing in model configuration
```

### EC2 and AWS Issues

#### Issue: Instance Connection Problems
**Solutions**:
```bash
# Check security group settings (ensure port 22 is open)
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx

# Verify key pair permissions
chmod 400 ~/.ssh/your-key.pem

# Check instance status
aws ec2 describe-instances --instance-ids i-xxxxxxxxx
```

#### Issue: Storage Full
**Solutions**:
```bash
# Check disk usage
df -h

# Clean temporary files
sudo apt autoremove
sudo apt autoclean
conda clean --all

# Remove old logs
find logs/ -type f -name "*.log" -mtime +7 -delete

# Move large files to S3
./scripts/s3_sync.sh full-backup
```

### Environment and Package Issues

#### Issue: Package Conflicts
**Solutions**:
```bash
# Create fresh environment
conda deactivate
conda remove -n maht-net --all
conda create -n maht-net python=3.10
conda activate maht-net

# Install packages one by one
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### Issue: Import Errors
**Solutions**:
```bash
# Check Python path
echo $PYTHONPATH

# Add project to Python path
export PYTHONPATH="/var/www/phd-researches/maht-net/src:$PYTHONPATH"

# Verify package installation
pip list | grep torch
```

### Performance Issues

#### Issue: Slow Training
**Diagnostic Commands**:
```bash
# Monitor system resources
./scripts/monitor_system.sh start

# Check GPU utilization
nvidia-smi dmon -s puc

# Profile memory usage
python -m memory_profiler your_training_script.py

# Check data loading bottlenecks
python -c "
import torch
from torch.utils.data import DataLoader
# Test dataloader speed
"
```

### Network and S3 Issues

#### Issue: S3 Access Problems
**Solutions**:
```bash
# Configure AWS credentials
aws configure

# Test S3 access
aws s3 ls s3://your-bucket-name

# Check IAM permissions
aws iam get-user
```

## Quick Start Checklist

### For New EC2 Instance Setup
```bash
# 1. Connect to instance
ssh -i ~/.ssh/your-key.pem ubuntu@your-ec2-ip

# 2. Run setup script
cd /var/www/phd-researches/maht-net
./scripts/validate_environment.sh

# 3. Start development environment
./scripts/setup_dev_env.sh

# 4. Verify everything works
./scripts/validate_environment.sh

# 5. Start remote development (optional)
./scripts/remote_access.sh all
```

### Daily Development Workflow
```bash
# 1. Activate environment
conda activate maht-net

# 2. Start monitoring
./scripts/monitor_system.sh start

# 3. Sync latest data
./scripts/s3_sync.sh download-data

# 4. Run your experiments
# ... your training/evaluation commands ...

# 5. Backup results
./scripts/s3_sync.sh full-backup

# 6. Stop monitoring
./scripts/monitor_system.sh stop

# 7. Setup auto-shutdown (optional, for cost savings)
./scripts/cost_optimization.sh auto-shutdown 8
```

## Next Steps and Additional Resources

### Immediate Next Steps
1. **Validate Environment**: Run the validation script to ensure everything is working
   ```bash
   ./scripts/validate_environment.sh
   ```

2. **Proceed to Dataset Preparation**: Follow [Dataset Preparation](03_dataset_preparation.md)

3. **Setup Development Workflow**: 
   - Configure remote access for Jupyter/TensorBoard
   - Test S3 synchronization
   - Run a simple training test

### Additional AWS Resources

#### CloudWatch Integration
```bash
# Install CloudWatch agent (optional for advanced monitoring)
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure basic metrics
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

#### Spot Instance Considerations
- Use Spot Instances for cost savings (up to 90% discount)
- Implement checkpoint saving every epoch
- Use `scripts/cost_optimization.sh` for graceful shutdowns

#### Auto Scaling Setup
- Consider using Auto Scaling Groups for multiple training runs
- Implement SQS-based job queuing for distributed experiments

### Advanced Configuration

#### Docker Deployment (Optional)
```bash
# Create Dockerfile for reproducible environments
cat > Dockerfile << 'EOF'
FROM nvidia/cuda:12.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3", "src/training/train.py"]
EOF

# Build and run
docker build -t maht-net .
docker run --gpus all -v $(pwd):/workspace maht-net
```

#### Kubernetes Deployment (Advanced)
- Consider EKS for large-scale distributed training
- Use Kubeflow for ML pipeline management

### Performance Optimization Tips

1. **Data Loading Optimization**:
   - Use multiple data loader workers
   - Implement data prefetching
   - Consider using NVIDIA DALI for faster data loading

2. **Training Optimization**:
   - Enable mixed precision training
   - Use gradient accumulation for large effective batch sizes
   - Implement gradient checkpointing for memory efficiency

3. **Storage Optimization**:
   - Use EBS optimized instances
   - Consider using EFS for shared storage across instances
   - Implement data caching strategies

### Cost Management

#### Monitoring Costs
```bash
# Setup cost alerts using AWS CLI
aws budgets create-budget --account-id 123456789012 --budget '{
    "BudgetName": "MAHT-Net-Monthly",
    "BudgetLimit": {"Amount": "100", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
}'
```

#### Cost Optimization Strategies
- Use Spot Instances for training (save up to 90%)
- Schedule automatic shutdown during non-working hours
- Use S3 Intelligent Tiering for data storage
- Monitor and right-size instances based on usage

### Support and Resources

#### AWS Support
- Use AWS Support for infrastructure issues
- Leverage AWS Documentation for best practices
- Consider AWS Professional Services for complex deployments

#### Community Resources
- PyTorch Forums for framework-specific issues
- Hugging Face Community for transformer-related questions
- Stack Overflow for general development issues

#### Project-Specific Support
- Check project documentation for detailed implementation guides
- Use GitHub Issues for bug reports and feature requests
- Refer to ablation studies for architecture decisions

This comprehensive setup ensures a robust, scalable, and cost-effective environment for MAHT-Net development and training on AWS EC2.
