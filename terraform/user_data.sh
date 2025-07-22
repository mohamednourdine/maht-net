#!/bin/bash
# MAHT-Net EC2 Instance Initialization Script
# This script sets up the complete environment for MAHT-Net research

set -e  # Exit on any error

# Update system
echo "🚀 Starting MAHT-Net instance setup..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y

# Install essential packages
echo "📦 Installing essential packages..."
apt-get install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    tmux \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip \
    python3-venv \
    awscli \
    jq \
    tree \
    ncdu \
    iotop \
    nethogs

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu
systemctl enable docker
systemctl start docker

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="2.21.0"
curl -L "https://github.com/docker/compose/releases/download/v$${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install NVIDIA drivers and CUDA
echo "🎮 Installing NVIDIA drivers and CUDA..."
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update

# Install NVIDIA driver
apt-get install -y nvidia-driver-535

# Install CUDA Toolkit 12.1
apt-get install -y cuda-toolkit-12-1

# Install cuDNN
apt-get install -y libcudnn8 libcudnn8-dev

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> /home/ubuntu/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc

# Install NVIDIA Container Toolkit for Docker
echo "🐳 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-docker-keyring.gpg
curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Setup data volume
echo "💾 Setting up data volume..."
if [ -b /dev/nvme1n1 ]; then
    DEVICE="/dev/nvme1n1"
elif [ -b /dev/xvdf ]; then
    DEVICE="/dev/xvdf"
else
    echo "⚠️  Data volume not found, skipping..."
    DEVICE=""
fi

if [ ! -z "$DEVICE" ]; then
    # Check if device is already formatted
    if ! blkid $DEVICE; then
        echo "Formatting data volume..."
        mkfs.ext4 $DEVICE
    fi
    
    # Create mount point and mount
    mkdir -p /data
    mount $DEVICE /data
    
    # Add to fstab for persistent mounting
    echo "$DEVICE /data ext4 defaults,nofail 0 2" >> /etc/fstab
    
    # Set ownership
    chown ubuntu:ubuntu /data
    chmod 755 /data
    
    # Create directories
    mkdir -p /data/{datasets,models,results,logs,backups}
    chown -R ubuntu:ubuntu /data
fi

# Install Miniconda
echo "🐍 Installing Miniconda..."
cd /home/ubuntu
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ubuntu/miniconda3
chown -R ubuntu:ubuntu /home/ubuntu/miniconda3

# Initialize conda for ubuntu user
sudo -u ubuntu /home/ubuntu/miniconda3/bin/conda init bash

# Create MAHT-Net environment
echo "🧬 Creating MAHT-Net conda environment..."
sudo -u ubuntu /home/ubuntu/miniconda3/bin/conda create -n maht-net python=3.9 -y

# Activate environment and install packages
echo "📚 Installing Python packages..."
sudo -u ubuntu bash -c "
source /home/ubuntu/miniconda3/bin/activate maht-net
pip install --upgrade pip setuptools wheel

# PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Core ML packages
pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    opencv-python==4.8.0.76 \
    pillow==10.0.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    tqdm==4.65.0 \
    tensorboard==2.13.0 \
    wandb==0.15.8

# Medical imaging
pip install \
    SimpleITK==2.2.1 \
    pydicom==2.4.2 \
    nibabel==5.1.0 \
    scikit-image==0.21.0

# Deep learning utilities
pip install \
    albumentations==1.3.1 \
    timm==0.9.2 \
    transformers==4.31.0 \
    einops==0.6.1 \
    torchmetrics==1.0.3

# Development tools
pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.5 \
    ipywidgets==8.1.0 \
    black==23.7.0 \
    flake8==6.0.0 \
    pytest==7.4.0 \
    pytest-cov==4.1.0 \
    mypy==1.5.1

# FastAPI for deployment
pip install \
    fastapi==0.101.1 \
    uvicorn==0.23.2 \
    pydantic==2.1.1 \
    python-multipart==0.0.6

# AWS SDK
pip install boto3==1.28.25 awscli==1.29.25
"

# Install Node.js for additional tools
echo "📦 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Clone MAHT-Net repository
echo "📁 Setting up MAHT-Net project..."
cd /home/ubuntu
if [ ! -d "maht-net" ]; then
    sudo -u ubuntu git clone https://github.com/mohamednourdine/maht-net.git || \
    sudo -u ubuntu mkdir -p maht-net
fi
chown -R ubuntu:ubuntu /home/ubuntu/maht-net

# Create symlink to data directory if it exists
if [ -d "/data" ]; then
    sudo -u ubuntu ln -sf /data /home/ubuntu/maht-net/data_volume
fi

# Setup Jupyter configuration
echo "📓 Configuring Jupyter..."
sudo -u ubuntu bash -c "
source /home/ubuntu/miniconda3/bin/activate maht-net
jupyter notebook --generate-config

cat > /home/ubuntu/.jupyter/jupyter_notebook_config.py << 'EOF'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = False
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.notebook_dir = '/home/ubuntu/maht-net'
EOF
"

# Setup systemd service for Jupyter
cat > /etc/systemd/system/jupyter.service << 'EOF'
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/maht-net
Environment=PATH=/home/ubuntu/miniconda3/envs/maht-net/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ubuntu/miniconda3/envs/maht-net/bin/jupyter notebook
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable jupyter
systemctl start jupyter

# Setup AWS CLI configuration
echo "☁️  Configuring AWS CLI..."
sudo -u ubuntu aws configure set region ${aws_region}
sudo -u ubuntu aws configure set output json

# Create helpful scripts
echo "📜 Creating utility scripts..."

# GPU monitoring script
cat > /home/ubuntu/gpu_monitor.sh << 'EOF'
#!/bin/bash
watch -n 1 nvidia-smi
EOF
chmod +x /home/ubuntu/gpu_monitor.sh

# Resource monitoring script
cat > /home/ubuntu/system_monitor.sh << 'EOF'
#!/bin/bash
echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "Disk Usage:"
df -h
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
EOF
chmod +x /home/ubuntu/system_monitor.sh

# Auto shutdown script (if enabled)
if [ "${auto_shutdown_time}" != "0" ]; then
    echo "⏰ Setting up auto-shutdown in ${auto_shutdown_time} minutes..."
    echo "shutdown -h +${auto_shutdown_time}" | at now
fi

# Setup log rotation
cat > /etc/logrotate.d/maht-net << 'EOF'
/home/ubuntu/maht-net/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    su ubuntu ubuntu
}
EOF

# Create welcome message
cat > /home/ubuntu/README_MAHT_NET.md << 'EOF'
# 🦷 MAHT-Net Research Instance Setup Complete! 🦷

## Quick Start Guide

### 1. Activate the MAHT-Net Environment
```bash
source ~/miniconda3/bin/activate maht-net
```

### 2. Access Services
- **Jupyter Notebook**: http://YOUR_IP:8888
- **TensorBoard**: http://YOUR_IP:6006  
- **FastAPI**: http://YOUR_IP:8000

### 3. Monitor Resources
```bash
# GPU monitoring
./gpu_monitor.sh

# System monitoring  
./system_monitor.sh

# Check CUDA
nvidia-smi
```

### 4. Navigate to Project
```bash
cd ~/maht-net
```

### 5. Data Storage
- **Local SSD**: `/data/` (high-performance storage)
- **S3 Bucket**: `${bucket_name}`

## Important Commands

### Environment Management
```bash
# Activate environment
conda activate maht-net

# Install additional packages
pip install package_name

# Update environment
conda env update -f environment.yml
```

### GPU Operations
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Data Management
```bash
# Sync data to S3
aws s3 sync /data/datasets/ s3://${bucket_name}/datasets/

# Download from S3
aws s3 sync s3://${bucket_name}/datasets/ /data/datasets/
```

## Project Structure
```
~/maht-net/
├── src/                 # Source code
├── data/               # Local datasets
├── models/             # Model checkpoints
├── results/            # Experiment results
├── logs/               # Training logs
├── scripts/            # Utility scripts
└── documentation/      # Project documentation
```

## Troubleshooting

### CUDA Issues
- Restart instance if CUDA not detected
- Check driver installation: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.get_device_name())"`

### Storage Issues
- Data volume: `/data/` (100GB+ available)
- Check usage: `df -h`
- Clean logs: `sudo logrotate -f /etc/logrotate.d/maht-net`

### Service Issues
```bash
# Restart Jupyter
sudo systemctl restart jupyter

# Check service status
sudo systemctl status jupyter
```

Happy researching! 🧬🤖
EOF

chown ubuntu:ubuntu /home/ubuntu/README_MAHT_NET.md

echo "✅ MAHT-Net instance setup complete!"
echo "🌐 Access Jupyter at: http://$(curl -s ifconfig.me):8888"
echo "📊 Monitor GPU with: nvidia-smi"
echo "📖 Read README_MAHT_NET.md for detailed instructions"

# Final system update and cleanup
apt-get autoremove -y
apt-get autoclean

# Reboot to ensure all drivers are properly loaded
echo "🔄 Rebooting to finalize driver installation..."
reboot
