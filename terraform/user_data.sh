#!/bin/bash
# MAHT-Net EC2 Instance Initialization Script
# This script sets up the development environment for MAHT-Net research

set -e  # Exit on any error

# Update system
echo "🚀 Starting MAHT-Net development instance setup..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y

# Install essential packages for development
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
    jq \
    tree \
    ncdu \
    iotop \
    nethogs \
    zip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker (for containerized development)
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

# Setup data volume for project storage
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
    
    # Create directories for project data
    mkdir -p /data/{datasets,models,results,logs,backups,notebooks}
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

# Activate environment and install packages for CPU-based development
echo "📚 Installing Python packages..."
sudo -u ubuntu bash -c "
source /home/ubuntu/miniconda3/bin/activate maht-net
pip install --upgrade pip setuptools wheel

# PyTorch CPU version (much faster to install and sufficient for development)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

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
    tensorboard==2.13.0

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
    mypy==1.5.1 \
    isort==5.12.0

# FastAPI for development
pip install \
    fastapi==0.101.1 \
    uvicorn==0.23.2 \
    pydantic==2.1.1 \
    python-multipart==0.0.6
"

# Install Node.js for additional development tools
echo "📦 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get install -y nodejs

# Setup MAHT-Net project directory
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

# Create project structure for local development
sudo -u ubuntu mkdir -p /home/ubuntu/maht-net/{data,models,results,logs,notebooks}
chown -R ubuntu:ubuntu /home/ubuntu/maht-net

# Setup Jupyter configuration (optional - primarily for VS Code development)
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

# Setup systemd service for Jupyter (optional)
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

# Create helpful development scripts
echo "📜 Creating utility scripts..."

# System monitoring script
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
echo "Process List:"
ps aux | head -10
EOF
chmod +x /home/ubuntu/system_monitor.sh

# VS Code setup helper script
cat > /home/ubuntu/setup_vscode.sh << 'EOF'
#!/bin/bash
echo "🔧 VS Code Remote Development Setup"
echo "To connect from VS Code:"
echo "1. Install 'Remote - SSH' extension in VS Code"
echo "2. Press Ctrl+Shift+P and select 'Remote-SSH: Connect to Host'"
echo "3. Enter: ubuntu@$(curl -s ifconfig.me)"
echo "4. Use your SSH key for authentication"
echo ""
echo "Project is located at: ~/maht-net"
echo "Python environment: conda activate maht-net"
EOF
chmod +x /home/ubuntu/setup_vscode.sh

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

# Create welcome message for VS Code development
cat > /home/ubuntu/README_MAHT_NET.md << 'EOF'
# 🦷 MAHT-Net Development Instance Setup Complete! 🦷

## VS Code Remote Development Setup

### 1. Connect from VS Code
```bash
# Install "Remote - SSH" extension in VS Code
# Press Ctrl+Shift+P → "Remote-SSH: Connect to Host"
# Enter: ubuntu@YOUR_INSTANCE_IP
```

### 2. Activate the MAHT-Net Environment
```bash
source ~/miniconda3/bin/activate maht-net
```

### 3. Navigate to Project
```bash
cd ~/maht-net
```

### 4. Development Setup
- **Python Environment**: `conda activate maht-net`
- **PyTorch**: CPU version installed (perfect for development)
- **Project Structure**: All folders created locally
- **Data Storage**: Use `/data/` for larger datasets or `~/maht-net/data/` for small files

## Quick Commands

### Environment Management
```bash
# Activate environment
conda activate maht-net

# Install additional packages
pip install package_name

# Run Python scripts
python src/train.py
```

### Development Workflow
```bash
# Check system resources
./system_monitor.sh

# Start Jupyter (optional)
jupyter notebook

# Run tests
pytest tests/

# Format code
black src/
```

## Project Structure
```
~/maht-net/
├── src/                 # Source code
├── data/               # Local datasets
├── models/             # Model checkpoints  
├── results/            # Experiment results
├── logs/               # Training logs
├── notebooks/          # Jupyter notebooks
└── documentation/      # Project documentation
```

## Instance Information
- **Instance Type**: t3.large (2 vCPUs, 8GB RAM)
- **Storage**: 50GB root + 50GB data volume
- **Python**: 3.9 with conda environment
- **PyTorch**: CPU version (suitable for development and small experiments)

## Development Tips

### VS Code Extensions (Recommended)
- Python
- Jupyter
- GitLens
- Remote - SSH
- Docker
- Pylance

### Cost Management
- Instance auto-shuts down after 8 hours by default
- Stop instance when not in use: `sudo shutdown -h now`
- Monitor costs in AWS console

### When Ready for GPU Training
When you're ready to move to GPU training:
1. Switch to g4dn.xlarge or g5.xlarge instance type
2. Install CUDA version of PyTorch
3. Transfer your working code

## Troubleshooting

### Connection Issues
```bash
# Check SSH service
sudo systemctl status ssh

# Restart SSH
sudo systemctl restart ssh
```

### Python Issues
```bash
# Reinstall environment
conda env remove -n maht-net
conda create -n maht-net python=3.9
```

### Storage Issues
```bash
# Check disk usage
df -h

# Clean cache
conda clean --all
pip cache purge
```

Happy coding with VS Code! 💻🚀
EOF

chown ubuntu:ubuntu /home/ubuntu/README_MAHT_NET.md

echo "✅ MAHT-Net development instance setup complete!"
echo "🌐 Instance IP: $(curl -s ifconfig.me)"
echo "� Connect with VS Code Remote SSH: ubuntu@$(curl -s ifconfig.me)"
echo "📖 Read README_MAHT_NET.md for VS Code setup instructions"

# Final system update and cleanup
apt-get autoremove -y
apt-get autoclean

echo "🔄 Setup complete! Ready for VS Code remote development."
