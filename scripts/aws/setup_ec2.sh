#!/bin/bash

# MAHT-Net AWS EC2 Setup Script
# Optimized for Ubuntu 20.04 LTS with GPU support (g5.xlarge, g4dn.xlarge)

set -e

echo "🚀 MAHT-Net AWS EC2 Environment Setup"
echo "======================================"

# System information
echo "📊 System Information:"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install essential dependencies
echo "🔧 Installing essential dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Python 3.9+
echo "🐍 Installing Python 3.9..."
sudo apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install NVIDIA drivers and CUDA 12.1
echo "🎮 Installing NVIDIA drivers and CUDA 12.1..."

# Remove any existing NVIDIA packages
sudo apt-get remove --purge -y '^nvidia-.*'
sudo apt-get remove --purge -y '^cuda-.*'
sudo apt-get autoremove -y

# Install NVIDIA driver
sudo apt-get install -y nvidia-driver-530

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y cuda-12-1

# Install cuDNN 8.9
echo "🧠 Installing cuDNN 8.9..."
wget https://developer.download.nvidia.com/compute/cudnn/secure/8.9.2/local_installers/12.x/cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.2.26_cuda12-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.9.2.26_cuda12-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# Set up environment variables
echo "🌐 Setting up environment variables..."
cat >> ~/.bashrc << EOF

# CUDA Environment Variables
export PATH=/usr/local/cuda-12.1/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1

# MAHT-Net Environment
export MAHT_NET_HOME=\$HOME/maht-net
export PYTHONPATH=\$MAHT_NET_HOME/src:\$PYTHONPATH

EOF

source ~/.bashrc

# Install Docker for containerized deployment
echo "🐳 Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
echo "🔧 Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install AWS CLI v2
echo "☁️ Installing AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Configure AWS CLI (interactive)
echo "🔑 AWS CLI Configuration:"
echo "Please configure your AWS credentials:"
aws configure

# Install monitoring tools
echo "📊 Installing monitoring tools..."
sudo apt-get install -y htop nvtop iotop nethogs

# Install Node.js for web dashboard (optional)
echo "🌐 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Setup swap file for memory management
echo "💾 Setting up swap file..."
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Configure system limits for ML workloads
echo "⚙️ Configuring system limits..."
cat << EOF | sudo tee -a /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
EOF

# Create directory structure
echo "📁 Creating MAHT-Net directory structure..."
mkdir -p $HOME/maht-net/{data,models,logs,results}
mkdir -p $HOME/maht-net/data/{raw,processed,augmented,splits}
mkdir -p $HOME/maht-net/models/{checkpoints,pretrained,configs}
mkdir -p $HOME/maht-net/results/{experiments,ablation,clinical_validation}
mkdir -p $HOME/maht-net/logs/{training,evaluation,deployment}

# Download pre-trained models (if available)
echo "📥 Setting up model directory..."
cd $HOME/maht-net/models/pretrained
wget -O efficientnet_b3_imagenet.pth https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth

# Setup TensorBoard service
echo "📊 Setting up TensorBoard service..."
sudo tee /etc/systemd/system/tensorboard.service > /dev/null <<EOF
[Unit]
Description=TensorBoard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/maht-net
ExecStart=/usr/bin/python3 -m tensorboard.main --logdir=logs/training --host=0.0.0.0 --port=6006
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tensorboard

# Setup firewall rules
echo "🔒 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 6006/tcp  # TensorBoard
sudo ufw allow 8000/tcp  # FastAPI
sudo ufw --force enable

# Final system optimizations
echo "⚡ Applying system optimizations..."

# GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 6001,1590  # Memory and graphics clocks for optimal ML performance

# CPU performance governor
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify installations
echo "✅ Verifying installations..."
echo "Python version: $(python3 --version)"
echo "CUDA version: $(nvcc --version | grep release)"
echo "Docker version: $(docker --version)"
echo "AWS CLI version: $(aws --version)"

# Test GPU availability
echo "🎮 Testing GPU availability..."
nvidia-smi

echo ""
echo "🎉 MAHT-Net AWS EC2 setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Reboot the system: sudo reboot"
echo "2. After reboot, verify GPU: nvidia-smi"
echo "3. Clone MAHT-Net repository: git clone <repository-url>"
echo "4. Run: make setup"
echo "5. Start training: make train"
echo ""
echo "🌐 Access TensorBoard at: http://<instance-ip>:6006"
echo "📊 Monitor GPU usage: watch -n 1 nvidia-smi"
echo ""
echo "💡 For production deployment, run: make deploy"

# Save setup completion timestamp
echo "$(date): MAHT-Net EC2 setup completed" >> $HOME/maht-net-setup.log
