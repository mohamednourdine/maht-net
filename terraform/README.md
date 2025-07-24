# MAHT-Net AWS Development Infrastructure

## Overview

This Terraform configuration creates a cost-effective AWS development environment for the MAHT-Net research project, including:

- **CPU-optimized EC2 instance** (t3.large) for development and testing
- **Complete ML environment** with PyTorch CPU, Jupyter, and development tools
- **VS Code Remote Development** support with all necessary extensions
- **Local storage** for datasets and models (no S3 dependency)
- **Security and cost management** features

## Prerequisites

### 1. Install Required Tools

```bash
# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### 2. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure
# Enter your AWS Access Key ID, Secret Access Key, region, and output format
```

### 3. Generate SSH Key Pair

```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/maht-net-key
chmod 400 ~/.ssh/maht-net-key

# Display public key (copy this for terraform.tfvars)
cat ~/.ssh/maht-net-key.pub
```

## Quick Deployment

### 1. Configure Variables

```bash
# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
vim terraform.tfvars
```

**Important**: Update the `ssh_public_key` variable with your actual SSH public key!

### 2. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Review the deployment plan
terraform plan

# Deploy infrastructure
terraform apply
```

### 3. Connect with VS Code

After deployment completes (5-10 minutes):

```bash
# Install VS Code Remote-SSH extension
# Press Ctrl+Shift+P → "Remote-SSH: Connect to Host"
# Enter: ubuntu@<PUBLIC_IP>
```

**VS Code Connection Steps:**
1. Install "Remote - SSH" extension
2. Open Command Palette (Ctrl+Shift+P)
3. Select "Remote-SSH: Connect to Host"
4. Enter connection string from terraform output
5. Select Linux platform
6. Open ~/maht-net folder

## Configuration Options

### Instance Types

| Instance Type | vCPUs | RAM | Use Case | Hourly Cost* |
|---------------|-------|-----|----------|-------------|
| t3.medium     | 2     | 4GB | Basic development | ~$0.04 |
| t3.large      | 2     | 8GB | Recommended dev | ~$0.08 |
| t3.xlarge     | 4     | 16GB| Enhanced dev | ~$0.17 |
| m5.large      | 2     | 8GB | Balanced compute | ~$0.10 |

*Prices are approximate and vary by region

### Storage Configuration

- **Root Volume**: 50GB (OS and software)
- **Data Volume**: 50GB (project data and datasets)
- **No S3**: All data stored locally for simplicity

### Cost Optimization

```hcl
# Enable auto-shutdown (recommended)
auto_shutdown_time = 480  # 8 hours

# Use spot instances for cost savings
enable_spot_instances = true
spot_price = "0.50"

# Minimal storage for development
data_volume_size = 30
```

## Development Workflow

### 1. VS Code Setup

```bash
# Connect to instance
# VS Code will automatically detect Python environment
# Open integrated terminal and activate conda:
conda activate maht-net
```

### 2. Development Commands

```bash
# Navigate to project
cd ~/maht-net

# Install additional packages
pip install package_name

# Run Python scripts
python src/train.py

# Run tests
pytest tests/

# Format code
black src/
```

### 3. Data Management

```bash
# Upload small datasets via VS Code
# For larger datasets, use scp:
scp -i ~/.ssh/id_rsa dataset.zip ubuntu@3.224.200.252:/home/ubuntu/maht-net/data/raw

# Extract and organize
cd ~/maht-net/data
unzip dataset.zip
```

## Security Considerations

### 1. Restrict SSH Access

```hcl
# In terraform.tfvars
allowed_ssh_cidrs = ["YOUR.IP.ADDRESS/32"]  # Your IP only
```

### 2. Use IAM Roles

The instance includes an IAM role with minimal S3 permissions. No AWS keys stored on instance.

### 3. Enable Encryption

All EBS volumes and S3 bucket are encrypted by default.

## Monitoring and Logging

### CloudWatch Integration

- **Instance metrics**: CPU, memory, disk, network
- **Custom logs**: Application and training logs
- **Cost monitoring**: Billing alerts and usage tracking

### Built-in Monitoring Scripts

```bash
# GPU monitoring
./gpu_monitor.sh

# System resources
./system_monitor.sh

# Check service status
sudo systemctl status jupyter
```

## Troubleshooting

### Common Issues

1. **CUDA not detected**
   ```bash
   # Reboot instance
   sudo reboot
   
   # Check driver installation
   nvidia-smi
   lsmod | grep nvidia
   ```

2. **Out of memory**
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Clear cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Permission issues**
   ```bash
   # Fix ownership
   sudo chown -R ubuntu:ubuntu /home/ubuntu/maht-net
   sudo chown -R ubuntu:ubuntu /data
   ```

4. **Jupyter not accessible**
   ```bash
   # Check service
   sudo systemctl status jupyter
   
   # Restart service
   sudo systemctl restart jupyter
   
   # Check port binding
   netstat -tulpn | grep 8888
   ```

### Cost Management

```bash
# Check current costs
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-31 --granularity MONTHLY --metrics BlendedCost

# Stop instance when not in use
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

## Cleanup

### Destroy Infrastructure

```bash
# Remove all resources
terraform destroy

# Confirm by typing 'yes'
```

**Warning**: This will delete all instances, volumes, and data. Backup important data to S3 first!

### Partial Cleanup

```bash
# Stop instance only
aws ec2 stop-instances --instance-ids $(terraform output -raw instance_id)

# Create snapshot before destroying
aws ec2 create-snapshot --volume-id $(terraform output -raw root_volume_id) --description "MAHT-Net backup"
```

## Advanced Configuration

### Multi-GPU Setup

```hcl
# For multiple GPUs
instance_type = "g4dn.12xlarge"  # 4x T4 GPUs
# or
instance_type = "p3.8xlarge"     # 4x V100 GPUs
```

### Custom AMI

```hcl
# Use custom AMI with pre-installed software
ami_id = "ami-12345678"  # Your custom AMI
```

### VPC Peering

```hcl
# Connect to existing VPC
vpc_id = "vpc-12345678"
subnet_id = "subnet-12345678"
```

## Support

For issues specific to the MAHT-Net project:
- 📧 Contact: Mohamed Nourdine
- 🏛️ Institution: Konya Technical University
- 📚 Documentation: `/var/www/phd-researches/maht-net/documentation/`

For AWS-specific issues:
- 📖 AWS Documentation: https://docs.aws.amazon.com/
- 💬 AWS Support: https://aws.amazon.com/support/

## Cost Estimation

### Monthly Costs (Approximate)

| Component | Cost (USD/month) |
|-----------|------------------|
| t3.large (24/7) | ~$60 |
| t3.large (8h/day) | ~$20 |
| EBS Storage (100GB) | ~$10 |

**Total for 8h/day usage**: ~$30/month

### Cost Optimization Tips

1. **Use auto-shutdown**: Reduces costs by 66%
2. **Spot instances**: Save up to 70% (less reliable)
3. **Right-size storage**: Only allocate what you need
4. **Monitor usage**: Set up billing alerts
5. **Stop when not developing**: Use AWS console or CLI
