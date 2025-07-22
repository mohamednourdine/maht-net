# MAHT-Net AWS Infrastructure Deployment Guide

## Overview

This Terraform configuration creates a complete AWS infrastructure for the MAHT-Net research project, including:

- **GPU-optimized EC2 instance** with NVIDIA drivers and CUDA 12.1
- **Complete ML environment** with PyTorch, Jupyter, and research tools
- **S3 bucket** for dataset and model storage
- **Security groups** and networking configuration
- **Monitoring and logging** setup
- **Cost optimization** features

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

### 3. Access Your Instance

After deployment completes (10-15 minutes), you'll get output with connection details:

```bash
# SSH to the instance
ssh -i ~/.ssh/maht-net-key ubuntu@<PUBLIC_IP>

# Access Jupyter Notebook
# Open http://<PUBLIC_IP>:8888 in your browser

# Access TensorBoard
# Open http://<PUBLIC_IP>:6006 in your browser
```

## Configuration Options

### Instance Types

| Instance Type | vCPUs | RAM | GPU | Use Case | Hourly Cost* |
|---------------|-------|-----|-----|----------|-------------|
| g4dn.xlarge   | 4     | 16GB| T4  | Development | ~$0.50 |
| g4dn.2xlarge  | 8     | 32GB| T4  | Training | ~$0.75 |
| g5.xlarge     | 4     | 16GB| A10G| Latest Gen | ~$1.00 |
| g5.2xlarge    | 8     | 32GB| A10G| Production | ~$1.20 |

*Prices are approximate and vary by region

### Storage Configuration

- **Root Volume**: 50GB (OS and software)
- **Data Volume**: 100GB+ (datasets and models)
- **S3 Bucket**: Unlimited (pay for usage)

### Cost Optimization

```hcl
# Enable auto-shutdown (recommended)
auto_shutdown_time = 480  # 8 hours

# Use spot instances for cost savings
enable_spot_instances = true
spot_price = "0.50"

# Smaller storage if budget is tight
data_volume_size = 50
```

## Post-Deployment Setup

### 1. Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Start Development

```bash
# Activate MAHT-Net environment
conda activate maht-net

# Navigate to project
cd ~/maht-net

# Start Jupyter (if not auto-started)
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### 3. Data Management

```bash
# Upload datasets to S3
aws s3 sync /local/datasets/ s3://your-bucket-name/datasets/

# Download to instance
aws s3 sync s3://your-bucket-name/datasets/ /data/datasets/
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
| g4dn.xlarge (24/7) | ~$360 |
| g4dn.xlarge (8h/day) | ~$120 |
| EBS Storage (150GB) | ~$15 |
| S3 Storage (100GB) | ~$2 |
| Data Transfer | ~$5-20 |

**Total for 8h/day usage**: ~$140-160/month

### Cost Optimization Tips

1. **Use auto-shutdown**: Reduces costs by 66%
2. **Spot instances**: Save up to 70% (less reliable)
3. **Right-size storage**: Only allocate what you need
4. **Monitor usage**: Set up billing alerts
5. **Use S3 for long-term storage**: Much cheaper than EBS
