# MAHT-Net Terraform Variables
# Configuration variables for AWS infrastructure deployment

variable "aws_region" {
  description = "AWS region for MAHT-Net infrastructure"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = contains([
      "us-west-1", "us-west-2", "us-east-1", "us-east-2",
      "eu-west-1", "eu-west-2", "eu-central-1",
      "ap-southeast-1", "ap-southeast-2", "ap-northeast-1"
    ], var.aws_region)
    error_message = "Please select a supported AWS region with GPU instance availability."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "instance_type" {
  description = "EC2 instance type for MAHT-Net training and inference"
  type        = string
  default     = "g4dn.xlarge"
  
  validation {
    condition = contains([
      "g4dn.xlarge",    # 4 vCPUs, 16 GB RAM, 1x NVIDIA T4 GPU - Cost-effective for development
      "g4dn.2xlarge",   # 8 vCPUs, 32 GB RAM, 1x NVIDIA T4 GPU - Better for training
      "g4dn.4xlarge",   # 16 vCPUs, 64 GB RAM, 1x NVIDIA T4 GPU - High performance
      "g5.xlarge",      # 4 vCPUs, 16 GB RAM, 1x NVIDIA A10G GPU - Latest generation
      "g5.2xlarge",     # 8 vCPUs, 32 GB RAM, 1x NVIDIA A10G GPU - Recommended for training
      "g5.4xlarge",     # 16 vCPUs, 64 GB RAM, 1x NVIDIA A10G GPU - High performance
      "p3.2xlarge",     # 8 vCPUs, 61 GB RAM, 1x NVIDIA V100 GPU - High-end training
      "p4d.24xlarge"    # 96 vCPUs, 1152 GB RAM, 8x NVIDIA A100 GPU - Enterprise scale
    ], var.instance_type)
    error_message = "Please select a supported GPU instance type for deep learning."
  }
}

variable "ssh_public_key" {
  description = "SSH public key for accessing the MAHT-Net instance"
  type        = string
  
  validation {
    condition     = can(regex("^ssh-(rsa|dss|ecdsa|ed25519)", var.ssh_public_key))
    error_message = "Please provide a valid SSH public key."
  }
}

variable "root_volume_size" {
  description = "Size of the root EBS volume in GB"
  type        = number
  default     = 50
  
  validation {
    condition     = var.root_volume_size >= 30 && var.root_volume_size <= 500
    error_message = "Root volume size must be between 30 and 500 GB."
  }
}

variable "data_volume_size" {
  description = "Size of the additional data EBS volume in GB for datasets and models"
  type        = number
  default     = 100
  
  validation {
    condition     = var.data_volume_size >= 50 && var.data_volume_size <= 1000
    error_message = "Data volume size must be between 50 and 1000 GB."
  }
}

variable "auto_shutdown_time" {
  description = "Time in minutes after which the instance should auto-shutdown (0 to disable)"
  type        = number
  default     = 480  # 8 hours
  
  validation {
    condition     = var.auto_shutdown_time >= 0 && var.auto_shutdown_time <= 1440
    error_message = "Auto shutdown time must be between 0 and 1440 minutes (24 hours)."
  }
}

variable "enable_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "allowed_ssh_cidrs" {
  description = "List of CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
  
  validation {
    condition = alltrue([
      for cidr in var.allowed_ssh_cidrs : can(cidrhost(cidr, 0))
    ])
    error_message = "All items must be valid CIDR blocks."
  }
}

variable "project_name" {
  description = "Name of the project for resource tagging"
  type        = string
  default     = "MAHT-Net"
}

variable "researcher_name" {
  description = "Name of the researcher for resource tagging"
  type        = string
  default     = "Mohamed Nourdine"
}

variable "university" {
  description = "University name for resource tagging"
  type        = string
  default     = "Konya Technical University"
}

variable "enable_spot_instances" {
  description = "Use spot instances for cost optimization (not recommended for long training jobs)"
  type        = bool
  default     = false
}

variable "spot_price" {
  description = "Maximum spot price per hour (only used if enable_spot_instances is true)"
  type        = string
  default     = "0.50"
}

variable "backup_retention_days" {
  description = "Number of days to retain EBS snapshots"
  type        = number
  default     = 7
  
  validation {
    condition     = var.backup_retention_days >= 1 && var.backup_retention_days <= 30
    error_message = "Backup retention must be between 1 and 30 days."
  }
}
