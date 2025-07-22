# MAHT-Net Terraform Configuration
# Multi-Stage Attention-enhanced Hybrid Transformer Network Infrastructure
#
# Konya Technical University
# PhD Research Project
# Developed by Mohamed Nourdine
# Supervisor: Dr. Öğr. Üyesi Betül Uzbaş

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "MAHT-Net"
      Environment = var.environment
      Owner       = "Mohamed Nourdine"
      University  = "Konya Technical University"
      Purpose     = "PhD Research - Cephalometric Landmark Detection"
      ManagedBy   = "Terraform"
    }
  }
}

# Random password for the instance
resource "random_password" "maht_net_password" {
  length  = 16
  special = true
}

# Data source for the latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Create VPC for the project
resource "aws_vpc" "maht_net_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "maht-net-vpc"
  }
}

# Create Internet Gateway
resource "aws_internet_gateway" "maht_net_igw" {
  vpc_id = aws_vpc.maht_net_vpc.id

  tags = {
    Name = "maht-net-igw"
  }
}

# Create public subnet
resource "aws_subnet" "maht_net_public_subnet" {
  vpc_id                  = aws_vpc.maht_net_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "maht-net-public-subnet"
  }
}

# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Create route table
resource "aws_route_table" "maht_net_route_table" {
  vpc_id = aws_vpc.maht_net_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.maht_net_igw.id
  }

  tags = {
    Name = "maht-net-route-table"
  }
}

# Associate route table with subnet
resource "aws_route_table_association" "maht_net_route_table_association" {
  subnet_id      = aws_subnet.maht_net_public_subnet.id
  route_table_id = aws_route_table.maht_net_route_table.id
}

# Security group for the MAHT-Net instance
resource "aws_security_group" "maht_net_sg" {
  name_prefix = "maht-net-sg"
  description = "Security group for MAHT-Net research instance"
  vpc_id      = aws_vpc.maht_net_vpc.id

  # SSH access (includes VS Code Remote Development)
  ingress {
    description = "SSH and VS Code Remote"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Jupyter notebook (optional)
  ingress {
    description = "Jupyter Notebook"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP for development servers
  ingress {
    description = "HTTP"
    from_port   = 8000
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "maht-net-security-group"
  }
}

# Key pair for SSH access
resource "aws_key_pair" "maht_net_key" {
  key_name   = "maht-net-key-${random_password.maht_net_password.id}"
  public_key = var.ssh_public_key

  tags = {
    Name = "maht-net-ssh-key"
  }
}

# IAM role for the EC2 instance (simplified - no S3 access needed)
resource "aws_iam_role" "maht_net_role" {
  name = "maht-net-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "maht-net-ec2-role"
  }
}

# Basic CloudWatch policy for logs
resource "aws_iam_role_policy" "maht_net_cloudwatch_policy" {
  name = "maht-net-cloudwatch-policy"
  role = aws_iam_role.maht_net_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "maht_net_profile" {
  name = "maht-net-instance-profile"
  role = aws_iam_role.maht_net_role.name

  tags = {
    Name = "maht-net-instance-profile"
  }
}

# EC2 Instance for MAHT-Net Development
resource "aws_instance" "maht_net_instance" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.maht_net_key.key_name
  vpc_security_group_ids = [aws_security_group.maht_net_sg.id]
  subnet_id              = aws_subnet.maht_net_public_subnet.id
  iam_instance_profile   = aws_iam_instance_profile.maht_net_profile.name

  # Root volume configuration
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size
    iops                  = 3000
    throughput            = 125
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "maht-net-root-volume"
    }
  }

  # Additional EBS volume for project data
  ebs_block_device {
    device_name           = "/dev/sdf"
    volume_type           = "gp3"
    volume_size           = var.data_volume_size
    iops                  = 3000
    throughput            = 125
    encrypted             = true
    delete_on_termination = false

    tags = {
      Name = "maht-net-data-volume"
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    aws_region         = var.aws_region
    auto_shutdown_time = var.auto_shutdown_time
  }))

  tags = {
    Name = "maht-net-development-instance"
    Type = "Development-Instance"
  }

  # Ensure the instance stops automatically after a certain time to save costs
  lifecycle {
    create_before_destroy = true
  }
}

# Elastic IP for consistent access
resource "aws_eip" "maht_net_eip" {
  instance = aws_instance.maht_net_instance.id
  domain   = "vpc"

  tags = {
    Name = "maht-net-elastic-ip"
  }

  depends_on = [aws_internet_gateway.maht_net_igw]
}

# CloudWatch Log Group for instance logs
resource "aws_cloudwatch_log_group" "maht_net_logs" {
  name              = "/aws/ec2/maht-net"
  retention_in_days = 30

  tags = {
    Name = "maht-net-cloudwatch-logs"
  }
}
