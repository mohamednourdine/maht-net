# MAHT-Net Infrastructure Outputs
# Important information about the deployed AWS resources

output "instance_id" {
  description = "ID of the MAHT-Net EC2 instance"
  value       = aws_instance.maht_net_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the MAHT-Net instance"
  value       = aws_eip.maht_net_eip.public_ip
}

output "instance_private_ip" {
  description = "Private IP address of the MAHT-Net instance"
  value       = aws_instance.maht_net_instance.private_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the MAHT-Net instance"
  value       = aws_instance.maht_net_instance.public_dns
}

output "ssh_connection_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/maht-net-key ubuntu@${aws_eip.maht_net_eip.public_ip}"
}

output "jupyter_url" {
  description = "URL to access Jupyter Notebook"
  value       = "http://${aws_eip.maht_net_eip.public_ip}:8888"
}

output "tensorboard_url" {
  description = "URL to access TensorBoard"
  value       = "http://${aws_eip.maht_net_eip.public_ip}:6006"
}

output "api_url" {
  description = "URL to access the FastAPI service"
  value       = "http://${aws_eip.maht_net_eip.public_ip}:8000"
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for data storage"
  value       = aws_s3_bucket.maht_net_bucket.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.maht_net_bucket.arn
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.maht_net_vpc.id
}

output "subnet_id" {
  description = "ID of the public subnet"
  value       = aws_subnet.maht_net_public_subnet.id
}

output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.maht_net_sg.id
}

output "key_pair_name" {
  description = "Name of the SSH key pair"
  value       = aws_key_pair.maht_net_key.key_name
}

output "iam_role_arn" {
  description = "ARN of the IAM role for the instance"
  value       = aws_iam_role.maht_net_role.arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.maht_net_logs.name
}

output "instance_type" {
  description = "Type of the EC2 instance"
  value       = aws_instance.maht_net_instance.instance_type
}

output "availability_zone" {
  description = "Availability zone of the instance"
  value       = aws_instance.maht_net_instance.availability_zone
}

output "root_volume_id" {
  description = "ID of the root EBS volume"
  value       = aws_instance.maht_net_instance.root_block_device[0].volume_id
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost for the infrastructure (USD)"
  value = {
    instance_cost = "Depends on instance type and usage hours"
    storage_cost  = "~$${(var.root_volume_size + var.data_volume_size) * 0.10} for EBS storage"
    data_transfer = "First 1GB outbound free, then $0.09/GB"
    total_estimate = "Check AWS Pricing Calculator for exact costs"
  }
}

output "quick_start_guide" {
  description = "Quick start commands and URLs"
  value = {
    ssh_command     = "ssh -i ~/.ssh/maht-net-key ubuntu@${aws_eip.maht_net_eip.public_ip}"
    jupyter_url     = "http://${aws_eip.maht_net_eip.public_ip}:8888"
    tensorboard_url = "http://${aws_eip.maht_net_eip.public_ip}:6006"
    api_url         = "http://${aws_eip.maht_net_eip.public_ip}:8000"
    s3_bucket       = aws_s3_bucket.maht_net_bucket.bucket
    setup_status    = "Instance will be ready in ~10-15 minutes after launch"
  }
}

output "security_information" {
  description = "Important security information"
  value = {
    vpc_cidr           = aws_vpc.maht_net_vpc.cidr_block
    public_subnet_cidr = aws_subnet.maht_net_public_subnet.cidr_block
    ssh_access         = "Configured for global access (0.0.0.0/0) - consider restricting"
    encryption         = "EBS volumes and S3 bucket are encrypted"
    iam_role          = "Instance has permissions for S3 bucket access only"
  }
}

output "monitoring_information" {
  description = "Monitoring and logging information"
  value = {
    cloudwatch_logs    = aws_cloudwatch_log_group.maht_net_logs.name
    instance_monitoring = var.enable_monitoring ? "Enabled" : "Basic monitoring only"
    log_retention      = "30 days"
    backup_retention   = "${var.backup_retention_days} days"
  }
}
