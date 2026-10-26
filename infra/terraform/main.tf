# AEGIS on AWS: one K3s workload node + one small load-generator node.
# Default VPC, public subnet, no NAT / load balancer / Elastic IP (cost traps).
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region"       { default = "ap-south-1" }
variable "my_ip"        { description = "your public IP, for SSH" }
variable "node_type"    { default = "m7i-flex.large" }  # 2 vCPU / 8 GiB, Free-plan eligible
variable "loadgen_type" { default = "t3.small" }        # 2 vCPU / 2 GiB, Free-plan eligible

data "aws_vpc" "default" { default = true }

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_key_pair" "aegis" {
  key_name   = "aegis"
  public_key = file(pathexpand("~/.ssh/aegis.pub"))
}

resource "aws_security_group" "aegis" {
  name        = "aegis"
  description = "AEGIS: SSH from my IP, everything between the two nodes"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH from my IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.my_ip}/32"]
  }
  ingress {
    description = "node to loadgen (HTTP, K8s API, Prometheus)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "node" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.node_type
  key_name               = aws_key_pair.aegis.key_name
  vpc_security_group_ids = [aws_security_group.aegis.id]
  user_data = <<-EOT
    #!/bin/bash
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--write-kubeconfig-mode 644" sh -
  EOT
  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }
  tags = { Name = "aegis-node", Project = "aegis" }
}

resource "aws_instance" "loadgen" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.loadgen_type
  key_name               = aws_key_pair.aegis.key_name
  vpc_security_group_ids = [aws_security_group.aegis.id]
  user_data = <<-EOT
    #!/bin/bash
    snap install k6 || true
    snap install kubectl --classic || true
    snap install helm --classic || true
    apt-get update -y && apt-get install -y python3 git
  EOT
  root_block_device {
    volume_size = 16
    volume_type = "gp3"
  }
  tags = { Name = "aegis-loadgen", Project = "aegis" }
}

output "node_ip"         { value = aws_instance.node.public_ip }
output "node_private_ip" { value = aws_instance.node.private_ip }
output "loadgen_ip"      { value = aws_instance.loadgen.public_ip }
output "instance_ids"    { value = "${aws_instance.node.id} ${aws_instance.loadgen.id}" }
