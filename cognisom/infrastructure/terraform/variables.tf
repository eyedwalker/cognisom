# ─── Core ────────────────────────────────────────────────────
variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "cognisom"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# ─── GPU Toggle ──────────────────────────────────────────────
variable "enable_gpu" {
  description = "Enable GPU mode (EC2 g4dn.xlarge). False = Fargate CPU only."
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU mode"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_spot_max_price" {
  description = "Max hourly spot price for GPU instance (on-demand is ~$0.526)"
  type        = string
  default     = "0.30"
}

# ─── Domain ──────────────────────────────────────────────────
variable "domain_name" {
  description = "Root domain name"
  type        = string
  default     = "cognisom.com"
}

# ─── Networking ──────────────────────────────────────────────
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

# ─── Container ───────────────────────────────────────────────
variable "container_cpu" {
  description = "Fargate CPU units (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "container_memory" {
  description = "Fargate memory in MB"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of ECS tasks to run"
  type        = number
  default     = 1
}

variable "flask_port" {
  description = "Flask API port"
  type        = number
  default     = 5000
}

variable "streamlit_port" {
  description = "Streamlit dashboard port"
  type        = number
  default     = 8501
}

# ─── Secrets ─────────────────────────────────────────────────
variable "nvidia_api_key" {
  description = "NVIDIA NIM API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "ngc_api_key" {
  description = "NVIDIA NGC API key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "secret_key" {
  description = "Flask session secret key"
  type        = string
  sensitive   = true
  default     = ""
}
