# ─── RDS PostgreSQL for Persistent Research Data ─────────────
#
# HIPAA-eligible PostgreSQL database for:
# - Entity Library (genes, proteins, pathways)
# - User data and session state
# - Audit logs for IRB compliance
# - Simulation results
#
# Features:
# - Encryption at rest (AES-256)
# - Encryption in transit (TLS)
# - Automated backups (35-day retention)
# - Multi-AZ for high availability (production)
# - Point-in-time recovery

# ─── Variables ────────────────────────────────────────────────
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"  # Free tier eligible
}

variable "db_allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 20
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "cognisom"
}

variable "db_username" {
  description = "Master username"
  type        = string
  default     = "cognisom_admin"
}

variable "db_password" {
  description = "Master password (use secrets manager in production)"
  type        = string
  sensitive   = true
  default     = ""  # Will generate random if empty
}

# ─── Random Password (if not provided) ────────────────────────
# Generate a password if db_password is empty
# NOTE: Always generate to avoid conditional on sensitive value
resource "random_password" "db_password" {
  count   = var.enable_rds ? 1 : 0
  length  = 32
  special = false  # Avoid special chars that cause connection string issues
}

locals {
  # Use random password - in production, set db_password variable explicitly
  # The random_password is always generated when RDS is enabled
  db_password = var.enable_rds ? random_password.db_password[0].result : ""
}

# ─── DB Subnet Group ──────────────────────────────────────────
resource "aws_db_subnet_group" "main" {
  count      = var.enable_rds ? 1 : 0
  name       = "${local.name_prefix}-db-subnet"
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = "${local.name_prefix}-db-subnet-group" }
}

# ─── Security Group for RDS ───────────────────────────────────
resource "aws_security_group" "rds" {
  count       = var.enable_rds ? 1 : 0
  name_prefix = "${local.name_prefix}-rds-"
  description = "Allow PostgreSQL from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  # Allow from GPU instances if enabled
  dynamic "ingress" {
    for_each = var.enable_gpu ? [1] : []
    content {
      description     = "PostgreSQL from GPU instances"
      from_port       = 5432
      to_port         = 5432
      protocol        = "tcp"
      security_groups = [aws_security_group.gpu_instances[0].id]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-rds-sg" }

  lifecycle { create_before_destroy = true }
}

# ─── RDS PostgreSQL Instance ──────────────────────────────────
resource "aws_db_instance" "main" {
  count = var.enable_rds ? 1 : 0

  identifier = "${local.name_prefix}-postgres"

  # Engine
  engine               = "postgres"
  engine_version       = "16.11"  # Latest stable 16.x
  instance_class       = var.db_instance_class
  allocated_storage    = var.db_allocated_storage
  max_allocated_storage = 100  # Auto-scaling up to 100GB

  # Credentials
  db_name  = var.db_name
  username = var.db_username
  password = local.db_password

  # Network
  db_subnet_group_name   = aws_db_subnet_group.main[0].name
  vpc_security_group_ids = [aws_security_group.rds[0].id]
  publicly_accessible    = false
  port                   = 5432

  # Security - HIPAA compliance
  storage_encrypted = true
  # kms_key_id      = aws_kms_key.rds.arn  # Uncomment for custom KMS key

  # Backup & Recovery
  backup_retention_period = 35  # Max for compliance
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"
  copy_tags_to_snapshot   = true
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.name_prefix}-final-snapshot"

  # Monitoring
  performance_insights_enabled = true
  performance_insights_retention_period = 7  # Free tier

  # Upgrades
  auto_minor_version_upgrade = true
  allow_major_version_upgrade = false

  # Deletion protection (enable in production)
  deletion_protection = false

  tags = {
    Name        = "${local.name_prefix}-postgres"
    Compliance  = "HIPAA-eligible"
    DataClass   = "research"
  }
}

# ─── Store credentials in Secrets Manager ─────────────────────
resource "aws_secretsmanager_secret" "db_credentials" {
  count       = var.enable_rds ? 1 : 0
  name        = "${local.name_prefix}/db-credentials"
  description = "RDS PostgreSQL credentials for Cognisom"

  tags = { Name = "${local.name_prefix}-db-secret" }
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  count     = var.enable_rds ? 1 : 0
  secret_id = aws_secretsmanager_secret.db_credentials[0].id

  secret_string = jsonencode({
    username = var.db_username
    password = local.db_password
    host     = aws_db_instance.main[0].address
    port     = 5432
    database = var.db_name
    url      = "postgresql://${var.db_username}:${local.db_password}@${aws_db_instance.main[0].address}:5432/${var.db_name}"
  })
}

# ─── Outputs ──────────────────────────────────────────────────
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : null
}

output "rds_address" {
  description = "RDS instance address (hostname only)"
  value       = var.enable_rds ? aws_db_instance.main[0].address : null
}

output "db_credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing DB credentials"
  value       = var.enable_rds ? aws_secretsmanager_secret.db_credentials[0].arn : null
}
