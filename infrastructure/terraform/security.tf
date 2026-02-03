# ─── ALB Security Group ──────────────────────────────────────
resource "aws_security_group" "alb" {
  name_prefix = "${local.name_prefix}-alb-"
  description = "Allow HTTP/HTTPS inbound to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-alb-sg" }

  lifecycle { create_before_destroy = true }
}

# ─── ECS Tasks Security Group ────────────────────────────────
resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${local.name_prefix}-ecs-"
  description = "Allow traffic from ALB to ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Flask API from ALB"
    from_port       = var.flask_port
    to_port         = var.flask_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Streamlit from ALB"
    from_port       = var.streamlit_port
    to_port         = var.streamlit_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All outbound (NIM APIs, ECR, etc.)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-ecs-sg" }

  lifecycle { create_before_destroy = true }
}

# ─── GPU EC2 Security Group (only in GPU mode) ──────────────
resource "aws_security_group" "gpu_instances" {
  count       = var.enable_gpu ? 1 : 0
  name_prefix = "${local.name_prefix}-gpu-"
  description = "GPU EC2 instances running ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Flask from ALB"
    from_port       = var.flask_port
    to_port         = var.flask_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Streamlit from ALB"
    from_port       = var.streamlit_port
    to_port         = var.streamlit_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  # Omniverse Nucleus ports (for demo/visualization)
  ingress {
    description     = "Nucleus Web UI"
    from_port       = 3009
    to_port         = 3009
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Nucleus API"
    from_port       = 3019
    to_port         = 3019
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Nucleus Collaboration"
    from_port       = 3030
    to_port         = 3030
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-gpu-sg" }

  lifecycle { create_before_destroy = true }
}
