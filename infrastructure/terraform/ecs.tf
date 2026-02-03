# ─── ECS Cluster ─────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = concat(
    ["FARGATE", "FARGATE_SPOT"],
    var.enable_gpu ? [aws_ecs_capacity_provider.gpu[0].name] : []
  )

  default_capacity_provider_strategy {
    capacity_provider = var.enable_gpu ? aws_ecs_capacity_provider.gpu[0].name : "FARGATE"
    weight            = 1
    base              = 1
  }
}

# ─── Container Definition (shared JSON) ──────────────────────
locals {
  container_environment = [
    { name = "FLASK_DEBUG", value = "false" },
    { name = "CORS_ORIGINS", value = "https://${var.domain_name}" },
    { name = "COGNISOM_GPU", value = var.enable_gpu ? "1" : "0" },
  ]

  container_secrets = [
    { name = "NVIDIA_API_KEY", valueFrom = aws_secretsmanager_secret.nvidia_api_key.arn },
    { name = "NGC_API_KEY", valueFrom = aws_secretsmanager_secret.ngc_api_key.arn },
    { name = "SECRET_KEY", valueFrom = aws_secretsmanager_secret.secret_key.arn },
  ]

  container_log_config = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
      "awslogs-region"        = var.aws_region
      "awslogs-stream-prefix" = "ecs"
    }
  }
}

# ─── CPU Task Definition (Fargate) ───────────────────────────
resource "aws_ecs_task_definition" "cpu" {
  count  = var.enable_gpu ? 0 : 1
  family = "${var.project_name}-cpu"

  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "cognisom"
    image     = local.ecr_image
    essential = true

    portMappings = [
      { containerPort = var.flask_port, protocol = "tcp" },
      { containerPort = var.streamlit_port, protocol = "tcp" },
    ]

    environment    = local.container_environment
    secrets        = local.container_secrets
    logConfiguration = local.container_log_config

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${var.flask_port}/api/health || exit 1"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 60
    }
  }])
}

# ─── GPU Task Definition (EC2) with Omniverse Nucleus ────────
resource "aws_ecs_task_definition" "gpu" {
  count  = var.enable_gpu ? 1 : 0
  family = "${var.project_name}-gpu"

  requires_compatibilities = ["EC2"]
  network_mode             = "bridge"
  cpu                      = 4096
  memory                   = 15360
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    # Main Cognisom application
    {
      name      = "cognisom"
      image     = local.ecr_image
      essential = true
      memory    = 8192

      portMappings = [
        { containerPort = var.flask_port, hostPort = var.flask_port, protocol = "tcp" },
        { containerPort = var.streamlit_port, hostPort = var.streamlit_port, protocol = "tcp" },
      ]

      resourceRequirements = [{ type = "GPU", value = "1" }]

      environment = concat(local.container_environment, [
        { name = "OMNIVERSE_URL", value = "omniverse://localhost:3019/cognisom" },
      ])
      secrets          = local.container_secrets
      logConfiguration = local.container_log_config

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.flask_port}/api/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }

      links = ["nucleus"]
    },

    # Omniverse Nucleus sidecar for 3D visualization
    {
      name      = "nucleus"
      image     = "nvcr.io/nvidia/omniverse/nucleus:2024.1"
      essential = false
      memory    = 4096

      portMappings = [
        { containerPort = 3009, hostPort = 3009, protocol = "tcp" },  # Web UI
        { containerPort = 3019, hostPort = 3019, protocol = "tcp" },  # API
        { containerPort = 3030, hostPort = 3030, protocol = "tcp" },  # Collaboration
      ]

      environment = [
        { name = "ACCEPT_EULA", value = "Y" },
        { name = "SECURITY_ENABLED", value = "false" },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "nucleus"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3009/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 120
      }
    }
  ])
}

# ─── CPU Service (Fargate) ───────────────────────────────────
resource "aws_ecs_service" "cpu" {
  count           = var.enable_gpu ? 0 : 1
  name            = "${local.name_prefix}-cpu"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.cpu[0].arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "cognisom"
    container_port   = var.streamlit_port
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.flask.arn
    container_name   = "cognisom"
    container_port   = var.flask_port
  }

  depends_on = [aws_lb_listener.https]
}

# ─── GPU Service (EC2) ──────────────────────────────────────
resource "aws_ecs_service" "gpu" {
  count           = var.enable_gpu ? 1 : 0
  name            = "${local.name_prefix}-gpu"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.gpu[0].arn
  desired_count   = var.desired_count

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu[0].name
    weight            = 1
    base              = 1
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.streamlit.arn
    container_name   = "cognisom"
    container_port   = var.streamlit_port
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.flask.arn
    container_name   = "cognisom"
    container_port   = var.flask_port
  }

  depends_on = [aws_lb_listener.https]
}
