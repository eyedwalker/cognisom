# All resources conditional on enable_gpu = true

# ─── ECS-Optimized GPU AMI ──────────────────────────────────
data "aws_ami" "ecs_gpu" {
  count       = var.enable_gpu ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# ─── Launch Template ─────────────────────────────────────────
resource "aws_launch_template" "gpu" {
  count         = var.enable_gpu ? 1 : 0
  name_prefix   = "${local.name_prefix}-gpu-"
  image_id      = data.aws_ami.ecs_gpu[0].id
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.gpu[0].name
  }

  vpc_security_group_ids = [aws_security_group.gpu_instances[0].id]

  user_data = base64encode(templatefile("${path.module}/gpu_userdata.sh.tpl", {
    cluster_name = aws_ecs_cluster.main.name
  }))

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 50
      volume_type = "gp3"
      encrypted   = true
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${local.name_prefix}-gpu"
    }
  }
}

# ─── Auto Scaling Group ─────────────────────────────────────
resource "aws_autoscaling_group" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-gpu-asg"

  min_size         = 0
  max_size         = 1
  desired_capacity = 1

  vpc_zone_identifier = aws_subnet.private[*].id

  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu[0].id
        version            = "$Latest"
      }

      override {
        instance_type = var.gpu_instance_type
      }

      override {
        instance_type = "g4dn.2xlarge"
      }
    }

    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized"
      spot_max_price                           = var.gpu_spot_max_price
    }
  }

  protect_from_scale_in = true

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  tag {
    key                 = "Name"
    value               = "${local.name_prefix}-gpu"
    propagate_at_launch = true
  }
}

# ─── ECS Capacity Provider ──────────────────────────────────
resource "aws_ecs_capacity_provider" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-gpu"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu[0].arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
    }
  }
}
