# ─── ECS Task Execution Role (pulls images, writes logs) ─────
resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.name_prefix}-ecs-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_policy" "secrets_read" {
  name = "${local.name_prefix}-secrets-read"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = concat([
        aws_secretsmanager_secret.nvidia_api_key.arn,
        aws_secretsmanager_secret.ngc_api_key.arn,
        aws_secretsmanager_secret.secret_key.arn,
      ],
      # Include database credentials if RDS is enabled
      var.enable_rds ? [aws_secretsmanager_secret.db_credentials[0].arn] : [])
    }]
  })
}

resource "aws_iam_role_policy_attachment" "secrets_read" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = aws_iam_policy.secrets_read.arn
}

# ─── ECS Task Role (what the container can do at runtime) ────
resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_policy" "s3_exports" {
  name = "${local.name_prefix}-s3-exports"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.exports.arn,
        "${aws_s3_bucket.exports.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "s3_exports" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.s3_exports.arn
}

# ─── Cognito Admin Access (for role mapping) ─────────────────
resource "aws_iam_policy" "cognito_admin" {
  count = var.enable_cognito ? 1 : 0
  name  = "${local.name_prefix}-cognito-admin"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "cognito-idp:AdminListGroupsForUser",
        "cognito-idp:AdminGetUser",
        "cognito-idp:ListUsers"
      ]
      Resource = aws_cognito_user_pool.main[0].arn
    }]
  })
}

resource "aws_iam_role_policy_attachment" "cognito_admin" {
  count      = var.enable_cognito ? 1 : 0
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.cognito_admin[0].arn
}

# ─── EC2 Instance Role (GPU mode only) ──────────────────────
resource "aws_iam_role" "ec2_gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-ec2-gpu"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ec2_ecs" {
  count      = var.enable_gpu ? 1 : 0
  role       = aws_iam_role.ec2_gpu[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  count      = var.enable_gpu ? 1 : 0
  role       = aws_iam_role.ec2_gpu[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu" {
  count = var.enable_gpu ? 1 : 0
  name  = "${local.name_prefix}-gpu-profile"
  role  = aws_iam_role.ec2_gpu[0].name
}
