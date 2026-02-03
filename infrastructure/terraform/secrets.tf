# ─── NVIDIA API Key ──────────────────────────────────────────
resource "aws_secretsmanager_secret" "nvidia_api_key" {
  name                    = "${var.project_name}/nvidia-api-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "nvidia_api_key" {
  secret_id     = aws_secretsmanager_secret.nvidia_api_key.id
  secret_string = var.nvidia_api_key

  lifecycle { ignore_changes = [secret_string] }
}

# ─── NGC API Key ─────────────────────────────────────────────
resource "aws_secretsmanager_secret" "ngc_api_key" {
  name                    = "${var.project_name}/ngc-api-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "ngc_api_key" {
  secret_id     = aws_secretsmanager_secret.ngc_api_key.id
  secret_string = var.ngc_api_key

  lifecycle { ignore_changes = [secret_string] }
}

# ─── Flask Secret Key ────────────────────────────────────────
resource "aws_secretsmanager_secret" "secret_key" {
  name                    = "${var.project_name}/secret-key"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "secret_key" {
  secret_id     = aws_secretsmanager_secret.secret_key.id
  secret_string = var.secret_key

  lifecycle { ignore_changes = [secret_string] }
}
