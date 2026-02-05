# ─── AWS Cognito for Research-Grade Authentication ───────────
#
# HIPAA-eligible authentication for:
# - User registration and login
# - MFA (multi-factor authentication)
# - Institutional SSO via SAML/OIDC
# - Role-based access control
# - Audit logs for compliance
#
# Features:
# - Email/password authentication
# - Social login (optional)
# - SAML federation for universities/hospitals
# - Password policies (complexity, expiration)
# - Account recovery

# ─── Variables ────────────────────────────────────────────────
variable "enable_cognito" {
  description = "Enable AWS Cognito authentication"
  type        = bool
  default     = true
}

variable "cognito_callback_urls" {
  description = "Allowed callback URLs after login"
  type        = list(string)
  default     = ["https://cognisom.com/auth/callback", "http://localhost:8501/auth/callback"]
}

variable "cognito_logout_urls" {
  description = "Allowed logout URLs"
  type        = list(string)
  default     = ["https://cognisom.com", "http://localhost:8501"]
}

# ─── Cognito User Pool ────────────────────────────────────────
resource "aws_cognito_user_pool" "main" {
  count = var.enable_cognito ? 1 : 0
  name  = "${local.name_prefix}-users"

  # Username configuration
  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  # Password policy (HIPAA-compliant)
  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 7
  }

  # MFA configuration
  mfa_configuration = "OPTIONAL"  # Change to "ON" for HIPAA production

  software_token_mfa_configuration {
    enabled = true
  }

  # Account recovery
  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  # User verification
  verification_message_template {
    default_email_option = "CONFIRM_WITH_CODE"
    email_subject        = "Cognisom - Verify your email"
    email_message        = "Your verification code is {####}"
  }

  # Email configuration (use SES for production)
  email_configuration {
    email_sending_account = "COGNITO_DEFAULT"
  }

  # Schema - custom attributes for research
  schema {
    name                     = "organization"
    attribute_data_type      = "String"
    mutable                  = true
    required                 = false
    string_attribute_constraints {
      min_length = 0
      max_length = 256
    }
  }

  schema {
    name                     = "role"
    attribute_data_type      = "String"
    mutable                  = true
    required                 = false
    string_attribute_constraints {
      min_length = 0
      max_length = 64
    }
  }

  schema {
    name                     = "research_area"
    attribute_data_type      = "String"
    mutable                  = true
    required                 = false
    string_attribute_constraints {
      min_length = 0
      max_length = 256
    }
  }

  # Admin create user config
  admin_create_user_config {
    allow_admin_create_user_only = false

    invite_message_template {
      email_subject = "Welcome to Cognisom"
      email_message = "You've been invited to Cognisom. Your username is {username} and temporary password is {####}"
      sms_message   = "Your Cognisom username is {username} and temporary password is {####}"
    }
  }

  # Device tracking (for security)
  device_configuration {
    challenge_required_on_new_device      = true
    device_only_remembered_on_user_prompt = true
  }

  # User pool add-ons
  user_pool_add_ons {
    advanced_security_mode = "AUDIT"  # Change to "ENFORCED" for production
  }

  tags = {
    Name       = "${local.name_prefix}-user-pool"
    Compliance = "HIPAA-eligible"
  }
}

# ─── User Pool Domain ─────────────────────────────────────────
resource "aws_cognito_user_pool_domain" "main" {
  count        = var.enable_cognito ? 1 : 0
  domain       = "${local.name_prefix}-auth"
  user_pool_id = aws_cognito_user_pool.main[0].id
}

# ─── App Client (Web Application) ─────────────────────────────
resource "aws_cognito_user_pool_client" "web" {
  count        = var.enable_cognito ? 1 : 0
  name         = "${local.name_prefix}-web-client"
  user_pool_id = aws_cognito_user_pool.main[0].id

  # Generate secret for server-side auth
  generate_secret = true

  # OAuth configuration
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["email", "openid", "profile"]
  supported_identity_providers         = ["COGNITO"]

  callback_urls = var.cognito_callback_urls
  logout_urls   = var.cognito_logout_urls

  # Token configuration
  access_token_validity  = 1   # hours
  id_token_validity      = 1   # hours
  refresh_token_validity = 30  # days

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }

  # Security
  prevent_user_existence_errors = "ENABLED"
  enable_token_revocation       = true

  # Allowed auth flows
  explicit_auth_flows = [
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_USER_PASSWORD_AUTH",  # For Streamlit integration
  ]

  # Read/write attributes
  read_attributes = [
    "email",
    "email_verified",
    "name",
    "custom:organization",
    "custom:role",
    "custom:research_area",
  ]

  write_attributes = [
    "email",
    "name",
    "custom:organization",
    "custom:research_area",
  ]
}

# ─── API Client (Backend Services) ────────────────────────────
resource "aws_cognito_user_pool_client" "api" {
  count        = var.enable_cognito ? 1 : 0
  name         = "${local.name_prefix}-api-client"
  user_pool_id = aws_cognito_user_pool.main[0].id

  generate_secret = true

  # Server-side only
  allowed_oauth_flows                  = ["client_credentials"]
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_scopes                 = ["cognisom/read", "cognisom/write"]
  supported_identity_providers         = ["COGNITO"]

  explicit_auth_flows = [
    "ALLOW_ADMIN_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]
}

# ─── Resource Server (API Scopes) ─────────────────────────────
resource "aws_cognito_resource_server" "api" {
  count        = var.enable_cognito ? 1 : 0
  identifier   = "cognisom"
  name         = "Cognisom API"
  user_pool_id = aws_cognito_user_pool.main[0].id

  scope {
    scope_name        = "read"
    scope_description = "Read access to Cognisom API"
  }

  scope {
    scope_name        = "write"
    scope_description = "Write access to Cognisom API"
  }

  scope {
    scope_name        = "admin"
    scope_description = "Admin access to Cognisom API"
  }
}

# ─── User Groups (Role-Based Access) ──────────────────────────
resource "aws_cognito_user_group" "admin" {
  count        = var.enable_cognito ? 1 : 0
  name         = "admin"
  description  = "Platform administrators"
  user_pool_id = aws_cognito_user_pool.main[0].id
  precedence   = 1
}

resource "aws_cognito_user_group" "researcher" {
  count        = var.enable_cognito ? 1 : 0
  name         = "researcher"
  description  = "Research users with full access"
  user_pool_id = aws_cognito_user_pool.main[0].id
  precedence   = 10
}

resource "aws_cognito_user_group" "viewer" {
  count        = var.enable_cognito ? 1 : 0
  name         = "viewer"
  description  = "Read-only access"
  user_pool_id = aws_cognito_user_pool.main[0].id
  precedence   = 100
}

# ─── Identity Pool (for AWS resource access) ──────────────────
resource "aws_cognito_identity_pool" "main" {
  count                            = var.enable_cognito ? 1 : 0
  identity_pool_name               = "${local.name_prefix}-identity"
  allow_unauthenticated_identities = false

  cognito_identity_providers {
    client_id               = aws_cognito_user_pool_client.web[0].id
    provider_name           = aws_cognito_user_pool.main[0].endpoint
    server_side_token_check = true
  }

  tags = { Name = "${local.name_prefix}-identity-pool" }
}

# ─── IAM Roles for Cognito Identity ───────────────────────────
resource "aws_iam_role" "cognito_authenticated" {
  count = var.enable_cognito ? 1 : 0
  name  = "${local.name_prefix}-cognito-authenticated"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = "cognito-identity.amazonaws.com"
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "cognito-identity.amazonaws.com:aud" = aws_cognito_identity_pool.main[0].id
          }
          "ForAnyValue:StringLike" = {
            "cognito-identity.amazonaws.com:amr" = "authenticated"
          }
        }
      }
    ]
  })

  tags = { Name = "${local.name_prefix}-cognito-auth-role" }
}

resource "aws_iam_role_policy" "cognito_authenticated" {
  count = var.enable_cognito ? 1 : 0
  name  = "cognito-authenticated-policy"
  role  = aws_iam_role.cognito_authenticated[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.data.arn}/users/$${cognito-identity.amazonaws.com:sub}/*"
        ]
      }
    ]
  })
}

resource "aws_cognito_identity_pool_roles_attachment" "main" {
  count            = var.enable_cognito ? 1 : 0
  identity_pool_id = aws_cognito_identity_pool.main[0].id

  roles = {
    "authenticated" = aws_iam_role.cognito_authenticated[0].arn
  }
}

# ─── Outputs ──────────────────────────────────────────────────
output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = var.enable_cognito ? aws_cognito_user_pool.main[0].id : null
}

output "cognito_user_pool_endpoint" {
  description = "Cognito User Pool endpoint"
  value       = var.enable_cognito ? aws_cognito_user_pool.main[0].endpoint : null
}

output "cognito_web_client_id" {
  description = "Cognito Web App Client ID"
  value       = var.enable_cognito ? aws_cognito_user_pool_client.web[0].id : null
}

output "cognito_domain" {
  description = "Cognito hosted UI domain"
  value       = var.enable_cognito ? "https://${aws_cognito_user_pool_domain.main[0].domain}.auth.${var.aws_region}.amazoncognito.com" : null
}

output "cognito_identity_pool_id" {
  description = "Cognito Identity Pool ID"
  value       = var.enable_cognito ? aws_cognito_identity_pool.main[0].id : null
}
