#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Cognisom AWS Deployment
# Usage: bash deploy/deploy.sh [--gpu] [--plan] [--destroy]
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TF_DIR="$PROJECT_ROOT/infrastructure/terraform"

ENABLE_GPU="false"
PLAN_ONLY=false
DESTROY=false

for arg in "$@"; do
  case $arg in
    --gpu)     ENABLE_GPU="true" ;;
    --plan)    PLAN_ONLY=true ;;
    --destroy) DESTROY=true ;;
    *)         echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

# ─── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ─── Step 0: Prerequisites ──────────────────────────────────
info "Checking prerequisites..."

command -v aws >/dev/null 2>&1 || { echo "ERROR: aws cli not found. Install: https://aws.amazon.com/cli/"; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "ERROR: terraform not found. Install: https://terraform.io/downloads"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found."; exit 1; }

aws sts get-caller-identity >/dev/null 2>&1 || { echo "ERROR: AWS credentials not configured. Run: aws configure"; exit 1; }
ok "AWS credentials valid"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(cd "$TF_DIR" && terraform output -raw aws_region 2>/dev/null || echo "us-east-1")
ok "Account: $ACCOUNT_ID, Region: $REGION"

# ─── Step 1: Terraform Init ─────────────────────────────────
info "Initializing Terraform..."
cd "$TF_DIR"
terraform init -upgrade

# ─── Handle --destroy ────────────────────────────────────────
if [ "$DESTROY" = true ]; then
  warn "DESTROYING all infrastructure..."
  terraform destroy -var="enable_gpu=$ENABLE_GPU"
  exit 0
fi

# ─── Step 2: Create ECR (needed before Docker push) ─────────
info "Ensuring ECR repository exists..."
terraform apply -target=aws_ecr_repository.main -var="enable_gpu=$ENABLE_GPU" -auto-approve

ECR_URI=$(terraform output -raw ecr_repository_url 2>/dev/null)
ok "ECR: $ECR_URI"

# ─── Step 3: Build and Push Docker Image ────────────────────
info "Building Docker image (enable_gpu=$ENABLE_GPU)..."

if [ "$ENABLE_GPU" = "true" ]; then
  DOCKERFILE="Dockerfile.gpu"
else
  DOCKERFILE="Dockerfile.prod"
fi

cd "$PROJECT_ROOT"
docker build -f "$DOCKERFILE" -t cognisom:latest .

info "Pushing to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_URI"
docker tag cognisom:latest "$ECR_URI:latest"
docker push "$ECR_URI:latest"
ok "Image pushed: $ECR_URI:latest"

# ─── Step 4: Terraform Plan / Apply ─────────────────────────
cd "$TF_DIR"

if [ "$PLAN_ONLY" = true ]; then
  info "Running terraform plan (dry run)..."
  terraform plan -var="enable_gpu=$ENABLE_GPU"
  exit 0
fi

info "Applying Terraform..."
terraform apply -var="enable_gpu=$ENABLE_GPU" -auto-approve

# ─── Step 5: Output Results ─────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Cognisom Deployment Complete"
echo "════════════════════════════════════════════════════════"
echo ""
echo "  Mode:     $(terraform output -raw mode)"
echo "  URL:      $(terraform output -raw app_url)"
echo "  ALB:      $(terraform output -raw alb_dns_name)"
echo "  ECR:      $(terraform output -raw ecr_repository_url)"
echo "  Cost:     $(terraform output -raw estimated_monthly_cost)"
echo ""
terraform output -raw nameserver_instructions
echo ""
