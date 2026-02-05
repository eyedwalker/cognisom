# ═══════════════════════════════════════════════════════════════
#  Cognisom HDT Platform — Command Reference
#  Run "make" or "make help" to see all available commands
# ═══════════════════════════════════════════════════════════════

.PHONY: help run test deploy deploy-gpu stop start status destroy logs login docker menu

# Default: show help
help:
	@echo ""
	@echo "  Cognisom HDT Platform"
	@echo "  ====================="
	@echo ""
	@echo "  LOCAL DEVELOPMENT"
	@echo "  -----------------"
	@echo "  make run          Start dashboard locally (Streamlit)"
	@echo "  make api          Start Flask API locally"
	@echo "  make menu         Launch interactive terminal menu"
	@echo "  make test         Run all tests"
	@echo "  make docker       Build and run Docker container"
	@echo ""
	@echo "  AWS DEPLOYMENT"
	@echo "  --------------"
	@echo "  make deploy       Deploy to AWS (CPU mode, ~\$$83/month)"
	@echo "  make deploy-gpu   Deploy to AWS (GPU mode, ~\$$170/month)"
	@echo "  make plan         Dry-run: show what Terraform would create"
	@echo ""
	@echo "  AWS CONTROL"
	@echo "  -----------"
	@echo "  make stop         Scale to 0 (pause site, stop compute charges)"
	@echo "  make start        Scale to 1 (resume site)"
	@echo "  make status       Check if site is running"
	@echo "  make logs         Tail CloudWatch logs"
	@echo "  make destroy      Tear down ALL AWS infrastructure"
	@echo ""
	@echo "  CREDENTIALS"
	@echo "  -----------"
	@echo "  Dashboard login:  admin / Admin1234!"
	@echo "  Change password on first login via Account page"
	@echo ""
	@echo "  DNS (after deploy)"
	@echo "  ------------------"
	@echo "  make dns          Show Route53 nameservers for GoDaddy"
	@echo ""

# ─── Local Development ───────────────────────────────────────

run:
	cd $(CURDIR) && streamlit run cognisom/dashboard/app.py --server.port 8501

api:
	cd $(CURDIR) && python -m gunicorn api.rest_server:app --bind 0.0.0.0:5000 --workers 2

menu:
	cd $(CURDIR) && python launch_platform.py

test:
	cd $(CURDIR) && python -m pytest cognisom/tests/ -v

docker:
	docker build -f Dockerfile.prod -t cognisom:latest . && \
	docker run --rm -p 8501:8501 -p 5050:5000 --env-file .env cognisom:latest

# ─── AWS Deployment ──────────────────────────────────────────

deploy:
	bash deploy/deploy.sh

deploy-gpu:
	bash deploy/deploy.sh --gpu

plan:
	bash deploy/deploy.sh --plan

# ─── AWS Control ─────────────────────────────────────────────

stop:
	bash deploy/scale.sh off

start:
	bash deploy/scale.sh on

status:
	bash deploy/scale.sh status

destroy:
	@echo "This will DESTROY all AWS infrastructure."
	@read -p "Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ] && bash deploy/deploy.sh --destroy || echo "Cancelled."

logs:
	aws logs tail /ecs/cognisom --follow --region $${AWS_REGION:-us-east-1}

dns:
	@cd infrastructure/terraform && terraform output nameserver_instructions 2>/dev/null || \
		echo "Run 'make deploy' first — nameservers are shown after deployment."

# ─── Login Info ──────────────────────────────────────────────

login:
	@echo ""
	@echo "  Dashboard: https://cognisom.com  (or http://localhost:8501 for local)"
	@echo "  Username:  admin"
	@echo "  Password:  Admin1234!"
	@echo ""
	@echo "  Change this immediately after first login via the Account page."
	@echo ""
