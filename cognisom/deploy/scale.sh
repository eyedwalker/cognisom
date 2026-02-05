#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Cognisom AWS Scale Control
#
# Turn the site on/off to save costs when not in use.
#
# Usage:
#   bash deploy/scale.sh off        # Scale to 0 (stops billing for compute)
#   bash deploy/scale.sh on         # Scale back to 1
#   bash deploy/scale.sh status     # Check current state
# ═══════════════════════════════════════════════════════════════

CLUSTER="cognisom-production-cluster"
REGION="${AWS_REGION:-us-east-1}"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Detect which service is running (CPU or GPU)
get_service() {
    local svc
    svc=$(aws ecs list-services --cluster "$CLUSTER" --region "$REGION" --query 'serviceArns[0]' --output text 2>/dev/null)
    if [ "$svc" = "None" ] || [ -z "$svc" ]; then
        echo ""
        return
    fi
    basename "$svc"
}

case "${1:-status}" in
    off|stop|down)
        SERVICE=$(get_service)
        if [ -z "$SERVICE" ]; then
            echo -e "${RED}No service found in cluster $CLUSTER${NC}"
            exit 1
        fi

        echo -e "${CYAN}Scaling $SERVICE to 0...${NC}"
        aws ecs update-service \
            --cluster "$CLUSTER" \
            --service "$SERVICE" \
            --desired-count 0 \
            --region "$REGION" \
            --query 'service.desiredCount' \
            --output text

        echo -e "${GREEN}Site is OFF. Compute charges stopped.${NC}"
        echo ""
        echo "Note: ALB, NAT gateway, and Route53 still incur ~\$50/month."
        echo "To fully destroy everything: bash deploy/deploy.sh --destroy"
        ;;

    on|start|up)
        SERVICE=$(get_service)
        if [ -z "$SERVICE" ]; then
            echo -e "${RED}No service found in cluster $CLUSTER${NC}"
            exit 1
        fi

        echo -e "${CYAN}Scaling $SERVICE to 1...${NC}"
        aws ecs update-service \
            --cluster "$CLUSTER" \
            --service "$SERVICE" \
            --desired-count 1 \
            --region "$REGION" \
            --query 'service.desiredCount' \
            --output text

        echo -e "${GREEN}Site is ON. Waiting for health check...${NC}"
        aws ecs wait services-stable \
            --cluster "$CLUSTER" \
            --services "$SERVICE" \
            --region "$REGION" 2>/dev/null && \
            echo -e "${GREEN}Service is healthy and ready.${NC}" || \
            echo -e "${CYAN}Service is starting (may need a minute).${NC}"
        ;;

    status)
        SERVICE=$(get_service)
        if [ -z "$SERVICE" ]; then
            echo -e "${RED}No service found in cluster $CLUSTER${NC}"
            exit 1
        fi

        DESIRED=$(aws ecs describe-services \
            --cluster "$CLUSTER" \
            --services "$SERVICE" \
            --region "$REGION" \
            --query 'services[0].desiredCount' \
            --output text 2>/dev/null)

        RUNNING=$(aws ecs describe-services \
            --cluster "$CLUSTER" \
            --services "$SERVICE" \
            --region "$REGION" \
            --query 'services[0].runningCount' \
            --output text 2>/dev/null)

        echo "Service:  $SERVICE"
        echo "Desired:  $DESIRED"
        echo "Running:  $RUNNING"

        if [ "$DESIRED" = "0" ]; then
            echo -e "\nStatus: ${RED}OFF${NC} (no compute charges)"
        elif [ "$RUNNING" = "$DESIRED" ]; then
            echo -e "\nStatus: ${GREEN}ON${NC} and healthy"
        else
            echo -e "\nStatus: ${CYAN}STARTING${NC} ($RUNNING/$DESIRED tasks running)"
        fi
        ;;

    *)
        echo "Usage: bash deploy/scale.sh [on|off|status]"
        exit 1
        ;;
esac
