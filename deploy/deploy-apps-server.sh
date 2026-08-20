#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Cognisom dashboard — production rollout on apps-server
#
# This is the deploy that actually runs in production. Note that the
# neighbouring deploy.sh is a *terraform/ECS* script targeting a stack that
# does not exist in any region; it does not launch this container.
#
# Until now the running container's configuration existed nowhere but in
# Docker's own state, so a rollout meant retyping a `docker run` from memory.
# That is how the 2026-08-20 outage happened. Two things this encodes so they
# cannot silently regress:
#
#   1. The image runs as the unprivileged `cognisom` user (uid/gid 999), but
#      /opt/cognisom/{data,exports} are host bind mounts. If they are
#      root-owned, entrypoint.sh's `mkdir -p` fails, `set -e` exits before
#      gunicorn starts, and the restart policy crash-loops it into a 502.
#
#   2. Auth lives in us-west-2. cognito_provider.py reads AWS_REGION -- not
#      AWS_DEFAULT_REGION -- so both are set. Launching without these vars
#      disables Cognito entirely and silently falls back to local file auth.
#
# Usage: bash deploy/deploy-apps-server.sh [--build] [--no-restart]
# ═══════════════════════════════════════════════════════════════

CONTAINER="cognisom-dashboard"
IMAGE="cognisom:latest"
DATA_DIR="/opt/cognisom/data"
EXPORTS_DIR="/opt/cognisom/exports"
RUNTIME_UID=999
RUNTIME_GID=999

# Auth — us-west-2 pool. See reference_infrastructure notes before changing.
COGNITO_REGION="us-west-2"
COGNITO_POOL="us-west-2_6lEXWY7IU"
COGNITO_CLIENT="v4ir490b43m86qatdijfec90t"
COGNITO_DOMAIN_NAME="cognisom-production-auth"

DO_BUILD=false
DO_RESTART=true
for arg in "$@"; do
  case $arg in
    --build)      DO_BUILD=true ;;
    --no-restart) DO_RESTART=false ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

info() { echo "[INFO] $1"; }
ok()   { echo "[OK]   $1"; }

# ─── 1. Volumes ──────────────────────────────────────────────
# Created and chowned every run: cheap, idempotent, and the one step whose
# absence takes the site down.
info "Preparing data volumes for uid ${RUNTIME_UID}..."
mkdir -p "$DATA_DIR"/{auth,scrna,research_cache,subscriptions,flywheel,feedback,agent_interactions,model_registry,distilled_models,calibration}
mkdir -p "$EXPORTS_DIR"
chown -R "${RUNTIME_UID}:${RUNTIME_GID}" "$DATA_DIR" "$EXPORTS_DIR"
ok "Volumes owned by ${RUNTIME_UID}:${RUNTIME_GID}"

# ─── 2. Image ────────────────────────────────────────────────
if [ "$DO_BUILD" = true ]; then
  info "Building $IMAGE from $(git rev-parse --short HEAD 2>/dev/null || echo 'working tree')..."
  docker build -f Dockerfile.prod \
    --build-arg COGNISOM_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo dev)" \
    --build-arg COGNISOM_BUILD_DATE="$(date -u +%Y-%m-%d)" \
    -t "$IMAGE" .
  ok "Image built"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: $IMAGE not present. Re-run with --build."; exit 1; }

# ─── 3. Pre-flight: can the runtime user actually write? ─────
# Checked before tearing down the healthy container, so a permissions problem
# fails here rather than leaving the site down.
info "Verifying uid ${RUNTIME_UID} can write to the volumes..."
docker run --rm -v "$DATA_DIR:/app/data" --user "${RUNTIME_UID}:${RUNTIME_GID}" \
  --entrypoint touch "$IMAGE" /app/data/.deploy-writetest
rm -f "$DATA_DIR/.deploy-writetest"
ok "Volumes writable by runtime user"

[ "$DO_RESTART" = false ] && { info "--no-restart given; leaving running container alone."; exit 0; }

# ─── 4. Swap the container ───────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  info "Stashing current container as ${CONTAINER}-prev (rollback)..."
  docker rm -f "${CONTAINER}-prev" >/dev/null 2>&1 || true
  docker stop "$CONTAINER" >/dev/null
  docker rename "$CONTAINER" "${CONTAINER}-prev"
fi

info "Starting $CONTAINER..."
docker run -d --name "$CONTAINER" --restart unless-stopped \
  -p 8501:8501 \
  -v "$DATA_DIR:/app/data" \
  -v "$EXPORTS_DIR:/app/exports" \
  -e AWS_REGION="$COGNITO_REGION" \
  -e AWS_DEFAULT_REGION="$COGNITO_REGION" \
  -e COGNITO_USER_POOL_ID="$COGNITO_POOL" \
  -e COGNITO_CLIENT_ID="$COGNITO_CLIENT" \
  -e COGNITO_DOMAIN="$COGNITO_DOMAIN_NAME" \
  "$IMAGE" >/dev/null

# ─── 5. Wait for health ──────────────────────────────────────
info "Waiting for health check..."
for i in $(seq 1 30); do
  STATUS=$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || echo starting)
  [ "$STATUS" = "healthy" ] && { ok "Container healthy"; break; }
  if [ "$(docker inspect "$CONTAINER" --format '{{.State.Restarting}}')" = "true" ]; then
    echo "ERROR: container is restarting. Logs:"
    docker logs --tail 30 "$CONTAINER"
    echo "Roll back: docker rm -f $CONTAINER && docker rename ${CONTAINER}-prev $CONTAINER && docker start $CONTAINER"
    exit 1
  fi
  sleep 5
done

[ "${STATUS:-}" = "healthy" ] || { echo "ERROR: not healthy after 150s"; docker logs --tail 30 "$CONTAINER"; exit 1; }

curl -fsS -m 10 http://127.0.0.1:8501 >/dev/null && ok "Streamlit responding on 8501"
echo
ok "Deployed: $(docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep COGNISOM_GIT_SHA || echo 'sha unknown')"
echo "Previous container retained as ${CONTAINER}-prev for rollback."
