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
#   3. The root volume is 40 G and SHARED with every other app on this box.
#      Each deploy leaves a ~4.3 G image behind and nothing ever removed one,
#      so on 2026-11-06 a pull died mid-extract with "no space left on device"
#      at 95% full (2.4 G free); the same pull succeeded at 12 G free. The
#      site survived only because the pull comes before the swap. Now: a
#      free-space check before pulling, and superseded cognisom images are
#      removed after every successful deploy. See prune_superseded for exactly
#      what is protected.
#
# Usage: bash deploy/deploy-apps-server.sh [--pull <sha>] [--build] [--no-restart]
#        bash deploy/deploy-apps-server.sh --reclaim-only [--dry-run]
#
#   --pull <sha>  fetch the image CI already built and pushed to ECR. This is
#                 the normal path: CI builds on every push to main, and that
#                 image has passed the in-image checks (MHCflurry loads as the
#                 runtime user, gunicorn's target imports). Prefer it.
#   --build       build locally instead. Note the checkout at
#                 /home/ec2-user/cognisom drifts from main, so a local build
#                 there can quietly produce an older image than you expect.
#   --reclaim-only  remove superseded cognisom images and exit; deploys nothing.
#   --dry-run       with --reclaim-only: list what would be removed, remove nothing.
#
#   COGNISOM_MIN_FREE_GB (default 8) is the free space required before a pull.
#
#   The registry side of the same problem is deploy/ecr-lifecycle-policy.json.
#   Preview before applying -- since 2026-11-06 every tag is an OCI index whose
#   real image is an UNTAGGED child, so a naive "expire untagged" rule would
#   delete the image under every tag:
#     aws ecr start-lifecycle-policy-preview --region us-west-2 \
#       --repository-name cognisom --lifecycle-policy-text file://deploy/ecr-lifecycle-policy.json
#     aws ecr put-lifecycle-policy --region us-west-2 \
#       --repository-name cognisom --lifecycle-policy-text file://deploy/ecr-lifecycle-policy.json
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

ECR_REGISTRY="780457123717.dkr.ecr.us-west-2.amazonaws.com/cognisom"
ECR_REGION="us-west-2"

DO_BUILD=false
DO_RESTART=true
PULL_SHA=""
RECLAIM_ONLY=false
DRY_RUN=false
while [ $# -gt 0 ]; do
  case $1 in
    --pull)       PULL_SHA="${2:-}"; [ -n "$PULL_SHA" ] || { echo "--pull needs an image tag"; exit 1; }; shift 2 ;;
    --build)      DO_BUILD=true; shift ;;
    --no-restart) DO_RESTART=false; shift ;;
    --reclaim-only) RECLAIM_ONLY=true; shift ;;
    --dry-run)    DRY_RUN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Written as an `if` rather than an `&&` chain on purpose: under `set -e` a
# chain like `[ -n "$x" ] && [ ... ] && { exit 1; }` survives only because a
# failure *before* the final && is exempt from -e. That is too subtle to rest
# a deploy on.
if [ -n "$PULL_SHA" ] && [ "$DO_BUILD" = true ]; then
  echo "ERROR: --pull and --build are mutually exclusive."
  exit 1
fi
# A dry run of a DEPLOY does not exist; refuse rather than let someone believe
# they rehearsed one.
if [ "$DRY_RUN" = true ] && [ "$RECLAIM_ONLY" = false ]; then
  echo "ERROR: --dry-run only applies to --reclaim-only."
  exit 1
fi

info() { echo "[INFO] $1"; }
ok()   { echo "[OK]   $1"; }
warn() { echo "[WARN] $1"; }

# ─── Disk ────────────────────────────────────────────────────
# Measured: the prod image is 4.27 G on disk; a pull FAILED with 2.4 G free
# and SUCCEEDED with 12 G free. 8 G is about twice the image, because docker
# holds the compressed layers while it extracts them.
MIN_FREE_GB="${COGNISOM_MIN_FREE_GB:-8}"

free_gb() {
  local root
  root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
  df -P -BG "$root" | awk 'NR==2 { gsub("G", "", $4); print $4 }'
}

# Image IDs that must survive a prune:
#   - every image a container uses, running OR stopped. That is the live site,
#     cognisom-dashboard-prev (the rollback this script creates), and any old
#     container someone kept on purpose (cognisom-dashboard-preregion is one);
#   - every tag containing "rollback" (cognisom:rollback-20260820 is the
#     artifact from the August outage);
#   - whatever $IMAGE points at now, and the image this run is deploying.
protected_ids() {
  local ids
  ids=$(docker ps -aq)
  if [ -n "$ids" ]; then
    # shellcheck disable=SC2086  # one argument per container id
    docker inspect --format '{{.Image}}' $ids
  fi
  docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' \
    | awk '$1 ~ /rollback/ { print $2 }'
  docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true
  if [ -n "$PULL_SHA" ]; then
    docker image inspect --format '{{.Id}}' "$ECR_REGISTRY:$PULL_SHA" 2>/dev/null || true
  fi
}

# Remove every tag of a cognisom image that is not protected. Two rules:
#   - only repositories NAMED cognisom (bare, or ending /cognisom). This box
#     hosts other apps; their images, dangling layers and build cache are
#     theirs. That is why there is no `docker image prune` / `system prune`.
#   - never `rmi -f`. Docker refusing to delete an image a container holds is
#     the last guard, and it fired on the first manual cleanup.
prune_superseded() {
  local dry="${1:-false}" protected repo tag id ref n=0
  protected=$(protected_ids | sort -u)
  while read -r repo tag id; do
    [ -n "$id" ] || continue
    if printf '%s\n' "$protected" | grep -qx "$id"; then
      continue
    fi
    if [ "$tag" = "<none>" ]; then ref="$id"; else ref="$repo:$tag"; fi
    n=$((n + 1))
    if [ "$dry" = true ]; then
      info "would remove $ref"
    elif docker rmi "$ref" >/dev/null 2>&1; then
      info "removed $ref"
    else
      warn "docker refused to remove $ref; left in place"
    fi
  done < <(docker images --no-trunc --format '{{.Repository}} {{.Tag}} {{.ID}}' \
             | awk '$1 == "cognisom" || $1 ~ /\/cognisom$/')
  info "$n superseded cognisom image reference(s)"
}

# Checked before anything is pulled or built, so a short disk fails HERE with
# the running container untouched -- not halfway through extracting a layer.
ensure_free_space() {
  local have
  have=$(free_gb)
  info "Disk: ${have} G free; a pull needs ${MIN_FREE_GB} G"
  if [ "$have" -ge "$MIN_FREE_GB" ]; then
    return 0
  fi
  info "Below ${MIN_FREE_GB} G -- removing superseded cognisom images first..."
  prune_superseded false
  have=$(free_gb)
  info "Disk: ${have} G free after pruning"
  if [ "$have" -lt "$MIN_FREE_GB" ]; then
    echo "ERROR: only ${have} G free after removing every unprotected cognisom image."
    echo "       The running site has NOT been touched. Look at 'docker system df';"
    echo "       the rest of this disk belongs to other apps, so free it by hand."
    exit 1
  fi
}

if [ "$RECLAIM_ONLY" = true ]; then
  info "Disk: $(free_gb) G free before"
  prune_superseded "$DRY_RUN"
  info "Disk: $(free_gb) G free after"
  exit 0
fi

# ─── 1. Volumes ──────────────────────────────────────────────
# Created and chowned every run: cheap, idempotent, and the one step whose
# absence takes the site down.
info "Preparing data volumes for uid ${RUNTIME_UID}..."
mkdir -p "$DATA_DIR"/{auth,scrna,research_cache,subscriptions,flywheel,feedback,agent_interactions,model_registry,distilled_models,calibration}
mkdir -p "$EXPORTS_DIR"
chown -R "${RUNTIME_UID}:${RUNTIME_GID}" "$DATA_DIR" "$EXPORTS_DIR"
ok "Volumes owned by ${RUNTIME_UID}:${RUNTIME_GID}"

# ─── 2. Image ────────────────────────────────────────────────
if [ -n "$PULL_SHA" ] || [ "$DO_BUILD" = true ]; then
  ensure_free_space
fi

if [ -n "$PULL_SHA" ]; then
  info "Pulling $PULL_SHA from ECR ($ECR_REGION)..."
  aws ecr get-login-password --region "$ECR_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
  docker pull "$ECR_REGISTRY:$PULL_SHA"
  docker tag "$ECR_REGISTRY:$PULL_SHA" "$IMAGE"
  ok "Pulled and tagged as $IMAGE"
fi

if [ "$DO_BUILD" = true ]; then
  info "Building $IMAGE from $(git rev-parse --short HEAD 2>/dev/null || echo 'working tree')..."
  docker build -f Dockerfile.prod \
    --build-arg COGNISOM_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo dev)" \
    --build-arg COGNISOM_BUILD_DATE="$(date -u +%Y-%m-%d)" \
    -t "$IMAGE" .
  ok "Image built"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: $IMAGE not present. Re-run with --pull <sha> (normal) or --build."
  exit 1
}

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

# ─── 6. Leave the disk as we would want to find it ───────────
# The container before -prev was removed in step 4, so its image is now
# unreferenced. Removing it here keeps the steady state at two cognisom images
# (live + rollback) instead of one more per deploy. A failure here does not
# fail a deploy that has already succeeded.
echo
info "Removing superseded cognisom images..."
prune_superseded false || warn "prune failed; the deploy is fine, the disk was not reclaimed"
info "Disk: $(free_gb) G free"
