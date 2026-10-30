#!/bin/bash
# Deploy the cognisom.sim Kit extension to the RTX box
# ====================================================
# The extension is bind-mounted read-only into the Isaac Sim container
# (-v <host dir>:/exts/cognisom.sim:ro), so deploying it means syncing
# the directory on the host and restarting the container.
#
# This exists because the extension had no deploy path at all: it was
# last placed on the box by hand in September 2026, and the stage
# endpoints added since then cannot reach RTX until the box has them.
#
# What it checks, in order:
#   1. every .py in the extension parses (locally, before anything ships)
#   2. the files sync
#   3. Kit comes back and answers /status
#   4. /cognisom/stage/status answers -- which is the proof that the NEW
#      code is live, not just that a container restarted
#
# Prerequisites:
#   - KIT_SERVER set to user@host for the box (there is NO default;
#     the old Elastic IP was released and may now be someone else's)
#   - SSH access to the GPU box, which must be RUNNING (it self-stops
#     after 15 idle minutes; start it from the dashboard or with
#     aws ec2 start-instances --instance-ids i-0ac9eb88c1b046163)
#
# Usage:
#   ./deploy_extension.sh              # check, sync, restart, verify
#   ./deploy_extension.sh --dry-run    # check and show what would sync
#   ./deploy_extension.sh --no-restart # sync only (Kit keeps the old code)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_DIR="${SCRIPT_DIR}/cognisom.sim"
SERVER="${KIT_SERVER:-}"
SSH_KEY="${KIT_SSH_KEY:-${HOME}/.ssh/wabah-key.pem}"
REMOTE_PATH="${KIT_REMOTE_PATH:-/home/ubuntu/exts/cognisom.sim}"
CONTAINER="${KIT_CONTAINER:-cognisom-kit}"
KIT_PORT="${KIT_PORT:-8600}"
HOST_ONLY="${SERVER##*@}"

# The Elastic IP this project used (52.32.247.131) was RELEASED when the
# GPU box was terminated, so that address may now belong to someone
# else's instance. Nothing here gets a default host: a deploy script
# that silently targets a released address uploads to a stranger.
if [[ -z "${KIT_SERVER:-}" ]]; then
    cat >&2 <<'USAGE'
KIT_SERVER is not set, and there is deliberately no default.

Set it to the GPU box you mean, as user@host:
    export KIT_SERVER=ubuntu@<the box's current address>

The previous box (i-0ac9eb88c1b046163, g6e.2xlarge, L40S) was terminated
and its Elastic IP released, so the old address is not safe to assume.
List what exists with:
    aws ec2 describe-instances --region us-west-2 \
        --query 'Reservations[].Instances[].{Id:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,Type:InstanceType,State:State.Name,IP:PublicIpAddress}' \
        --output table
USAGE
    exit 2
fi

DRY_RUN=0
RESTART=1
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=1 ;;
        --no-restart) RESTART=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "=== cognisom.sim -> ${SERVER}:${REMOTE_PATH} ==="

# ── 1. parse every module before shipping any of them ──
# A SyntaxError in an extension module takes the whole extension down,
# and the box is the worst place to discover one.
echo "--- checking the extension parses ---"
PY="$(command -v python3 || true)"
if [[ -z "$PY" ]]; then
    echo "no python3 on PATH; cannot pre-check the extension" >&2
    exit 1
fi
FAILED=0
while IFS= read -r -d '' file; do
    if ! "$PY" -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$file" 2>/dev/null; then
        echo "  SYNTAX ERROR: ${file#"$SCRIPT_DIR"/}"
        "$PY" -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$file" || true
        FAILED=1
    fi
done < <(find "$EXT_DIR" -name '*.py' -print0)
if [[ "$FAILED" -ne 0 ]]; then
    echo "refusing to deploy: fix the above first" >&2
    exit 1
fi
echo "  all modules parse"

# ── 2. sync ──
RSYNC_FLAGS=(-az --delete --exclude '__pycache__' --exclude '*.pyc')
if [[ "$DRY_RUN" -eq 1 ]]; then
    RSYNC_FLAGS+=(--dry-run -v)
    echo "--- dry run: what would sync ---"
else
    echo "--- syncing ---"
fi
ssh -i "$SSH_KEY" "$SERVER" "mkdir -p '${REMOTE_PATH}'"
rsync "${RSYNC_FLAGS[@]}" -e "ssh -i ${SSH_KEY}" \
    "${EXT_DIR}/" "${SERVER}:${REMOTE_PATH}/"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "dry run complete; nothing was changed."
    exit 0
fi

# ── 3. restart Kit ──
if [[ "$RESTART" -eq 0 ]]; then
    echo "--no-restart: files are in place, Kit is still running the old code."
    exit 0
fi
echo "--- restarting ${CONTAINER} ---"
ssh -i "$SSH_KEY" "$SERVER" "docker restart '${CONTAINER}'"

echo "--- waiting for Kit (RTX takes 1-2 minutes to come up) ---"
STATUS_URL="http://${HOST_ONLY}:${KIT_PORT}/status"
STAGE_URL="http://${HOST_ONLY}:${KIT_PORT}/cognisom/stage/status"
for attempt in $(seq 1 40); do
    if curl -fsS --max-time 5 "$STATUS_URL" >/dev/null 2>&1; then
        echo "  Kit answered /status after ${attempt} attempt(s)"
        break
    fi
    sleep 10
    if [[ "$attempt" -eq 40 ]]; then
        echo "Kit did not answer ${STATUS_URL} within ~7 minutes." >&2
        echo "Check the container log: ssh -i ${SSH_KEY} ${SERVER} docker logs --tail 100 ${CONTAINER}" >&2
        exit 1
    fi
done

# ── 4. the only check that proves the NEW code is live ──
echo "--- verifying the stage endpoint ---"
if STAGE_JSON="$(curl -fsS --max-time 10 "$STAGE_URL" 2>/dev/null)"; then
    echo "  /cognisom/stage/status -> ${STAGE_JSON}"
else
    echo "Kit is up but ${STAGE_URL} does not answer." >&2
    echo "The container is running code without UsdSceneManager -- check that" >&2
    echo "${REMOTE_PATH} is the directory actually bind-mounted into it:" >&2
    echo "  ssh -i ${SSH_KEY} ${SERVER} docker inspect ${CONTAINER} --format '{{json .Mounts}}'" >&2
    exit 1
fi

echo ""
echo "=== deployed ==="
echo "Push a run to it with:"
echo "  python -m cognisom.omniverse.rtx_client <run.usdc> \\"
echo "      --url http://${HOST_ONLY}:${KIT_PORT} --play --save frame.jpg"
