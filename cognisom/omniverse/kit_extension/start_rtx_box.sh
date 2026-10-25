#!/bin/bash
# Get an L40S running, whatever AWS has today
# ===========================================
# L40S capacity in us-west-2 is intermittent: on 2026-10-24 a
# g6e.2xlarge was unavailable in ALL FOUR AZs and only a .4xlarge in
# us-west-2d could be had; on 2026-10-25 there was none at all, and a
# STOPPED instance could not restart because it is pinned to its AZ.
#
# So this does what has to be done by hand otherwise: try to start the
# existing box, and failing that sweep every g6e size across every
# public subnet until one takes.
#
# Starting the existing instance is always preferred -- it carries the
# 22.6 GB Isaac Sim image, which a fresh launch has to pull again.
#
# Usage:
#   ./start_rtx_box.sh            # one pass
#   ./start_rtx_box.sh --watch    # keep retrying until capacity appears
#
# Prints the IP and the export line to paste. Does NOT deploy; run
# deploy_extension.sh afterwards.

set -uo pipefail

REGION="${GPU_REGION:-us-west-2}"
INSTANCE="${GPU_INSTANCE_ID:-i-033332fd431971335}"
AMI="${GPU_AMI:-ami-020344111958d86a5}"
SG="${GPU_SG:-sg-0e56caac0e415420f}"
PROFILE="${GPU_PROFILE:-cognisom-gpu-test-profile}"
KEY="${GPU_KEY_NAME:-wabah-key}"
SUBNETS="${GPU_SUBNETS:-subnet-04646a6df8ff74dca subnet-09117a186e3e491d0 subnet-06bfc51a1a9374f2f subnet-007e07f7adef80da7}"
TYPES="${GPU_TYPES:-g6e.2xlarge g6e.4xlarge g6e.xlarge}"
WATCH=0
SLEEP="${GPU_RETRY_SECONDS:-180}"
[[ "${1:-}" == "--watch" ]] && WATCH=1

state_of() {
    aws ec2 describe-instances --instance-ids "$1" --region "$REGION" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null
}

ip_of() {
    aws ec2 describe-instances --instance-ids "$1" --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null
}

report() {
    local id="$1" ip
    for _ in $(seq 1 30); do
        ip="$(ip_of "$id")"
        [[ -n "$ip" && "$ip" != "None" ]] && break
        sleep 5
    done
    local type
    type=$(aws ec2 describe-instances --instance-ids "$id" --region "$REGION" \
        --query 'Reservations[0].Instances[0].InstanceType' --output text 2>/dev/null)
    echo ""
    echo "=== up: $id  $type  $ip ==="
    echo "Allow your current address through the security group:"
    echo "  MYIP=\$(curl -s https://checkip.amazonaws.com)"
    for port in 22 8600; do
        echo "  aws ec2 authorize-security-group-ingress --region $REGION \\"
        echo "      --group-id $SG --protocol tcp --port $port --cidr \$MYIP/32"
    done
    echo ""
    echo "Then:"
    echo "  ssh-keyscan -H $ip >> ~/.ssh/known_hosts"
    echo "  export KIT_SERVER=ubuntu@$ip"
    echo "  cognisom/omniverse/kit_extension/deploy_extension.sh"
    echo ""
    echo "REMEMBER to stop it when you are done -- it bills by the hour:"
    echo "  aws ec2 stop-instances --instance-ids $id --region $REGION"
}

attempt() {
    # 1. the existing box, which already has the Isaac Sim image
    local state
    state="$(state_of "$INSTANCE")"
    if [[ "$state" == "running" ]]; then
        echo "already running"; report "$INSTANCE"; return 0
    fi
    if [[ "$state" == "stopped" ]]; then
        for t in $TYPES; do
            aws ec2 modify-instance-attribute --instance-id "$INSTANCE" \
                --region "$REGION" --instance-type "{\"Value\": \"$t\"}" >/dev/null 2>&1
            if aws ec2 start-instances --instance-ids "$INSTANCE" \
                    --region "$REGION" >/dev/null 2>&1; then
                echo "started $INSTANCE as $t (keeps the cached Isaac Sim image)"
                report "$INSTANCE"; return 0
            fi
            echo "  no capacity to start as $t in its AZ"
        done
    elif [[ -n "$state" && "$state" != "None" ]]; then
        echo "existing instance is '$state'; not touching it"
    fi

    # 2. a fresh one wherever capacity exists -- costs a 22.6 GB re-pull
    echo "sweeping for capacity (a fresh box must re-pull Isaac Sim)..."
    for t in $TYPES; do
        for s in $SUBNETS; do
            local out
            out=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" \
                --instance-type "$t" --key-name "$KEY" --security-group-ids "$SG" \
                --subnet-id "$s" --associate-public-ip-address \
                --iam-instance-profile "Name=$PROFILE" \
                --metadata-options "HttpTokens=optional,HttpPutResponseHopLimit=2,HttpEndpoint=enabled" \
                --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cognisom-rtx},{Key=project,Value=cognisom}]' \
                --output json 2>&1)
            if grep -q InstanceId <<<"$out"; then
                local id
                id=$(python3 -c "import json,sys;print(json.load(sys.stdin)['Instances'][0]['InstanceId'])" <<<"$out")
                echo "launched $id as $t in $s"
                report "$id"; return 0
            fi
        done
        echo "  no $t capacity in any subnet"
    done
    return 1
}

if [[ "$WATCH" -eq 1 ]]; then
    while true; do
        attempt && exit 0
        echo "--- no L40S capacity; retrying in ${SLEEP}s (ctrl-c to stop) ---"
        sleep "$SLEEP"
    done
fi
attempt || { echo "no L40S capacity in $REGION right now. Try --watch."; exit 1; }
