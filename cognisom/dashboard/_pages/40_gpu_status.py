"""
Page 40: GPU Pipeline Status — HealthOmics Run Monitor
========================================================

Real-time monitoring of AWS HealthOmics Parabricks GPU pipeline runs.
Shows active/completed/failed runs with task-level detail, cost tracking,
and one-click VCF handoff to MAD Agent.
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="GPU Pipeline Status", page_icon="📡", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("40_gpu_status")

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

st.title("GPU Pipeline Status")
st.caption(
    "Real-time monitoring of AWS HealthOmics Parabricks runs. "
    "Track alignment, variant calling, and VCF output on GPU hardware."
)

# ══════════════════════════════════════════════════════════════════════
# AWS CLIENT
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(ttl=60)
def _get_omics_client():
    try:
        import boto3
        return boto3.client("omics", region_name="us-west-2")
    except Exception as e:
        return None

@st.cache_resource(ttl=60)
def _get_s3_client():
    try:
        import boto3
        return boto3.client("s3", region_name="us-west-2")
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════
# STATUS DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════

STATUS_COLORS = {
    "PENDING": "#6b7280",
    "STARTING": "#f59e0b",
    "RUNNING": "#3b82f6",
    "STOPPING": "#f59e0b",
    "COMPLETED": "#10b981",
    "FAILED": "#ef4444",
    "CANCELLED": "#6b7280",
    "DELETED": "#6b7280",
}

STATUS_ICONS = {
    "PENDING": "hourglass_not_done",
    "STARTING": "rocket_launch",
    "RUNNING": "sync",
    "STOPPING": "stop_circle",
    "COMPLETED": "check_circle",
    "FAILED": "error",
    "CANCELLED": "cancel",
}

COST_PER_GPU_HOUR = 6.50  # Approximate HealthOmics GPU cost


def _format_elapsed(start_time, stop_time=None):
    """Format elapsed time as human-readable string."""
    end = stop_time or datetime.now(timezone.utc)
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    elapsed = (end - start_time).total_seconds()
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _estimate_cost(start_time, stop_time=None, gpus=4):
    """Estimate run cost based on GPU-hours."""
    end = stop_time or datetime.now(timezone.utc)
    if isinstance(start_time, str):
        start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    hours = (end - start_time).total_seconds() / 3600
    return hours * COST_PER_GPU_HOUR * (gpus / 4)


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

omics = _get_omics_client()
s3 = _get_s3_client()

if omics is None:
    st.error("AWS credentials not available. Cannot connect to HealthOmics.")
    st.info("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or use IAM roles.")
    st.stop()

# ── Controls ──
col_refresh, col_start = st.columns([1, 3])
with col_refresh:
    if st.button("Refresh", icon=":material/refresh:", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col_start:
    st.markdown("")  # Spacer

# ── Fetch runs ──
try:
    runs_resp = omics.list_runs(maxResults=20)
    runs = runs_resp.get("items", [])
except Exception as e:
    st.error(f"Failed to list runs: {e}")
    runs = []

if not runs:
    st.info("No HealthOmics runs found. Start a pipeline from the **DNA to Decision** page.")
    st.stop()

# ── Categorize ──
active_runs = [r for r in runs if r.get("status") in ("PENDING", "STARTING", "RUNNING", "STOPPING")]
completed_runs = [r for r in runs if r.get("status") == "COMPLETED"]
failed_runs = [r for r in runs if r.get("status") in ("FAILED", "CANCELLED")]

# ── Summary cards ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Runs", len(runs))
c2.metric("Active", len(active_runs), delta=None)
c3.metric("Completed", len(completed_runs))
c4.metric("Failed", len(failed_runs))

st.divider()

# ══════════════════════════════════════════════════════════════════════
# ACTIVE RUNS (expanded by default)
# ══════════════════════════════════════════════════════════════════════

if active_runs:
    st.subheader("Active Runs", divider="blue")

    for run in active_runs:
        run_id = run["id"]
        name = run.get("name", f"Run {run_id}")
        status = run.get("status", "UNKNOWN")
        start = run.get("startTime")
        color = STATUS_COLORS.get(status, "#6b7280")

        with st.expander(f":{STATUS_ICONS.get(status, 'sync')}: **{name}** — {status}", expanded=True):
            # Get detailed run info
            try:
                detail = omics.get_run(id=run_id)
            except Exception:
                detail = run

            # Run info columns
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Run ID", run_id)
            r2.metric("Status", status)
            elapsed = _format_elapsed(start) if start else "N/A"
            r3.metric("Elapsed", elapsed)

            # Cost estimate
            if start:
                cost = _estimate_cost(start)
                r4.metric("Est. Cost", f"${cost:.2f}")

            # Tags
            tags = run.get("tags", detail.get("tags", {}))
            if tags:
                tag_str = " | ".join(f"**{k}**: {v}" for k, v in tags.items())
                st.caption(tag_str)

            # Task-level detail
            try:
                tasks_resp = omics.list_run_tasks(id=run_id, maxResults=20)
                tasks = tasks_resp.get("items", [])

                if tasks:
                    st.markdown("**Tasks:**")
                    for task in tasks:
                        t_name = task.get("name", "unknown")
                        t_status = task.get("status", "UNKNOWN")
                        t_cpus = task.get("cpus", 0)
                        t_mem = task.get("memory", 0)
                        t_gpus = task.get("gpus", 0)
                        t_start = task.get("startTime")
                        t_elapsed = _format_elapsed(t_start) if t_start else "..."

                        t_color = STATUS_COLORS.get(t_status, "#6b7280")
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;"
                            f"<span style='color:{t_color};font-weight:bold;'>{t_status}</span> "
                            f"**{t_name}** — {t_elapsed} "
                            f"({t_cpus} CPUs, {t_mem} GB RAM, {t_gpus} GPUs)",
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.caption(f"Could not fetch tasks: {e}")

            # Progress indicator
            if status == "RUNNING" and start:
                elapsed_min = (datetime.now(timezone.utc) - start).total_seconds() / 60
                # Estimate progress (30x WGS typically ~150 min)
                estimated_total = 150
                progress = min(elapsed_min / estimated_total, 0.99)
                st.progress(progress, text=f"~{elapsed_min:.0f} / ~{estimated_total} min estimated")

            # Output URI
            output_uri = detail.get("outputUri", "")
            if output_uri:
                st.caption(f"Output: `{output_uri}`")

# ══════════════════════════════════════════════════════════════════════
# COMPLETED RUNS
# ══════════════════════════════════════════════════════════════════════

if completed_runs:
    st.subheader("Completed Runs", divider="green")

    for run in completed_runs:
        run_id = run["id"]
        name = run.get("name", f"Run {run_id}")
        start = run.get("startTime")
        stop = run.get("stopTime")

        with st.expander(f":check_circle: **{name}**", expanded=False):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Run ID", run_id)
            elapsed = _format_elapsed(start, stop) if start else "N/A"
            r2.metric("Duration", elapsed)
            cost = _estimate_cost(start, stop) if start else 0
            r3.metric("Cost", f"${cost:.2f}")

            # List output files
            try:
                detail = omics.get_run(id=run_id)
                output_uri = detail.get("outputUri", "")
                if output_uri and s3:
                    # Parse S3 URI
                    parts = output_uri.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    prefix = f"{parts[1]}{run_id}/" if len(parts) > 1 else f"{run_id}/"

                    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
                    objects = resp.get("Contents", [])

                    if objects:
                        st.markdown("**Output Files:**")
                        vcf_key = None
                        for obj in objects:
                            key = obj["Key"]
                            size_mb = obj["Size"] / 1e6
                            icon = "description"
                            if key.endswith(".vcf.gz") or key.endswith(".vcf"):
                                icon = "genetics"
                                vcf_key = key
                            elif key.endswith(".bam"):
                                icon = "storage"
                            elif key.endswith(".html"):
                                icon = "web"

                            st.markdown(
                                f"&nbsp;&nbsp;:{icon}: `{key.split('/')[-1]}` ({size_mb:.1f} MB)"
                            )

                        # Handoff button
                        if vcf_key:
                            st.divider()
                            if st.button(
                                f"Analyze VCF with MAD Agent",
                                key=f"handoff_{run_id}",
                                icon=":material/groups:",
                                type="primary",
                            ):
                                st.session_state["mad_vcf_s3_key"] = vcf_key
                                st.session_state["mad_vcf_s3_bucket"] = bucket
                                st.session_state["mad_vcf_run_id"] = run_id
                                st.switch_page(str(
                                    __import__("pathlib").Path(__file__).parent / "37_mad_agent.py"
                                ))
                    else:
                        st.info("No output files found yet.")
            except Exception as e:
                st.caption(f"Could not list outputs: {e}")

# ══════════════════════════════════════════════════════════════════════
# FAILED RUNS
# ══════════════════════════════════════════════════════════════════════

if failed_runs:
    st.subheader("Failed Runs", divider="red")

    for run in failed_runs:
        run_id = run["id"]
        name = run.get("name", f"Run {run_id}")
        status = run.get("status", "FAILED")

        with st.expander(f":error: **{name}** — {status}", expanded=False):
            try:
                detail = omics.get_run(id=run_id)
                msg = detail.get("statusMessage", "No error message available")
                st.error(msg)

                r1, r2 = st.columns(2)
                r1.metric("Run ID", run_id)
                start = detail.get("startTime")
                stop = detail.get("stopTime")
                if start and stop:
                    r2.metric("Duration before failure", _format_elapsed(start, stop))
            except Exception:
                st.error(f"Run {run_id}: {status}")

# ══════════════════════════════════════════════════════════════════════
# QUICK LAUNCH
# ══════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Quick Launch")

with st.form("launch_run"):
    st.markdown("Start a new HealthOmics Parabricks pipeline run.")

    pipeline_type = st.selectbox(
        "Pipeline",
        ["Germline DeepVariant 30x", "Somatic DeepSomatic (matched tumor-normal)"],
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fastq_r1 = st.text_input(
            "FASTQ R1 (S3 URI)",
            placeholder="s3://cognisom-genomics/fastq/sample/R1.fastq.gz",
        )
    with col_r2:
        fastq_r2 = st.text_input(
            "FASTQ R2 (S3 URI)",
            placeholder="s3://cognisom-genomics/fastq/sample/R2.fastq.gz",
        )

    if pipeline_type.startswith("Somatic"):
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            normal_r1 = st.text_input(
                "Normal FASTQ R1 (S3 URI)",
                placeholder="s3://cognisom-genomics/fastq/normal/R1.fastq.gz",
            )
        with col_n2:
            normal_r2 = st.text_input(
                "Normal FASTQ R2 (S3 URI)",
                placeholder="s3://cognisom-genomics/fastq/normal/R2.fastq.gz",
            )

    sample_name = st.text_input("Sample Name", placeholder="NA12878")

    submitted = st.form_submit_button("Start Pipeline", icon=":material/play_arrow:", type="primary")

    if submitted:
        if not fastq_r1 or not fastq_r2:
            st.error("Both FASTQ R1 and R2 URIs are required.")
        elif not sample_name:
            st.error("Sample name is required.")
        else:
            try:
                workflow_id = "7330987" if pipeline_type.startswith("Germline") else "9596598"
                role_arn = "arn:aws:iam::780457123717:role/OmicsServiceRole"
                output_uri = f"s3://cognisom-genomics/results/{sample_name}-{'germline' if pipeline_type.startswith('Germline') else 'somatic'}/"

                params = {
                    "inputFASTQ_1": fastq_r1,
                    "inputFASTQ_2": fastq_r2,
                }

                resp = omics.start_run(
                    workflowId=workflow_id,
                    workflowType="READY2RUN",
                    name=f"cognisom-{sample_name}-{pipeline_type.split()[0].lower()}",
                    roleArn=role_arn,
                    outputUri=output_uri,
                    parameters=params,
                    tags={
                        "project": "cognisom",
                        "sample": sample_name,
                        "pipeline": pipeline_type.split()[0].lower(),
                    },
                )
                st.success(f"Pipeline started! Run ID: **{resp['id']}**")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start pipeline: {e}")

# ── Auto-refresh for active runs ──
if active_runs:
    st.caption("Page auto-refreshes every 60 seconds while runs are active.")
    time.sleep(0.1)  # Prevent immediate rerun
    import time as _time
    # Use st.empty() to schedule rerun
    _placeholder = st.empty()
    _placeholder.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
