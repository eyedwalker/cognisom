"""
Page 38: Real Patient Analysis — DNA to Decision
====================================================

Full GPU pipeline: FASTQ → Parabricks → VCF → MAD Agent → Recommendation.
Real-time monitoring with DNA helix visualization.

Supports:
  - Matched tumor + normal (germline) — GOLD STANDARD
  - Tumor-only (somatic) — FALLBACK
  - Pre-called VCF upload — FASTEST
  - HealthOmics job monitoring (existing runs)
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config
import pandas as pd
import json
import logging
import time

safe_set_page_config(page_title="DNA to Decision", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("38_real_patient")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# DNA HELIX CSS + ANIMATION
# ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.dna-pipeline {
    background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 50%, #0a2a3e 100%);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    color: white;
}
.pipeline-step {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 12px;
    transition: all 0.3s ease;
}
.step-pending { background: rgba(255,255,255,0.05); }
.step-running { background: rgba(59,130,246,0.2); border-left: 4px solid #3b82f6; }
.step-complete { background: rgba(34,197,94,0.15); border-left: 4px solid #22c55e; }
.step-failed { background: rgba(239,68,68,0.15); border-left: 4px solid #ef4444; }
.step-icon { font-size: 24px; margin-right: 12px; min-width: 36px; text-align: center; }
.step-info { flex: 1; }
.step-name { font-weight: 600; font-size: 15px; }
.step-detail { color: rgba(255,255,255,0.6); font-size: 13px; }
.step-time { color: rgba(255,255,255,0.4); font-size: 12px; min-width: 80px; text-align: right; }
.helix-icon { animation: spin 3s linear infinite; display: inline-block; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.pulse { animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

st.title("🧬 DNA to Decision")
st.caption(
    "Full GPU pipeline: Raw sequencing data through NVIDIA Parabricks "
    "to MAD Agent immunotherapy recommendation."
)

# ─────────────────────────────────────────────────────────────────────────
# PIPELINE STEP DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    {"key": "upload", "name": "FASTQ Upload", "icon": "📤",
     "detail": "Raw sequencing reads to S3", "gpu": False},
    {"key": "align_tumor", "name": "Align Tumor DNA", "icon": "🧬",
     "detail": "Parabricks fq2bam — BWA-MEM + BQSR", "gpu": True},
    {"key": "align_normal", "name": "Align Normal DNA", "icon": "🩸",
     "detail": "Parabricks fq2bam — germline baseline", "gpu": True},
    {"key": "somatic_call", "name": "Somatic Calling", "icon": "🔬",
     "detail": "Mutect2 — tumor vs. normal comparison", "gpu": True},
    {"key": "hla_typing", "name": "HLA Typing", "icon": "🔑",
     "detail": "Identify immune receptor alleles", "gpu": False},
    {"key": "neoantigen", "name": "Neoantigen Prediction", "icon": "🎯",
     "detail": "MHCflurry — peptide-MHC binding", "gpu": False},
    {"key": "mad_board", "name": "MAD Board", "icon": "🏛️",
     "detail": "3-agent deliberation + consensus", "gpu": False},
    {"key": "report", "name": "Clinical Report", "icon": "📋",
     "detail": "Evidence-traced recommendation", "gpu": False},
]


def render_pipeline(step_states: dict):
    """Render the DNA pipeline visualization."""
    html_parts = ['<div class="dna-pipeline">']
    html_parts.append('<div style="text-align:center;margin-bottom:16px;">')
    html_parts.append('<span style="font-size:28px;" class="helix-icon">🧬</span> ')
    html_parts.append('<span style="font-size:20px;font-weight:700;">Genomics Pipeline</span>')
    html_parts.append('</div>')

    for step in PIPELINE_STEPS:
        state = step_states.get(step["key"], "pending")
        css_class = f"step-{state}"

        if state == "running":
            icon_html = f'<span class="pulse">{step["icon"]}</span>'
        elif state == "complete":
            icon_html = "✅"
        elif state == "failed":
            icon_html = "❌"
        else:
            icon_html = step["icon"]

        time_str = step_states.get(f"{step['key']}_time", "")
        gpu_badge = ' <span style="background:#7c3aed;padding:2px 8px;border-radius:4px;font-size:10px;">GPU</span>' if step["gpu"] and state == "running" else ""

        html_parts.append(f'''
        <div class="pipeline-step {css_class}">
            <div class="step-icon">{icon_html}</div>
            <div class="step-info">
                <div class="step-name">{step["name"]}{gpu_badge}</div>
                <div class="step-detail">{step["detail"]}</div>
            </div>
            <div class="step-time">{time_str}</div>
        </div>
        ''')

    html_parts.append('</div>')
    return "".join(html_parts)


# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────

tab_run, tab_monitor, tab_vcf, tab_guide = st.tabs([
    "Launch Pipeline", "Monitor Jobs", "Upload VCF", "Guide",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: LAUNCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────

with tab_run:
    mode = st.radio(
        "Analysis Mode",
        ["Matched Tumor-Normal (Gold Standard)", "Germline Only", "Tumor Only"],
        horizontal=True,
    )

    if "Matched" in mode:
        # Demo data selector
        demo = st.selectbox("Load Demo Data", [
            "(Custom — enter S3 paths below)",
            "SEQC2 WES — HCC1395 breast cancer + matched normal (~12 GB, ~30 min)",
            "SEQC2 WGS — HCC1395 breast cancer + matched normal (~400 GB, ~3.5 hrs)",
        ], key="demo_data")

        if "WES" in demo:
            _t_r1 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395_WES_R1.fastq.gz"
            _t_r2 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395_WES_R2.fastq.gz"
            _n_r1 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395BL_WES_R1.fastq.gz"
            _n_r2 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395BL_WES_R2.fastq.gz"
            _pid = "SEQC2-HCC1395-WES"
        elif "WGS" in demo:
            _t_r1 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395_WGS_R1.fastq.gz"
            _t_r2 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395_WGS_R2.fastq.gz"
            _n_r1 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395BL_WGS_R1.fastq.gz"
            _n_r2 = "s3://cognisom-genomics/fastq/SEQC2/HCC1395BL_WGS_R2.fastq.gz"
            _pid = "SEQC2-HCC1395-WGS"
        else:
            _t_r1 = _t_r2 = _n_r1 = _n_r2 = ""
            _pid = ""

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Tumor Sample")
            tumor_r1 = st.text_input("Tumor R1", value=_t_r1, placeholder="s3://cognisom-genomics/fastq/.../tumor_R1.fastq.gz", key="t_r1")
            tumor_r2 = st.text_input("Tumor R2", value=_t_r2, placeholder="s3://cognisom-genomics/fastq/.../tumor_R2.fastq.gz", key="t_r2")
        with col2:
            st.markdown("##### Normal Sample (Blood/Saliva)")
            normal_r1 = st.text_input("Normal R1", value=_n_r1, placeholder="s3://cognisom-genomics/fastq/.../normal_R1.fastq.gz", key="n_r1")
            normal_r2 = st.text_input("Normal R2", value=_n_r2, placeholder="s3://cognisom-genomics/fastq/.../normal_R2.fastq.gz", key="n_r2")
    elif "Germline" in mode:
        tumor_r1 = tumor_r2 = normal_r1 = normal_r2 = ""
        st.markdown("##### Germline Sample")
        germline_r1 = st.text_input("FASTQ R1", value="s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R1.fastq.gz", key="g_r1")
        germline_r2 = st.text_input("FASTQ R2", value="s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R2.fastq.gz", key="g_r2")
    else:
        normal_r1 = normal_r2 = ""
        st.markdown("##### Tumor Sample")
        tumor_r1 = st.text_input("Tumor R1", key="to_r1")
        tumor_r2 = st.text_input("Tumor R2", key="to_r2")

    _default_pid = _pid if "Matched" in mode and "_pid" in dir() else ""
    patient_id = st.text_input("Patient ID", value=_default_pid, key="pid")

    col_exec, col_launch = st.columns([2, 1])
    with col_exec:
        exec_mode = st.selectbox("Execution", [
            "AWS HealthOmics (Serverless, ~$14-27, no GPU to manage)",
            "Self-Managed GPU (L40S, ~$3-6, faster)",
        ])
    with col_launch:
        st.markdown("<br>", unsafe_allow_html=True)
        launch = st.button("Launch Pipeline", type="primary", use_container_width=True)

    if launch:
        use_healthomics = "HealthOmics" in exec_mode

        if not patient_id:
            st.error("Enter a patient ID.")
        elif "Germline" in mode and not germline_r1:
            st.error("Provide FASTQ paths.")
        elif "Matched" in mode and not all([tumor_r1, tumor_r2, normal_r1, normal_r2]):
            st.error("Provide all 4 FASTQ paths for matched analysis.")
        else:
            # Show pipeline visualization
            pipeline_container = st.empty()
            status_text = st.empty()
            states = {s["key"]: "pending" for s in PIPELINE_STEPS}

            if use_healthomics:
                try:
                    import boto3
                    omics = boto3.client("omics", region_name="us-west-2")

                    if "Germline" in mode:
                        # Get correct workflow params
                        workflow_id = "7330987"  # DeepVariant 30x
                        params = {
                            "inputFASTQ_1": germline_r1,
                            "inputFASTQ_2": germline_r2,
                        }
                        states["upload"] = "complete"
                        states["align_tumor"] = "pending"
                        states["align_normal"] = "pending"
                    elif "Matched" in mode:
                        workflow_id = "4974161"  # fq2bam 30x (start with alignment)
                        params = {
                            "inputFASTQ_1": tumor_r1,
                            "inputFASTQ_2": tumor_r2,
                        }
                    else:  # Tumor only
                        workflow_id = "7330987"
                        params = {
                            "inputFASTQ_1": tumor_r1,
                            "inputFASTQ_2": tumor_r2,
                        }

                    states["upload"] = "complete"
                    if "Germline" in mode:
                        states["align_tumor"] = "running"
                    else:
                        states["align_tumor"] = "running"
                    pipeline_container.markdown(render_pipeline(states), unsafe_allow_html=True)

                    resp = omics.start_run(
                        workflowId=workflow_id,
                        workflowType="READY2RUN",
                        name=f"cognisom-{patient_id}-{int(time.time())}",
                        roleArn="arn:aws:iam::780457123717:role/OmicsServiceRole",
                        outputUri=f"s3://cognisom-genomics/results/{patient_id}/",
                        parameters=params,
                        tags={"project": "cognisom", "sample": patient_id},
                    )

                    run_id = resp["id"]
                    st.session_state["healthomics_run_id"] = run_id
                    st.session_state["healthomics_patient_id"] = patient_id
                    st.session_state["healthomics_mode"] = mode

                    states["align_tumor"] = "running"
                    states["align_tumor_time"] = "Started"
                    pipeline_container.markdown(render_pipeline(states), unsafe_allow_html=True)

                    st.success(f"Pipeline launched! Run ID: **{run_id}**")
                    st.info(
                        f"Estimated completion: ~45-60 min for germline, ~2 hrs for somatic. "
                        f"Switch to the **Monitor Jobs** tab to track progress."
                    )

                except Exception as e:
                    st.error(f"Launch failed: {e}")
            else:
                # ── Self-Managed GPU (L40S via SSM) ──
                try:
                    import boto3

                    ec2 = boto3.client("ec2", region_name="us-west-2")
                    ssm = boto3.client("ssm", region_name="us-west-2")

                    GPU_INSTANCE_ID = "i-0ac9eb88c1b046163"

                    # Check instance state
                    resp = ec2.describe_instances(InstanceIds=[GPU_INSTANCE_ID])
                    gpu_state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]

                    if gpu_state == "stopped":
                        states["upload"] = "running"
                        states["upload_time"] = "Starting GPU..."
                        pipeline_container.markdown(render_pipeline(states), unsafe_allow_html=True)
                        ec2.start_instances(InstanceIds=[GPU_INSTANCE_ID])
                        status_text.info("GPU instance starting... This takes ~60 seconds.")

                        # Wait for running
                        import time as _time
                        for _ in range(24):
                            _time.sleep(5)
                            r = ec2.describe_instances(InstanceIds=[GPU_INSTANCE_ID])
                            if r["Reservations"][0]["Instances"][0]["State"]["Name"] == "running":
                                break
                        _time.sleep(15)  # SSM agent startup

                    states["upload"] = "complete"
                    states["upload_time"] = "GPU ready"
                    pipeline_container.markdown(render_pipeline(states), unsafe_allow_html=True)

                    # Build the SSM script based on mode
                    is_matched = "Matched" in mode
                    is_germline = "Germline" in mode

                    script_header = f"""#!/bin/bash
set -e
crontab -l 2>/dev/null | grep -v "idle\\|watchdog\\|self.stop\\|auto.stop" | crontab - 2>/dev/null || true
docker stop cognisom-kit 2>/dev/null || true
docker stop vllm-fast 2>/dev/null || true
mkdir -p /opt/cognisom/fastq/{patient_id} /opt/cognisom/results/{patient_id}
echo "STAGE_FASTQ"
"""

                    if is_matched:
                        # Matched tumor-normal: download 4 FASTQs
                        script_stage = f"""
aws s3 cp {tumor_r1} /opt/cognisom/fastq/{patient_id}/tumor_R1.fastq.gz --quiet 2>/dev/null || true
aws s3 cp {tumor_r2} /opt/cognisom/fastq/{patient_id}/tumor_R2.fastq.gz --quiet 2>/dev/null || true
aws s3 cp {normal_r1} /opt/cognisom/fastq/{patient_id}/normal_R1.fastq.gz --quiet 2>/dev/null || true
aws s3 cp {normal_r2} /opt/cognisom/fastq/{patient_id}/normal_R2.fastq.gz --quiet 2>/dev/null || true
"""
                        # Align tumor
                        script_align = f"""
echo "FQ2BAM_TUMOR_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/fastq/{patient_id}:/fastq \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun fq2bam \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-fq /fastq/tumor_R1.fastq.gz /fastq/tumor_R2.fastq.gz \\
    --out-bam /results/tumor.bam \\
    --knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz \\
    --out-recal-file /results/tumor.recal.txt \\
    --num-gpus 1 \\
    2>&1 | tail -20

echo "FQ2BAM_NORMAL_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/fastq/{patient_id}:/fastq \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun fq2bam \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-fq /fastq/normal_R1.fastq.gz /fastq/normal_R2.fastq.gz \\
    --out-bam /results/normal.bam \\
    --knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz \\
    --out-recal-file /results/normal.recal.txt \\
    --num-gpus 1 \\
    2>&1 | tail -20

echo "MUTECT2_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun mutectcaller \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --tumor-name tumor \\
    --in-tumor-bam /results/tumor.bam \\
    --normal-name normal \\
    --in-normal-bam /results/normal.bam \\
    --out-vcf /results/{patient_id}.somatic.vcf \\
    --num-gpus 1 \\
    2>&1 | tail -20
"""
                        script_upload = f"""
echo "UPLOAD_RESULTS"
aws s3 cp /opt/cognisom/results/{patient_id}/{patient_id}.somatic.vcf s3://cognisom-genomics/results/{patient_id}/{patient_id}.somatic.vcf --quiet
"""
                    elif is_germline:
                        fq_r1 = germline_r1
                        fq_r2 = germline_r2
                        script_stage = f"""
aws s3 cp {fq_r1} /opt/cognisom/fastq/{patient_id}/R1.fastq.gz --quiet 2>/dev/null || true
aws s3 cp {fq_r2} /opt/cognisom/fastq/{patient_id}/R2.fastq.gz --quiet 2>/dev/null || true
"""
                        script_align = f"""
echo "FQ2BAM_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/fastq/{patient_id}:/fastq \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun fq2bam \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-fq /fastq/R1.fastq.gz /fastq/R2.fastq.gz \\
    --out-bam /results/{patient_id}.bam \\
    --knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz \\
    --out-recal-file /results/{patient_id}.recal.txt \\
    --num-gpus 1 \\
    2>&1 | tail -20

echo "DEEPVARIANT_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun deepvariant \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-bam /results/{patient_id}.bam \\
    --out-variants /results/{patient_id}.deepvariant.vcf \\
    --num-gpus 1 \\
    2>&1 | tail -20
"""
                        script_upload = f"""
echo "UPLOAD_RESULTS"
aws s3 cp /opt/cognisom/results/{patient_id}/{patient_id}.deepvariant.vcf s3://cognisom-genomics/results/{patient_id}/{patient_id}.deepvariant.vcf --quiet
"""
                    else:
                        # Tumor only — same as germline but label differently
                        fq_r1 = tumor_r1
                        fq_r2 = tumor_r2
                        script_stage = f"""
aws s3 cp {fq_r1} /opt/cognisom/fastq/{patient_id}/R1.fastq.gz --quiet 2>/dev/null || true
aws s3 cp {fq_r2} /opt/cognisom/fastq/{patient_id}/R2.fastq.gz --quiet 2>/dev/null || true
"""
                        script_align = f"""
echo "FQ2BAM_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/fastq/{patient_id}:/fastq \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun fq2bam \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-fq /fastq/R1.fastq.gz /fastq/R2.fastq.gz \\
    --out-bam /results/{patient_id}.bam \\
    --knownSites /ref/Homo_sapiens_assembly38.known_indels.vcf.gz \\
    --out-recal-file /results/{patient_id}.recal.txt \\
    --num-gpus 1 \\
    2>&1 | tail -20

echo "DEEPVARIANT_START"
docker run --rm --gpus all \\
  -v /opt/cognisom/ref:/ref \\
  -v /opt/cognisom/results/{patient_id}:/results \\
  nvcr.io/nvidia/clara/clara-parabricks:4.3.0-1 \\
  pbrun deepvariant \\
    --ref /ref/Homo_sapiens_assembly38.fasta \\
    --in-bam /results/{patient_id}.bam \\
    --out-variants /results/{patient_id}.deepvariant.vcf \\
    --num-gpus 1 \\
    2>&1 | tail -20
"""
                        script_upload = f"""
echo "UPLOAD_RESULTS"
aws s3 cp /opt/cognisom/results/{patient_id}/{patient_id}.deepvariant.vcf s3://cognisom-genomics/results/{patient_id}/{patient_id}.deepvariant.vcf --quiet
"""

                    script_footer = f"""
echo "PIPELINE_COMPLETE"
ls -lh /opt/cognisom/results/{patient_id}/
docker start cognisom-kit 2>/dev/null || true
"""

                    ssm_script = script_header + script_stage + script_align + script_upload + script_footer

                    # Send the command
                    ssm_resp = ssm.send_command(
                        InstanceIds=[GPU_INSTANCE_ID],
                        DocumentName="AWS-RunShellScript",
                        Parameters={"commands": [ssm_script]},
                        TimeoutSeconds=7200,
                    )
                    ssm_cmd_id = ssm_resp["Command"]["CommandId"]

                    st.session_state["gpu_ssm_cmd_id"] = ssm_cmd_id
                    st.session_state["gpu_patient_id"] = patient_id
                    st.session_state["gpu_pipeline_mode"] = mode

                    states["align_tumor"] = "running"
                    states["align_tumor_time"] = "Started"
                    pipeline_container.markdown(render_pipeline(states), unsafe_allow_html=True)

                    st.success(f"GPU pipeline launched on L40S! SSM Command: `{ssm_cmd_id[:12]}...`")
                    st.info(
                        "Estimated: ~90 min for 30x WGS (fq2bam ~70 min + DeepVariant ~20 min). "
                        "Switch to **Monitor Jobs** tab to track progress."
                    )

                except Exception as e:
                    st.error(f"GPU launch failed: {e}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: MONITOR JOBS
# ─────────────────────────────────────────────────────────────────────────

with tab_monitor:
    backend = st.radio(
        "Backend",
        ["L40S GPU (SSM)", "HealthOmics"],
        horizontal=True,
    )

    if backend == "L40S GPU (SSM)":
        st.subheader("L40S GPU Job Monitor")

        ssm_cmd = st.text_input(
            "SSM Command ID",
            value=st.session_state.get("gpu_ssm_cmd_id", ""),
            key="monitor_ssm_cmd",
        )

        if st.button("Check GPU Status", type="primary", use_container_width=True) and ssm_cmd:
            try:
                import boto3
                ssm_client = boto3.client("ssm", region_name="us-west-2")
                r = ssm_client.get_command_invocation(
                    CommandId=ssm_cmd,
                    InstanceId="i-0ac9eb88c1b046163",
                )
                cmd_status = r["Status"]
                output = r.get("StandardOutputContent", "")

                # Determine pipeline stage from stdout markers
                stages_detected = []
                for marker in ["STAGE_FASTQ", "FQ2BAM_START", "FQ2BAM_TUMOR_START",
                               "FQ2BAM_NORMAL_START", "MUTECT2_START",
                               "DEEPVARIANT_START", "UPLOAD_RESULTS", "PIPELINE_COMPLETE"]:
                    if marker in output:
                        stages_detected.append(marker)

                last_stage = stages_detected[-1] if stages_detected else "PENDING"

                # Map to pipeline step states
                states = {s["key"]: "pending" for s in PIPELINE_STEPS}

                # Build progress based on detected stages
                stage_progress = {
                    "PENDING": (0, "Initializing..."),
                    "STAGE_FASTQ": (8, "Staging FASTQ from S3..."),
                    "FQ2BAM_START": (20, "Aligning reads (BWA-MEM + BQSR)..."),
                    "FQ2BAM_TUMOR_START": (15, "Aligning tumor reads..."),
                    "FQ2BAM_NORMAL_START": (40, "Aligning normal reads..."),
                    "MUTECT2_START": (65, "Somatic variant calling (Mutect2)..."),
                    "DEEPVARIANT_START": (70, "Variant calling (DeepVariant CNN)..."),
                    "UPLOAD_RESULTS": (92, "Uploading VCF to S3..."),
                    "PIPELINE_COMPLETE": (100, "Complete!"),
                }

                progress_pct, stage_label = stage_progress.get(last_stage, (0, "Unknown"))

                if cmd_status == "InProgress":
                    # Update step states for pipeline viz
                    states["upload"] = "complete" if "STAGE_FASTQ" in stages_detected else "running"
                    if "FQ2BAM_START" in stages_detected or "FQ2BAM_TUMOR_START" in stages_detected:
                        states["upload"] = "complete"
                        states["align_tumor"] = "running" if "FQ2BAM_NORMAL_START" not in stages_detected else "complete"
                    if "FQ2BAM_NORMAL_START" in stages_detected:
                        states["align_tumor"] = "complete"
                        states["align_normal"] = "running" if "MUTECT2_START" not in stages_detected else "complete"
                    if "MUTECT2_START" in stages_detected or "DEEPVARIANT_START" in stages_detected:
                        states["align_tumor"] = "complete"
                        states["align_normal"] = "complete"
                        states["somatic_call"] = "running"

                    # DNA Helix Progress Animation
                    import math
                    num_pairs = 12
                    pairs_html = ""
                    for i in range(num_pairs):
                        pair_pct = (i + 1) / num_pairs * 100
                        top = i * 15
                        offset_x = math.sin(i * 0.5) * 15
                        if pair_pct <= progress_pct:
                            cls = "completed"
                        elif pair_pct <= progress_pct + 10:
                            cls = "running active"
                        else:
                            cls = "pending"
                        pairs_html += f'<div class="base-pair {cls}" style="top:{top}px;transform:translateX({offset_x:.0f}px) rotateY({i*30}deg);"></div>'

                    particles = ""
                    for i in range(6):
                        particles += f'<div class="dna-bg-particle" style="left:{10+i*15}%;top:{5+i*14}%;animation-delay:{i*0.4}s;"></div>'

                    st.markdown(f"""
                    <style>
                    @keyframes helixSpin {{ 0% {{ transform: rotateY(0deg); }} 100% {{ transform: rotateY(360deg); }} }}
                    @keyframes helixPulse {{ 0%,100% {{ opacity: 0.3; }} 50% {{ opacity: 1; }} }}
                    @keyframes basePairGlow {{ 0%,100% {{ box-shadow: 0 0 3px rgba(0,212,170,0.3); }} 50% {{ box-shadow: 0 0 12px rgba(0,212,170,0.8); }} }}
                    .dna-progress-container {{ display:flex;align-items:center;justify-content:center;gap:24px;padding:24px;
                        background:linear-gradient(135deg,#0a0a2e,#1a1a4e,#0a2a3e);border-radius:16px;margin:16px 0;min-height:200px;position:relative;overflow:hidden; }}
                    .dna-helix {{ width:60px;height:180px;position:relative;perspective:400px;transform-style:preserve-3d;animation:helixSpin 4s linear infinite; }}
                    .base-pair {{ position:absolute;width:50px;height:4px;left:5px;border-radius:2px;transition:all 0.5s ease; }}
                    .base-pair.active {{ animation:basePairGlow 1.5s ease-in-out infinite; }}
                    .base-pair.completed {{ background:linear-gradient(90deg,#10b981,#00d4aa)!important;box-shadow:0 0 8px rgba(16,185,129,0.5); }}
                    .base-pair.pending {{ background:rgba(255,255,255,0.08); }}
                    .base-pair.running {{ background:linear-gradient(90deg,#3b82f6,#6366f1);animation:basePairGlow 1.5s ease-in-out infinite; }}
                    .dna-progress-info {{ text-align:left;color:white;flex:1;max-width:400px; }}
                    .dna-progress-bar-bg {{ width:100%;height:8px;background:rgba(255,255,255,0.1);border-radius:4px;overflow:hidden;margin:8px 0; }}
                    .dna-progress-bar-fill {{ height:100%;background:linear-gradient(90deg,#00d4aa,#6366f1);border-radius:4px;transition:width 1s ease; }}
                    .dna-bg-particle {{ position:absolute;width:3px;height:3px;border-radius:50%;background:rgba(0,212,170,0.2);animation:helixPulse 3s ease-in-out infinite; }}
                    </style>
                    <div class="dna-progress-container">
                        {particles}
                        <div class="dna-helix">{pairs_html}</div>
                        <div class="dna-progress-info">
                            <div style="font-size:1.3rem;font-weight:700;background:linear-gradient(135deg,#00d4aa,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                                Genomics Pipeline Running
                            </div>
                            <div style="font-size:0.9rem;color:rgba(255,255,255,0.7);margin:4px 0;">{stage_label}</div>
                            <div class="dna-progress-bar-bg">
                                <div class="dna-progress-bar-fill" style="width:{progress_pct}%;"></div>
                            </div>
                            <div style="display:flex;justify-content:space-between;">
                                <span style="font-size:2rem;font-weight:800;color:#00d4aa;">{progress_pct}%</span>
                                <span style="font-size:0.8rem;color:rgba(255,255,255,0.5);align-self:end;">L40S GPU</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Also show the step-by-step pipeline
                    st.markdown(render_pipeline(states), unsafe_allow_html=True)

                elif cmd_status == "Success":
                    if "PIPELINE_COMPLETE" in output:
                        for s in PIPELINE_STEPS[:4]:
                            states[s["key"]] = "complete"

                        # Completed DNA helix (all green)
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#0a0a2e,#1a1a4e);border-radius:16px;padding:24px;text-align:center;margin:16px 0;">
                            <div style="font-size:3rem;">✅</div>
                            <div style="font-size:1.5rem;font-weight:700;color:#10b981;margin:8px 0;">Pipeline Complete</div>
                            <div style="color:rgba(255,255,255,0.6);">VCF uploaded to S3 — ready for MAD Agent analysis</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(render_pipeline(states), unsafe_allow_html=True)

                        # Show output files
                        last_lines = output.strip().split("\n")[-10:]
                        with st.expander("Output Files"):
                            for line in last_lines:
                                if line.strip():
                                    st.text(line)

                        # Full end-to-end analysis: VCF → MAD Agent → 3D Target
                        pid = st.session_state.get("gpu_patient_id", "")
                        if pid:
                            is_somatic = "somatic" in output.lower() or "MUTECT2_START" in output
                            vcf_name = f"{pid}.somatic.vcf" if is_somatic else f"{pid}.deepvariant.vcf"
                            vcf_s3 = f"s3://cognisom-genomics/results/{pid}/{vcf_name}"
                            st.info(f"VCF: `{vcf_s3}`")

                            if st.button("Run Full Analysis: MAD Agent + Target Visualization",
                                         type="primary", icon=":material/biotech:",
                                         use_container_width=True):
                                with st.spinner("Downloading VCF from S3..."):
                                    import boto3 as _b3
                                    import tempfile
                                    _s3 = _b3.client("s3", region_name="us-west-2")
                                    bucket = "cognisom-genomics"
                                    key = vcf_s3.replace(f"s3://{bucket}/", "")

                                    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp:
                                        _s3.download_file(bucket, key, tmp.name)
                                        vcf_text = open(tmp.name).read()

                                st.success(f"VCF loaded: {len(vcf_text)//1024:,} KB")

                                # Build patient profile
                                with st.spinner("Building patient profile..."):
                                    from cognisom.genomics.patient_profile import PatientProfileBuilder
                                    _builder = PatientProfileBuilder()
                                    _profile = _builder.from_vcf_text(vcf_text, pid)
                                    st.session_state["patient_profile"] = _profile

                                st.markdown(f"**Variants:** {len(_profile.variants)} | "
                                           f"**Drivers:** {len(_profile.cancer_driver_mutations)} | "
                                           f"**TMB:** {_profile.tumor_mutational_burden:.1f}")

                                # Digital twin + treatment simulation
                                with st.spinner("Running treatment simulation..."):
                                    from cognisom.genomics.twin_config import DigitalTwinConfig
                                    from cognisom.genomics.treatment_simulator import TreatmentSimulator
                                    _twin = DigitalTwinConfig.from_profile_only(_profile)
                                    _sim = TreatmentSimulator()
                                    _recommended = _sim.get_recommended_treatments(_twin)
                                    _results = _sim.compare_treatments(_recommended, _twin)
                                    st.session_state["digital_twin"] = _twin
                                    st.session_state["treatment_results"] = _results

                                # MAD Board
                                with st.spinner("MAD Board deliberating..."):
                                    from cognisom.mad.board import BoardModerator
                                    _moderator = BoardModerator()
                                    _decision = _moderator.run_full_analysis(
                                        patient_id=pid, profile=_profile,
                                        twin=_twin, treatment_results=_results,
                                    )
                                    st.session_state["mad_decision"] = _decision

                                # Show recommendation
                                st.markdown("---")
                                st.markdown(f"""
                                <div style="background:linear-gradient(135deg,#0a0a2e,#1a1a4e);border-radius:16px;padding:24px;margin:16px 0;">
                                    <div style="font-size:1.5rem;font-weight:700;color:#00d4aa;margin-bottom:8px;">
                                        MAD Board Recommendation
                                    </div>
                                    <div style="font-size:1.2rem;color:white;font-weight:600;">
                                        {_decision.recommended_treatment_name}
                                    </div>
                                    <div style="color:rgba(255,255,255,0.6);margin-top:4px;">
                                        Consensus: {_decision.consensus_level} ({_decision.confidence:.0%} confidence)
                                        | Evidence: {len(_decision.evidence_chain)} items
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Show rationale
                                st.markdown(f"**Rationale:** {_decision.unified_rationale}")

                                # Neoantigen target visualization
                                if _profile.predicted_neoantigens:
                                    top_neos = [n for n in _profile.predicted_neoantigens
                                                if n.is_strong_binder][:3]
                                    if not top_neos:
                                        top_neos = [n for n in _profile.predicted_neoantigens
                                                    if n.is_weak_binder][:3]

                                    if top_neos:
                                        st.markdown("---")
                                        st.subheader("Top Neoantigen Target")
                                        _neo = top_neos[0]
                                        st.markdown(
                                            f"**{_neo.source_gene}** {_neo.mutation} | "
                                            f"Peptide: `{_neo.peptide}` | "
                                            f"HLA: **{_neo.best_hla_allele}** | "
                                            f"IC50: **{_neo.binding_affinity_nm:.1f} nM**"
                                        )

                                        with st.spinner("Fetching peptide-MHC structure..."):
                                            try:
                                                from cognisom.genomics.target_structure import TargetStructureBuilder
                                                _tsb = TargetStructureBuilder()
                                                _struct = _tsb.build(_neo)

                                                import streamlit.components.v1 as components
                                                _pdb = _struct.pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                                                _pc = _struct.peptide_chain_id
                                                _mp = _neo.mutation_position_in_peptide

                                                _viewer = f"""
                                                <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
                                                <div id="target-pmhc" style="width:100%;height:450px;border-radius:12px;
                                                     overflow:hidden;border:1px solid rgba(128,128,128,0.2);background:#0a0a2e;"></div>
                                                <script>
                                                (function(){{
                                                    var v=$3Dmol.createViewer("target-pmhc",{{backgroundColor:"0x0a0a2e"}});
                                                    v.addModel(`{_pdb}`,"pdb");
                                                    v.setStyle({{chain:"A"}},{{cartoon:{{color:"0xcccccc",opacity:0.5}}}});
                                                    v.addSurface($3Dmol.SurfaceType.VDW,{{opacity:0.12,color:"0xdddddd"}},{{chain:"A"}});
                                                    v.setStyle({{chain:"B"}},{{cartoon:{{color:"0x6366f1",opacity:0.25}}}});
                                                    v.setStyle({{chain:"{_pc}"}},{{stick:{{radius:0.25,colorscheme:"Jmol"}},sphere:{{radius:0.5,colorscheme:"Jmol"}}}});
                                                    v.addStyle({{chain:"{_pc}",resi:{_mp+1}}},{{sphere:{{radius:0.8,color:"0xff4444"}}}});
                                                    "DEFGHIJKLMNOPQRSTUVWXYZ".split("").forEach(function(c){{v.setStyle({{chain:c}},{{cartoon:{{color:"0x333",opacity:0.1}}}});}});
                                                    v.zoomTo({{chain:"{_pc}"}},600);
                                                    v.spin("y",0.3);
                                                    v.render();
                                                }})();
                                                </script>
                                                """
                                                components.html(_viewer, height=470)

                                                _method = {
                                                    "rcsb_template": f"RCSB PDB ({_struct.template_pdb_id})",
                                                    "alphafold2_multimer": "AlphaFold2-Multimer",
                                                    "peptide_only": "Extended peptide",
                                                }.get(_struct.method, _struct.method)

                                                c1, c2, c3, c4 = st.columns(4)
                                                c1.metric("Peptide", _neo.peptide)
                                                c2.metric("HLA", _neo.best_hla_allele)
                                                c3.metric("IC50", f"{_neo.binding_affinity_nm:.1f} nM")
                                                c4.metric("Structure", _method)

                                                st.caption(
                                                    f"Gray surface: MHC groove | Purple: b2m | "
                                                    f"Colored atoms: neoantigen peptide | Red: mutation site"
                                                )
                                            except Exception as e:
                                                st.warning(f"3D visualization unavailable: {e}")

                                # Update all pipeline steps to complete
                                for s in PIPELINE_STEPS:
                                    states[s["key"]] = "complete"
                                st.markdown(render_pipeline(states), unsafe_allow_html=True)
                    else:
                        st.warning("Command completed but pipeline may not have finished.")
                        st.text(output[-500:])

                elif cmd_status == "Failed":
                    st.error("Pipeline failed.")
                    st.text(r.get("StandardErrorContent", "")[-500:])

                elif cmd_status == "TimedOut":
                    st.warning("SSM command timed out — pipeline may still be running on the GPU box.")

            except Exception as e:
                st.error(f"Monitor failed: {e}")

        # Also show GPU instance state
        try:
            import boto3
            ec2 = boto3.client("ec2", region_name="us-west-2")
            resp = ec2.describe_instances(InstanceIds=["i-0ac9eb88c1b046163"])
            gpu_state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
            instance_type = resp["Reservations"][0]["Instances"][0]["InstanceType"]
            st.caption(f"GPU Instance: **{gpu_state}** ({instance_type}, L40S 48GB)")
        except Exception:
            pass

    else:
        st.subheader("HealthOmics Job Monitor")

        run_id_input = st.text_input(
            "Run ID",
            value=st.session_state.get("healthomics_run_id", ""),
            key="monitor_run_id",
        )

        col_check, col_list = st.columns(2)

        with col_check:
            if st.button("Check Status", type="primary") and run_id_input:
                try:
                    import boto3
                    omics = boto3.client("omics", region_name="us-west-2")
                    resp = omics.get_run(id=run_id_input)

                    status = resp["status"]
                    name = resp.get("name", "")
                    start_time = resp.get("startTime", "")
                    stop_time = resp.get("stopTime", "")
                    output_uri = resp.get("outputUri", "")

                    # Pipeline visualization based on status
                    states = {s["key"]: "pending" for s in PIPELINE_STEPS}
                    states["upload"] = "complete"

                    if status == "PENDING":
                        states["align_tumor"] = "pending"
                    elif status == "STARTING":
                        states["align_tumor"] = "running"
                    elif status == "RUNNING":
                        if start_time:
                            from datetime import datetime, timezone
                            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() / 60
                            if elapsed < 25:
                                states["align_tumor"] = "running"
                                states["align_tumor_time"] = f"{elapsed:.0f} min"
                            elif elapsed < 50:
                                states["align_tumor"] = "complete"
                                states["align_tumor_time"] = "~25 min"
                                states["align_normal"] = "running" if "somatic" in name.lower() else "complete"
                            else:
                                states["align_tumor"] = "complete"
                                states["align_normal"] = "complete"
                                states["somatic_call"] = "running"
                                states["somatic_call_time"] = f"{elapsed-50:.0f} min"
                    elif status == "COMPLETED":
                        for s in PIPELINE_STEPS[:4]:
                            states[s["key"]] = "complete"
                    elif status in ("FAILED", "CANCELLED"):
                        states["align_tumor"] = "failed"

                    st.markdown(render_pipeline(states), unsafe_allow_html=True)

                    status_colors = {
                        "PENDING": "🟡", "STARTING": "🟡", "RUNNING": "🔵",
                        "COMPLETED": "🟢", "FAILED": "🔴", "CANCELLED": "⚫",
                    }
                    st.markdown(f"### {status_colors.get(status, '')} {status}")
                    st.caption(f"Run: {name} | Started: {start_time}")

                    if status == "COMPLETED":
                        st.success("Pipeline complete! Ready for MAD Agent analysis.")
                except Exception as e:
                    st.error(f"Status check failed: {e}")

        with col_list:
            if st.button("List Recent Jobs"):
                try:
                    import boto3
                    omics = boto3.client("omics", region_name="us-west-2")
                    runs = omics.list_runs(maxResults=10)
                    items = runs.get("items", [])
                    if items:
                        for run in items:
                            status_icon = {"COMPLETED": "🟢", "RUNNING": "🔵",
                                           "FAILED": "🔴", "PENDING": "🟡"}.get(run.get("status", ""), "⚪")
                            st.caption(
                                f"{status_icon} **{run.get('id')}** — {run.get('name', '')} "
                                f"({run.get('status', '')})"
                            )
                    else:
                        st.info("No recent HealthOmics runs.")
                except Exception as e:
                    st.error(f"List failed: {e}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: VCF UPLOAD
# ─────────────────────────────────────────────────────────────────────────

with tab_vcf:
    st.subheader("Upload Pre-Called VCF")
    st.info(
        "Skip GPU processing — upload a VCF from any variant caller "
        "(DeepVariant, Mutect2, GATK, Strelka2) and go straight to MAD Agent."
    )

    vcf_upload = st.file_uploader("Upload VCF", type=["vcf", "gz"], key="vcf_up")
    vcf_text_input = st.text_area("Or paste VCF", height=100, key="vcf_txt")
    vcf_pid = st.text_input("Patient ID", key="vcf_pid")

    if st.button("Analyze VCF", type="primary", key="analyze_vcf"):
        vcf_text = ""
        if vcf_upload:
            content = vcf_upload.read()
            if vcf_upload.name.endswith(".gz"):
                import gzip
                vcf_text = gzip.decompress(content).decode("utf-8")
            else:
                vcf_text = content.decode("utf-8")
        elif vcf_text_input:
            vcf_text = vcf_text_input

        if not vcf_text or not vcf_pid:
            st.error("Provide VCF and patient ID.")
        else:
            states = {s["key"]: "pending" for s in PIPELINE_STEPS}
            for s in PIPELINE_STEPS[:4]:
                states[s["key"]] = "complete"
            states["hla_typing"] = "running"
            pipeline_viz = st.empty()
            pipeline_viz.markdown(render_pipeline(states), unsafe_allow_html=True)

            with st.spinner("Running MAD Agent pipeline..."):
                from cognisom.core.orchestrator import CognisomOrchestrator
                orch = CognisomOrchestrator()
                result = orch.run(patient_id=vcf_pid, vcf_text=vcf_text)

                st.session_state["patient_profile"] = result.profile
                st.session_state["digital_twin"] = result.twin
                st.session_state["treatment_results"] = result.treatments
                st.session_state["mad_decision"] = result.mad_decision

                for s in PIPELINE_STEPS:
                    states[s["key"]] = "complete"
                pipeline_viz.markdown(render_pipeline(states), unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Variants", result.variants_found)
                with c2:
                    st.metric("Drivers", result.drivers_found)
                with c3:
                    st.metric("Neoantigens", result.neoantigens_found)
                with c4:
                    if result.mad_decision:
                        st.metric("Confidence", f"{result.mad_decision.confidence:.0%}")

                if result.mad_decision:
                    d = result.mad_decision
                    st.success(f"**{d.recommended_treatment_name}** — {d.consensus_level} consensus")

# ─────────────────────────────────────────────────────────────────────────
# TAB 4: GUIDE
# ─────────────────────────────────────────────────────────────────────────

with tab_guide:
    st.subheader("How It Works: DNA to Decision")

    st.markdown("""
    ### The 4-Layer Unified Patient Profile

    | Layer | Input | Tool | Output | Time |
    |-------|-------|------|--------|------|
    | **1. Personal Baseline** | Blood/saliva FASTQ | Parabricks + OptiType | HLA alleles (immune "locks") | ~30 min |
    | **2. Malignant Shift** | Tumor biopsy FASTQ | Parabricks Mutect2 | Somatic variants (cancer "keys") | ~70 min |
    | **3. Binding Logic** | HLA + mutations | MHCflurry (14,847 alleles) | Peptide-MHC binding (lock-key fit) | ~2 min |
    | **4. Evidence Match** | Variants + binding | OncoKB + ClinicalTrials.gov | Actionable drugs + recruiting trials | ~30 sec |

    ### Cost Comparison

    | Pipeline | HealthOmics | Self-Managed GPU | Time |
    |----------|-------------|-----------------|------|
    | Germline only | ~$14 | ~$1.40 | ~45 min |
    | Matched somatic | ~$27 | ~$2.90 | ~95 min |
    | VCF analysis only | Free | Free | ~30 sec |

    ### Sample Data Available

    | Dataset | Type | Size | Status |
    |---------|------|------|--------|
    | NA12878 (GIAB) | Germline FASTQ, 30x WGS | 27 GB | On S3 |
    | SEQC2 HCC1395 | Matched tumor-normal | 200 GB | Available (NCBI) |
    | SU2C mCRPC 2019 | 429 patients, mutations | 98 MB | Integrated |
    """)

    st.caption("FOR RESEARCH USE ONLY")
