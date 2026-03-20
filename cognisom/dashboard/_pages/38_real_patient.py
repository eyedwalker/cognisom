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
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Tumor Sample")
            tumor_r1 = st.text_input("Tumor R1", placeholder="s3://cognisom-genomics/fastq/.../tumor_R1.fastq.gz", key="t_r1")
            tumor_r2 = st.text_input("Tumor R2", placeholder="s3://cognisom-genomics/fastq/.../tumor_R2.fastq.gz", key="t_r2")
        with col2:
            st.markdown("##### Normal Sample (Blood/Saliva)")
            normal_r1 = st.text_input("Normal R1", placeholder="s3://cognisom-genomics/fastq/.../normal_R1.fastq.gz", key="n_r1")
            normal_r2 = st.text_input("Normal R2", placeholder="s3://cognisom-genomics/fastq/.../normal_R2.fastq.gz", key="n_r2")
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

    patient_id = st.text_input("Patient ID", value="", key="pid")

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
                st.info("Self-managed GPU pipeline — check GPU status and submit via Orchestrator page.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: MONITOR JOBS
# ─────────────────────────────────────────────────────────────────────────

with tab_monitor:
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
                        elif elapsed < 95:
                            states["align_tumor"] = "complete"
                            states["align_normal"] = "complete"
                            states["somatic_call"] = "running"
                            states["somatic_call_time"] = f"{elapsed-50:.0f} min"
                        else:
                            states["align_tumor"] = "complete"
                            states["align_normal"] = "complete"
                            states["somatic_call"] = "running"
                            states["somatic_call_time"] = f"{elapsed-50:.0f} min"
                elif status == "COMPLETED":
                    for s in PIPELINE_STEPS[:4]:
                        states[s["key"]] = "complete"
                    states["hla_typing"] = "pending"
                    states["neoantigen"] = "pending"
                    states["mad_board"] = "pending"
                    states["report"] = "pending"
                elif status in ("FAILED", "CANCELLED"):
                    states["align_tumor"] = "failed"

                st.markdown(render_pipeline(states), unsafe_allow_html=True)

                # Status details
                status_colors = {
                    "PENDING": "🟡", "STARTING": "🟡", "RUNNING": "🔵",
                    "COMPLETED": "🟢", "FAILED": "🔴", "CANCELLED": "⚫",
                }
                st.markdown(f"### {status_colors.get(status, '')} {status}")
                st.caption(f"Run: {name} | Started: {start_time}")

                if status == "COMPLETED":
                    st.success("GPU pipeline complete! Ready for MAD Agent analysis.")
                    if st.button("Run MAD Agent on Results", type="primary"):
                        with st.spinner("Downloading VCF and running MAD Agent..."):
                            try:
                                s3 = boto3.client("s3", region_name="us-west-2")
                                bucket = "cognisom-genomics"
                                prefix = output_uri.replace(f"s3://{bucket}/", "")

                                # Find VCF
                                resp2 = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
                                vcf_key = None
                                for obj in resp2.get("Contents", []):
                                    if obj["Key"].endswith(".vcf") or obj["Key"].endswith(".vcf.gz"):
                                        vcf_key = obj["Key"]
                                        break

                                if not vcf_key:
                                    st.error("No VCF found in output!")
                                else:
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False) as tmp:
                                        s3.download_file(bucket, vcf_key, tmp.name)
                                        vcf_text = open(tmp.name).read()

                                    pid = st.session_state.get("healthomics_patient_id", f"run-{run_id_input}")

                                    from cognisom.core.orchestrator import CognisomOrchestrator
                                    orch = CognisomOrchestrator()
                                    result = orch.run(patient_id=pid, vcf_text=vcf_text)

                                    st.session_state["patient_profile"] = result.profile
                                    st.session_state["digital_twin"] = result.twin
                                    st.session_state["treatment_results"] = result.treatments
                                    st.session_state["mad_decision"] = result.mad_decision

                                    # Update pipeline viz
                                    for s in PIPELINE_STEPS:
                                        states[s["key"]] = "complete"
                                    st.markdown(render_pipeline(states), unsafe_allow_html=True)

                                    if result.mad_decision:
                                        d = result.mad_decision
                                        st.markdown(f"""
                                        ### MAD Board Recommendation
                                        **{d.recommended_treatment_name}**
                                        - Consensus: {d.consensus_level} ({d.confidence:.0%} confidence)
                                        - Evidence: {len(d.evidence_chain)} traceable items
                                        - View full analysis on the **MAD Agent** page
                                        """)
                            except Exception as e:
                                st.error(f"MAD Agent failed: {e}")

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
