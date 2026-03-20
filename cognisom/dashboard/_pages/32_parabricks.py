"""
Page 32: GPU Genomics Pipeline (NVIDIA Parabricks)
====================================================

Visual pipeline: FASTQ → Alignment → Variant Calling → VCF →
Annotation → HLA Typing → Neoantigen → Digital Twin → Treatment

Features:
- Step-by-step pipeline visualization with progress
- System readiness check (GPU, Parabricks, reference genome)
- Germline / Somatic / RNA-seq pipeline selection
- Job monitoring with status polling
- Quality metrics after completion
- One-click VCF → Genomic Twin handoff
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(
    page_title="GPU Genomics | Cognisom",
    page_icon="\U0001f9ec",
    layout="wide",
)

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("32_parabricks")

import logging

logger = logging.getLogger(__name__)

st.title("GPU Genomics Pipeline")
st.caption(
    "NVIDIA Parabricks on L40S — raw sequencing reads to personalized "
    "treatment prediction in under an hour."
)

# ══════════════════════════════════════════════════════════════════════
# PIPELINE VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

def _pipeline_step(label, icon, status="pending", detail=""):
    """Render one step of the pipeline visualization."""
    colors = {
        "completed": ("#10b981", "#10b98133", "\u2713"),
        "running": ("#f59e0b", "#f59e0b33", "\u25b6"),
        "pending": ("#6b7280", "#6b728022", "\u25cb"),
        "error": ("#ef4444", "#ef444433", "\u2717"),
    }
    color, bg, symbol = colors.get(status, colors["pending"])
    st.markdown(
        f'<div style="background: {bg}; border: 1px solid {color}40; '
        f'border-radius: 8px; padding: 0.6rem 0.8rem; text-align: center; '
        f'min-height: 80px; display: flex; flex-direction: column; '
        f'align-items: center; justify-content: center;">'
        f'<div style="font-size: 1.5rem;">{icon}</div>'
        f'<div style="font-size: 0.75rem; font-weight: 600; color: {color}; '
        f'margin-top: 0.2rem;">{symbol} {label}</div>'
        + (f'<div style="font-size: 0.6rem; opacity: 0.6; margin-top: 0.1rem;">{detail}</div>' if detail else '')
        + '</div>',
        unsafe_allow_html=True,
    )


def _arrow():
    st.markdown(
        '<div style="text-align: center; font-size: 1.2rem; color: #6b7280; '
        'padding: 0 0.2rem;">→</div>',
        unsafe_allow_html=True,
    )


# Get current pipeline state from session
pipeline_state = st.session_state.get("pipeline_state", "idle")

# Draw the pipeline
st.markdown("### Pipeline Flow")
cols = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1])

with cols[0]:
    s = "completed" if pipeline_state in ("aligned", "called", "annotated", "complete") else (
        "running" if pipeline_state == "aligning" else "pending")
    _pipeline_step("FASTQ\nInput", "\U0001f4c4", s, "Raw reads")
with cols[1]:
    _arrow()
with cols[2]:
    s = "completed" if pipeline_state in ("called", "annotated", "complete") else (
        "running" if pipeline_state == "aligning" else "pending")
    _pipeline_step("GPU\nAlignment", "\u26a1", s, "BWA-MEM • L40S")
with cols[3]:
    _arrow()
with cols[4]:
    s = "completed" if pipeline_state in ("annotated", "complete") else (
        "running" if pipeline_state == "calling" else "pending")
    _pipeline_step("Variant\nCalling", "\U0001f52c", s, "DeepVariant/Mutect2")
with cols[5]:
    _arrow()
with cols[6]:
    s = "completed" if pipeline_state == "complete" else (
        "running" if pipeline_state == "annotating" else "pending")
    _pipeline_step("Annotation\n& HLA", "\U0001f9ec", s, "14 driver genes")
with cols[7]:
    _arrow()
with cols[8]:
    s = "completed" if pipeline_state == "complete" else "pending"
    _pipeline_step("Digital\nTwin", "\U0001f4ca", s, "Treatment sim")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# SYSTEM READINESS
# ══════════════════════════════════════════════════════════════════════

from cognisom.infrastructure.gpu_connector import get_gpu_instance_state, start_gpu_instance

gpu_state = get_gpu_instance_state()

with st.expander("System Readiness", expanded=(gpu_state != "running")):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if gpu_state == "running":
            st.markdown(
                '<div style="text-align:center">'
                '<div style="font-size:2rem">\u2705</div>'
                '<div style="font-size:0.8rem; font-weight:600; color:#10b981;">GPU Online</div>'
                '<div style="font-size:0.65rem; opacity:0.5;">L40S 48GB</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="text-align:center">'
                '<div style="font-size:2rem">\u26a0\ufe0f</div>'
                '<div style="font-size:0.8rem; font-weight:600; color:#f59e0b;">GPU Stopped</div></div>',
                unsafe_allow_html=True)
            if st.button("Start GPU", type="primary", use_container_width=True):
                ok, msg = start_gpu_instance()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    if gpu_state == "running":
        try:
            from cognisom.genomics.parabricks_runner import ParabricksRunner
            runner = ParabricksRunner()
            pb_status = runner.check_parabricks_ready()
        except Exception:
            pb_status = {}

        with c2:
            if pb_status.get("parabricks_installed"):
                st.markdown(
                    '<div style="text-align:center">'
                    '<div style="font-size:2rem">\u2705</div>'
                    '<div style="font-size:0.8rem; font-weight:600; color:#10b981;">Parabricks</div>'
                    '<div style="font-size:0.65rem; opacity:0.5;">v4.3.0-1</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:center">'
                    '<div style="font-size:2rem">\u274c</div>'
                    '<div style="font-size:0.8rem; font-weight:600; color:#ef4444;">Not Installed</div></div>',
                    unsafe_allow_html=True)

        with c3:
            if pb_status.get("reference_genome"):
                st.markdown(
                    '<div style="text-align:center">'
                    '<div style="font-size:2rem">\u2705</div>'
                    '<div style="font-size:0.8rem; font-weight:600; color:#10b981;">GRCh38</div>'
                    '<div style="font-size:0.65rem; opacity:0.5;">3.0 GB + index</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:center">'
                    '<div style="font-size:2rem">\u274c</div>'
                    '<div style="font-size:0.8rem; font-weight:600; color:#ef4444;">No Reference</div></div>',
                    unsafe_allow_html=True)

        with c4:
            st.markdown(
                '<div style="text-align:center">'
                '<div style="font-size:2rem">\u2705</div>'
                '<div style="font-size:0.8rem; font-weight:600; color:#10b981;">Entity Library</div>'
                '<div style="font-size:0.65rem; opacity:0.5;">294 entities • 23 drugs</div></div>',
                unsafe_allow_html=True)
    else:
        for c in [c2, c3, c4]:
            with c:
                st.markdown(
                    '<div style="text-align:center">'
                    '<div style="font-size:2rem">\u23f8\ufe0f</div>'
                    '<div style="font-size:0.8rem; opacity:0.5;">Start GPU first</div></div>',
                    unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# PIPELINE SELECTION & EXECUTION
# ══════════════════════════════════════════════════════════════════════

st.subheader("Run Pipeline")

tab_germline, tab_somatic, tab_rnaseq = st.tabs([
    "\U0001f9ec Germline (DeepVariant)",
    "\U0001f3af Somatic (Mutect2)",
    "\U0001f9ea RNA-seq (STAR-Fusion)",
])

with tab_germline:
    st.markdown("**Single-sample germline variant calling** — identifies inherited variants.")

    # Execution path toggle
    exec_path = st.radio(
        "Execution Path",
        [
            "Self-Managed GPU (L40S + Parabricks Docker)",
            "AWS HealthOmics (Serverless Ready2Run)",
        ],
        horizontal=True,
        key="germline_exec_path",
    )
    is_healthomics = "HealthOmics" in exec_path

    if is_healthomics:
        st.info(
            "**AWS HealthOmics** — Fully serverless, zero infrastructure. "
            "~$8.84 for 30x WGS alignment. FASTQ must be in S3."
        )
        st.markdown("""
        | Step | Workflow | Est. Cost | Est. Time |
        |------|---------|-----------|-----------|
        | Alignment | Parabricks FQ2BAM (Ready2Run) | ~$8.84 | ~1:40 |
        | Variant Calling | Parabricks DeepVariant (Ready2Run) | ~$5-10 | ~1:00 |
        | **Total** | | **~$14-19** | **~2:40** |
        """)
    else:
        st.markdown("""
        | Step | Tool | GPU Time |
        |------|------|----------|
        | Alignment | BWA-MEM (fq2bam) | ~25 min |
        | Variant Calling | DeepVariant | ~20 min |
        | **Total** | | **~45 min** |
        """)

    col1, col2 = st.columns(2)
    with col1:
        placeholder_r1 = "s3://cognisom-genomics/fastq/sample/R1.fastq.gz" if is_healthomics \
            else "/opt/cognisom/jobs/sample_R1.fastq.gz"
        g_r1 = st.text_input("FASTQ R1", placeholder=placeholder_r1, key="g_r1")
    with col2:
        placeholder_r2 = "s3://cognisom-genomics/fastq/sample/R2.fastq.gz" if is_healthomics \
            else "/opt/cognisom/jobs/sample_R2.fastq.gz"
        g_r2 = st.text_input("FASTQ R2", placeholder=placeholder_r2, key="g_r2")
    g_sample = st.text_input("Sample ID", value="SAMPLE-001", key="g_sample")

    btn_label = "\U0001f680 Run via HealthOmics" if is_healthomics else "\u26a1 Run Germline Pipeline"
    btn_disabled = False if is_healthomics else (gpu_state != "running")

    if st.button(btn_label, type="primary", disabled=btn_disabled, key="run_germline"):
        if not g_r1 or not g_r2:
            st.error("Both R1 and R2 FASTQ paths required")
        elif is_healthomics:
            st.session_state["pipeline_state"] = "aligning"
            with st.spinner("Submitting to AWS HealthOmics..."):
                try:
                    from cognisom.infrastructure.healthomics import HealthOmicsRunner
                    ho = HealthOmicsRunner()
                    run_id = ho.run_germline_pipeline(g_r1, g_r2, g_sample)
                    if run_id:
                        st.session_state["healthomics_run_id"] = run_id
                        st.success(f"HealthOmics run submitted: `{run_id}`")
                    else:
                        st.error("HealthOmics submission failed")
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.session_state["pipeline_state"] = "aligning"
            with st.spinner("Submitting to GPU..."):
                try:
                    runner = ParabricksRunner()
                    job_id = runner.run_germline(g_r1, g_r2, g_sample)
                    st.session_state["parabricks_job_id"] = job_id
                    st.success(f"Job submitted: `{job_id}`")
                except Exception as e:
                    st.error(f"Failed: {e}")

with tab_somatic:
    st.markdown("""
    **Tumor-normal somatic variant calling** — identifies cancer-specific mutations.

    | Step | Tool | GPU Time |
    |------|------|----------|
    | Tumor alignment | BWA-MEM | ~25 min |
    | Normal alignment | BWA-MEM | ~25 min |
    | Somatic calling | Mutect2 | ~45 min |
    | **Total** | | **~95 min** |
    """)

    st.markdown("##### Tumor Sample")
    tc1, tc2 = st.columns(2)
    with tc1:
        s_tr1 = st.text_input("Tumor R1", key="s_tr1")
    with tc2:
        s_tr2 = st.text_input("Tumor R2", key="s_tr2")

    st.markdown("##### Normal Sample")
    nc1, nc2 = st.columns(2)
    with nc1:
        s_nr1 = st.text_input("Normal R1", key="s_nr1")
    with nc2:
        s_nr2 = st.text_input("Normal R2", key="s_nr2")

    s_sample = st.text_input("Sample ID", value="TUMOR-001", key="s_sample")

    if st.button("\u26a1 Run Somatic Pipeline", type="primary",
                 disabled=(gpu_state != "running"), key="run_somatic"):
        if not all([s_tr1, s_tr2, s_nr1, s_nr2]):
            st.error("All 4 FASTQ paths required")
        else:
            st.session_state["pipeline_state"] = "aligning"
            with st.spinner("Submitting to GPU..."):
                try:
                    runner = ParabricksRunner()
                    job_id = runner.run_somatic(s_tr1, s_tr2, s_nr1, s_nr2, s_sample)
                    st.session_state["parabricks_job_id"] = job_id
                    st.success(f"Job submitted: `{job_id}`")
                except Exception as e:
                    st.error(f"Failed: {e}")

with tab_rnaseq:
    st.markdown("""
    **RNA-seq alignment + fusion detection** — identifies gene fusions
    like TMPRSS2-ERG (present in ~50% of prostate cancers).

    | Step | Tool | GPU Time |
    |------|------|----------|
    | RNA alignment | STAR (rna_fq2bam) | ~15 min |
    """)

    rc1, rc2 = st.columns(2)
    with rc1:
        rna_r1 = st.text_input("RNA FASTQ R1", key="rna_r1")
    with rc2:
        rna_r2 = st.text_input("RNA FASTQ R2", key="rna_r2")

    rna_sample = st.text_input("Sample ID", value="RNA-001", key="rna_sample")

    if st.button("\u26a1 Run RNA-seq Pipeline", type="primary",
                 disabled=(gpu_state != "running"), key="run_rnaseq"):
        if not rna_r1 or not rna_r2:
            st.error("Both R1 and R2 FASTQ paths required")
        else:
            with st.spinner("Submitting to GPU..."):
                try:
                    runner = ParabricksRunner()
                    job_id = runner.run_rnaseq_fusion(rna_r1, rna_r2, rna_sample)
                    st.session_state["parabricks_job_id"] = job_id
                    st.success(f"Job submitted: `{job_id}`")
                except Exception as e:
                    st.error(f"Failed: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# JOB MONITOR
# ══════════════════════════════════════════════════════════════════════

st.subheader("Job Monitor")

job_id = st.session_state.get("parabricks_job_id", "")
check_job = st.text_input("Job ID", value=job_id, key="check_job")

if check_job and gpu_state == "running":
    col_check, col_load = st.columns([1, 1])

    with col_check:
        if st.button("Check Status", use_container_width=True):
            try:
                runner = ParabricksRunner()
                status = runner.get_job_status(check_job)
                state = status.get("state", "unknown")

                if state == "completed":
                    st.success(f"\u2705 **{status.get('pipeline', 'Pipeline')}** completed!")
                    st.session_state["pipeline_state"] = "complete"
                elif state == "running":
                    st.info(f"\u25b6 **{status.get('pipeline', 'Pipeline')}** running...")
                    st.caption(f"Started: {status.get('start_time', 'unknown')}")
                elif state == "not_found":
                    st.warning(f"Job `{check_job}` not found")
                else:
                    st.code(str(status))
            except Exception as e:
                st.error(str(e))

    with col_load:
        if st.button("Load VCF \u2192 Genomic Twin", type="primary",
                      use_container_width=True):
            try:
                runner = ParabricksRunner()
                vcf_text = runner.get_result_vcf(check_job)
                if vcf_text and vcf_text.startswith("##"):
                    from cognisom.genomics.patient_profile import PatientProfileBuilder
                    builder = PatientProfileBuilder()
                    profile = builder.from_vcf_text(vcf_text, patient_id=check_job)
                    st.session_state["patient_profile"] = profile
                    st.session_state["patient_profile_dict"] = profile.to_dict()
                    st.session_state["pipeline_state"] = "complete"

                    # Show summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Variants", len(profile.variants))
                    m2.metric("Drivers", len(profile.cancer_driver_mutations))
                    m3.metric("TMB", f"{profile.tumor_mutational_burden:.1f}")
                    m4.metric("Neoantigens", len(profile.predicted_neoantigens))

                    st.success(
                        f"Loaded {len(profile.variants)} variants. "
                        f"Go to **Genomic Profile** to continue the digital twin pipeline."
                    )
                else:
                    st.error("No VCF output found. Check if the pipeline completed.")
            except Exception as e:
                st.error(f"Failed to load VCF: {e}")

# List jobs
if gpu_state == "running":
    with st.expander("All Jobs on GPU"):
        try:
            runner = ParabricksRunner()
            jobs = runner.list_jobs()
            if jobs:
                for job in jobs:
                    icon = "\u2705" if job["state"] == "completed" else (
                        "\u25b6" if job["state"] == "running" else "\u25cb")
                    st.markdown(f"{icon} `{job['job_id']}` — {job['pipeline']} — **{job['state']}**")
            else:
                st.caption("No jobs found")
        except Exception:
            st.caption("Could not list jobs")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════

with st.expander("Complete Pipeline Architecture"):
    st.code("""
FASTQ Files (Raw Sequencing Reads)
    |
    v
[NVIDIA Parabricks on L40S GPU]
    |
    +-- Germline: fq2bam (BWA-MEM) --> DeepVariant --> VCF
    |
    +-- Somatic:  fq2bam x2 (tumor+normal) --> Mutect2 --> VCF
    |
    +-- RNA-seq:  rna_fq2bam (STAR) --> Fusion calls
    |
    v
VCF (Variant Call Format)
    |
    v
[Cognisom Genomics Pipeline]
    |
    +-- Variant Annotator (14 cancer driver genes)
    +-- TMB / MSI calculation
    +-- HLA Typing (6 alleles)
    +-- Neoantigen Prediction (MHC-I binding)
    |
    v
Patient Profile
    |
    v
[Entity-Driven Digital Twin]
    |
    +-- 294 entities with physics parameters
    +-- 23 drug treatment profiles from entity library
    +-- Gene immune effects aggregated from entities
    |
    v
Treatment Simulation
    |
    +-- 9 therapy regimens + entity-derived drugs
    +-- RECIST response classification
    +-- irAE risk prediction
    +-- Neoantigen vaccine design
    |
    v
Personalized Treatment Recommendation
    """, language=None)

with st.expander("GPU Performance Benchmarks"):
    st.markdown("""
    | Pipeline | Input | L40S GPU | CPU (24 cores) | Speedup |
    |----------|-------|----------|----------------|---------|
    | BWA-MEM alignment | 30x WGS FASTQ (~90GB) | ~25 min | ~8 hours | **19x** |
    | DeepVariant | 30x WGS BAM | ~20 min | ~6 hours | **18x** |
    | Mutect2 somatic | Tumor + Normal BAMs | ~45 min | ~24 hours | **32x** |
    | STAR RNA alignment | RNA-seq FASTQ | ~15 min | ~2 hours | **8x** |
    | **Complete germline** | **FASTQ → VCF** | **~45 min** | **~14 hours** | **19x** |
    | **Complete somatic** | **2x FASTQ → VCF** | **~95 min** | **~40 hours** | **25x** |

    L40S: 48GB VRAM, 18,176 CUDA cores, 568 Tensor cores
    """)
