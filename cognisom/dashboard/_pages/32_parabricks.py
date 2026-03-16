"""
Page 32: GPU Genomics Pipeline (NVIDIA Parabricks)
====================================================

Run NVIDIA Parabricks GPU-accelerated genomics pipelines:
- Germline variant calling (fq2bam + DeepVariant)
- Somatic variant calling (fq2bam + Mutect2)
- RNA-seq fusion detection (STAR-Fusion)

Input: FASTQ files on GPU box or S3
Output: VCF → feeds into Genomic Twin pipeline
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

import time
import logging

logger = logging.getLogger(__name__)

st.title("GPU Genomics Pipeline")
st.markdown(
    "**NVIDIA Parabricks** — GPU-accelerated variant calling on L40S. "
    "Process raw FASTQ files into variant calls (VCF) in minutes, not hours."
)

# ── Status Check ─────────────────────────────────────────────────

st.subheader("System Status")

try:
    from cognisom.genomics.parabricks_runner import ParabricksRunner
    runner = ParabricksRunner()
except Exception as e:
    st.error(f"Parabricks runner initialization failed: {e}")
    st.stop()

# Check GPU and Parabricks status
from cognisom.infrastructure.gpu_connector import get_gpu_instance_state, start_gpu_instance

gpu_state = get_gpu_instance_state()

col_gpu, col_pb, col_ref = st.columns(3)

with col_gpu:
    if gpu_state == "running":
        st.success("GPU Instance: Running")
    elif gpu_state == "stopped":
        st.warning("GPU Instance: Stopped")
        if st.button("Start GPU", type="primary"):
            ok, msg = start_gpu_instance()
            if ok:
                st.success(msg)
                st.info("Wait 2-3 minutes, then refresh.")
            else:
                st.error(msg)
    else:
        st.info(f"GPU Instance: {gpu_state}")

if gpu_state == "running":
    pb_status = runner.check_parabricks_ready()

    with col_pb:
        if pb_status.get("parabricks_installed"):
            st.success("Parabricks: Installed")
        else:
            st.error("Parabricks: Not installed")
            st.caption("Run: `docker pull nvcr.io/nvidia/clara/clara-parabricks:4.3.1`")

    with col_ref:
        if pb_status.get("reference_genome"):
            st.success("Reference Genome: GRCh38")
        else:
            st.error("Reference: Not found")
            st.caption("Download GRCh38 to /opt/cognisom/ref/")

    if pb_status.get("gpu_info"):
        st.caption(f"GPU: {pb_status['gpu_info'].strip()}")
else:
    with col_pb:
        st.info("Start GPU to check Parabricks status")
    with col_ref:
        st.info("Start GPU to check reference genome")

st.divider()

# ── Pipeline Selection ───────────────────────────────────────────

st.subheader("Run Pipeline")

pipeline = st.radio(
    "Select pipeline",
    ["Germline (DeepVariant)", "Somatic (Mutect2)", "RNA-seq (STAR-Fusion)"],
    horizontal=True,
)

if pipeline == "Germline (DeepVariant)":
    st.markdown("""
    **Germline Variant Calling**: Identifies inherited variants from a single sample.

    Pipeline: `fq2bam` (BWA-MEM alignment) → `DeepVariant` (AI variant calling)

    - Input: Paired-end FASTQ (R1 + R2)
    - Output: VCF with germline variants
    - Time: ~45 minutes for 30x WGS on L40S
    """)

    col1, col2 = st.columns(2)
    with col1:
        fastq_r1 = st.text_input("FASTQ R1 path (on GPU box)",
                                  placeholder="/opt/cognisom/jobs/sample_R1.fastq.gz")
    with col2:
        fastq_r2 = st.text_input("FASTQ R2 path (on GPU box)",
                                  placeholder="/opt/cognisom/jobs/sample_R2.fastq.gz")

    sample_id = st.text_input("Sample ID", value="SAMPLE-001")

    if st.button("Run Germline Pipeline", type="primary", disabled=(gpu_state != "running")):
        if not fastq_r1 or not fastq_r2:
            st.error("Please provide both R1 and R2 FASTQ paths")
        else:
            with st.spinner("Submitting germline pipeline to GPU..."):
                job_id = runner.run_germline(fastq_r1, fastq_r2, sample_id)
                st.success(f"Job submitted: `{job_id}`")
                st.session_state["parabricks_job_id"] = job_id

elif pipeline == "Somatic (Mutect2)":
    st.markdown("""
    **Somatic Variant Calling**: Identifies tumor-specific mutations by comparing
    tumor vs matched normal samples.

    Pipeline: `fq2bam` (tumor) → `fq2bam` (normal) → `Mutect2` (somatic calling)

    - Input: 2 paired-end FASTQ sets (tumor + normal)
    - Output: VCF with somatic variants
    - Time: ~95 minutes for 30x WGS on L40S
    """)

    st.markdown("**Tumor Sample**")
    col1, col2 = st.columns(2)
    with col1:
        tumor_r1 = st.text_input("Tumor FASTQ R1", key="tumor_r1",
                                  placeholder="/opt/cognisom/jobs/tumor_R1.fastq.gz")
    with col2:
        tumor_r2 = st.text_input("Tumor FASTQ R2", key="tumor_r2")

    st.markdown("**Normal Sample**")
    col3, col4 = st.columns(2)
    with col3:
        normal_r1 = st.text_input("Normal FASTQ R1", key="normal_r1",
                                   placeholder="/opt/cognisom/jobs/normal_R1.fastq.gz")
    with col4:
        normal_r2 = st.text_input("Normal FASTQ R2", key="normal_r2")

    sample_id = st.text_input("Sample ID", value="TUMOR-001", key="somatic_sample")

    if st.button("Run Somatic Pipeline", type="primary", disabled=(gpu_state != "running")):
        if not all([tumor_r1, tumor_r2, normal_r1, normal_r2]):
            st.error("All 4 FASTQ paths required (tumor R1/R2 + normal R1/R2)")
        else:
            with st.spinner("Submitting somatic pipeline to GPU..."):
                job_id = runner.run_somatic(tumor_r1, tumor_r2, normal_r1, normal_r2, sample_id)
                st.success(f"Job submitted: `{job_id}`")
                st.session_state["parabricks_job_id"] = job_id

elif pipeline == "RNA-seq (STAR-Fusion)":
    st.markdown("""
    **RNA-seq Fusion Detection**: Aligns RNA-seq reads and detects gene fusions
    (e.g., TMPRSS2-ERG in prostate cancer).

    Pipeline: `rna_fq2bam` (STAR alignment) → fusion calling

    - Input: Paired-end RNA FASTQ
    - Output: BAM + fusion calls
    - Time: ~15 minutes on L40S
    """)

    col1, col2 = st.columns(2)
    with col1:
        rna_r1 = st.text_input("RNA FASTQ R1", key="rna_r1")
    with col2:
        rna_r2 = st.text_input("RNA FASTQ R2", key="rna_r2")

    sample_id = st.text_input("Sample ID", value="RNA-001", key="rna_sample")

    if st.button("Run RNA-seq Pipeline", type="primary", disabled=(gpu_state != "running")):
        if not rna_r1 or not rna_r2:
            st.error("Both R1 and R2 FASTQ paths required")
        else:
            with st.spinner("Submitting RNA-seq pipeline to GPU..."):
                job_id = runner.run_rnaseq_fusion(rna_r1, rna_r2, sample_id)
                st.success(f"Job submitted: `{job_id}`")
                st.session_state["parabricks_job_id"] = job_id

st.divider()

# ── Job Monitor ──────────────────────────────────────────────────

st.subheader("Job Monitor")

job_id = st.session_state.get("parabricks_job_id", "")
manual_job_id = st.text_input("Job ID to check", value=job_id)

if manual_job_id and gpu_state == "running":
    if st.button("Check Status"):
        status = runner.get_job_status(manual_job_id)
        state = status.get("state", "unknown")

        if state == "completed":
            st.success(f"Pipeline **{status.get('pipeline', '')}** completed!")

            # Offer to load into Genomic Twin
            if st.button("Load VCF into Genomic Twin", type="primary"):
                vcf_text = runner.get_result_vcf(manual_job_id)
                if vcf_text:
                    from cognisom.genomics.patient_profile import PatientProfileBuilder
                    builder = PatientProfileBuilder()
                    profile = builder.from_vcf_text(vcf_text, patient_id=manual_job_id)
                    st.session_state["patient_profile"] = profile
                    st.session_state["patient_profile_dict"] = profile.to_dict()
                    st.success(
                        f"Loaded {len(profile.variants)} variants, "
                        f"{len(profile.cancer_driver_mutations)} drivers. "
                        f"Go to **Genomic Profile** to continue."
                    )
                else:
                    st.error("Could not retrieve VCF from GPU box")

        elif state == "running":
            st.info(f"Pipeline **{status.get('pipeline', '')}** is running...")
            st.caption(f"Started: {status.get('start_time', 'unknown')}")

        elif state == "not_found":
            st.warning(f"Job `{manual_job_id}` not found on GPU box")

        else:
            st.code(str(status))

# List existing jobs
if gpu_state == "running":
    with st.expander("All Jobs"):
        jobs = runner.list_jobs()
        if jobs:
            for job in jobs:
                st.markdown(
                    f"- `{job['job_id']}` — {job['pipeline']} — **{job['state']}**"
                )
        else:
            st.caption("No jobs found")

st.divider()

# ── Processing Times ─────────────────────────────────────────────

with st.expander("Estimated Processing Times (L40S 48GB)"):
    st.markdown("""
    | Pipeline | Input | GPU Time | CPU Time (comparison) |
    |----------|-------|----------|----------------------|
    | fq2bam (alignment) | 30x WGS FASTQ | ~25 min | ~8 hours |
    | DeepVariant (germline) | 30x BAM | ~20 min | ~6 hours |
    | Mutect2 (somatic) | Tumor + Normal | ~45 min | ~24 hours |
    | STAR-Fusion (RNA-seq) | RNA FASTQ | ~15 min | ~2 hours |
    | **Total germline** | FASTQ → VCF | **~45 min** | ~14 hours |
    | **Total somatic** | 2x FASTQ → VCF | **~95 min** | ~40 hours |

    GPU acceleration: **10-30x faster** than CPU-only processing.
    """)
