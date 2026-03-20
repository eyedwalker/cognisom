"""
Page 35: Cognisom Orchestrator
================================

One-click autonomous pipeline: patient data → complete analysis.
Upload VCF or provide FASTQ paths → the orchestrator handles everything:
GPU startup, variant calling, annotation, HLA, neoantigens, digital twin,
treatment simulation, and clinical report generation.
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Orchestrator | Cognisom", page_icon="\U0001f916", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("35_orchestrator")

import json
import time
from datetime import datetime

st.title("\U0001f916 Cognisom Orchestrator")
st.markdown(
    "**One command, full analysis.** Upload patient data and the orchestrator "
    "autonomously runs the complete pipeline: variant calling → annotation → "
    "HLA typing → neoantigen prediction → digital twin → treatment simulation → "
    "MAD Board (3-agent consensus) → clinical report."
)

# ── Pipeline Steps Visualization ──────────────────────────────────

def _step_indicator(label, icon, status):
    colors = {
        "completed": ("#10b981", "\u2713"),
        "running": ("#f59e0b", "\u25b6"),
        "pending": ("#374151", "\u25cb"),
        "failed": ("#ef4444", "\u2717"),
        "skipped": ("#6b7280", "\u2212"),
    }
    color, sym = colors.get(status, colors["pending"])
    return (
        f'<div style="text-align:center; padding:0.3rem;">'
        f'<div style="font-size:1.2rem;">{icon}</div>'
        f'<div style="font-size:0.6rem; color:{color}; font-weight:600;">{sym} {label}</div>'
        f'</div>'
    )

# Get orchestrator state
orch_state = st.session_state.get("orch_steps", {})

steps_row = st.columns(11)
step_defs = [
    ("Init", "\U0001f4cb", "init"),
    ("GPU", "\u26a1", "gpu_start"),
    ("Align", "\U0001f9ec", "parabricks"),
    ("Parse", "\U0001f4c4", "vcf_parse"),
    ("Annotate", "\U0001f50d", "variant_annotate"),
    ("HLA", "\U0001f3af", "hla_typing"),
    ("Neoantigen", "\U0001f489", "neoantigen"),
    ("Twin", "\U0001f9d1\u200d\U0001f4bb", "digital_twin"),
    ("Treat", "\U0001f48a", "treatment_sim"),
    ("MAD", "\U0001f3db\ufe0f", "mad_board"),
    ("Report", "\U0001f4ca", "clinical_report"),
]
for col, (label, icon, key) in zip(steps_row, step_defs):
    with col:
        status = orch_state.get(key, "pending")
        st.markdown(_step_indicator(label, icon, status), unsafe_allow_html=True)

st.divider()

# ── Input Selection ──────────────────────────────────────────────

st.subheader("Patient Input")

input_mode = st.radio(
    "Input type",
    [
        "VCF Text (paste or synthetic)",
        "VCF File Upload",
        "FASTQ on GPU (Self-Managed)",
        "FASTQ via HealthOmics (Serverless)",
    ],
    horizontal=True,
)

patient_id = st.text_input("Patient ID", value=f"PATIENT-{datetime.now().strftime('%H%M%S')}")

vcf_text = None
fastq_r1 = None
fastq_r2 = None
use_healthomics = False

if input_mode == "VCF Text (paste or synthetic)":
    use_synthetic = st.checkbox("Use synthetic prostate cancer demo", value=True)
    if use_synthetic:
        from cognisom.genomics.synthetic_vcf import get_synthetic_vcf
        vcf_text = get_synthetic_vcf()
        st.success("Synthetic VCF loaded (51 variants, 18 cancer drivers)")
    else:
        vcf_text = st.text_area("Paste VCF content", height=150)

elif input_mode == "VCF File Upload":
    uploaded = st.file_uploader("Upload VCF (.vcf or .vcf.gz)", type=["vcf", "gz"])
    if uploaded:
        vcf_text = uploaded.read().decode("utf-8", errors="replace")
        st.success(f"Loaded {uploaded.name}")

elif input_mode == "FASTQ on GPU (Self-Managed)":
    st.info("Provide FASTQ paths on the GPU instance (requires GPU to be running)")

    use_test = st.checkbox("Use Parabricks sample data (NA12878)", value=True)
    if use_test:
        fastq_r1 = "/opt/cognisom/jobs/test_fastq/parabricks_sample/Data/sample_1.fq.gz"
        fastq_r2 = "/opt/cognisom/jobs/test_fastq/parabricks_sample/Data/sample_2.fq.gz"
        st.success(f"Test data: `sample_1.fq.gz` (2.5 GB) + `sample_2.fq.gz` (2.7 GB)")
    else:
        fastq_r1 = ""
        fastq_r2 = ""

    c1, c2 = st.columns(2)
    with c1:
        fastq_r1 = st.text_input("FASTQ R1 path", value=fastq_r1, key="orch_r1")
    with c2:
        fastq_r2 = st.text_input("FASTQ R2 path", value=fastq_r2, key="orch_r2")

elif input_mode == "FASTQ via HealthOmics (Serverless)":
    use_healthomics = True
    st.info(
        "**AWS HealthOmics Ready2Run** — Fully serverless GPU genomics. "
        "No GPU instance needed. FASTQ must be in S3."
    )
    use_na12878 = st.checkbox("Use NA12878 benchmark data (in S3)", value=True)
    if use_na12878:
        fastq_r1 = "s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R1.fastq.gz"
        fastq_r2 = "s3://cognisom-genomics/fastq/NA12878/NA12878_30x_R2.fastq.gz"
        st.success("NA12878 30x WGS from S3 (12.1 + 13.4 GB)")
    else:
        fastq_r1 = ""
        fastq_r2 = ""

    c1, c2 = st.columns(2)
    with c1:
        fastq_r1 = st.text_input("S3 URI R1", value=fastq_r1, key="orch_ho_r1")
    with c2:
        fastq_r2 = st.text_input("S3 URI R2", value=fastq_r2, key="orch_ho_r2")

st.divider()

# ── Run Pipeline ─────────────────────────────────────────────────

if st.button("\U0001f680 Run Complete Pipeline", type="primary",
             use_container_width=True, disabled=(not vcf_text and not fastq_r1)):

    from cognisom.core.orchestrator import CognisomOrchestrator

    # Progress callback updates session state
    def on_progress(step, status, message):
        st.session_state.setdefault("orch_steps", {})[step] = status
        st.session_state["orch_message"] = f"{step}: {message}"

    orchestrator = CognisomOrchestrator(progress_callback=on_progress)

    # Reset state
    st.session_state["orch_steps"] = {}
    st.session_state["orch_result"] = None

    with st.spinner("Running autonomous pipeline..."):
        progress_bar = st.progress(0, text="Initializing...")

        result = orchestrator.run(
            patient_id=patient_id,
            vcf_text=vcf_text,
            fastq_r1=fastq_r1,
            fastq_r2=fastq_r2,
            auto_stop_gpu=False,
            use_healthomics=use_healthomics,
        )

        st.session_state["orch_result"] = result

        # Store profile in session for other pages
        if result.profile:
            st.session_state["patient_profile"] = result.profile
            st.session_state["patient_profile_dict"] = result.profile.to_dict()

    # Update step indicators
    for step_result in result.steps:
        st.session_state.setdefault("orch_steps", {})[step_result.step.value] = step_result.status.value

    st.rerun()

# ── Results Display ──────────────────────────────────────────────

result = st.session_state.get("orch_result")

if result:
    st.divider()

    if result.status == "completed":
        st.success(
            f"\u2705 **Pipeline complete** for {result.patient_id} "
            f"in {result.total_duration_seconds:.1f} seconds"
        )
    elif result.status == "failed":
        st.error(f"\u274c Pipeline failed for {result.patient_id}")
    else:
        st.info(f"Pipeline status: {result.status}")

    # Summary metrics
    st.subheader("Results Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Variants", result.variants_found)
    m2.metric("Drivers", result.drivers_found)
    m3.metric("Neoantigens", result.neoantigens_found)
    m4.metric("Treatments", result.treatments_simulated)
    m5.metric("Best Response", f"{result.best_response} ({result.best_treatment[:15]}...)"
              if result.best_treatment else "N/A")

    # Step-by-step log
    with st.expander("Pipeline Log", expanded=True):
        for step in result.steps:
            icon = {
                "completed": "\u2705", "failed": "\u274c",
                "skipped": "\u23ed\ufe0f", "running": "\u25b6",
            }.get(step.status.value, "\u25cb")
            time_str = f" ({step.duration_seconds:.1f}s)" if step.duration_seconds > 0 else ""
            st.markdown(f"{icon} **{step.step.value}**{time_str} — {step.message}")

    # Treatment comparison
    if result.treatments:
        st.subheader("Treatment Comparison")

        import plotly.graph_objects as go

        fig = go.Figure()
        for t in sorted(result.treatments, key=lambda x: x.best_response)[:8]:
            fig.add_trace(go.Scatter(
                y=t.tumor_response_curve,
                mode="lines",
                name=f"{t.treatment_name[:20]} ({t.response_category})",
            ))
        fig.add_hline(y=0.7, line_dash="dash", line_color="gray",
                      annotation_text="PR threshold")
        fig.update_layout(
            xaxis_title="Days", yaxis_title="Tumor Volume",
            height=350, legend=dict(font=dict(size=8)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)

    with c1:
        if result.report:
            st.download_button(
                "\U0001f4cb Download Clinical Report (JSON)",
                data=json.dumps(result.report, indent=2, default=str),
                file_name=f"cognisom_report_{result.patient_id}.json",
                mime="application/json",
                use_container_width=True,
            )

    with c2:
        st.download_button(
            "\U0001f4ca Download Pipeline Summary",
            data=json.dumps(result.to_dict(), indent=2, default=str),
            file_name=f"cognisom_pipeline_{result.patient_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    with c3:
        st.markdown("**Next Steps:**")
        st.markdown(
            "- \U0001f9ec **Genomic Profile** — Explore variants\n"
            "- \U0001f489 **Neoantigen Vaccine** — Design mRNA vaccine\n"
            "- \U0001f4cb **Clinical Report** — Detailed report view"
        )
