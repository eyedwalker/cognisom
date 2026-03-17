"""
Page 36: TCGA Validation & VC Demo
=====================================

Two-in-one page:
1. Run TCGA-PRAD validation (100 patients from cBioPortal)
2. VC-friendly visualization of platform capabilities and results
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(page_title="Validation & Demo | Cognisom", page_icon="\U0001f4ca", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("36_validation_demo")

import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.title("\U0001f4ca TCGA Validation & Platform Demo")

tab_demo, tab_validate = st.tabs(["\U0001f3af Platform Overview (VC Demo)", "\U0001f52c TCGA Validation"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1: VC DEMO
# ══════════════════════════════════════════════════════════════════════

with tab_demo:
    # Hero metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a3e, #0a2a4a); border-radius: 16px;
                padding: 2rem; margin-bottom: 1.5rem; text-align: center;">
        <div style="font-size: 2rem; font-weight: 800; color: #e2e8f0;">
            Cognisom: Personalized Molecular Digital Twin
        </div>
        <div style="font-size: 1rem; color: rgba(255,255,255,0.6); margin-top: 0.5rem;">
            From patient DNA to personalized treatment prediction in 8 minutes on GPU
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key numbers
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Entity Library", "294", help="PhD-level biological entities with 3D visualization")
    k2.metric("Drug Treatments", "23", help="Entity-driven, add new drugs without code changes")
    k3.metric("3D Structures", "97%", help="285/294 entities have real PDB crystal structures")
    k4.metric("Pipeline Speed", "8 min", help="FASTQ → treatment on L40S GPU")
    k5.metric("PDB Mappings", "221", help="Unique crystal structures mapped to entities")

    st.divider()

    # The pipeline visual
    st.subheader("End-to-End Pipeline")
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem;
                padding: 1rem; flex-wrap: wrap;">
        <div style="background: #1e3a5f; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\U0001f4c4</div>
            <div style="font-size: 0.75rem; font-weight: 600;">Patient DNA</div>
            <div style="font-size: 0.6rem; opacity: 0.5;">FASTQ / VCF</div>
        </div>
        <div style="font-size: 1.5rem; color: #6366f1;">\u2192</div>
        <div style="background: #1e3a5f; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\u26a1</div>
            <div style="font-size: 0.75rem; font-weight: 600;">GPU Genomics</div>
            <div style="font-size: 0.6rem; opacity: 0.5;">Parabricks L40S</div>
        </div>
        <div style="font-size: 1.5rem; color: #6366f1;">\u2192</div>
        <div style="background: #1e3a5f; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\U0001f9ec</div>
            <div style="font-size: 0.75rem; font-weight: 600;">294 Entities</div>
            <div style="font-size: 0.6rem; opacity: 0.5;">Entity Library</div>
        </div>
        <div style="font-size: 1.5rem; color: #6366f1;">\u2192</div>
        <div style="background: #1e3a5f; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\U0001f9d1\u200d\U0001f4bb</div>
            <div style="font-size: 0.75rem; font-weight: 600;">Digital Twin</div>
            <div style="font-size: 0.6rem; opacity: 0.5;">Entity-Driven</div>
        </div>
        <div style="font-size: 1.5rem; color: #6366f1;">\u2192</div>
        <div style="background: #1e3a5f; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\U0001f48a</div>
            <div style="font-size: 0.75rem; font-weight: 600;">23 Treatments</div>
            <div style="font-size: 0.6rem; opacity: 0.5;">Simulation</div>
        </div>
        <div style="font-size: 1.5rem; color: #6366f1;">\u2192</div>
        <div style="background: #10b981; padding: 0.8rem 1.2rem; border-radius: 8px; text-align: center;">
            <div style="font-size: 1.5rem;">\U0001f4cb</div>
            <div style="font-size: 0.75rem; font-weight: 600; color: white;">Report</div>
            <div style="font-size: 0.6rem; color: rgba(255,255,255,0.7);">Clinical</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Live demo — run synthetic patient
    st.subheader("Live Demo: Synthetic Patient Analysis")

    if st.button("\U0001f680 Run Demo Analysis (6 seconds)", type="primary",
                 use_container_width=True):
        import os, tempfile
        os.environ.setdefault('COGNISOM_DATA_DIR',
                              os.environ.get('COGNISOM_DATA_DIR', '/app/data'))

        with st.spinner("Running autonomous pipeline..."):
            from cognisom.core.orchestrator import CognisomOrchestrator
            from cognisom.genomics.synthetic_vcf import get_synthetic_vcf

            orch = CognisomOrchestrator()
            result = orch.run("DEMO-VC", vcf_text=get_synthetic_vcf(), auto_stop_gpu=False)

        if result.status == "completed":
            st.success(f"Analysis complete in {result.total_duration_seconds:.1f}s")

            # Results in VC-friendly format
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Variants Found", result.variants_found)
            r2.metric("Cancer Drivers", result.drivers_found)
            r3.metric("Neoantigens", result.neoantigens_found)
            r4.metric("Best Treatment", result.best_treatment[:20] if result.best_treatment else "N/A")

            # Treatment comparison
            if result.treatments:
                fig = go.Figure()
                for t in sorted(result.treatments, key=lambda x: x.best_response)[:6]:
                    reduction = (1 - t.best_response) * 100
                    fig.add_trace(go.Bar(
                        x=[t.treatment_name[:25]],
                        y=[reduction],
                        marker_color="#10b981" if t.response_category in ("CR", "PR") else "#f59e0b",
                        text=f"{reduction:.0f}%",
                        textposition="outside",
                    ))
                fig.update_layout(
                    title="Predicted Tumor Reduction by Treatment",
                    yaxis_title="Tumor Reduction (%)",
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Competitive landscape
    st.subheader("Competitive Landscape")
    comp_data = pd.DataFrame({
        "Platform": ["Cognisom", "PhysiCell", "VCell", "COPASI", "CompuCell3D"],
        "Entity Library": [294, 0, 0, 0, 0],
        "GPU Accelerated": ["Yes (L40S)", "No", "No", "No", "No"],
        "FASTQ→Treatment": ["8 min", "N/A", "N/A", "N/A", "N/A"],
        "Entity-Driven Sim": ["Yes", "No", "No", "No", "No"],
        "3D Visualization": ["97% PDB", "Limited", "2D", "None", "Limited"],
        "Neoantigen Vaccine": ["Yes", "No", "No", "No", "No"],
    })
    st.dataframe(comp_data, use_container_width=True, hide_index=True)

    st.divider()

    # Technology stack
    st.subheader("Technology Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        **NVIDIA**
        - Parabricks (GPU genomics)
        - BioNeMo NIMs (11 models)
        - Isaac Sim (RTX rendering)
        - Warp (GPU physics)
        - Inception Program member
        """)
    with t2:
        st.markdown("""
        **AWS**
        - L40S GPU (g6e.2xlarge)
        - Split architecture (CPU/GPU)
        - Cognito authentication
        - S3 data storage
        - SSM orchestration
        """)
    with t3:
        st.markdown("""
        **Data Sources**
        - RCSB PDB (221 structures)
        - SABIO-RK (enzyme kinetics)
        - cBioPortal (TCGA validation)
        - KEGG / Reactome / STRING
        - UniProt / NCBI / PubChem
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: TCGA VALIDATION
# ══════════════════════════════════════════════════════════════════════

with tab_validate:
    st.subheader("TCGA-PRAD Validation")
    st.markdown(
        "Validate Cognisom predictions against **TCGA prostate adenocarcinoma** "
        "clinical outcomes. Downloads mutation and clinical data from cBioPortal "
        "(494 patients available, free API)."
    )

    n_patients = st.slider("Number of patients to validate", 10, 200, 50, step=10)

    if st.button(f"\U0001f52c Run Validation on {n_patients} TCGA Patients",
                 type="primary", use_container_width=True):

        from cognisom.validation.tcga_validator import TCGAValidator
        import os
        os.environ.setdefault('COGNISOM_DATA_DIR',
                              os.environ.get('COGNISOM_DATA_DIR', '/app/data'))

        validator = TCGAValidator()

        progress_bar = st.progress(0, text="Downloading TCGA data from cBioPortal...")
        status_text = st.empty()
        results_container = st.container()

        def on_progress(current, total, patient_id, status):
            if total > 0:
                progress_bar.progress(current / total,
                                      text=f"Patient {current}/{total}: {patient_id}")

        summary = validator.run_validation(n_patients, progress_callback=on_progress)
        progress_bar.progress(1.0, text="Validation complete!")

        st.session_state["tcga_summary"] = summary

        with results_container:
            st.success(
                f"Validated {summary.n_completed}/{summary.n_patients} patients "
                f"in {summary.total_time_seconds:.0f}s"
            )

    # Display results
    summary = st.session_state.get("tcga_summary")

    if summary:
        st.divider()

        # Summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Patients", f"{summary.n_completed}/{summary.n_patients}")
        m2.metric("Avg Mutations", f"{summary.mean_mutations:.0f}")
        m3.metric("Avg Drivers", f"{summary.mean_drivers:.1f}")
        m4.metric("Avg TMB", f"{summary.mean_tmb:.1f}")
        m5.metric("Avg Time/Patient", f"{summary.mean_processing_time:.2f}s")

        st.divider()

        # Visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            # Subtype distribution
            if summary.subtypes:
                fig_sub = go.Figure(go.Pie(
                    labels=list(summary.subtypes.keys()),
                    values=list(summary.subtypes.values()),
                    hole=0.4,
                ))
                fig_sub.update_layout(title="Molecular Subtypes", height=350)
                st.plotly_chart(fig_sub, use_container_width=True)

            # Treatment distribution
            if summary.treatment_distribution:
                treats = sorted(summary.treatment_distribution.items(), key=lambda x: -x[1])
                fig_treat = go.Figure(go.Bar(
                    x=[t[0][:20] for t in treats],
                    y=[t[1] for t in treats],
                    marker_color="#6366f1",
                ))
                fig_treat.update_layout(
                    title="Predicted Best Treatment Distribution",
                    height=350,
                )
                st.plotly_chart(fig_treat, use_container_width=True)

        with col_right:
            # Response distribution
            if summary.response_distribution:
                resp_colors = {"CR": "#10b981", "PR": "#3b82f6", "SD": "#f59e0b", "PD": "#ef4444"}
                fig_resp = go.Figure(go.Pie(
                    labels=list(summary.response_distribution.keys()),
                    values=list(summary.response_distribution.values()),
                    marker_colors=[resp_colors.get(k, "#6b7280")
                                   for k in summary.response_distribution.keys()],
                    hole=0.4,
                ))
                fig_resp.update_layout(title="Response Category Distribution", height=350)
                st.plotly_chart(fig_resp, use_container_width=True)

            # Driver mutations per patient
            driver_counts = [r.n_drivers for r in summary.results if r.status == "completed"]
            if driver_counts:
                fig_drivers = go.Figure(go.Histogram(
                    x=driver_counts,
                    nbinsx=max(driver_counts) + 1 if driver_counts else 10,
                    marker_color="#00d4aa",
                ))
                fig_drivers.update_layout(
                    title="Cancer Driver Mutations per Patient",
                    xaxis_title="Number of Drivers",
                    yaxis_title="Patients",
                    height=350,
                )
                st.plotly_chart(fig_drivers, use_container_width=True)

        # Per-patient results table
        with st.expander("Per-Patient Results"):
            if summary.results:
                df = pd.DataFrame([
                    {
                        "Patient": r.patient_id,
                        "Mutations": r.n_mutations,
                        "Drivers": r.n_drivers,
                        "TMB": round(r.predicted_tmb, 1),
                        "Neoantigens": r.n_neoantigens,
                        "Best Treatment": r.predicted_best_treatment[:25] if r.predicted_best_treatment else "",
                        "Response": r.predicted_best_response,
                        "Subtype": r.subtype[:15] if r.subtype else "",
                        "OS (months)": r.os_months,
                        "Time (s)": round(r.processing_seconds, 2),
                        "Status": r.status,
                    }
                    for r in summary.results
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

        # Export
        st.download_button(
            "\U0001f4be Download Validation Results (JSON)",
            data=json.dumps(summary.to_dict(), indent=2),
            file_name="cognisom_tcga_validation.json",
            mime="application/json",
        )

        # Required citations
        st.divider()
        st.markdown("### Data Citations")
        st.markdown(
            "This validation uses data from cBioPortal. Per their terms of use, "
            "the following papers must be cited in any publication or presentation:"
        )
        st.markdown("""
1. Cerami et al. **The cBio Cancer Genomics Portal: An Open Platform for Exploring
   Multidimensional Cancer Genomics Data.** *Cancer Discovery.* May 2012; 2:401.
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/22588877/)

2. Gao et al. **Integrative analysis of complex cancer genomics and clinical profiles
   using the cBioPortal.** *Sci. Signal.* 6, pl1 (2013).
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/23550210/)

3. de Bruijn et al. **Analysis and Visualization of Longitudinal Genomic and Clinical
   Data from the AACR Project GENIE Biopharma Collaborative in cBioPortal.**
   *Cancer Res* (2023). [PubMed](https://pubmed.ncbi.nlm.nih.gov/37668528/)
        """)
