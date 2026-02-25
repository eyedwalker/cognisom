"""
Page 28: Personalized Molecular Digital Twin
=============================================

THE CENTERPIECE of the new Cognisom platform.

Combines patient genomic profile (Page 26) + immune landscape (Page 27)
into a personalized digital twin. Simulates immunotherapy, targeted therapy,
and combination regimens to predict treatment response.

Phase 3 of the Molecular Digital Twin pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import logging

st.set_page_config(page_title="Digital Twin", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("28_digital_twin")

from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.genomics.treatment_simulator import (
    TreatmentSimulator, TreatmentResult, TREATMENT_PROFILES,
)
from cognisom.genomics.patient_profile import PatientProfile, PatientProfileBuilder
from cognisom.genomics.cell_state_classifier import ImmuneClassification
from cognisom.genomics.synthetic_vcf import get_synthetic_vcf

logger = logging.getLogger(__name__)

st.title("🧬 Personalized Molecular Digital Twin")
st.caption(
    "Combine your genomic profile with immune landscape analysis to predict "
    "personalized treatment response. Powered by BioNeMo NIMs + Cell2Sentence."
)

# ─────────────────────────────────────────────────────────────────────────
# BUILD OR LOAD DIGITAL TWIN
# ─────────────────────────────────────────────────────────────────────────

twin = None
profile = None
classification = None

# Check session state for data from Pages 26 and 27
has_profile = "patient_profile" in st.session_state and isinstance(
    st.session_state["patient_profile"], PatientProfile
)
has_classification = "cell_classification" in st.session_state and isinstance(
    st.session_state["cell_classification"], ImmuneClassification
)

with st.sidebar:
    st.header("Data Sources")

    if has_profile:
        profile = st.session_state["patient_profile"]
        st.success(f"Genomic profile: {profile.patient_id}")
        st.caption(f"{len(profile.cancer_driver_mutations)} drivers, TMB={profile.tumor_mutational_burden:.1f}")
    else:
        st.warning("No genomic profile loaded")
        if st.button("Load Synthetic Demo"):
            builder = PatientProfileBuilder()
            profile = builder.from_vcf_text(get_synthetic_vcf(), patient_id="COGNISOM-DEMO-001")
            st.session_state["patient_profile"] = profile
            st.rerun()

    if has_classification:
        classification = st.session_state["cell_classification"]
        st.success(f"Immune analysis: {classification.immune_score}")
        st.caption(f"Exhaustion: {classification.mean_exhaustion:.2f}")
    else:
        st.info("No immune analysis — using genomic-only estimates")

    st.divider()
    st.header("Treatment Options")

    available_treatments = list(TREATMENT_PROFILES.keys())
    selected_treatments = st.multiselect(
        "Treatments to simulate:",
        available_treatments,
        default=None,  # Will auto-select recommended
        format_func=lambda k: TREATMENT_PROFILES[k]["name"],
    )

    duration = st.slider("Simulation duration (days)", 30, 365, 180, step=30)

# ─────────────────────────────────────────────────────────────────────────
# BUILD TWIN CONFIG
# ─────────────────────────────────────────────────────────────────────────

if profile is None:
    st.info(
        "No patient data loaded. Go to **Page 26: Genomic Twin** to upload "
        "a VCF file, or click **Load Synthetic Demo** in the sidebar."
    )
    st.stop()

# Build twin config
if classification:
    twin = DigitalTwinConfig.from_profile_and_classification(profile, classification)
else:
    twin = DigitalTwinConfig.from_profile_only(profile)

# Store in session state
st.session_state["digital_twin_config"] = twin

# ─────────────────────────────────────────────────────────────────────────
# TWIN OVERVIEW
# ─────────────────────────────────────────────────────────────────────────

st.header("Digital Twin Overview")

# Key metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

score_colors = {"hot": "🔴", "cold": "🔵", "excluded": "🟡", "suppressed": "🟣", "unknown": "⚪"}

with col1:
    st.metric("Immune Score", f"{score_colors.get(twin.immune_score, '⚪')} {twin.immune_score.upper()}")
with col2:
    tmb_emoji = "🔴" if twin.tumor_mutational_burden >= 10 else "🟢"
    st.metric("TMB", f"{twin.tumor_mutational_burden:.1f}/Mb {tmb_emoji}")
with col3:
    st.metric("MSI", twin.microsatellite_instability)
with col4:
    st.metric("Exhaustion", f"{twin.mean_exhaustion:.0%}")
with col5:
    st.metric("Neoantigens", f"~{twin.neoantigen_count}")
with col6:
    st.metric("PD-L1", f"{twin.pd_l1_expression:.0%}")

# Genomic vulnerabilities
st.subheader("Genomic Vulnerabilities")

vuln_cols = st.columns(5)
vulnerabilities = [
    ("DNA Repair Defect", twin.has_dna_repair_defect, "PARP inhibitors"),
    ("AR Mutation", twin.has_ar_mutation, "AR antagonists (resistance risk)"),
    ("PTEN Loss", twin.has_pten_loss, "PI3K/AKT inhibitors"),
    ("MHC-I Loss", twin.mhc1_downregulation > 0.3, "NK cell therapy"),
    ("TMB-High", twin.tumor_mutational_burden >= 10, "Checkpoint inhibitors"),
]

for i, (name, present, therapy) in enumerate(vulnerabilities):
    with vuln_cols[i]:
        if present:
            st.markdown(f"**{name}** ✅")
            st.caption(f"→ {therapy}")
        else:
            st.markdown(f"**{name}** ❌")

# ─────────────────────────────────────────────────────────────────────────
# TREATMENT SIMULATION
# ─────────────────────────────────────────────────────────────────────────

st.header("Treatment Simulation")

simulator = TreatmentSimulator()

# Auto-recommend if none selected
if not selected_treatments:
    selected_treatments = simulator.get_recommended_treatments(twin)
    st.info(
        f"Auto-selected recommended treatments: "
        f"{', '.join(TREATMENT_PROFILES[t]['name'] for t in selected_treatments)}"
    )

# Run simulations
results = []
with st.spinner("Running treatment simulations..."):
    for treatment_key in selected_treatments:
        result = simulator.simulate(treatment_key, twin, duration_days=duration)
        results.append(result)

# ─────────────────────────────────────────────────────────────────────────
# TUMOR RESPONSE CURVES
# ─────────────────────────────────────────────────────────────────────────

if results:
    st.subheader("Predicted Tumor Response")

    fig = go.Figure()

    response_colors = {
        "CR": "#22cc22", "PR": "#88cc22",
        "SD": "#cccc22", "PD": "#cc2222",
    }

    for result in results:
        color = response_colors.get(result.response_category, "#888888")
        fig.add_trace(go.Scatter(
            x=list(range(len(result.tumor_response_curve))),
            y=result.tumor_response_curve,
            mode="lines",
            name=f"{result.treatment_name} ({result.response_category})",
            line=dict(width=2.5),
        ))

    # Baseline
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="Baseline")

    # RECIST thresholds
    fig.add_hline(y=0.7, line_dash="dot", line_color="green",
                  annotation_text="PR threshold (-30%)")
    fig.add_hline(y=1.2, line_dash="dot", line_color="red",
                  annotation_text="PD threshold (+20%)")

    fig.update_layout(
        title="Predicted Tumor Volume Over Time",
        xaxis_title="Days",
        yaxis_title="Relative Tumor Volume (1.0 = baseline)",
        height=500,
        yaxis=dict(range=[0, max(2.0, max(r.tumor_response_curve[-1] for r in results) * 1.1)]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────

    st.subheader("Treatment Comparison")

    comparison_data = []
    for r in sorted(results, key=lambda x: x.best_response):
        comparison_data.append({
            "Treatment": r.treatment_name,
            "Class": r.treatment_class,
            "Response": r.response_category,
            "Best Response": f"{(1 - r.best_response) * 100:.0f}% reduction",
            "Time to Best (days)": r.time_to_best_response_days,
            "PFS (days)": r.progression_free_days,
            "T-cell Reactivation": f"{r.t_cell_reactivation_fraction:.0%}",
            "irAE Risk": f"{r.immune_related_adverse_events:.0%}",
            "Confidence": f"{r.confidence:.0%}",
        })

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────
    # DETAILED RESULTS PER TREATMENT
    # ─────────────────────────────────────────────────────────────────

    st.subheader("Detailed Results")

    for result in results:
        resp_emoji = {"CR": "🟢", "PR": "🟡", "SD": "🟠", "PD": "🔴"}.get(result.response_category, "⚪")

        with st.expander(
            f"{resp_emoji} {result.treatment_name} — {result.response_category}",
            expanded=False,
        ):
            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown(f"**Mechanism:** {result.mechanism}")
                st.markdown(f"**Rationale:** {result.rationale}")

            with col_d2:
                st.metric("Best Response", f"{(1 - result.best_response) * 100:.0f}% reduction")
                st.metric("PFS", f"{result.progression_free_days} days")
                st.metric("Post-treatment Exhaustion", f"{result.final_exhaustion:.2f}")

# ─────────────────────────────────────────────────────────────────────────
# THERAPY RECOMMENDATIONS (from genomic profile)
# ─────────────────────────────────────────────────────────────────────────

st.header("Genomic-Based Therapy Recommendations")

recommendations = profile.get_therapy_recommendations()
if recommendations:
    for rec in recommendations:
        confidence_emoji = {"high": "🟢", "moderate": "🟡", "low": "🔴"}.get(
            rec.get("confidence", ""), "⚪"
        )
        with st.expander(
            f"{confidence_emoji} {rec['therapy_class']}",
            expanded=False,
        ):
            st.markdown(f"**Drugs:** {', '.join(rec['drugs'])}")
            st.markdown(f"**Rationale:** {rec['rationale']}")
            st.markdown(f"**Evidence:** {rec['evidence_level']}")

# ─────────────────────────────────────────────────────────────────────────
# DRIVER MUTATIONS
# ─────────────────────────────────────────────────────────────────────────

with st.expander("Cancer Driver Mutations", expanded=False):
    drivers = profile.get_driver_details()
    if drivers:
        driver_df = pd.DataFrame(drivers)
        display_cols = ["gene", "full_name", "mutation", "consequence", "role"]
        available_cols = [c for c in display_cols if c in driver_df.columns]
        st.dataframe(driver_df[available_cols], use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────

st.header("Export Digital Twin")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    twin_json = json.dumps(twin.to_dict(), indent=2)
    st.download_button(
        "Download Twin Config (JSON)",
        data=twin_json,
        file_name=f"{profile.patient_id}_twin.json",
        mime="application/json",
    )

with col_exp2:
    if results:
        results_data = [r.summary() for r in results]
        results_json = json.dumps(results_data, indent=2)
        st.download_button(
            "Download Treatment Results (JSON)",
            data=results_json,
            file_name=f"{profile.patient_id}_treatments.json",
            mime="application/json",
        )

with col_exp3:
    sim_params = twin.to_simulation_params()
    sim_json = json.dumps(sim_params, indent=2)
    st.download_button(
        "Download Simulation Params (JSON)",
        data=sim_json,
        file_name=f"{profile.patient_id}_sim_params.json",
        mime="application/json",
    )

# ─────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS (for developers)
# ─────────────────────────────────────────────────────────────────────────

with st.expander("Simulation Parameters (Developer)", expanded=False):
    st.json(twin.to_simulation_params())
