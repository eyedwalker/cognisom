"""
Page 31: Neoantigen Vaccine Designer
=====================================

Predict tumor neoantigens from the patient's genomic profile and HLA type,
then design a personalized mRNA vaccine targeting the strongest MHC-I binders.

Requires a patient profile from Page 26 (Genomic Twin).

Pipeline: VCF -> HLA Typing -> Neoantigen Prediction -> Vaccine Design
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

safe_set_page_config(
    page_title="Neoantigen Vaccine",
    page_icon="\U0001f489",
    layout="wide",
)

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("31_neoantigen_vaccine")

from cognisom.genomics.hla_typer import HLATyper
from cognisom.genomics.neoantigen_predictor import NeoantigenPredictor, Neoantigen

logger = logging.getLogger(__name__)

st.title("Neoantigen Vaccine Designer")
st.markdown(
    "Predict tumor neoantigens from patient DNA and HLA type, "
    "then design a personalized mRNA cancer vaccine."
)

# ── Check for patient profile ──────────────────────────────────
profile = st.session_state.get("patient_profile")

if profile is None:
    st.warning(
        "No patient profile loaded. Go to **Genomic Twin** (page 26) "
        "to upload a VCF file or load the synthetic demo."
    )
    st.stop()

# ── HLA Typing ─────────────────────────────────────────────────
st.header("1. HLA Typing")

hla_alleles = profile.hla_alleles or []
if not hla_alleles:
    st.error("No HLA alleles found in patient profile.")
    st.stop()

hla_display = HLATyper.format_alleles(hla_alleles)
st.markdown(f"**Patient HLA Type:** {hla_display}")

col_hla1, col_hla2, col_hla3 = st.columns(3)
by_locus = {}
for allele in hla_alleles:
    locus = allele.split("*")[0]  # "HLA-A"
    by_locus.setdefault(locus, []).append(allele)

for col, locus in zip([col_hla1, col_hla2, col_hla3], ["HLA-A", "HLA-B", "HLA-C"]):
    with col:
        alleles_for_locus = by_locus.get(locus, [])
        st.metric(locus, " / ".join(a.split("*")[1] for a in alleles_for_locus) or "N/A")

st.divider()

# ── Neoantigen Predictions ─────────────────────────────────────
st.header("2. Predicted Neoantigens")

neoantigens = profile.predicted_neoantigens
if not neoantigens:
    st.info("No neoantigens predicted. This may indicate insufficient missense mutations.")
    st.stop()

# Summary metrics
summary = NeoantigenPredictor.summary(neoantigens)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Predicted", summary["total"])
m2.metric("Strong Binders", summary["strong_binders"], help="IC50 < 50 nM")
m3.metric("Weak Binders", summary["weak_binders"], help="IC50 < 500 nM")
m4.metric("Vaccine Candidates", summary["vaccine_candidates"])
m5.metric("Genes Targeted", len(summary["genes_with_neoantigens"]))

st.divider()

# ── Binding Affinity Distribution ──────────────────────────────
st.subheader("Binding Affinity Distribution")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    # Histogram of binding affinities
    affinities = [n.binding_affinity_nm for n in neoantigens]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=affinities,
        nbinsx=30,
        marker_color="rgba(0, 212, 170, 0.7)",
        name="All peptides",
    ))
    fig_hist.add_vline(x=50, line_dash="dash", line_color="red",
                       annotation_text="Strong binder (50 nM)")
    fig_hist.add_vline(x=500, line_dash="dash", line_color="orange",
                       annotation_text="Weak binder (500 nM)")
    fig_hist.update_layout(
        title="Predicted Binding Affinity (IC50)",
        xaxis_title="IC50 (nM)",
        yaxis_title="Count",
        xaxis_type="log",
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_chart2:
    # Scatter: affinity vs agretopicity
    binders = [n for n in neoantigens if n.is_weak_binder]
    if binders:
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=[n.binding_affinity_nm for n in binders],
            y=[n.agretopicity for n in binders],
            mode="markers",
            marker=dict(
                size=10,
                color=[n.foreignness for n in binders],
                colorscale="Viridis",
                colorbar=dict(title="Foreignness"),
                showscale=True,
            ),
            text=[f"{n.source_gene} {n.mutation}<br>{n.peptide}" for n in binders],
            hovertemplate="%{text}<br>IC50: %{x:.0f} nM<br>Agretopicity: %{y:.2f}<extra></extra>",
        ))
        fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="gray",
                              annotation_text="Agretopicity = 1.0")
        fig_scatter.update_layout(
            title="Binder Quality (Affinity vs Agretopicity)",
            xaxis_title="IC50 (nM, lower = better)",
            yaxis_title="Agretopicity (mutant/WT ratio)",
            xaxis_type="log",
            height=350,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Neoantigen Table ───────────────────────────────────────────
st.subheader("Neoantigen Rankings")

tab_vaccine, tab_all = st.tabs(["Vaccine Candidates", "All Predictions"])

with tab_vaccine:
    vaccine_neos = [n for n in neoantigens if n.include_in_vaccine]
    if vaccine_neos:
        df_vaccine = pd.DataFrame([
            {
                "Priority": n.vaccine_priority,
                "Gene": n.source_gene,
                "Mutation": n.mutation,
                "Peptide": n.peptide,
                "Length": n.peptide_length,
                "Best HLA": n.best_hla_allele,
                "IC50 (nM)": round(n.binding_affinity_nm, 1),
                "Binding": n.binding_level,
                "Agretopicity": round(n.agretopicity, 2),
                "Foreignness": round(n.foreignness, 2),
            }
            for n in vaccine_neos
        ])
        st.dataframe(df_vaccine, use_container_width=True, hide_index=True)
        st.caption(
            f"{len(vaccine_neos)} neoantigens selected for vaccine inclusion. "
            f"Criteria: < 500 nM binding, agretopicity >= 1.0, top 20 by affinity."
        )
    else:
        st.info("No neoantigens met vaccine inclusion criteria.")

with tab_all:
    df_all = pd.DataFrame([
        {
            "Rank": n.vaccine_priority,
            "Gene": n.source_gene,
            "Mutation": n.mutation,
            "Peptide": n.peptide,
            "WT Peptide": n.wild_type_peptide,
            "Length": n.peptide_length,
            "Best HLA": n.best_hla_allele,
            "IC50 (nM)": round(n.binding_affinity_nm, 1),
            "Binding": n.binding_level,
            "Agretopicity": round(n.agretopicity, 2),
            "Foreignness": round(n.foreignness, 2),
            "Vaccine": "Yes" if n.include_in_vaccine else "",
        }
        for n in neoantigens[:50]
    ])
    st.dataframe(df_all, use_container_width=True, hide_index=True)

st.divider()

# ── Per-Gene Neoantigen Breakdown ──────────────────────────────
st.subheader("Neoantigens by Gene")

gene_counts = {}
for n in neoantigens:
    if n.is_weak_binder:
        gene_counts[n.source_gene] = gene_counts.get(n.source_gene, 0) + 1

if gene_counts:
    fig_genes = go.Figure()
    genes_sorted = sorted(gene_counts.items(), key=lambda x: -x[1])
    fig_genes.add_trace(go.Bar(
        x=[g[0] for g in genes_sorted],
        y=[g[1] for g in genes_sorted],
        marker_color="rgba(99, 102, 241, 0.7)",
    ))
    fig_genes.update_layout(
        title="MHC-I Binders by Source Gene",
        xaxis_title="Gene",
        yaxis_title="Number of Binding Peptides",
        height=300,
    )
    st.plotly_chart(fig_genes, use_container_width=True)

st.divider()

# ── HLA Allele Coverage ───────────────────────────────────────
st.subheader("HLA Allele Coverage")

allele_counts = {}
for n in neoantigens:
    if n.is_weak_binder:
        allele_counts[n.best_hla_allele] = allele_counts.get(n.best_hla_allele, 0) + 1

if allele_counts:
    fig_alleles = go.Figure()
    alleles_sorted = sorted(allele_counts.items(), key=lambda x: -x[1])
    fig_alleles.add_trace(go.Bar(
        x=[a[0] for a in alleles_sorted],
        y=[a[1] for a in alleles_sorted],
        marker_color="rgba(236, 72, 153, 0.7)",
    ))
    fig_alleles.update_layout(
        title="Binding Peptides per HLA Allele",
        xaxis_title="HLA Allele",
        yaxis_title="Number of Binders",
        height=300,
    )
    st.plotly_chart(fig_alleles, use_container_width=True)

    st.caption(
        "Good vaccine coverage requires binders across multiple HLA alleles "
        "to ensure robust antigen presentation."
    )

st.divider()

# ── Vaccine Design Summary ─────────────────────────────────────
st.header("3. Vaccine Design Summary")

vaccine_candidates = [n for n in neoantigens if n.include_in_vaccine]
n_candidates = len(vaccine_candidates)
target_genes = sorted(set(n.source_gene for n in vaccine_candidates))

if n_candidates >= 3:
    st.success(
        f"**Vaccine feasible**: {n_candidates} neoantigen targets identified "
        f"from {len(target_genes)} genes ({', '.join(target_genes)})"
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("##### mRNA Vaccine Design Parameters")
        st.markdown(f"- **Neoantigen payload**: {n_candidates} epitopes")
        st.markdown(f"- **Target genes**: {', '.join(target_genes)}")
        st.markdown(f"- **HLA coverage**: {len(allele_counts)} alleles")
        st.markdown("- **Delivery**: LNP-encapsulated mRNA")
        st.markdown("- **Optimization**: LinearDesign (MFE + CAI co-optimization)")
        st.markdown("- **Dosing**: Prime + boost regimen (q3w x 9 cycles)")

    with col_v2:
        st.markdown("##### Clinical Considerations")
        st.markdown(
            "- **Combination**: Consider with pembrolizumab (anti-PD-1) for "
            "synergistic effect (mRNA-4157/V940 paradigm)"
        )
        if profile.has_dna_repair_defect:
            st.markdown(
                "- **HRD status**: DNA repair defect may increase neoantigen "
                "generation over time (favorable for vaccine)"
            )
        if profile.immunotherapy_candidate:
            st.markdown(
                "- **Immunotherapy eligible**: TMB-high/MSI-H status supports "
                "checkpoint inhibitor combination"
            )
        st.markdown(
            "- **Manufacturing**: Cell-free DNA template (Elegen ENFINIA) + "
            "automated mRNA synthesis (BioXp / Nutcracker ACORN)"
        )

    # Pipeline overview
    st.markdown("---")
    st.markdown("##### End-to-End Pipeline")
    st.code("""
Patient VCF
    |
    v
[Variant Annotation] --> Cancer Driver Mutations (14 genes)
    |
    v
[HLA Typing] --> Patient MHC-I Alleles (6 alleles)
    |
    v
[Neoantigen Prediction] --> Peptide-MHC Binding Affinity
    |
    v
[Vaccine Candidate Selection] --> Top binders (< 500 nM, agretopicity >= 1.0)
    |
    v
[mRNA Sequence Design] --> LinearDesign optimization (stability + translation)
    |
    v
[LNP Formulation] --> Microfluidic encapsulation
    |
    v
[Administration] --> Prime + boost with anti-PD-1 combination
    """, language=None)

else:
    st.warning(
        f"Only {n_candidates} vaccine candidates found (minimum 3 required). "
        "Consider expanding mutation analysis to include passenger mutations "
        "and non-coding regions."
    )

# ── Export ──────────────────────────────────────────────────────
st.divider()
with st.expander("Export Neoantigen Data"):
    import json
    export_data = {
        "patient_id": profile.patient_id,
        "hla_alleles": hla_alleles,
        "neoantigen_summary": summary,
        "vaccine_candidates": [n.to_dict() for n in vaccine_candidates],
        "all_predictions": [n.to_dict() for n in neoantigens[:50]],
    }
    st.download_button(
        "Download JSON",
        data=json.dumps(export_data, indent=2),
        file_name=f"neoantigens_{profile.patient_id}.json",
        mime="application/json",
    )
