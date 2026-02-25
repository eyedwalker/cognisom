"""
Page 26: Genomic Digital Twin — Patient DNA Profile
====================================================

Upload a VCF file (or use synthetic demo data) to build a patient
genomic profile. Identifies cancer driver mutations, affected proteins,
TMB/MSI biomarkers, and generates therapy recommendations.

Phase 1 of the Molecular Digital Twin pipeline.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import logging

st.set_page_config(page_title="Genomic Twin", page_icon="🧬", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("26_genomic_twin")

from cognisom.genomics.vcf_parser import VCFParser
from cognisom.genomics.variant_annotator import VariantAnnotator, PROSTATE_CANCER_DRIVERS
from cognisom.genomics.patient_profile import PatientProfileBuilder, PatientProfile
from cognisom.genomics.synthetic_vcf import get_synthetic_vcf, get_synthetic_profile_description

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────

st.title("🧬 Genomic Digital Twin")
st.caption(
    "Upload patient VCF data to build a personalized genomic profile — "
    "identify cancer drivers, predict therapy response, and feed into "
    "the molecular digital twin simulation."
)

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR: DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data Source")

    data_source = st.radio(
        "Choose data source:",
        ["Synthetic Demo (Prostate Cancer)", "Upload VCF File", "Paste VCF Text"],
        index=0,
    )

    patient_id = st.text_input("Patient ID", value="COGNISOM-DEMO-001")
    cancer_type = st.selectbox("Cancer Type", ["prostate"], index=0)

    st.divider()
    st.header("Analysis Options")
    min_impact = st.selectbox(
        "Minimum variant impact",
        ["HIGH", "MODERATE", "LOW", "MODIFIER"],
        index=1,
    )
    show_noncoding = st.checkbox("Show non-coding variants", value=False)

# ─────────────────────────────────────────────────────────────────────────
# LOAD VCF DATA
# ─────────────────────────────────────────────────────────────────────────

vcf_text = None

if data_source == "Synthetic Demo (Prostate Cancer)":
    vcf_text = get_synthetic_vcf()
    with st.expander("About Synthetic Demo Data", expanded=False):
        st.markdown(get_synthetic_profile_description())

elif data_source == "Upload VCF File":
    uploaded = st.file_uploader(
        "Upload VCF file (.vcf or .vcf.gz)",
        type=["vcf", "gz"],
    )
    if uploaded:
        vcf_text = uploaded.read().decode("utf-8", errors="replace")
        st.success(f"Loaded {uploaded.name} ({len(vcf_text):,} bytes)")
    else:
        st.info("Upload a VCF file to begin analysis.")

elif data_source == "Paste VCF Text":
    vcf_text = st.text_area(
        "Paste VCF content:",
        height=200,
        placeholder="##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\t...",
    )
    if not vcf_text.strip():
        vcf_text = None

# ─────────────────────────────────────────────────────────────────────────
# BUILD PROFILE
# ─────────────────────────────────────────────────────────────────────────

if vcf_text:
    with st.spinner("Building genomic profile..."):
        builder = PatientProfileBuilder(cancer_type=cancer_type)
        profile = builder.from_vcf_text(vcf_text, patient_id=patient_id)

    # Store in session state for other pages
    st.session_state["patient_profile"] = profile
    st.session_state["patient_profile_dict"] = profile.to_dict()

    # ─────────────────────────────────────────────────────────────────
    # OVERVIEW METRICS
    # ─────────────────────────────────────────────────────────────────

    st.header("Patient Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Variants", len(profile.variants))
    with col2:
        st.metric("Coding Variants", len(profile.coding_variants))
    with col3:
        st.metric("Cancer Drivers", len(profile.cancer_driver_mutations))
    with col4:
        tmb_color = "🔴" if profile.is_tmb_high else "🟢"
        st.metric("TMB", f"{profile.tumor_mutational_burden:.1f}/Mb {tmb_color}")
    with col5:
        msi_color = "🔴" if profile.msi_status == "MSI-H" else "🟢"
        st.metric("MSI Status", f"{profile.msi_status} {msi_color}")

    # ─────────────────────────────────────────────────────────────────
    # CANCER DRIVER MUTATIONS
    # ─────────────────────────────────────────────────────────────────

    st.header("Cancer Driver Mutations")

    drivers = profile.get_driver_details()
    if drivers:
        driver_df = pd.DataFrame(drivers)
        display_cols = [
            "gene", "full_name", "mutation", "consequence", "impact",
            "role", "clinical_significance",
        ]
        available_cols = [c for c in display_cols if c in driver_df.columns]
        st.dataframe(
            driver_df[available_cols],
            use_container_width=True,
            hide_index=True,
        )

        # Driver gene impact visualization
        fig_drivers = go.Figure()
        for i, d in enumerate(drivers):
            color = "#ff4444" if d["impact"] == "HIGH" else (
                "#ffaa44" if d["impact"] == "MODERATE" else "#44aa44"
            )
            fig_drivers.add_trace(go.Bar(
                x=[d["gene"]],
                y=[1],
                name=d.get("mutation", ""),
                marker_color=color,
                text=d.get("mutation", ""),
                textposition="inside",
                hovertext=d.get("clinical_significance", ""),
            ))
        fig_drivers.update_layout(
            title="Cancer Driver Mutations by Gene",
            showlegend=False,
            yaxis_visible=False,
            height=300,
        )
        st.plotly_chart(fig_drivers, use_container_width=True)
    else:
        st.info("No cancer driver mutations identified.")

    # ─────────────────────────────────────────────────────────────────
    # THERAPY RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────

    st.header("Therapy Recommendations")

    recommendations = profile.get_therapy_recommendations()
    if recommendations:
        for rec in recommendations:
            confidence_emoji = {
                "high": "🟢", "moderate": "🟡", "low": "🔴"
            }.get(rec.get("confidence", ""), "⚪")

            with st.expander(
                f"{confidence_emoji} {rec['therapy_class']} — "
                f"{', '.join(rec['drugs'][:3])}",
                expanded=True,
            ):
                st.markdown(f"**Rationale:** {rec['rationale']}")
                st.markdown(f"**Evidence:** {rec['evidence_level']}")
                st.markdown(f"**Drugs:** {', '.join(rec['drugs'])}")
    else:
        st.info("No specific therapy recommendations based on current genomic profile.")

    # ─────────────────────────────────────────────────────────────────
    # VARIANT SUMMARY CHARTS
    # ─────────────────────────────────────────────────────────────────

    st.header("Variant Summary")

    col_a, col_b = st.columns(2)

    with col_a:
        # Chromosome distribution
        by_chrom = profile.variant_summary.get("by_chromosome", {})
        if by_chrom:
            chrom_df = pd.DataFrame([
                {"Chromosome": k, "Count": v}
                for k, v in sorted(by_chrom.items(),
                                   key=lambda x: _chrom_sort_key(x[0]))
            ])
            fig_chrom = px.bar(
                chrom_df, x="Chromosome", y="Count",
                title="Variants by Chromosome",
                color="Count",
                color_continuous_scale="Reds",
            )
            fig_chrom.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_chrom, use_container_width=True)

    with col_b:
        # Consequence distribution
        by_consequence = profile.variant_summary.get("by_consequence", {})
        if by_consequence:
            cons_df = pd.DataFrame([
                {"Consequence": k, "Count": v}
                for k, v in sorted(by_consequence.items(),
                                   key=lambda x: -x[1])
            ])
            fig_cons = px.pie(
                cons_df, names="Consequence", values="Count",
                title="Variant Consequences",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_cons.update_layout(height=350)
            st.plotly_chart(fig_cons, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # AFFECTED PROTEINS
    # ─────────────────────────────────────────────────────────────────

    st.header("Affected Proteins")

    if profile.affected_proteins:
        protein_data = []
        for gene, prot in profile.affected_proteins.items():
            protein_data.append({
                "Gene": gene,
                "Protein": prot.protein_name,
                "UniProt ID": prot.uniprot_id,
                "Length (AA)": prot.length,
                "Function": prot.function[:80] + "..." if len(prot.function) > 80 else prot.function,
            })
        prot_df = pd.DataFrame(protein_data)
        st.dataframe(prot_df, use_container_width=True, hide_index=True)

        # Structure prediction section
        st.subheader("Predict Mutant Protein Structures")
        st.markdown(
            "Select a protein to predict its 3D structure using BioNeMo NIMs "
            "(AlphaFold2 / OpenFold3). Requires `NVIDIA_API_KEY` environment variable."
        )

        selected_gene = st.selectbox(
            "Select protein for structure prediction:",
            list(profile.affected_proteins.keys()),
        )

        if selected_gene:
            protein = profile.affected_proteins[selected_gene]
            # Find mutations for this gene
            gene_mutations = [
                v for v in profile.cancer_driver_mutations
                if v.gene == selected_gene and v.protein_change
            ]

            st.markdown(f"**{protein.protein_name}** ({protein.length} AA)")
            if gene_mutations:
                st.markdown(
                    "**Mutations:** " +
                    ", ".join(v.protein_change for v in gene_mutations if v.protein_change)
                )

            col_wt, col_mut = st.columns(2)
            with col_wt:
                if st.button(f"Predict Wild-Type {selected_gene} Structure",
                           key=f"wt_{selected_gene}"):
                    _predict_structure(protein.sequence, f"{selected_gene} (wild-type)")

            with col_mut:
                if gene_mutations and st.button(
                    f"Predict Mutant {selected_gene} Structure",
                    key=f"mut_{selected_gene}",
                ):
                    # Apply first mutation
                    from cognisom.genomics.gene_protein_mapper import GeneProteinMapper
                    mapper = GeneProteinMapper()
                    mut = gene_mutations[0]
                    aa_change = mut.protein_change.replace("p.", "") if mut.protein_change else None
                    if aa_change:
                        mutant = mapper.apply_mutation(protein, aa_change)
                        if mutant:
                            _predict_structure(
                                mutant.sequence,
                                f"{selected_gene} ({aa_change})",
                            )
    else:
        st.info("No protein data available for affected genes.")

    # ─────────────────────────────────────────────────────────────────
    # ALL VARIANTS TABLE
    # ─────────────────────────────────────────────────────────────────

    st.header("All Variants")

    variants_to_show = profile.variants if show_noncoding else profile.coding_variants
    if variants_to_show:
        var_data = []
        for v in variants_to_show:
            var_data.append({
                "Location": v.location_str,
                "Gene": v.gene or "-",
                "Ref": v.ref,
                "Alt": v.alt,
                "Type": v.variant_type,
                "Consequence": v.consequence or "-",
                "Protein Change": v.protein_change or "-",
                "Impact": v.impact,
                "Genotype": v.genotype,
                "QUAL": v.qual,
                "Driver": "Yes" if v.is_cancer_driver else "",
            })
        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No variants to display.")

    # ─────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────

    st.header("Export")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        profile_json = json.dumps(profile.to_dict(), indent=2)
        st.download_button(
            "Download Patient Profile (JSON)",
            data=profile_json,
            file_name=f"{patient_id}_profile.json",
            mime="application/json",
        )
    with col_exp2:
        st.markdown(
            "Use this profile in **Page 28: Digital Twin** to run "
            "personalized immune simulations and predict treatment response."
        )

else:
    # No data loaded yet
    st.info(
        "Select a data source in the sidebar to begin. "
        "Choose **Synthetic Demo** for an instant preview."
    )

    # Show cancer driver gene database
    st.header("Prostate Cancer Driver Gene Database")
    st.markdown(
        "Built-in database of known prostate cancer driver genes "
        "used for variant annotation."
    )

    gene_data = []
    for gene, info in PROSTATE_CANCER_DRIVERS.items():
        gene_data.append({
            "Gene": gene,
            "Full Name": info["full_name"],
            "Role": info["role"],
            "Chromosome": info["chromosome"],
            "Therapies": ", ".join(info.get("therapies", [])[:3]) or "-",
        })
    gene_df = pd.DataFrame(gene_data)
    st.dataframe(gene_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────

def _chrom_sort_key(chrom: str) -> tuple:
    """Sort chromosomes numerically (chr1, chr2, ... chr22, chrX, chrY)."""
    chrom = chrom.replace("chr", "")
    if chrom == "X":
        return (23,)
    elif chrom == "Y":
        return (24,)
    elif chrom == "M" or chrom == "MT":
        return (25,)
    try:
        return (int(chrom),)
    except ValueError:
        return (99,)


def _predict_structure(sequence: str, label: str):
    """Run structure prediction via BioNeMo NIM."""
    import os
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        st.error(
            "NVIDIA_API_KEY not set. Add it to your .env file or "
            "set it as an environment variable to use BioNeMo NIMs."
        )
        return

    with st.spinner(f"Predicting structure for {label}..."):
        try:
            from cognisom.bridge.pipeline import DiscoveryPipeline
            pipeline = DiscoveryPipeline(api_key=api_key)

            # Truncate to first 1024 AA for ESM2 compatibility
            seq = sequence[:1024] if len(sequence) > 1024 else sequence

            result = pipeline.run_structure_prediction_pipeline(
                sequence=seq, method="openfold3"
            )

            st.success(f"Structure predicted for {label}")

            if "confidence" in result:
                st.metric("Confidence", f"{result['confidence']:.2f}")
            if "plddt" in result:
                st.metric("Mean pLDDT", f"{result['plddt']:.1f}")
            if "structure" in result:
                st.text_area(
                    "Structure (mmCIF/PDB)",
                    value=result["structure"][:2000],
                    height=200,
                )
                # Store for molecular viewer
                st.session_state[f"structure_{label}"] = result

        except Exception as e:
            st.error(f"Structure prediction failed: {e}")
