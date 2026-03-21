"""
Page 37: MAD Agent — Molecular AI Decision Support
====================================================

Multi-agent Molecular Tumor Board for immunotherapy treatment selection.
Three specialist agents (Genomics, Immune, Clinical) independently analyze
patient data, then a Board Moderator synthesizes a consensus decision.

FDA-compliant: Non-Device CDS with traceable evidence chains.
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import logging
import time

safe_set_page_config(page_title="MAD Agent", page_icon="🏛️", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("37_mad_agent")

from cognisom.genomics.twin_config import DigitalTwinConfig
from cognisom.genomics.treatment_simulator import (
    TreatmentSimulator, TREATMENT_PROFILES,
)
from cognisom.genomics.patient_profile import PatientProfile, PatientProfileBuilder
from cognisom.genomics.synthetic_vcf import get_synthetic_vcf
from cognisom.mad.board import BoardModerator, BoardDecision
from cognisom.mad.agents import GenomicsAgent, ImmuneAgent, ClinicalAgent
from cognisom.mad.compliance import ContextOfUse, NonDeviceCDSChecker, CredibilityFramework
from cognisom.mad.model_cards import get_all_model_cards
from cognisom.mad.audit import AuditStore, AuditRecord
from cognisom.mad.provenance import DataProvenance

logger = logging.getLogger(__name__)

st.title("🏛️ MAD Agent — Molecular Tumor Board")
st.caption(
    "Multi-agent AI decision support for immunotherapy selection. "
    "Three specialist agents deliberate independently, then a Board "
    "Moderator synthesizes a consensus recommendation with traceable evidence."
)

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────

tab_board, tab_evidence, tab_explain, tab_compliance, tab_audit, tab_study = st.tabs([
    "Board Decision", "Evidence Chain", "Explainability",
    "FDA Compliance", "Audit Trail", "Research Study",
])

# ─────────────────────────────────────────────────────────────────────────
# BUILD / LOAD PATIENT
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Patient Data")

    # Check for existing data from Page 26/28
    profile = st.session_state.get("patient_profile")
    twin = st.session_state.get("digital_twin")
    treatment_results = st.session_state.get("treatment_results")

    if profile and twin:
        st.success(f"Patient: {profile.patient_id}")
        st.caption(f"TMB: {profile.tumor_mutational_burden:.1f} | "
                   f"Drivers: {len(profile.cancer_driver_mutations)}")
    else:
        st.info("No patient loaded. Use synthetic demo or load from Page 26.")
        if st.button("Load Synthetic Demo Patient", type="primary"):
            with st.spinner("Building patient profile..."):
                builder = PatientProfileBuilder()
                vcf_text = get_synthetic_vcf()
                profile = builder.from_vcf_text(vcf_text, "COGNISOM-DEMO-001")
                twin = DigitalTwinConfig.from_profile_only(profile)

                simulator = TreatmentSimulator()
                recommended = simulator.get_recommended_treatments(twin)
                treatment_results = simulator.compare_treatments(recommended, twin)

                st.session_state["patient_profile"] = profile
                st.session_state["digital_twin"] = twin
                st.session_state["treatment_results"] = treatment_results
            st.rerun()

    st.divider()
    st.caption("FOR RESEARCH USE ONLY")

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: BOARD DECISION
# ─────────────────────────────────────────────────────────────────────────

with tab_board:
    if not profile or not twin:
        st.warning("Load a patient to run the MAD Board analysis.")
        st.stop()

    # Run MAD Board
    if st.button("Convene MAD Board", type="primary", key="run_mad"):
        with st.spinner("Agents deliberating..."):
            start_time = time.time()

            # Progress tracking
            progress = st.progress(0, text="Initializing agents...")

            # Run agents
            progress.progress(10, text="Genomics Agent analyzing...")
            genomics_agent = GenomicsAgent()
            genomics_opinion = genomics_agent.analyze(profile=profile, twin=twin)

            progress.progress(40, text="Immune Agent analyzing...")
            immune_agent = ImmuneAgent()
            immune_opinion = immune_agent.analyze(twin=twin, profile=profile)

            progress.progress(70, text="Clinical Agent analyzing...")
            clinical_agent = ClinicalAgent()
            clinical_opinion = clinical_agent.analyze(
                twin=twin,
                treatment_results=treatment_results or [],
            )

            progress.progress(90, text="Board Moderator synthesizing...")
            moderator = BoardModerator()
            decision = moderator.convene(
                patient_id=profile.patient_id,
                genomics_opinion=genomics_opinion,
                immune_opinion=immune_opinion,
                clinical_opinion=clinical_opinion,
            )

            elapsed = time.time() - start_time
            progress.progress(100, text=f"Complete in {elapsed:.2f}s")

            st.session_state["mad_decision"] = decision

            # Audit
            try:
                audit_store = AuditStore()
                record = AuditRecord.from_board_decision(
                    decision,
                    user_id=getattr(user, "username", "local"),
                    input_data=str(profile.patient_id),
                )
                audit_store.record(record)
            except Exception:
                pass  # Audit failure is non-fatal

    # Display decision
    decision = st.session_state.get("mad_decision")
    if decision:
        # Hero metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recommended", decision.recommended_treatment_name.split("(")[0].strip())
        with col2:
            consensus_colors = {"unanimous": "🟢", "majority": "🟡", "split": "🔴"}
            st.metric("Consensus", f"{consensus_colors.get(decision.consensus_level, '')} {decision.consensus_level.title()}")
        with col3:
            st.metric("Confidence", f"{decision.confidence:.0%}")
        with col4:
            st.metric("Evidence Items", len(decision.evidence_chain))

        # Unified rationale
        st.subheader("Board Rationale")
        st.info(decision.unified_rationale)

        # Agreement matrix heatmap
        st.subheader("Agent Agreement Matrix")
        if decision.agreement_matrix:
            matrix_data = []
            for treatment, agents in decision.agreement_matrix.items():
                for agent_name, agrees in agents.items():
                    matrix_data.append({
                        "Treatment": treatment.replace("_", " ").title(),
                        "Agent": agent_name.title(),
                        "In Top 3": 1 if agrees else 0,
                    })

            if matrix_data:
                df_matrix = pd.DataFrame(matrix_data)
                fig = px.imshow(
                    df_matrix.pivot(index="Treatment", columns="Agent", values="In Top 3"),
                    color_continuous_scale=["#f0f0f0", "#1f77b4"],
                    labels={"color": "In Top 3"},
                    title="Treatment × Agent Agreement",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Per-agent rankings
        st.subheader("Agent Rankings")
        for opinion in decision.agent_opinions:
            with st.expander(f"{opinion.agent_name.title()} Agent (confidence: {opinion.confidence:.0%})"):
                for ranking in opinion.treatment_rankings[:5]:
                    score_bar = "█" * int(ranking.score * 20)
                    st.markdown(
                        f"**#{ranking.rank} {ranking.treatment_name}** "
                        f"(score: {ranking.score:.2f}) {score_bar}"
                    )
                    if ranking.contributions:
                        for c in ranking.contributions[:3]:
                            icon = "↑" if c.direction == "positive" else "↓"
                            st.caption(f"  {icon} {c.description}")

                if opinion.dissenting_notes:
                    st.warning("Dissenting: " + "; ".join(opinion.dissenting_notes))

        # Alternatives
        if decision.alternative_treatments:
            st.subheader("Alternative Treatments")
            for alt in decision.alternative_treatments:
                st.write(f"- {BoardModerator._get_display_name(alt)}")

        # Warnings
        if decision.warnings:
            st.subheader("Warnings")
            for w in decision.warnings:
                st.warning(w)

        # Neoantigen Target Visualization
        profile_for_viz = st.session_state.get("patient_profile")
        if profile_for_viz and profile_for_viz.predicted_neoantigens:
            top_neos = [n for n in profile_for_viz.predicted_neoantigens if n.is_strong_binder][:3]
            if not top_neos:
                top_neos = [n for n in profile_for_viz.predicted_neoantigens if n.is_weak_binder][:3]

            if top_neos:
                st.subheader("Top Neoantigen Target")
                neo = top_neos[0]
                st.markdown(
                    f"**{neo.source_gene}** {neo.mutation} | "
                    f"Peptide: `{neo.peptide}` | "
                    f"HLA: **{neo.best_hla_allele}** | "
                    f"IC50: **{neo.binding_affinity_nm:.1f} nM**"
                )
                if st.button("View 3D Peptide-MHC Complex", icon=":material/view_in_ar:"):
                    with st.spinner("Fetching structure from RCSB PDB..."):
                        try:
                            from cognisom.genomics.target_structure import TargetStructureBuilder
                            builder = TargetStructureBuilder()
                            structure = builder.build(neo)
                            st.session_state["mad_pmhc_structure"] = structure
                            st.session_state["mad_pmhc_neo"] = neo
                        except Exception as e:
                            st.error(f"Structure build failed: {e}")

                if st.session_state.get("mad_pmhc_structure"):
                    import streamlit.components.v1 as components
                    s = st.session_state["mad_pmhc_structure"]
                    n = st.session_state.get("mad_pmhc_neo", neo)
                    pdb_esc = s.pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
                    pc = s.peptide_chain_id
                    mp = n.mutation_position_in_peptide

                    html = f"""
                    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
                    <div id="mad-pmhc" style="width:100%;height:400px;border-radius:12px;
                         overflow:hidden;border:1px solid rgba(128,128,128,0.2);background:#0a0a2e;"></div>
                    <script>
                    (function(){{
                        var v=$3Dmol.createViewer("mad-pmhc",{{backgroundColor:"0x0a0a2e"}});
                        v.addModel(`{pdb_esc}`,"pdb");
                        v.setStyle({{chain:"A"}},{{cartoon:{{color:"0xcccccc",opacity:0.5}}}});
                        v.addSurface($3Dmol.SurfaceType.VDW,{{opacity:0.12,color:"0xdddddd"}},{{chain:"A"}});
                        v.setStyle({{chain:"B"}},{{cartoon:{{color:"0x6366f1",opacity:0.25}}}});
                        v.setStyle({{chain:"{pc}"}},{{stick:{{radius:0.25,colorscheme:"Jmol"}},sphere:{{radius:0.5,colorscheme:"Jmol"}}}});
                        v.addStyle({{chain:"{pc}",resi:{mp+1}}},{{sphere:{{radius:0.8,color:"0xff4444"}}}});
                        "DEFGHIJKLMNOPQRSTUVWXYZ".split("").forEach(function(c){{v.setStyle({{chain:c}},{{cartoon:{{color:"0x333333",opacity:0.1}}}});}});
                        v.zoomTo({{chain:"{pc}"}},600);
                        v.spin("y",0.3);
                        v.render();
                    }})();
                    </script>
                    """
                    components.html(html, height=420)
                    st.caption(
                        f"Structure: {s.method} | "
                        f"Template: {s.template_pdb_id or 'predicted'} | "
                        f"Peptide chain: {pc}"
                    )

        # Limitations
        with st.expander("Limitations & Disclaimers"):
            for lim in decision.limitations:
                st.caption(f"- {lim}")
            st.caption(f"\n{decision.context_of_use}")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: EVIDENCE CHAIN
# ─────────────────────────────────────────────────────────────────────────

with tab_evidence:
    decision = st.session_state.get("mad_decision")
    if not decision:
        st.info("Run the MAD Board first to see the evidence chain.")
    else:
        st.subheader(f"Evidence Chain ({len(decision.evidence_chain)} items)")

        for ev in decision.evidence_chain:
            icon_map = {
                "clinical_trial": "🔬",
                "guideline": "📋",
                "biomarker": "🧬",
                "simulation": "💻",
                "literature": "📚",
                "validation": "✅",
            }
            icon = icon_map.get(ev.source_type, "📄")

            with st.expander(f"{icon} [{ev.strength}] {ev.source_name}"):
                st.markdown(f"**Claim:** {ev.claim}")
                st.caption(f"Source ID: {ev.source_id} | Type: {ev.source_type}")
                if ev.supporting_data:
                    st.json(ev.supporting_data)

        # Dissenting views
        if decision.dissenting_views:
            st.subheader("Dissenting Views")
            for dv in decision.dissenting_views:
                st.warning(dv)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3: EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────

with tab_explain:
    decision = st.session_state.get("mad_decision")
    if not decision:
        st.info("Run the MAD Board first to see explainability details.")
    else:
        st.subheader("Biomarker Contribution Analysis")

        # Show contributions for top recommended treatment
        for opinion in decision.agent_opinions:
            top_ranking = next(
                (r for r in opinion.treatment_rankings
                 if r.treatment_key == decision.recommended_treatment),
                None,
            )
            if not top_ranking or not top_ranking.contributions:
                continue

            st.markdown(f"### {opinion.agent_name.title()} Agent")

            # Waterfall chart
            contribs = top_ranking.contributions
            names = [c.feature_name.replace("_", " ").title() for c in contribs]
            deltas = [c.delta for c in contribs]
            colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

            fig = go.Figure(go.Bar(
                x=deltas,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{d:+.3f}" for d in deltas],
                textposition="auto",
            ))
            fig.update_layout(
                title=f"Feature Contributions to {decision.recommended_treatment_name}",
                xaxis_title="Effect on Score",
                yaxis_title="",
                height=max(250, len(contribs) * 40),
            )
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 4: FDA COMPLIANCE
# ─────────────────────────────────────────────────────────────────────────

with tab_compliance:
    st.subheader("FDA Compliance Status")

    # Context of Use
    cou = ContextOfUse()
    with st.expander("Context of Use (COU)", expanded=True):
        st.markdown(f"**{cou.product_name}** v{cou.version}")
        st.info(cou.statement)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Intended Users:** {cou.intended_users}")
            st.markdown(f"**Intended Setting:** {cou.intended_setting}")
        with col2:
            st.markdown(f"**Decision Consequence:** {cou.decision_consequence}")
            st.markdown(f"**Model Influence:** {cou.model_influence}")
            st.markdown(f"**Human in the Loop:** {'Yes' if cou.human_in_the_loop else 'No'}")

    # Non-Device CDS
    checker = NonDeviceCDSChecker()
    result = checker.check_all()
    with st.expander("Non-Device CDS Criteria (21st Century Cures Act)"):
        if result["all_criteria_met"]:
            st.success(f"Classification: {result['classification']}")
        else:
            st.error("One or more criteria NOT met")

        for criterion in result["criteria"]:
            status_icon = "✅" if criterion["status"] == "PASS" else "❌"
            st.markdown(f"{status_icon} **Criterion {criterion['id']}:** {criterion['text']}")
            st.caption(criterion["rationale"])

    # 7-Step Framework
    framework = CredibilityFramework()
    status = framework.get_status()
    with st.expander("7-Step AI Credibility Framework"):
        st.progress(status["completed"] / 7, text=f"{status['completed']}/7 steps completed")

        for step in status["steps"]:
            icon = "✅" if step["status"] in ("defined", "executed") else "⏳"
            st.markdown(f"{icon} **Step {step['step']}: {step['name']}**")
            # Show whatever description field exists
            for key in ("question", "description", "rationale", "plan", "execution",
                        "documentation", "assessment"):
                if key in step:
                    st.caption(step[key])

    # Model Cards
    with st.expander("Model Cards"):
        cards = get_all_model_cards()
        for name, card in cards.items():
            st.markdown(f"### {card.name} ({card.version})")
            st.markdown(f"**Type:** {card.model_type}")
            st.markdown(f"**Purpose:** {card.purpose}")
            st.markdown(f"**Population:** {card.intended_population}")
            st.markdown(f"**Validation:** {card.validation_citation}")
            st.caption(f"Limitations: {'; '.join(card.known_limitations[:3])}")
            st.divider()

# ─────────────────────────────────────────────────────────────────────────
# TAB 5: AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────

with tab_audit:
    st.subheader("MAD Agent Audit Trail")

    try:
        audit_store = AuditStore()
        records = audit_store.get_recent(50)

        if records:
            st.metric("Total Sessions", audit_store.count())

            df = pd.DataFrame([r.to_dict() for r in records])
            display_cols = [
                "session_id", "timestamp", "patient_id",
                "recommended_treatment", "consensus_level",
                "confidence", "n_evidence_items",
            ]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available], use_container_width=True)
        else:
            st.info("No audit records yet. Run the MAD Board to create records.")
    except Exception as e:
        st.warning(f"Audit store not available: {e}")
        st.caption("Audit database will be created when the MAD Board is first run.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 6: RESEARCH STUDY
# ─────────────────────────────────────────────────────────────────────────

with tab_study:
    st.subheader("MAD Agent Research Study")
    st.markdown("""
    **Study:** Retrospective Validation of MAD Agent for Immunotherapy Selection in mCRPC

    | Parameter | Value |
    |-----------|-------|
    | **Design** | Retrospective concordance, multi-cohort |
    | **Primary Cohort** | SU2C/PCF 2019 (429 patients) |
    | **Validation Cohort** | TCGA-PRAD (494 patients) |
    | **Primary Endpoint** | Treatment-Biomarker Concordance Rate |
    | **Secondary Endpoints** | Biomarker sensitivity/specificity, agent agreement, TMB calibration |
    """)

    study_results = st.session_state.get("mad_study_results")

    col1, col2 = st.columns(2)
    with col1:
        n_patients = st.number_input(
            "Patients to analyze", min_value=1, max_value=429,
            value=50, step=10, key="study_n",
        )
    with col2:
        if st.button("Run Study", type="primary"):
            try:
                from cognisom.validation.mad_study import MADStudy

                study = MADStudy()
                progress_bar = st.progress(0, text="Loading cohort...")

                def update_progress(current, total, pid):
                    progress_bar.progress(
                        current / total,
                        text=f"Patient {current}/{total}: {pid}",
                    )

                study_results = study.run_full_study(
                    n_patients=n_patients,
                    progress_callback=update_progress,
                )
                st.session_state["mad_study_results"] = study_results
                progress_bar.progress(1.0, text="Complete!")
            except FileNotFoundError:
                st.error("SU2C data not found. Download from cBioPortal first.")
            except Exception as e:
                st.error(f"Study failed: {e}")

    if study_results:
        st.divider()

        # Hero metrics
        c1, c2, c3, c4 = st.columns(4)
        conc = study_results.concordance
        with c1:
            st.metric("Patients Analyzed", study_results.n_patients)
        with c2:
            if conc:
                st.metric("Biomarker Concordance", f"{conc.biomarker_concordance_rate:.0%}")
        with c3:
            if conc:
                st.metric("Unanimous Rate", f"{conc.unanimous_rate:.0%}")
        with c4:
            avg_time = study_results.total_processing_seconds / max(1, study_results.n_patients)
            st.metric("Avg Time/Patient", f"{avg_time:.2f}s")

        # Concordance details
        if conc:
            st.subheader("Concordance Metrics")
            st.json(conc.to_dict())

        # Biomarker accuracy
        if study_results.biomarker_accuracy:
            st.subheader("Biomarker Detection Accuracy")
            for ba in study_results.biomarker_accuracy:
                st.json(ba.to_dict())

        # TMB calibration
        if study_results.tmb_calibration:
            st.subheader("TMB Calibration")
            st.json(study_results.tmb_calibration)

        # Export
        if st.button("Export Results (JSON)"):
            st.download_button(
                "Download",
                json.dumps(study_results.to_dict(), indent=2),
                file_name="mad_study_results.json",
                mime="application/json",
            )
