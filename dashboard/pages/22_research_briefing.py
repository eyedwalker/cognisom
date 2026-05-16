"""
Page 22: Research Collaboration Briefing
=========================================

Briefing page for academic research conversations — what Cognisom is,
what it does today, and where research partnerships fit.
"""

import streamlit as st

st.set_page_config(
    page_title="Research Briefing | Cognisom",
    page_icon="🔬",
    layout="wide",
)

try:
    from cognisom.auth.middleware import streamlit_page_gate
    user = streamlit_page_gate("22_research_briefing")
except Exception:
    user = None

# ── Styling ──────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .pillar-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .pillar-card h3 { margin-top: 0; }
    .tag {
        display: inline-block;
        background: rgba(80, 180, 255, 0.15);
        color: #7dc6ff;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .tag-green { background: rgba(80, 220, 130, 0.15); color: #7ddfa0; }
    .tag-amber { background: rgba(255, 190, 80, 0.15); color: #ffc864; }
    .quote {
        border-left: 3px solid #7dc6ff;
        padding: 0.6rem 1rem;
        margin: 1rem 0;
        font-style: italic;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ─────────────────────────────────────────────────────────

st.title("Cognisom — Research Collaboration Briefing")
st.caption(
    "GPU-accelerated, mechanistic simulation of cancer–immune biology, "
    "paired with an FDA-compliant clinical AI layer."
)

st.markdown(
    """
    <div class="quote">
    Cognisom simulates cancer–immune interactions across three scales —
    molecular, cellular, and tissue — and pairs the simulation with a
    multi-agent AI (MAD Agent) that ranks immunotherapies and matches
    patients to trials, with a full evidence audit trail.
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Simulation modules", "6", help="Molecular, Cellular, Immune, Vascular, Lymphatic, Spatial")
c2.metric("Event throughput", "243k/s", help="Event-driven core, single-host benchmark")
c3.metric("MAD Agent LOC", "3.4k", help="Genomics + Immune + Clinical agents with consensus")
c4.metric("Real patient datasets wired in", "2", help="SEQC2 breast cancer; SU2C mCRPC (429 patients)")

st.divider()

# ── Three pillars ───────────────────────────────────────────────

st.subheader("Three pillars")

p1, p2, p3 = st.columns(3, gap="large")

with p1:
    st.markdown(
        """
        <div class="pillar-card">
        <h3>🧬 Entity Library</h3>
        <p>The biological knowledge graph that grounds every simulation
        and every recommendation.</p>
        <ul>
          <li>Real DNA / RNA sequences from NCBI (not abstract counts)</li>
          <li>Oncogenic variants: KRAS G12D, TP53 R175H, BRAF V600E, …</li>
          <li>Pathways from BioModels / KEGG (SBML-native)</li>
          <li>Immune entities: CD8⁺ T cells, NK cells, M1/M2 macrophages, dendritic cells</li>
          <li>HLA typing (OptiType) + MHC-I peptide binding (MHCflurry 2.1.5, GPU)</li>
          <li>Clinical annotations: OncoKB, ClinicalTrials.gov, PubMed, Reactome</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        """
        <div class="pillar-card">
        <h3>🔬 Cellular Simulation</h3>
        <p>Agent-based + stochastic + PDE diffusion, integrated end-to-end.</p>
        <ul>
          <li>Three timescales: ms (reactions) → hours (cell cycle) → days (immune infiltration)</li>
          <li>Cell-cycle, Warburg metabolism, apoptosis / necrosis</li>
          <li>T-cell killing kinetics, NK missing-self, M1/M2 polarization</li>
          <li>Immune evasion: MHC-I loss, PD-L1 upregulation, TGF-β / IL-10 suppression</li>
          <li>3D diffusion fields: O₂, glucose, IFN-γ, IL-2, IL-10, TGF-β, CXCL9/10, CCL2/5</li>
          <li>Stochastic engine: Gillespie SSA / tau-leaping; PDE solver for diffusion</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
        <div class="pillar-card">
        <h3>🧠 Clinical Intelligence (MAD Agent)</h3>
        <p>Multi-agent decision board that turns a patient VCF into a
        ranked treatment plan with evidence chains.</p>
        <ul>
          <li><b>GenomicsAgent</b> — driver/passenger calls, TMB, MSI / dMMR</li>
          <li><b>ImmuneAgent</b> — neoantigen prediction, infiltration scoring, TCR clonality</li>
          <li><b>ClinicalAgent</b> — treatment simulation, resistance modeling, trial matching</li>
          <li><b>BoardModerator</b> — synthesizes consensus → ranked therapies</li>
          <li>Every recommendation traces to evidence (audit trail)</li>
          <li>Designed under <b>21st Century Cures Act § 3060(a)</b> non-device CDS rules</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── End-to-end flow ─────────────────────────────────────────────

st.subheader("End-to-end research flow")

st.markdown(
    """
    ```
    Patient VCF (germline + somatic)
        │
        ├──▶  GenomicsAgent     ── driver mutations, TMB, MSI status
        │
        ├──▶  ImmuneAgent       ── neoantigens (MHCflurry GPU), infiltration profile
        │
        ├──▶  ClinicalAgent     ── treatment sim, resistance, trial matches
        │
        ▼
    BoardModerator   ──▶   Ranked treatments + evidence chain + matching trials
        │
        ▼
    Cellular simulation under each candidate therapy
        │
        ▼
    Predicted tumor + immune dynamics over days
    ```
    """
)

st.divider()

# ── Working today vs roadmap ────────────────────────────────────

st.subheader("Where it stands today")

t1, t2 = st.columns(2)

with t1:
    st.markdown("##### ✅ Working today")
    st.markdown(
        """
        - 6-module integrated simulator (~6k LOC core)
        - MAD Agent with 3-agent consensus and evidence chain (3.4k LOC)
        - VCF → MHCflurry neoantigen prediction (GPU, validated on SEQC2 breast cancer — 5 strong binders)
        - 38-page Streamlit interface with FASTQ → BAM → VCF → Recommendations pipeline
        - Parabricks 4.7.0 integration (matched tumor / normal variant calling)
        - Docker / Kubernetes deployment, cloud-agnostic
        - Validation study protocol drafted against the SU2C mCRPC cohort (429 patients)
        """
    )

with t2:
    st.markdown("##### 🛠 In development")
    st.markdown(
        """
        - Multi-GPU domain decomposition for tissue-scale simulation (100k+ cells)
        - ML surrogate models for metabolism and diffusion fields
        - Prospective clinical validation
        - TCR repertoire dynamics module
        - Tertiary lymphoid structure modeling
        - B-cell maturation / germinal center dynamics
        """
    )

st.divider()

# ── Where it sits in the landscape ──────────────────────────────

st.subheader("Where Cognisom sits in the landscape")

st.markdown(
    """
    | Compared to | Cognisom adds |
    |---|---|
    | **PhysiCell** | Detailed immune system, real molecular sequences, exosome transfer |
    | **VCell** | GPU acceleration (10–100×), integrated immune layer |
    | **Whole-cell models** (e.g. M. genitalium) | Multicellular + tissue scale, immune focus |
    | **ML-only platforms** (e.g. AlphaFold-class) | Mechanistic + interpretable; every prediction traceable |

    The combination — **real molecular sequences + GPU scaling + detailed
    immune system + clinical-validation pathway** — is, as far as we can
    tell, not available anywhere else.
    """
)

st.divider()

# ── Non-profit thesis ───────────────────────────────────────────

st.subheader("Why we are pursuing this as a non-profit")

st.markdown(
    """
    The hypothesis we want to pressure-test:

    1. **The most valuable form of this platform is open infrastructure for
       academic immunology and oncology research**, not a commercial SaaS.
    2. Mechanistic + interpretable wins for *hypothesis generation*; black-box
       ML wins for raw predictive accuracy. Cognisom serves the first need.
    3. A non-profit lets us partner with academic medical centers (TCGA,
       SU2C, ICGC, dbGaP) without IP-licensing friction.
    4. Sustainability comes from foundation grants, NIH SBIR/STTR, and
       fee-for-service compute — not from licensing the platform.
    """
)

st.divider()

# ── Discussion topics ───────────────────────────────────────────

st.subheader("Topics we'd value your read on")

st.markdown(
    """
    1. **Hypothesis-generation fit.** If you wanted to test a hypothesis about
       NK-cell exhaustion in the tumor microenvironment tomorrow, would you
       reach for a tool like this — and if not, what is missing?

    2. **Smallest credible result.** What is the smallest scientific output
       that would make computational immunology peers take this seriously?
       A wet-lab–validated mechanistic prediction? A retrospective cohort
       reanalysis? A challenge-dataset benchmark?

    3. **Beta users.** Who in your network is doing computational immunology
       and has a workflow gap this could fill — not as a favor, but because
       the tool removes real friction?

    4. **Sustainability model.** Non-profit infrastructure vs. open-source-
       with-services-arm — given current NIH funding pressure, which model
       do you actually see surviving in academic biology?

    5. **Missing immunology features.** Which immune-system feature do you
       wish existed but currently doesn't —
       TCR repertoire dynamics?
       Tertiary lymphoid structures?
       Tissue-resident memory T cells?
       B-cell maturation / germinal centers?
       Cytokine storm / systemic inflammation?

    6. **Validation cohorts.** Beyond SU2C mCRPC and SEQC2 breast cancer,
       which public datasets would carry the most weight as validation
       targets for an immunology audience?
    """
)

st.divider()

# ── Quick links to live pages ───────────────────────────────────

st.subheader("Live in this dashboard")

l1, l2, l3, l4 = st.columns(4)
with l1:
    st.page_link("pages/14_entity_library.py", label="📚 Entity Library", icon="🔗")
with l2:
    st.page_link("pages/3_simulation.py", label="🔬 Cellular Simulation", icon="🔗")
with l3:
    st.page_link("pages/11_validation.py", label="✅ Validation & Benchmarks", icon="🔗")
with l4:
    st.page_link("pages/19_prostate_metastasis.py", label="🧬 Prostate Metastasis Case", icon="🔗")

st.caption(
    "Cognisom — research-grade prototype, deployed at cognisom.wubba.ai. "
    "Source code, validation protocols, and study designs available on request."
)
