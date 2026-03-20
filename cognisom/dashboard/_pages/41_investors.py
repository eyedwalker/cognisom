"""
Investor Overview
=================

Password-protected investor data room for Cognisom Therapeutics.
"""

import streamlit as st
from cognisom.dashboard.page_config import safe_set_page_config

safe_set_page_config(
    page_title="Cognisom | Investor Overview",
    page_icon=":material/trending_up:",
    layout="wide",
)

# ── Password Gate ──────────────────────────────────────────────────
_INVESTOR_PASSWORD = "investinme2026"

if not st.session_state.get("investor_auth"):
    st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    _pad1, _center, _pad2 = st.columns([1, 2, 1])
    with _center:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 0 2rem 0;">
            <div style="font-size: 2rem; font-weight: 700; color: var(--text-color, #e2e8f0);
                        margin-bottom: 0.3rem;">Cognisom Therapeutics</div>
            <div style="font-size: 1rem; opacity: 0.5; margin-bottom: 2rem;">Investor Data Room</div>
            <div style="height: 3px; width: 80px; margin: 0 auto 2rem auto;
                        background: linear-gradient(90deg, #00d4aa, #6366f1);
                        border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)

        pwd = st.text_input(
            "Access Code",
            type="password",
            placeholder="Enter investor access code",
            key="investor_pw_input",
        )
        if st.button("Enter", use_container_width=True, type="primary"):
            if pwd == _INVESTOR_PASSWORD:
                st.session_state["investor_auth"] = True
                st.rerun()
            else:
                st.error("Invalid access code.")

        st.markdown("""
        <div style="text-align: center; margin-top: 2rem; font-size: 0.8rem; opacity: 0.4;">
            Request access at david@cognisom.com
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════════
# AUTHENTICATED INVESTOR PAGE
# ════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hero ── */
.inv-hero {
    background: linear-gradient(-45deg, #0a0e27, #1a1f4e, #0d2137, #0a1628);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    border-radius: 16px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.inv-hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -15%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
}

.inv-hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #6366f1, #00d4aa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 5s ease infinite;
    position: relative;
    z-index: 1;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}

.inv-hero-sub {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.3rem;
    position: relative;
    z-index: 1;
}

.inv-hero-byline {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.5rem;
    position: relative;
    z-index: 1;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Sections ── */
.inv-section {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-color, #e2e8f0);
    margin-bottom: 0.2rem;
    margin-top: 2.5rem;
}

.inv-section-sub {
    font-size: 0.9rem;
    color: var(--text-color, rgba(255,255,255,0.5));
    opacity: 0.55;
    margin-bottom: 1rem;
}

.inv-line {
    height: 3px;
    background: linear-gradient(90deg, #00d4aa, #6366f1, transparent);
    border-radius: 2px;
    margin-bottom: 1.5rem;
    width: 100px;
}

/* ── Cards ── */
.inv-card {
    background: var(--secondary-background-color, rgba(255,255,255,0.03));
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 12px;
    padding: 1.5rem 1.3rem;
    height: 100%;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.inv-card:hover {
    border-color: rgba(0,212,170,0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

.inv-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4aa, #6366f1);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.inv-card:hover::before {
    opacity: 1;
}

.inv-card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-color, #e2e8f0);
    margin-bottom: 0.4rem;
}

.inv-card-text {
    font-size: 0.85rem;
    color: var(--text-color, rgba(255,255,255,0.6));
    opacity: 0.7;
    line-height: 1.6;
}

/* ── Stat row ── */
.inv-stat {
    text-align: center;
    padding: 1rem 0.5rem;
}

.inv-stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.inv-stat-label {
    font-size: 0.72rem;
    color: var(--text-color, rgba(255,255,255,0.5));
    opacity: 0.55;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.15rem;
}

/* ── Tech badges ── */
.inv-badge {
    display: inline-block;
    background: var(--secondary-background-color, rgba(255,255,255,0.04));
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 8px;
    padding: 0.7rem 1.2rem;
    margin: 0.3rem;
    font-size: 0.82rem;
    color: var(--text-color, rgba(255,255,255,0.65));
    transition: all 0.3s ease;
}

.inv-badge:hover {
    border-color: rgba(0,212,170,0.3);
    color: var(--text-color, #e2e8f0);
}

/* ── Comp table ── */
.inv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.5rem;
}

.inv-table th {
    background: var(--secondary-background-color, rgba(0,212,170,0.08));
    color: var(--text-color, #e2e8f0);
    padding: 0.7rem 0.8rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid rgba(0,212,170,0.2);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.inv-table td {
    padding: 0.65rem 0.8rem;
    border-bottom: 1px solid rgba(128,128,128,0.1);
    color: var(--text-color, rgba(255,255,255,0.7));
    vertical-align: top;
    line-height: 1.5;
}

.inv-table tr:hover td {
    background: var(--secondary-background-color, rgba(255,255,255,0.02));
}

.inv-table .co-highlight {
    color: #00d4aa;
    font-weight: 600;
}

/* ── Roadmap timeline ── */
.inv-rm-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.8rem;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    background: var(--secondary-background-color, rgba(255,255,255,0.02));
    border: 1px solid rgba(128,128,128,0.08);
}

.inv-rm-q {
    background: linear-gradient(135deg, #00d4aa, #6366f1);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    white-space: nowrap;
    min-width: 52px;
    text-align: center;
}

.inv-rm-text {
    font-size: 0.88rem;
    color: var(--text-color, rgba(255,255,255,0.7));
    line-height: 1.5;
}

/* ── Funds breakdown ── */
.inv-fund-bar {
    height: 8px;
    border-radius: 4px;
    margin-bottom: 0.6rem;
}

.inv-fund-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-color, rgba(255,255,255,0.65));
    margin-bottom: 1rem;
}

.inv-fund-pct {
    font-weight: 700;
    color: var(--text-color, #e2e8f0);
}

/* ── Contact footer ── */
.inv-contact {
    background: linear-gradient(-45deg, #0a0e27, #1a1f4e, #0d2137);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem 0 1rem 0;
}

.inv-contact-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}

.inv-contact-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

.inv-contact-link {
    color: #00d4aa;
    font-weight: 600;
    font-size: 1rem;
    text-decoration: none;
}

.inv-footer {
    text-align: center;
    padding: 1rem;
    font-size: 0.7rem;
    color: var(--text-color, rgba(255,255,255,0.25));
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="inv-hero">
    <p class="inv-hero-title">Cognisom Therapeutics</p>
    <p class="inv-hero-sub">Investor Overview</p>
    <p class="inv-hero-byline">eyentelligence inc. &mdash; Confidential</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: THE OPPORTUNITY
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">The Opportunity</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

opp1, opp2, opp3 = st.columns(3)

with opp1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-stat-val" style="font-size: 2.2rem;">$200B+</div>
        <div class="inv-stat-label" style="margin-bottom: 0.8rem;">Immunotherapy market by 2030</div>
        <div class="inv-card-text">
            Immuno-oncology is the fastest-growing segment in oncology,
            driven by checkpoint inhibitors, cell therapies, and
            personalized cancer vaccines.
        </div>
    </div>
    """, unsafe_allow_html=True)

with opp2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-stat-val" style="font-size: 2.2rem;">~25%</div>
        <div class="inv-stat-label" style="margin-bottom: 0.8rem;">Patients who respond today</div>
        <div class="inv-card-text">
            Immunotherapy only works for roughly 1 in 4 patients.
            The other 75% face trial-and-error medicine because
            current tools cannot reconcile each patient's unique
            tumor biology with the immune system's complexity.
        </div>
    </div>
    """, unsafe_allow_html=True)

with opp3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-stat-val" style="font-size: 2.2rem;">75%</div>
        <div class="inv-stat-label" style="margin-bottom: 0.8rem;">Addressable gap</div>
        <div class="inv-card-text">
            Cognisom is an AI evidence platform that makes immunotherapy
            effective for the other 75% &mdash; by giving clinicians
            the molecular intelligence they need to match the right
            patient to the right therapy, with traceable evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: THE SOLUTION
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">The Solution</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
Cognisom is an **AI precision-oncology intelligence and evidence platform** that transforms
raw patient DNA into actionable, evidence-traced treatment recommendations.
""")

sol1, sol2 = st.columns(2)

with sol1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Matched Tumor-Normal Analysis</div>
        <div class="inv-card-text">
            Gold-standard somatic variant calling from matched tumor + germline
            sequencing. GPU-accelerated via NVIDIA Parabricks on AWS HealthOmics.
            Full pipeline: alignment, variant calling, HLA typing, neoantigen
            prediction &mdash; in under 90 minutes for 30x WGS.
        </div>
    </div>
    """, unsafe_allow_html=True)

with sol2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">MAD Agent: AI Molecular Tumor Board</div>
        <div class="inv-card-text">
            Three specialist AI agents (Genomics, Immune, Clinical) deliberate
            on each case and produce a consensus recommendation with full
            evidence chain, dissent tracking, and audit trail &mdash; designed
            for independent clinician review.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

sol3, sol4 = st.columns(2)

with sol3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Evidence-Traced Recommendations</div>
        <div class="inv-card-text">
            Every recommendation links back to its source: OncoKB evidence levels,
            MHCflurry binding scores, published clinical trial data, and
            ClinicalTrials.gov matching. Clinicians can independently verify
            every step in the reasoning chain.
        </div>
    </div>
    """, unsafe_allow_html=True)

with sol4:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">FDA 2026 AI Credibility Framework</div>
        <div class="inv-card-text">
            Designed from the ground up as a Non-Device Clinical Decision Support
            tool under the FDA's 2026 AI Credibility Framework. Context of Use
            documentation, model cards, data provenance, and full audit trail
            are built into the platform architecture.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: TRACTION & VALIDATION
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Traction & Validation</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">429</div>
        <div class="inv-stat-label">Real Patients Validated</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            SU2C/PCF 2019 mCRPC cohort. Real clinical outcomes data.
        </div>
    </div>
    """, unsafe_allow_html=True)

with v2:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">r=0.987</div>
        <div class="inv-stat-label">TMB Correlation</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            100% biomarker concordance across the validation cohort.
        </div>
    </div>
    """, unsafe_allow_html=True)

with v3:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">700+</div>
        <div class="inv-stat-label">Actionable Genes (OncoKB)</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            Memorial Sloan Kettering evidence levels integrated.
        </div>
    </div>
    """, unsafe_allow_html=True)

with v4:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">14,847</div>
        <div class="inv-stat-label">HLA Alleles (MHCflurry)</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            >90% binding prediction accuracy. Neural network model.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

v5, v6, v7 = st.columns(3)

with v5:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">4-GPU</div>
        <div class="inv-stat-label">Parabricks Pipeline</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            Real 30x WGS through NVIDIA Parabricks on AWS HealthOmics.
            DeepVariant + Mutect2 somatic variant calling.
        </div>
    </div>
    """, unsafe_allow_html=True)

with v6:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">Live</div>
        <div class="inv-stat-label">ClinicalTrials.gov Matching</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            Real-time trial matching based on patient biomarkers,
            mutations, and HLA type.
        </div>
    </div>
    """, unsafe_allow_html=True)

with v7:
    st.markdown("""
    <div class="inv-card" style="text-align: center;">
        <div class="inv-stat-val">Live</div>
        <div class="inv-stat-label">Platform at cognisom.com</div>
        <div class="inv-card-text" style="margin-top: 0.5rem;">
            Production deployment with AWS Cognito authentication,
            TLS, and split GPU architecture.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: TECHNOLOGY STACK
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Technology Stack</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <span class="inv-badge">NVIDIA Parabricks</span>
    <span class="inv-badge">AWS HealthOmics</span>
    <span class="inv-badge">OncoKB (MSK)</span>
    <span class="inv-badge">MHCflurry</span>
    <span class="inv-badge">ClinicalTrials.gov</span>
    <span class="inv-badge">NVIDIA BioNeMo</span>
    <span class="inv-badge">OpenUSD / Isaac Sim</span>
</div>
""", unsafe_allow_html=True)

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">GPU Genomics</div>
        <div class="inv-card-text">
            NVIDIA Parabricks on AWS HealthOmics for 40x faster variant calling.
            4-GPU DeepVariant, Mutect2, and HaplotypeCaller workflows.
            OptiType HLA typing from germline reads.
        </div>
    </div>
    """, unsafe_allow_html=True)

with tech2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Clinical Evidence</div>
        <div class="inv-card-text">
            OncoKB (700+ genes, FDA evidence levels), MHCflurry (14,847 HLA alleles,
            neural network binding), ClinicalTrials.gov API for real-time
            recruiting trial matching.
        </div>
    </div>
    """, unsafe_allow_html=True)

with tech3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">AI & Visualization</div>
        <div class="inv-card-text">
            11 NVIDIA BioNeMo NIMs for drug discovery. Multi-agent
            deliberation architecture. OpenUSD and Isaac Sim for
            RTX-rendered molecular and tissue visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: BUSINESS MODEL
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Business Model</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

bm1, bm2, bm3 = st.columns(3)

with bm1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Clinical</div>
        <div class="inv-card-text" style="margin-bottom: 0.8rem;">
            SaaS for molecular tumor boards. Per-patient, per-analysis pricing.
            Evidence-traced recommendations that integrate into existing
            clinical workflows. Designed for CLIA/CAP lab partnerships.
        </div>
        <div style="font-size: 0.75rem; color: #00d4aa; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;">
            Near-term revenue
        </div>
    </div>
    """, unsafe_allow_html=True)

with bm2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Sponsor</div>
        <div class="inv-card-text" style="margin-bottom: 0.8rem;">
            Evidence-as-a-Service for biopharma. Trial enrichment
            (identify likely responders), companion diagnostics support,
            and real-world evidence packages for regulatory submissions.
        </div>
        <div style="font-size: 0.75rem; color: #6366f1; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;">
            High-margin contracts
        </div>
    </div>
    """, unsafe_allow_html=True)

with bm3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Therapeutics</div>
        <div class="inv-card-text" style="margin-bottom: 0.8rem;">
            Future expansion into AI-guided therapeutic design.
            Neoantigen vaccine candidate selection, combination therapy
            optimization, and target identification from digital twin
            simulations.
        </div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.35);
                    text-transform: uppercase; letter-spacing: 0.5px;">
            Long-term upside
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: COMPETITIVE LANDSCAPE
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Competitive Landscape</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
<table class="inv-table">
<thead>
<tr>
    <th>Company</th>
    <th>Primary Focus</th>
    <th>AI Architecture</th>
    <th>Regulatory Stance</th>
    <th>Cognisom Advantage</th>
</tr>
</thead>
<tbody>
<tr>
    <td><strong>Moderna / BioNTech</strong></td>
    <td>mRNA vaccine manufacturing</td>
    <td>Proprietary, closed-loop</td>
    <td>Therapeutic (IND/BLA pathway)</td>
    <td class="co-highlight">Platform-agnostic evidence layer; not locked to one modality</td>
</tr>
<tr>
    <td><strong>Gritstone Bio</strong></td>
    <td>Neoantigen prediction for vaccines</td>
    <td>EDGE model, single-agent</td>
    <td>Therapeutic development</td>
    <td class="co-highlight">Multi-agent deliberation; broader scope beyond vaccine targets</td>
</tr>
<tr>
    <td><strong>Tempus AI</strong></td>
    <td>Clinical genomics + data licensing</td>
    <td>Proprietary ML, retrospective</td>
    <td>LDT / 510(k) for select panels</td>
    <td class="co-highlight">Real-time evidence chain; FDA CDS framework; GPU-native pipeline</td>
</tr>
<tr>
    <td><strong>Adaptive Biotechnologies</strong></td>
    <td>TCR sequencing + immune profiling</td>
    <td>Immunosequencing platform</td>
    <td>CLIA/CAP lab certified</td>
    <td class="co-highlight">Full tumor + immune integration; treatment simulation, not just profiling</td>
</tr>
</tbody>
</table>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: 12-MONTH ROADMAP
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">12-Month Roadmap</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

rm1, rm2 = st.columns(2)

with rm1:
    st.markdown("""
    <div class="inv-rm-item">
        <div class="inv-rm-q">Q1</div>
        <div class="inv-rm-text">
            <strong>Molecular Tumor Board pilot</strong> &mdash; Partner with 2-3 oncology
            practices for real-world clinical validation of the MAD Agent workflow.
        </div>
    </div>
    <div class="inv-rm-item">
        <div class="inv-rm-q">Q2</div>
        <div class="inv-rm-text">
            <strong>FDA Biomarker Letter of Intent</strong> &mdash; Submit LOI for
            Non-Device CDS designation under the 2026 AI Credibility Framework.
            Complete 7-step credibility dossier.
        </div>
    </div>
    """, unsafe_allow_html=True)

with rm2:
    st.markdown("""
    <div class="inv-rm-item">
        <div class="inv-rm-q">Q3</div>
        <div class="inv-rm-text">
            <strong>Biopharma trial enrichment deal</strong> &mdash; First
            Evidence-as-a-Service contract for clinical trial patient
            stratification and biomarker-driven enrollment optimization.
        </div>
    </div>
    <div class="inv-rm-item">
        <div class="inv-rm-q">Q4</div>
        <div class="inv-rm-text">
            <strong>5,000 matched sample validation</strong> &mdash; Scale validation
            from 429 to 5,000 patients across multiple tumor types.
            Publish concordance and outcomes data.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: TEAM
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Team</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

team1, team2 = st.columns([2, 1])

with team1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title" style="font-size: 1.1rem;">David Walker</div>
        <div style="font-size: 0.82rem; color: #00d4aa; font-weight: 600;
                    margin-bottom: 0.6rem;">Founder & CEO</div>
        <div class="inv-card-text">
            Strategic Technology Executive with deep expertise in AI/ML systems,
            cloud infrastructure, and enterprise platform architecture. Harvard
            Extension School. Built and scaled technology platforms across
            healthcare, finance, and defense sectors.
        </div>
    </div>
    """, unsafe_allow_html=True)

with team2:
    st.markdown("""
    <div class="inv-card" style="display: flex; flex-direction: column; justify-content: center;">
        <div style="text-align: center;">
            <div style="font-size: 0.78rem; color: var(--text-color, rgba(255,255,255,0.5));
                        opacity: 0.55; text-transform: uppercase; letter-spacing: 1px;
                        margin-bottom: 0.6rem;">Program Membership</div>
            <div style="font-size: 1rem; font-weight: 600; color: var(--text-color, #e2e8f0);
                        margin-bottom: 0.3rem;">NVIDIA Inception Program</div>
            <div style="font-size: 0.82rem; color: var(--text-color, rgba(255,255,255,0.5));
                        opacity: 0.6;">
                Access to NVIDIA engineering support, DGX Cloud credits,
                and go-to-market resources.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: THE ASK
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">The Ask</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

ask1, ask2 = st.columns([1, 2])

with ask1:
    st.markdown("""
    <div class="inv-card" style="text-align: center; display: flex; flex-direction: column;
                justify-content: center; min-height: 220px;">
        <div style="font-size: 1.1rem; font-weight: 700; color: var(--text-color, #e2e8f0);
                    margin-bottom: 0.3rem;">Seed Round</div>
        <div style="font-size: 0.85rem; color: var(--text-color, rgba(255,255,255,0.5));
                    opacity: 0.6; margin-bottom: 1.2rem;">
            Pre-revenue, post-validation
        </div>
        <div style="font-size: 0.9rem; color: #00d4aa; font-weight: 600;">
            Contact for details
        </div>
        <div style="font-size: 0.75rem; color: var(--text-color, rgba(255,255,255,0.4));
                    margin-top: 0.3rem;">
            david@cognisom.com
        </div>
    </div>
    """, unsafe_allow_html=True)

with ask2:
    st.markdown("""
    <div class="inv-card" style="min-height: 220px;">
        <div class="inv-card-title" style="margin-bottom: 1rem;">Use of Funds</div>

        <div class="inv-fund-label">
            <span>Platform Development</span>
            <span class="inv-fund-pct">40%</span>
        </div>
        <div class="inv-fund-bar" style="background: linear-gradient(90deg, #00d4aa 0%, #00d4aa 100%); width: 40%;"></div>

        <div class="inv-fund-label">
            <span>Clinical Validation</span>
            <span class="inv-fund-pct">30%</span>
        </div>
        <div class="inv-fund-bar" style="background: linear-gradient(90deg, #6366f1 0%, #6366f1 100%); width: 30%;"></div>

        <div class="inv-fund-label">
            <span>Regulatory & Compliance</span>
            <span class="inv-fund-pct">15%</span>
        </div>
        <div class="inv-fund-bar" style="background: linear-gradient(90deg, #818cf8 0%, #818cf8 100%); width: 15%;"></div>

        <div class="inv-fund-label">
            <span>Team & Operations</span>
            <span class="inv-fund-pct">15%</span>
        </div>
        <div class="inv-fund-bar" style="background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0.3) 100%); width: 15%;"></div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 10: PITCH DECK
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Pitch Deck</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="inv-card" style="text-align: center; padding: 2.5rem 2rem;">
    <div style="font-size: 0.9rem; color: var(--text-color, rgba(255,255,255,0.6));
                margin-bottom: 0.5rem;">
        Pitch deck available upon request
    </div>
    <div style="font-size: 1rem; color: #00d4aa; font-weight: 600;">
        david@cognisom.com
    </div>
    <div style="font-size: 0.75rem; color: var(--text-color, rgba(255,255,255,0.35));
                margin-top: 0.8rem;">
        Full deck includes detailed financials, technical architecture deep-dive,
        clinical validation methodology, and regulatory strategy.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: CONTACT
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-contact">
    <div class="inv-contact-title">Get in Touch</div>
    <div class="inv-contact-sub">We welcome conversations with aligned investors and strategic partners.</div>
    <div style="margin-bottom: 0.8rem;">
        <a href="mailto:david@cognisom.com" class="inv-contact-link">david@cognisom.com</a>
    </div>
    <div style="font-size: 0.82rem; color: rgba(255,255,255,0.5);">
        cognisom.com
    </div>
    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.3); margin-top: 0.8rem;">
        Harvard Innovation Labs Member &nbsp;&bull;&nbsp; NVIDIA Inception Program Member
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="inv-footer">
    Confidential &mdash; For intended recipients only. &copy; 2026 eyentelligence inc.
</div>
""", unsafe_allow_html=True)
