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
            Request access at david@eyentelligence.com
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
    <p class="inv-hero-title">Cognisom</p>
    <p class="inv-hero-sub">The Operating System for Decentralized Precision Oncology</p>
    <p class="inv-hero-byline">eyentelligence inc. &mdash; Confidential</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: WHY THIS OPPORTUNITY MATTERS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Why This Opportunity Matters</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

opp1, opp2 = st.columns(2)

with opp1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-stat-val" style="font-size: 2.2rem;">$200B+</div>
        <div class="inv-stat-label" style="margin-bottom: 0.8rem;">Immunotherapy market by 2030</div>
        <div class="inv-card-text">
            Large and growing oncology market driven by checkpoint inhibitors,
            cell therapies, and personalized cancer vaccines.
        </div>
    </div>
    """, unsafe_allow_html=True)

with opp2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Fragmented Infrastructure</div>
        <div class="inv-card-text">
            Precision medicine infrastructure remains fragmented.
            AI can improve interpretation and operational coordination
            across the oncology workflow.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

opp3, opp4 = st.columns(2)

with opp3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Decentralized Execution</div>
        <div class="inv-card-text">
            Decentralized execution models could reshape delivery economics,
            bringing precision oncology closer to where patients are treated.
        </div>
    </div>
    """, unsafe_allow_html=True)

with opp4:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Outcome-Aligned Finance</div>
        <div class="inv-card-text">
            Outcome-aligned financial structures could unlock payer relevance,
            connecting clinical decision-making with reimbursement accountability.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: INVESTMENT THESIS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Investment Thesis</div>
<div class="inv-section-sub">An integrated approach to a massive market</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
Cognisom is designed around the belief that the next major opportunity in oncology is not just
inventing a therapy, but building the infrastructure that makes individualized therapy more
**actionable**, **scalable**, and **accountable**.
""")

pil1, pil2, pil3 = st.columns(3)

with pil1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom</div>
        <div style="font-size: 0.75rem; color: #00d4aa; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;
                    margin-bottom: 0.6rem;">Data & Intelligence Layer</div>
        <div class="inv-card-text">
            AI-guided molecular interpretation built on matched tumor-normal
            analysis, multimodal biological data, and compounding oncology insight.
        </div>
    </div>
    """, unsafe_allow_html=True)

with pil2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Therapeutics</div>
        <div style="font-size: 0.75rem; color: #6366f1; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;
                    margin-bottom: 0.6rem;">Execution & Manufacturing Readiness</div>
        <div class="inv-card-text">
            A workflow and quality framework designed to support partner-enabled
            therapeutic execution now and distributed manufacturing integration
            in the future.
        </div>
    </div>
    """, unsafe_allow_html=True)

with pil3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Assurance</div>
        <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;
                    margin-bottom: 0.6rem;">Risk & Payment Alignment</div>
        <div class="inv-card-text">
            A model designed to connect oncology decision-making with financial
            accountability, predictive triage, and future outcome-aligned
            reimbursement structures.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: PLATFORM VALIDATION
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Platform Validation</div>
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
            AI-guided molecular interpretation to support clinically informed
            therapy selection. Designed for independent clinician review.
        </div>
        <div style="font-size: 0.75rem; color: #00d4aa; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;">
            Interpretation & Decision Support
        </div>
    </div>
    """, unsafe_allow_html=True)

with bm2:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Sponsor</div>
        <div class="inv-card-text" style="margin-bottom: 0.8rem;">
            Evidence-as-a-Service helping biopharma with biomarker strategy,
            patient stratification, trial enrichment, companion-diagnostic
            support, and RWD/RWE packages.
        </div>
        <div style="font-size: 0.75rem; color: #6366f1; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;">
            Biopharma Evidence Services
        </div>
    </div>
    """, unsafe_allow_html=True)

with bm3:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Cognisom Therapeutics</div>
        <div class="inv-card-text" style="margin-bottom: 0.8rem;">
            Execution and manufacturing readiness. Partner-enabled therapeutic
            workflow. Future distributed manufacturing integration as
            technologies mature.
        </div>
        <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.5px;">
            Execution & Manufacturing
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
# SECTION 7: USE OF CAPITAL
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">What Early Capital Supports</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

cap1, cap2 = st.columns([1, 2])

with cap1:
    st.markdown("""
    <div class="inv-card">
        <div class="inv-card-title">Focus Areas</div>
        <div class="inv-card-text" style="line-height: 2.2;">
            &bull; Platform and workflow development<br>
            &bull; Clinical and scientific partnerships<br>
            &bull; Regulatory and quality architecture<br>
            &bull; Validation and partner readiness<br>
            &bull; Corporate, IP, and operating infrastructure
        </div>
    </div>
    """, unsafe_allow_html=True)

with cap2:
    st.markdown("**Fund Allocation**")
    st.progress(40, text="Platform Development — 40%")
    st.progress(30, text="Clinical Validation — 30%")
    st.progress(15, text="Regulatory & Quality — 15%")
    st.progress(15, text="Team & Operations — 15%")


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: 12-MONTH ROADMAP
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
# SECTION 9: TEAM
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
# SECTION 10: CURRENT FOCUS
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-section">Current Focus</div>
<div class="inv-line"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="inv-card" style="text-align: center; padding: 2.5rem 2rem;">
    <div class="inv-card-text" style="font-size: 0.95rem; opacity: 0.85; max-width: 700px; margin: 0 auto;">
        Cognisom is currently focused on platform formation, strategic partnerships,
        workflow architecture, and validation planning to support long-term execution
        across its three core platform pillars.
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 11: CONTACT
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="inv-contact">
    <div class="inv-contact-title">Request the Investor Deck</div>
    <div class="inv-contact-sub">
        We welcome conversations with cancer centers, translational researchers,
        manufacturers, payer organizations, strategic investors, and infrastructure partners.
    </div>
    <div style="margin-bottom: 0.8rem;">
        <a href="mailto:david@eyentelligence.com" class="inv-contact-link">david@eyentelligence.com</a>
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
    Confidential &mdash; For intended recipients only.
</div>
""", unsafe_allow_html=True)
