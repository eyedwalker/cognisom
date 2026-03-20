"""
Cognisom HDT Platform
======================

Landing page + authenticated dashboard for the Cognisom platform.

Visitors see a public landing page with product information,
subscription tiers, and sign-in/registration.

Authenticated users see the platform dashboard with metrics and status.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import os

# ── Check auth state FIRST (before navigation) ──────────────────────

try:
    from cognisom.auth.middleware import get_current_user, _get_auth_manager
    user = get_current_user()
except Exception:
    user = None

# Force password change if needed
if user is not None and getattr(user, 'must_change_password', False):
    try:
        from cognisom.auth.middleware import _streamlit_force_password_change
        _streamlit_force_password_change(user)
    except Exception:
        user = None


# ── Navigation setup (MUST be called before any other st.* calls) ──
_pages_dir = Path(__file__).resolve().parent / "_pages"

if user is not None:
    # Authenticated: grouped navigation
    pages = {
        "": [
            st.Page(str(_pages_dir / "0_home.py"), title="Home", icon=":material/home:", default=True),
            st.Page(str(_pages_dir / "35_orchestrator.py"), title="Orchestrator", icon=":material/smart_toy:"),
            st.Page(str(_pages_dir / "36_validation_demo.py"), title="Demo & Validation", icon=":material/analytics:"),
        ],
        "Clinical Intelligence": [
            st.Page(str(_pages_dir / "38_real_patient.py"), title="DNA to Decision", icon=":material/labs:"),
            st.Page(str(_pages_dir / "37_mad_agent.py"), title="MAD Agent", icon=":material/groups:"),
            st.Page(str(_pages_dir / "33_clinical_report.py"), title="Clinical Report", icon=":material/description:"),
            st.Page(str(_pages_dir / "39_pipeline_guide.py"), title="Pipeline Guide", icon=":material/menu_book:"),
        ],
        "GPU Pipeline": [
            st.Page(str(_pages_dir / "32_parabricks.py"), title="GPU Genomics", icon=":material/biotech:"),
            st.Page(str(_pages_dir / "40_gpu_status.py"), title="Pipeline Status", icon=":material/monitor_heart:"),
        ],
        "Digital Twin": [
            st.Page(str(_pages_dir / "26_genomic_twin.py"), title="Genomic Profile", icon=":material/genetics:"),
            st.Page(str(_pages_dir / "27_cell_states.py"), title="Immune Landscape", icon=":material/biotech:"),
            st.Page(str(_pages_dir / "28_digital_twin.py"), title="Treatment Simulator", icon=":material/medication:"),
            st.Page(str(_pages_dir / "31_neoantigen_vaccine.py"), title="Neoantigen Vaccine", icon=":material/vaccines:"),
            st.Page(str(_pages_dir / "34_pathway_editor.py"), title="Pathway Editor", icon=":material/hub:"),
        ],
        "Visualization": [
            st.Page(str(_pages_dir / "29_molecular_viewer.py"), title="Molecular Viewer", icon=":material/view_in_ar:"),
            st.Page(str(_pages_dir / "30_spatial_tissue.py"), title="Spatial Tissue", icon=":material/map:"),
            st.Page(str(_pages_dir / "25_diapedesis.py"), title="Diapedesis", icon=":material/bloodtype:"),
            st.Page(str(_pages_dir / "12_3d_visualization.py"), title="3D Cells & Fields", icon=":material/scatter_plot:"),
        ],
        "Drug Discovery": [
            st.Page(str(_pages_dir / "2_discovery.py"), title="NIM Discovery", icon=":material/science:"),
            st.Page(str(_pages_dir / "5_molecular_lab.py"), title="Molecular Lab", icon=":material/labs:"),
        ],
        "Simulation": [
            st.Page(str(_pages_dir / "3_simulation.py"), title="9-Module Engine", icon=":material/settings:"),
            st.Page(str(_pages_dir / "15_physics_engine.py"), title="Physics Engine", icon=":material/bolt:"),
            st.Page(str(_pages_dir / "19_prostate_metastasis.py"), title="Cancer Progression", icon=":material/target:"),
            st.Page(str(_pages_dir / "20_vcell_solvers.py"), title="VCell Solvers", icon=":material/calculate:"),
            st.Page(str(_pages_dir / "21_tissue_scale.py"), title="Tissue Scale", icon=":material/memory:"),
        ],
        "Research": [
            st.Page(str(_pages_dir / "7_research_agent.py"), title="Research Agent", icon=":material/smart_toy:"),
            st.Page(str(_pages_dir / "6_research_feed.py"), title="Research Feed", icon=":material/newspaper:"),
            st.Page(str(_pages_dir / "8_subscriptions.py"), title="Subscriptions", icon=":material/mail:"),
            st.Page(str(_pages_dir / "18_external_databases.py"), title="Databases", icon=":material/database:"),
            st.Page(str(_pages_dir / "1_ingestion.py"), title="Data Ingestion", icon=":material/download:"),
            st.Page(str(_pages_dir / "24_data_pipeline.py"), title="Data Pipeline", icon=":material/sync:"),
        ],
        "Publish": [
            st.Page(str(_pages_dir / "22_researcher.py"), title="Researcher", icon=":material/edit_note:"),
            st.Page(str(_pages_dir / "23_paper_studio.py"), title="Paper Studio", icon=":material/article:"),
            st.Page(str(_pages_dir / "11_validation.py"), title="Validation", icon=":material/verified:"),
            st.Page(str(_pages_dir / "14_entity_library.py"), title="Entity Library", icon=":material/library_books:"),
            st.Page(str(_pages_dir / "16_usd_omniverse.py"), title="USD & Omniverse", icon=":material/public:"),
            st.Page(str(_pages_dir / "17_flywheel_monitor.py"), title="Flywheel Monitor", icon=":material/autorenew:"),
        ],
        "Admin": [
            st.Page(str(_pages_dir / "9_account.py"), title="Account", icon=":material/person:"),
            st.Page(str(_pages_dir / "13_organization.py"), title="Organization", icon=":material/business:"),
            st.Page(str(_pages_dir / "10_security.py"), title="Security", icon=":material/lock:"),
            st.Page(str(_pages_dir / "4_admin.py"), title="Platform Admin", icon=":material/admin_panel_settings:"),
            st.Page(str(_pages_dir / "41_investors.py"), title="Investor Overview", icon=":material/trending_up:"),
        ],
    }
    nav = st.navigation(pages)

    # Sidebar user info (visible on all authenticated pages)
    st.sidebar.markdown(f"**{st.session_state.get('username', 'user')}**")
    st.sidebar.markdown(
        '<span style="color: #f97316; font-weight: 600; font-size: 0.7rem;">BETA</span>'
        ' <span style="font-size: 0.65rem; opacity: 0.5;">Research only</span>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Log out", key="main_logout"):
        auth = _get_auth_manager()
        session_id = st.session_state.get("session_id")
        if session_id:
            auth.logout(session_id)
        for key in ["session_id", "username", "cognito_access_token",
                    "cognito_refresh_token", "cognito_token_expires"]:
            st.session_state.pop(key, None)
        st.rerun()
    st.sidebar.divider()

    nav.run()
    st.stop()  # Navigation handles everything for authenticated users


# ════════════════════════════════════════════════════════════════════
# PUBLIC LANDING PAGE (not logged in — st.navigation not used)
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Cognisom | Human Digital Twin Platform",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
)

if user is None:

    # ── Load assets ───────────────────────────────────────────────
    _app_dir = Path(__file__).resolve().parent
    _pkg_dir = _app_dir.parent  # cognisom package dir
    _root_dir = _pkg_dir.parent  # project root

    # NVIDIA Inception badge — use PNG for reliable st.image() rendering
    _inception_png_path = _pkg_dir / "nvidia-inception-program-badge-rgb-for-screen.png"
    _has_inception_badge = _inception_png_path.exists()

    # Demo video path
    _demo_video_path = _root_dir / "cognisom-demo.mp4"

    # ── Global Styles ─────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Hide default Streamlit header & footer for cleaner look ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Animated gradient background for hero ── */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes rotateHelix {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 170, 0.3); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 170, 0.6), 0 0 40px rgba(0, 212, 170, 0.2); }
    }

    /* ── Hero Section ── */
    .hero-container {
        background: linear-gradient(-45deg, #0a0e27, #1a1f4e, #0d2137, #0a1628);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        border-radius: 16px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
        animation: float 8s ease-in-out infinite;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
        animation: float 10s ease-in-out infinite 2s;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0;
        background: linear-gradient(135deg, #00d4aa, #6366f1, #00d4aa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 5s ease infinite;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }

    .hero-sub {
        font-size: 1.4rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
        animation: slideInLeft 1s ease-out;
    }

    .hero-byline {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.45);
        margin-top: 0.3rem;
        position: relative;
        z-index: 1;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── DNA Helix Particles ── */
    .dna-particles {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }

    .dna-dot {
        position: absolute;
        width: 4px;
        height: 4px;
        background: rgba(0,212,170,0.4);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }

    .dna-dot:nth-child(1) { top: 15%; left: 80%; animation-delay: 0s; background: rgba(99,102,241,0.4); }
    .dna-dot:nth-child(2) { top: 35%; left: 85%; animation-delay: 1s; width: 6px; height: 6px; }
    .dna-dot:nth-child(3) { top: 55%; left: 75%; animation-delay: 2s; background: rgba(99,102,241,0.3); }
    .dna-dot:nth-child(4) { top: 25%; left: 90%; animation-delay: 3s; width: 3px; height: 3px; }
    .dna-dot:nth-child(5) { top: 70%; left: 88%; animation-delay: 0.5s; }
    .dna-dot:nth-child(6) { top: 45%; left: 92%; animation-delay: 1.5s; width: 5px; height: 5px; background: rgba(99,102,241,0.3); }
    .dna-dot:nth-child(7) { top: 80%; left: 82%; animation-delay: 2.5s; width: 3px; height: 3px; }
    .dna-dot:nth-child(8) { top: 10%; left: 72%; animation-delay: 4s; background: rgba(236,72,153,0.3); }

    /* ── Connecting lines between particles ── */
    .dna-line {
        position: absolute;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,170,0.15), transparent);
        animation: pulse 4s ease-in-out infinite;
    }

    .dna-line:nth-child(9) { top: 25%; left: 72%; width: 120px; transform: rotate(25deg); animation-delay: 0.5s; }
    .dna-line:nth-child(10) { top: 50%; left: 78%; width: 100px; transform: rotate(-15deg); animation-delay: 1.5s; }
    .dna-line:nth-child(11) { top: 65%; left: 75%; width: 90px; transform: rotate(40deg); animation-delay: 2.5s; }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--secondary-background-color, rgba(255,255,255,0.03));
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        animation: slideInUp 0.8s ease-out;
        position: relative;
        z-index: 1;
    }

    .metric-card:hover {
        border-color: rgba(0,212,170,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--text-color, rgba(255,255,255,0.55));
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color, #e2e8f0);
        margin-bottom: 0.3rem;
    }

    .section-sub {
        font-size: 1rem;
        color: var(--text-color, rgba(255,255,255,0.5));
        opacity: 0.6;
        margin-bottom: 1.5rem;
    }

    .gradient-line {
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #6366f1, transparent);
        border-radius: 2px;
        margin-bottom: 2rem;
        width: 120px;
    }

    /* ── Capability Cards ── */
    .cap-card {
        background: var(--secondary-background-color, rgba(255,255,255,0.04));
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        height: 100%;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .cap-card:hover {
        border-color: rgba(0,212,170,0.4);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    .cap-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4aa, #6366f1);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .cap-card:hover::before {
        opacity: 1;
    }

    .cap-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        display: block;
    }

    .cap-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-color, #e2e8f0);
        margin-bottom: 0.6rem;
    }

    .cap-text {
        font-size: 0.88rem;
        color: var(--text-color, rgba(255,255,255,0.6));
        opacity: 0.75;
        line-height: 1.6;
    }

    .cap-tag {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        color: #00d4aa;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 4px;
        margin-top: 4px;
    }

    .cap-tag-purple {
        background: rgba(99,102,241,0.1);
        color: #818cf8;
    }

    .cap-tag-pink {
        background: rgba(236,72,153,0.1);
        color: #f472b6;
    }

    /* ── Roadmap Timeline ── */
    .roadmap-item {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.2rem;
        padding: 1rem;
        border-radius: 12px;
        background: var(--secondary-background-color, rgba(255,255,255,0.02));
        border: 1px solid rgba(128,128,128,0.1);
        transition: all 0.3s ease;
    }

    .roadmap-item:hover {
        border-color: rgba(128,128,128,0.25);
    }

    .roadmap-num {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        min-width: 36px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 0.85rem;
        flex-shrink: 0;
    }

    .roadmap-done {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
    }

    .roadmap-progress {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        animation: glow 3s ease-in-out infinite;
    }

    .roadmap-planned {
        background: rgba(99,102,241,0.2);
        color: #818cf8;
        border: 1px solid rgba(99,102,241,0.3);
    }

    .roadmap-name {
        font-weight: 700;
        color: var(--text-color, #e2e8f0);
        font-size: 0.95rem;
    }

    .roadmap-desc {
        color: var(--text-color, rgba(255,255,255,0.5));
        opacity: 0.6;
        font-size: 0.8rem;
        margin-top: 2px;
    }

    .roadmap-badge {
        font-size: 0.65rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-done { background: rgba(16,185,129,0.15); color: #34d399; }
    .badge-progress { background: rgba(245,158,11,0.15); color: #fbbf24; }
    .badge-planned { background: rgba(99,102,241,0.15); color: #818cf8; }

    /* ── Tier Cards ── */
    .tier-card {
        background: var(--secondary-background-color, rgba(255,255,255,0.04));
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.4s ease;
        position: relative;
        height: 100%;
    }

    .tier-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 50px rgba(0,0,0,0.15);
    }

    .tier-featured {
        border-color: rgba(0,212,170,0.4);
    }

    .tier-featured::before {
        content: 'MOST POPULAR';
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #00d4aa, #6366f1);
        color: white;
        font-size: 0.6rem;
        font-weight: 700;
        padding: 3px 12px;
        border-radius: 10px;
        letter-spacing: 1px;
    }

    .tier-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-color, #e2e8f0);
        margin-bottom: 0.5rem;
    }

    .tier-price {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .tier-period {
        font-size: 0.75rem;
        color: var(--text-color, rgba(255,255,255,0.4));
        opacity: 0.5;
        margin-bottom: 1.2rem;
    }

    .tier-features {
        text-align: left;
        font-size: 0.85rem;
        color: var(--text-color, rgba(255,255,255,0.65));
        opacity: 0.75;
        line-height: 1.8;
    }

    .tier-check {
        color: #00d4aa;
        margin-right: 6px;
    }

    /* ── Architecture Diagram ── */
    .arch-container {
        background: linear-gradient(145deg, rgba(10,14,39,0.8), rgba(13,33,55,0.8));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: rgba(0,212,170,0.8);
        overflow-x: auto;
        line-height: 1.6;
    }

    .arch-highlight { color: #818cf8; }
    .arch-accent { color: #f472b6; }
    .arch-dim { color: rgba(255,255,255,0.3); }

    /* ── Video Section ── */
    .video-section {
        background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
    }

    /* ── Stats Banner ── */
    .stats-banner {
        background: var(--secondary-background-color, rgba(0,212,170,0.08));
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .stat-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: #00d4aa;
    }

    .stat-label {
        font-size: 0.7rem;
        color: var(--text-color, rgba(255,255,255,0.5));
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── CTA Section ── */
    .cta-section {
        background: linear-gradient(-45deg, #0a0e27, #1a1f4e, #0d2137);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }

    .cta-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }

    .cta-sub {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* ── Partner logos / badges ── */
    .partner-badge {
        display: inline-block;
        background: var(--secondary-background-color, rgba(255,255,255,0.05));
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        font-size: 0.8rem;
        color: var(--text-color, rgba(255,255,255,0.6));
        opacity: 0.7;
        transition: all 0.3s ease;
    }

    .partner-badge:hover {
        border-color: rgba(0,212,170,0.3);
        opacity: 1;
    }

    /* ── Molecular Visualization Placeholder ── */
    .mol-viz {
        background: radial-gradient(ellipse at center, rgba(0,212,170,0.05) 0%, transparent 70%);
        border-radius: 50%;
        width: 200px;
        height: 200px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }

    .mol-orbit {
        position: absolute;
        border: 1px solid rgba(0,212,170,0.15);
        border-radius: 50%;
        animation: rotateHelix 20s linear infinite;
    }

    .mol-orbit:nth-child(1) { width: 120px; height: 120px; }
    .mol-orbit:nth-child(2) { width: 160px; height: 160px; animation-duration: 25s; animation-direction: reverse; border-color: rgba(99,102,241,0.15); }
    .mol-orbit:nth-child(3) { width: 200px; height: 200px; animation-duration: 30s; border-color: rgba(236,72,153,0.1); }

    .mol-orbit::after {
        content: '';
        position: absolute;
        top: -3px;
        left: 50%;
        width: 6px;
        height: 6px;
        background: #00d4aa;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(0,212,170,0.6);
    }

    .mol-orbit:nth-child(2)::after { background: #6366f1; box-shadow: 0 0 8px rgba(99,102,241,0.6); }
    .mol-orbit:nth-child(3)::after { background: #f472b6; box-shadow: 0 0 8px rgba(236,72,153,0.6); top: auto; bottom: -3px; }

    .mol-center {
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #00d4aa, #6366f1);
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(0,212,170,0.4);
        z-index: 1;
    }

    /* ── Footer ── */
    .nvidia-footer {
        text-align: center;
        padding: 1.5rem 1rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(128,128,128,0.15);
        color: var(--text-color, rgba(255,255,255,0.3));
        opacity: 0.4;
        font-size: 0.7rem;
        line-height: 1.5;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Section ──────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <div class="dna-particles">
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-dot"></div>
            <div class="dna-line"></div>
            <div class="dna-line"></div>
            <div class="dna-line"></div>
        </div>
        <p class="hero-title">Cognisom</p>
        <p class="hero-sub">Personalized Molecular Digital Twin Platform for Precision Oncology</p>
        <p class="hero-byline">by eyentelligence inc.</p>
    </div>
    """, unsafe_allow_html=True)

    # NVIDIA Inception badge — rendered via st.image for reliability
    if _has_inception_badge:
        badge_col1, badge_col2, badge_col3 = st.columns([2, 1, 2])
        with badge_col2:
            st.image(str(_inception_png_path), width=200)

    # ── Development Disclaimer ────────────────────────────────────
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(234,179,8,0.05));
                border: 1px solid rgba(234,179,8,0.3); border-radius: 10px;
                padding: 0.8rem 1.2rem; margin: 1rem auto; max-width: 700px; text-align: center;">
        <span style="color: #eab308; font-weight: 600;">Under Active Development</span>
        <span style="font-size: 0.85rem; opacity: 0.7;">
            &mdash; This application is intended for designated testers and research collaborators only.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Demo Video (uncomment when YouTube account is active) ────
    # vid_col1, vid_col2, vid_col3 = st.columns([1, 3, 1])
    # with vid_col2:
    #     st.markdown("""
    #     <div style="border-radius: 12px; overflow: hidden; border: 1px solid rgba(128,128,128,0.15);">
    #         <iframe width="100%" height="400"
    #             src="https://www.youtube.com/embed/YOUR_VIDEO_ID?rel=0"
    #             title="Cognisom Platform Overview"
    #             frameborder="0" allowfullscreen></iframe>
    #     </div>
    #     """, unsafe_allow_html=True)

    # ── Hero Metrics Row ──────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">700+</div>
            <div class="metric-label">Actionable Genes (OncoKB)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">14,847</div>
            <div class="metric-label">HLA Alleles (MHCflurry)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">Treatment Regimens</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">MAD Agent Specialists</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">429</div>
            <div class="metric-label">Patients Validated (SU2C)</div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">4 GPU</div>
            <div class="metric-label">Parabricks Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── What is Cognisom ──────────────────────────────────────────
    st.markdown("""
    <div class="section-header">What is Cognisom?</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    col_what1, col_what2 = st.columns([3, 2])

    with col_what1:
        st.markdown("""
Cognisom is an **AI precision-oncology intelligence and evidence platform** that
combines matched patient germline DNA + tumor DNA, real-time clinical evidence,
and multi-agent AI to help clinicians rank therapies and match clinical trials.

**From raw DNA to treatment recommendation:**
Upload matched tumor-normal sequencing (FASTQ) and Cognisom runs the full
**GPU-accelerated Parabricks pipeline** — alignment, variant calling, HLA typing,
neoantigen prediction — then feeds results through a **3-agent Molecular AI
Decision (MAD) Board** that produces an explainable, evidence-traced recommendation.

The MAD Board uses **OncoKB** (700+ actionable genes), **MHCflurry** (14,847 HLA alleles),
**ClinicalTrials.gov** (live trial matching), and published trial evidence to generate
recommendations that a clinician can independently review — designed for the
**FDA's 2026 AI Credibility Framework** as a Non-Device CDS.

Validated against 429 real mCRPC patients from the SU2C/PCF 2019 cohort.
        """)

    with col_what2:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; padding: 1rem;">
            <div class="mol-viz">
                <div class="mol-orbit"></div>
                <div class="mol-orbit"></div>
                <div class="mol-orbit"></div>
                <div class="mol-center"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Technology Partners Banner ────────────────────────────────
    if _has_inception_badge:
        badge_b1, badge_b2, badge_b3 = st.columns([2, 1, 2])
        with badge_b2:
            st.image(str(_inception_png_path), width=180)

    st.markdown("""
    <div class="stats-banner">
        <span class="partner-badge">NVIDIA Parabricks</span>
        <span class="partner-badge">AWS HealthOmics</span>
        <span class="partner-badge">OncoKB (MSK)</span>
        <span class="partner-badge">MHCflurry</span>
        <span class="partner-badge">ClinicalTrials.gov</span>
        <span class="partner-badge">SU2C / PCF</span>
        <span class="partner-badge">NVIDIA BioNeMo</span>
        <span class="partner-badge">OpenUSD</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Core Capabilities ─────────────────────────────────────────
    st.markdown("""
    <div class="section-header">Core Capabilities</div>
    <div class="section-sub">From patient DNA to personalized treatment prediction</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    cap1, cap2, cap3 = st.columns(3)

    with cap1:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#129516;</span>
            <div class="cap-title">Tumor DNA Analysis</div>
            <div class="cap-text">
                Upload patient VCF files to identify cancer driver mutations
                across 14 genes (AR, TP53, PTEN, BRCA1/2, SPOP, and more).
                Automated variant annotation with protein impact prediction,
                tumor mutational burden (TMB), and microsatellite instability (MSI).
                <br><br>
                Mutation-informed therapy guidance: immunotherapy eligibility,
                PARP inhibitor sensitivity, and AR-targeted therapy selection.
            </div>
            <br>
            <span class="cap-tag">14 Driver Genes</span>
            <span class="cap-tag">TMB / MSI</span>
            <span class="cap-tag">VCF Ingestion</span>
            <span class="cap-tag">Variant Annotation</span>
        </div>
        """, unsafe_allow_html=True)

    with cap2:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#129514;</span>
            <div class="cap-title">Immune Landscape Profiling</div>
            <div class="cap-text">
                Single-cell resolution immune characterization of the tumor
                microenvironment. T-cell exhaustion scoring with PD-1/TIM-3/LAG-3
                co-expression analysis. M1 vs M2 macrophage polarization.
                <br><br>
                Spatial transcriptomics maps immune infiltration patterns,
                identifies hot zones via Moran's I autocorrelation, and
                classifies tumor core, stroma, and immune border regions.
            </div>
            <br>
            <span class="cap-tag cap-tag-purple">T-Cell Exhaustion</span>
            <span class="cap-tag cap-tag-purple">Macrophage M1/M2</span>
            <span class="cap-tag cap-tag-purple">Spatial TME</span>
        </div>
        """, unsafe_allow_html=True)

    with cap3:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#128300;</span>
            <div class="cap-title">Personalized Digital Twin</div>
            <div class="cap-text">
                Combines genomic profile + immune landscape to create a
                patient-specific computational model. Simulates 7 treatment
                regimens: pembrolizumab, nivolumab, ipilimumab, dual checkpoint
                blockade, olaparib, enzalutamide, and combination therapies.
                <br><br>
                RECIST-like response classification, tumor dynamics curves,
                immune-related adverse event (irAE) risk, and survival projections.
            </div>
            <br>
            <span class="cap-tag cap-tag-pink">7 Therapies</span>
            <span class="cap-tag cap-tag-pink">RECIST Response</span>
            <span class="cap-tag cap-tag-pink">irAE Risk</span>
            <span class="cap-tag cap-tag-pink">Survival Curves</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Additional Capabilities Row ────────────────────────────────
    cap4, cap5, cap6 = st.columns(3)

    with cap4:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#9881;</span>
            <div class="cap-title">Multi-Scale Simulation</div>
            <div class="cap-text">
                9 physics modules: cellular dynamics, immune surveillance,
                vascular O2, lymphatic metastasis, epigenetics, circadian rhythms,
                morphogen gradients, molecular interactions, receptor signaling.
                GPU-accelerated with NVIDIA Warp. Multi-GPU tissue-scale (100K-5M cells).
            </div>
            <br>
            <span class="cap-tag">9 Modules</span>
            <span class="cap-tag">GPU Warp</span>
            <span class="cap-tag">Multi-GPU</span>
        </div>
        """, unsafe_allow_html=True)

    with cap5:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#128172;</span>
            <div class="cap-title">AI Drug Discovery</div>
            <div class="cap-text">
                11 NVIDIA BioNeMo NIM endpoints: molecule generation (MolMIM, GenMol),
                protein design (RFdiffusion, ProteinMPNN), docking (DiffDock),
                structure prediction (OpenFold3, Boltz-2, AlphaFold2), and
                genomic modeling (Evo2). Target to lead compound in one workflow.
            </div>
            <br>
            <span class="cap-tag cap-tag-purple">11 NIMs</span>
            <span class="cap-tag cap-tag-purple">DiffDock</span>
            <span class="cap-tag cap-tag-purple">Evo2</span>
        </div>
        """, unsafe_allow_html=True)

    with cap6:
        st.markdown("""
        <div class="cap-card">
            <span class="cap-icon">&#127912;</span>
            <div class="cap-title">RTX 3D Visualization</div>
            <div class="cap-text">
                Real-time RTX rendering via NVIDIA Isaac Sim and OpenUSD.
                Bio-USD schema (16 prim types, 99-entity catalog).
                3Dmol.js protein viewer, Three.js tissue visualization,
                MJPEG streaming, and leukocyte diapedesis simulation.
            </div>
            <br>
            <span class="cap-tag cap-tag-pink">Omniverse</span>
            <span class="cap-tag cap-tag-pink">Bio-USD</span>
            <span class="cap-tag cap-tag-pink">RTX Path Tracing</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Vision Section ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">The Vision</div>
    <div class="section-sub">Building the future of computational oncology</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("""
**The Intelligence Layer of Precision Oncology**

Immunotherapy works for ~25% of patients. The other 75% face
trial-and-error medicine because current tools can't reconcile the
immune system's complexity with each patient's unique tumor biology.

**What Cognisom delivers today:**
- **GPU variant calling** — Parabricks on HealthOmics, real 30x WGS
- **OncoKB evidence** — 700+ actionable genes with FDA-level annotations
- **MHCflurry binding** — Neural network, 14,847 HLA alleles, >90% accuracy
- **MAD Agent Board** — 3 specialist agents, evidence-traced consensus
- **Clinical trials** — Live ClinicalTrials.gov matching
- **429-patient validation** — SU2C mCRPC cohort, real outcomes
        """)

    with vis_col2:
        st.markdown("""
**From Raw DNA to Treatment Recommendation**

The full evidence-traced pipeline:

1. Upload matched tumor + normal FASTQ to S3
2. Parabricks GPU alignment + variant calling (~90 min)
3. OncoKB variant actionability (700+ genes, evidence levels 1-4)
4. OptiType HLA typing from germline reads
5. MHCflurry neoantigen binding prediction (14,847 alleles)
6. MAD Board: 3 agents deliberate → consensus recommendation
7. ClinicalTrials.gov: matching recruiting trials
8. Traceable evidence chain for clinician independent review

Designed for the **FDA 2026 AI Credibility Framework**
as a **Non-Device Clinical Decision Support** tool.
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 10-Phase Roadmap ───────────────────────────────────────────
    st.markdown("""
    <div class="section-header">Roadmap</div>
    <div class="section-sub">10 phases from simulation engine to clinically validated personalized digital twins</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    phases = [
        ("Deploy & Stabilize", "Live platform on AWS with Cognito auth, TLS, split architecture", "Done"),
        ("Genomic Digital Twin", "VCF ingestion, 14 cancer driver genes, variant annotation, TMB/MSI", "Done"),
        ("Immune Landscape", "Cell2Sentence, T-cell exhaustion, macrophage polarization, spatial TME", "Done"),
        ("Treatment Simulation", "9 therapy regimens including neoantigen mRNA vaccine + combinations", "Done"),
        ("GPU Parabricks Pipeline", "HealthOmics DeepVariant on real 30x WGS, 4-GPU variant calling", "Done"),
        ("Clinical Intelligence", "OncoKB (700+ genes), MHCflurry (14,847 alleles), ClinicalTrials.gov", "Done"),
        ("MAD Agent Board", "3-agent consensus (Genomics, Immune, Clinical) with FDA CDS compliance", "Done"),
        ("429-Patient Validation", "SU2C mCRPC cohort: TMB r=0.987, 100% biomarker concordance", "Done"),
        ("Matched Tumor-Normal", "Full FASTQ-to-recommendation on real patient data (SEQC2 benchmark)", "In Progress"),
        ("FDA Credibility Dossier", "7-step framework: COU, model cards, provenance, audit trail", "In Progress"),
    ]

    phase_col1, phase_col2 = st.columns(2)

    for i, (name, desc, status) in enumerate(phases):
        col = phase_col1 if i < 5 else phase_col2
        if status == "Done":
            num_class = "roadmap-done"
            badge_class = "badge-done"
        elif status == "In Progress":
            num_class = "roadmap-progress"
            badge_class = "badge-progress"
        else:
            num_class = "roadmap-planned"
            badge_class = "badge-planned"

        with col:
            st.markdown(f"""
            <div class="roadmap-item">
                <div class="roadmap-num {num_class}">{i+1}</div>
                <div>
                    <div class="roadmap-name">{name} <span class="roadmap-badge {badge_class}">{status}</span></div>
                    <div class="roadmap-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Subscription Tiers ────────────────────────────────────────
    st.markdown("""
    <div class="section-header">Subscription Plans</div>
    <div class="section-sub">Start free. Scale as your research grows. No credit card required.</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown("""
        <div class="tier-card">
            <div class="tier-name">Free</div>
            <div class="tier-price">$0</div>
            <div class="tier-period">forever</div>
            <div class="tier-features">
                <span class="tier-check">&#10003;</span> 1 user<br>
                <span class="tier-check">&#10003;</span> Research feed<br>
                <span class="tier-check">&#10003;</span> Platform overview<br>
                <span class="tier-check">&#10003;</span> Community support
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="tier-card tier-featured">
            <div class="tier-name">Researcher</div>
            <div class="tier-price">Contact Us</div>
            <div class="tier-period">per month</div>
            <div class="tier-features">
                <span class="tier-check">&#10003;</span> Up to 3 users<br>
                <span class="tier-check">&#10003;</span> Full simulation engine<br>
                <span class="tier-check">&#10003;</span> Drug discovery pipeline<br>
                <span class="tier-check">&#10003;</span> Molecular lab + 3D viewer<br>
                <span class="tier-check">&#10003;</span> Research agent<br>
                <span class="tier-check">&#10003;</span> API access
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="tier-card">
            <div class="tier-name">Institution</div>
            <div class="tier-price">Contact Us</div>
            <div class="tier-period">per month</div>
            <div class="tier-features">
                <span class="tier-check">&#10003;</span> Up to 25 users<br>
                <span class="tier-check">&#10003;</span> Everything in Researcher<br>
                <span class="tier-check">&#10003;</span> Admin & security tools<br>
                <span class="tier-check">&#10003;</span> Validation suite<br>
                <span class="tier-check">&#10003;</span> 3D visualization<br>
                <span class="tier-check">&#10003;</span> Org management
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t4:
        st.markdown("""
        <div class="tier-card">
            <div class="tier-name">Enterprise</div>
            <div class="tier-price">Custom</div>
            <div class="tier-period">pricing</div>
            <div class="tier-features">
                <span class="tier-check">&#10003;</span> Unlimited users<br>
                <span class="tier-check">&#10003;</span> Everything in Institution<br>
                <span class="tier-check">&#10003;</span> GPU acceleration<br>
                <span class="tier-check">&#10003;</span> Priority support<br>
                <span class="tier-check">&#10003;</span> Custom integrations<br>
                <span class="tier-check">&#10003;</span> On-premise deployment
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Architecture ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">Platform Architecture</div>
    <div class="section-sub">From patient DNA to personalized treatment prediction</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-container">
<span class="arch-dim">GPU PIPELINE (NVIDIA Parabricks on AWS HealthOmics)</span>
Tumor FASTQ <span class="arch-dim">──┐</span>
             <span class="arch-dim">├──></span> <span class="arch-highlight">Parabricks fq2bam</span> <span class="arch-dim">──></span> <span class="arch-highlight">DeepVariant / Mutect2</span> <span class="arch-dim">──></span> Somatic VCF
Normal FASTQ <span class="arch-dim">─┘</span>  (4 GPUs, 48 CPUs, 179 GB)                    <span class="arch-dim">│</span>
                                                                  <span class="arch-dim">v</span>
<span class="arch-dim">CLINICAL INTELLIGENCE LAYER</span>
  VCF <span class="arch-dim">──></span> Variant Annotator <span class="arch-dim">──></span> <span class="arch-highlight">OncoKB (700+ genes)</span> <span class="arch-dim">────────────────┐</span>
  Normal BAM <span class="arch-dim">──></span> OptiType <span class="arch-dim">──></span> <span class="arch-highlight">HLA Typing (6 alleles)</span> <span class="arch-dim">────────────┤</span>
  Mutations <span class="arch-dim">──></span> <span class="arch-highlight">MHCflurry (14,847 alleles)</span> <span class="arch-dim">──></span> Neoantigen Binding <span class="arch-dim">─┤</span>
  Biomarkers <span class="arch-dim">──></span> <span class="arch-highlight">ClinicalTrials.gov API</span> <span class="arch-dim">──></span> Trial Matching <span class="arch-dim">──────┘</span>
                                                                  <span class="arch-dim">│</span>
<span class="arch-dim">MAD BOARD (Multi-Agent Decision)</span>                                  <span class="arch-dim">v</span>
  <span class="arch-accent">Genomics Agent</span> <span class="arch-dim">──┐</span>
  <span class="arch-accent">Immune Agent</span> <span class="arch-dim">────┤──></span> Board Moderator <span class="arch-dim">──></span> <span class="arch-highlight">Consensus Recommendation</span>
  <span class="arch-accent">Clinical Agent</span> <span class="arch-dim">──┘</span>     + Evidence Chain + Dissent + Audit Trail
                                                                  <span class="arch-dim">│</span>
<span class="arch-dim">DIGITAL TWIN</span>                                                     <span class="arch-dim">v</span>
  Genomic Profile + Immune Landscape <span class="arch-dim">──></span> Treatment Simulator (9 regimens)
    <span class="arch-dim">──></span> RECIST Response + Survival + irAE Risk + Neoantigen Vaccine Design

<span class="arch-dim">FDA COMPLIANCE (Non-Device CDS, 2026 Credibility Framework)</span>
  Context of Use <span class="arch-dim">│</span> Model Cards <span class="arch-dim">│</span> Data Provenance <span class="arch-dim">│</span> Audit Trail
  429-patient SU2C validation <span class="arch-dim">│</span> TMB r=0.987 <span class="arch-dim">│</span> 100% biomarker concordance
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CTA / Get Started ─────────────────────────────────────────
    st.markdown("""
    <div class="cta-section">
        <div class="cta-title">Ready to Accelerate Your Research?</div>
        <div class="cta-sub">
            Create an organization to start running simulations,
            or join your team with an invite code.
        </div>
        <div style="margin-top: 0.8rem; font-size: 0.8rem; color: rgba(255,255,255,0.4);">
            This application is under active development and is intended for designated testers
            and research collaborators only. Not for clinical use.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer (rendered before login gate since st.stop() halts execution) ──
    from cognisom.dashboard.footer import render_footer, VERSION, BUILD_DATE
    if _has_inception_badge:
        footer_b1, footer_b2, footer_b3 = st.columns([3, 1, 3])
        with footer_b2:
            st.image(str(_inception_png_path), width=140)

    st.markdown(f"""
    <div class="nvidia-footer">
        eyentelligence inc. | Cognisom v{VERSION} | Built {BUILD_DATE}<br>
        NVIDIA Inception Program Member<br>
        &copy; 2026 NVIDIA, the NVIDIA logo, and NVIDIA Inception are trademarks
        and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries.
    </div>
    """, unsafe_allow_html=True)

    # ── Sign In / Register ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">Sign In</div>
    <div class="section-sub">Log in to access the platform, or create an account to get started</div>
    <div class="gradient-line"></div>
    """, unsafe_allow_html=True)

    from cognisom.auth.middleware import streamlit_login_gate
    streamlit_login_gate()


# ════════════════════════════════════════════════════════════════════
# (Authenticated users are handled above by st.navigation + st.stop)
# ════════════════════════════════════════════════════════════════════
