"""
Shared Footer Component
=======================

Displays version number, date, and NVIDIA Inception badge on all pages.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import os

# Version info - can be overridden by environment variables from CI/CD
VERSION = os.environ.get("COGNISOM_VERSION", "0.3.0")
BUILD_DATE = os.environ.get("COGNISOM_BUILD_DATE", datetime.now().strftime("%Y-%m-%d"))
GIT_SHA = os.environ.get("COGNISOM_GIT_SHA", "dev")[:7]  # Short SHA

# NVIDIA Inception badge path
_pkg_dir = Path(__file__).resolve().parent.parent
_inception_png_path = _pkg_dir / "nvidia-inception-program-badge-rgb-for-screen.png"


def render_footer():
    """Render the standard Cognisom footer with version, date, and NVIDIA Inception badge."""

    st.markdown("---")

    # NVIDIA Inception badge
    if _inception_png_path.exists():
        col1, col2, col3 = st.columns([3, 1, 3])
        with col2:
            st.image(str(_inception_png_path), width=120)

    # Footer text with version and date
    sha_display = f" ({GIT_SHA})" if GIT_SHA != "dev" else ""
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
        <strong>Cognisom</strong> v{VERSION}{sha_display} | Built {BUILD_DATE}<br>
        eyentelligence inc. | NVIDIA Inception Program Member<br>
        <span style="font-size: 0.65rem; color: rgba(255,255,255,0.25);">
        NVIDIA, the NVIDIA logo, and NVIDIA Inception are trademarks of NVIDIA Corporation.
        </span>
    </div>
    """, unsafe_allow_html=True)
