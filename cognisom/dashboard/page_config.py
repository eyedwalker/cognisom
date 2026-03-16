"""
Safe page config wrapper.

When using st.navigation(), the main app.py calls st.set_page_config() once.
Individual pages should use safe_set_page_config() instead, which silently
skips if page config has already been set.
"""

import streamlit as st


def safe_set_page_config(**kwargs):
    """Call st.set_page_config() only if it hasn't been called yet."""
    try:
        st.set_page_config(**kwargs)
    except st.errors.StreamlitAPIException:
        pass  # Already set by navigation
