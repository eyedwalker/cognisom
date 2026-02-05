"""Account – registration, login, profile, and API key management."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Account | Cognisom", page_icon="👤", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("9_account")

from cognisom.auth import AuthManager, UserRole
from cognisom.auth.middleware import get_current_user

# ── Shared auth manager ─────────────────────────────────────────────

AUTH_DIR = Path(_project_root) / "data" / "auth"


@st.cache_resource
def get_auth() -> AuthManager:
    return AuthManager(data_dir=str(AUTH_DIR))


auth = get_auth()

# ── Page header ─────────────────────────────────────────────────────

st.title("Account")

user = get_current_user()

# ── Not logged in: show login + registration ────────────────────────

if user is None:
    tab_login, tab_register = st.tabs(["Log In", "Register"])

    with tab_login:
        st.subheader("Log In")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary")

        if submitted:
            if not username or not password:
                st.error("Username and password are required.")
            else:
                session, msg = auth.login(username.strip(), password)
                if session:
                    st.session_state["session_id"] = session.session_id
                    st.session_state["username"] = session.username
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("---")
        st.caption("Default admin account: `admin` / `Admin1234!` (change this after first login)")

    with tab_register:
        st.subheader("Create Account")
        with st.form("register_form"):
            reg_username = st.text_input("Username (3+ chars)", key="reg_user")
            reg_email = st.text_input("Email", key="reg_email")
            reg_display = st.text_input("Display name (optional)", key="reg_display")
            reg_password = st.text_input("Password (8+ chars, 1 uppercase, 1 digit)", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            reg_submit = st.form_submit_button("Register", type="primary")

        if reg_submit:
            if reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = auth.register(
                    reg_username, reg_email, reg_password,
                    role=UserRole.RESEARCHER,
                    display_name=reg_display,
                )
                if ok:
                    st.success(f"{msg} You can now log in.")
                else:
                    st.error(msg)

        st.markdown("---")
        st.markdown("**Password requirements:**")
        st.markdown("- At least 8 characters\n- At least 1 uppercase letter\n- At least 1 digit")

# ── Logged in: show profile ─────────────────────────────────────────

else:
    st.success(f"Logged in as **{user.display_name}** ({user.role.value})")

    tab_profile, tab_security, tab_api = st.tabs(["Profile", "Security", "API Key"])

    # ── Profile ──
    with tab_profile:
        st.subheader("Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Username", user.username)
        col2.metric("Role", user.role.value.title())
        col3.metric("Email", user.email)

        st.markdown(f"**Display name:** {user.display_name}")
        if user.created_at:
            st.markdown(f"**Joined:** {datetime.fromtimestamp(user.created_at):%Y-%m-%d %H:%M}")
        if user.last_login:
            st.markdown(f"**Last login:** {datetime.fromtimestamp(user.last_login):%Y-%m-%d %H:%M}")

        st.markdown("---")

        # Permissions
        st.markdown("**Your permissions:**")
        from cognisom.auth.models import ROLE_PERMISSIONS
        perms = ROLE_PERMISSIONS.get(user.role, [])
        for p in sorted(perms):
            st.markdown(f"- `{p}`")

    # ── Security ──
    with tab_security:
        st.subheader("Change Password")
        with st.form("change_pw_form"):
            old_pw = st.text_input("Current password", type="password", key="old_pw")
            new_pw = st.text_input("New password", type="password", key="new_pw")
            new_pw2 = st.text_input("Confirm new password", type="password", key="new_pw2")
            pw_submit = st.form_submit_button("Change Password", type="primary")

        if pw_submit:
            if new_pw != new_pw2:
                st.error("New passwords do not match.")
            else:
                ok, msg = auth.change_password(user.username, old_pw, new_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

        st.markdown("---")
        if st.button("Log Out", type="secondary"):
            session_id = st.session_state.get("session_id")
            if session_id:
                auth.logout(session_id)
            # Clear all auth-related session state (local + Cognito)
            for key in ["session_id", "username", "cognito_access_token",
                        "cognito_refresh_token", "cognito_token_expires"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── API Key ──
    with tab_api:
        st.subheader("Personal API Key")
        st.markdown(
            "Use this key for programmatic access to the Cognisom API. "
            "Include it as `X-API-Key` header or `api_key` query parameter."
        )

        if user.api_key:
            st.code(user.api_key, language=None)
        else:
            st.info("No API key generated yet.")

        if st.button("Regenerate API Key", type="primary"):
            new_key = auth.regenerate_api_key(user.username)
            if new_key:
                st.success("New API key generated. Copy it now — it won't be shown again after page reload.")
                st.code(new_key, language=None)
            else:
                st.error("Failed to generate API key.")

        st.markdown("---")
        st.markdown("**API usage example:**")
        st.code(
            'curl -H "X-API-Key: YOUR_KEY" http://localhost:5000/api/health',
            language="bash",
        )

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
