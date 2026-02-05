"""Security & User Management – admin-only user and session management."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Security | Cognisom", page_icon="🔒", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate
user = streamlit_page_gate("10_security")

from cognisom.auth import AuthManager, UserRole
from cognisom.auth.middleware import get_current_user

# Require admin role
if user.role not in (UserRole.ADMIN, UserRole.ORG_ADMIN):
    st.error("This page is restricted to **admin** users.")
    st.stop()

# ── Shared auth manager ─────────────────────────────────────────────

AUTH_DIR = Path(_project_root) / "data" / "auth"


@st.cache_resource
def get_auth() -> AuthManager:
    return AuthManager(data_dir=str(AUTH_DIR))


auth = get_auth()

# ── Page header ─────────────────────────────────────────────────────

st.title("Security & User Management")
st.markdown(f"Logged in as **{user.display_name}** (admin)")

tab_users, tab_sessions, tab_audit, tab_settings = st.tabs([
    "Users", "Active Sessions", "Security Audit", "Settings",
])

# ====================================================================
# TAB 1: User Management
# ====================================================================

with tab_users:
    st.subheader("Registered Users")

    users = auth.list_users()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total users", len(users))
    col2.metric("Active", sum(1 for u in users if u.is_active))
    col3.metric("Admins", sum(1 for u in users if u.role == UserRole.ADMIN))
    col4.metric("Researchers", sum(1 for u in users if u.role == UserRole.RESEARCHER))

    st.markdown("---")

    # User table
    for u in users:
        with st.expander(
            f"{'🟢' if u.is_active else '🔴'} **{u.username}** — {u.role.value} — {u.email}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Display name:** {u.display_name}")
            c2.markdown(f"**Role:** {u.role.value}")
            c3.markdown(f"**Active:** {'Yes' if u.is_active else 'No'}")

            if u.created_at:
                st.caption(f"Created: {datetime.fromtimestamp(u.created_at):%Y-%m-%d %H:%M}")
            if u.last_login:
                st.caption(f"Last login: {datetime.fromtimestamp(u.last_login):%Y-%m-%d %H:%M}")

            # Actions
            act1, act2, act3 = st.columns(3)

            with act1:
                new_role = st.selectbox(
                    "Change role",
                    [r.value for r in UserRole],
                    index=[r.value for r in UserRole].index(u.role.value),
                    key=f"role_{u.username}",
                )
                if new_role != u.role.value:
                    if st.button("Apply role", key=f"apply_role_{u.username}"):
                        auth.update_role(u.username, UserRole(new_role))
                        st.success(f"Role updated to {new_role}")
                        st.rerun()

            with act2:
                if u.is_active and u.username != "admin":
                    if st.button("Deactivate", key=f"deact_{u.username}", type="secondary"):
                        auth.deactivate_user(u.username)
                        st.warning(f"Deactivated {u.username}")
                        st.rerun()
                elif not u.is_active:
                    if st.button("Activate", key=f"act_{u.username}", type="primary"):
                        auth.activate_user(u.username)
                        st.success(f"Activated {u.username}")
                        st.rerun()

            with act3:
                if u.api_key:
                    st.caption(f"API key: {u.api_key[:12]}…")

    # Create new user
    st.markdown("---")
    st.subheader("Create New User")
    with st.form("admin_create_user"):
        new_username = st.text_input("Username", key="admin_new_user")
        new_email = st.text_input("Email", key="admin_new_email")
        new_display = st.text_input("Display name", key="admin_new_display")
        new_password = st.text_input("Password", type="password", key="admin_new_pw")
        new_role = st.selectbox("Role", [r.value for r in UserRole], index=1, key="admin_new_role")
        create_submit = st.form_submit_button("Create User", type="primary")

    if create_submit:
        ok, msg = auth.register(
            new_username, new_email, new_password,
            role=UserRole(new_role),
            display_name=new_display,
        )
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

# ====================================================================
# TAB 2: Active Sessions
# ====================================================================

with tab_sessions:
    st.subheader("Active Sessions")

    sessions = auth.active_sessions()
    st.metric("Active sessions", len(sessions))

    if sessions:
        for s in sessions:
            with st.expander(f"**{s.username}** — expires {datetime.fromtimestamp(s.expires_at):%Y-%m-%d %H:%M}"):
                st.markdown(f"- **Session ID:** `{s.session_id[:16]}…`")
                st.markdown(f"- **Created:** {datetime.fromtimestamp(s.created_at):%Y-%m-%d %H:%M}")
                st.markdown(f"- **Expires:** {datetime.fromtimestamp(s.expires_at):%Y-%m-%d %H:%M}")
                if s.ip_address:
                    st.markdown(f"- **IP:** {s.ip_address}")

                if st.button("Revoke session", key=f"revoke_{s.session_id[:16]}"):
                    auth.logout(s.session_id)
                    st.warning("Session revoked.")
                    st.rerun()
    else:
        st.info("No active sessions.")

# ====================================================================
# TAB 3: Security Audit
# ====================================================================

with tab_audit:
    st.subheader("Security Audit")
    st.markdown("Quick checks for common security issues.")

    import os

    checks = []

    # 1. Debug mode
    rest_server = Path(_project_root) / "api" / "rest_server.py"
    if rest_server.exists():
        content = rest_server.read_text()
        debug_on = "debug=True" in content
        checks.append(("Flask debug mode OFF", not debug_on,
                        "CRITICAL: debug=True found in rest_server.py — allows remote code execution"))
    else:
        checks.append(("Flask rest_server.py exists", False, "File not found"))

    # 2. .env not in git
    env_file = Path(_project_root) / ".env"
    gitignore = Path(_project_root) / ".gitignore"
    env_in_gitignore = False
    if gitignore.exists():
        env_in_gitignore = ".env" in gitignore.read_text()
    checks.append((".env in .gitignore", env_in_gitignore,
                    "Ensure .env is not committed to version control"))

    # 3. NVIDIA API key set
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    checks.append(("NVIDIA API key configured", bool(api_key) and not api_key.startswith("your-"),
                    "Set NVIDIA_API_KEY environment variable"))

    # 4. Default admin password changed
    admin_user = auth.get_user("admin")
    default_pw_still = admin_user.check_password("Admin1234!") if admin_user else False
    checks.append(("Default admin password changed", not default_pw_still,
                    "Change the default admin password immediately"))

    # 5. CORS configuration
    if rest_server.exists():
        content = rest_server.read_text()
        open_cors = "CORS(app)" in content and "origins" not in content
        checks.append(("CORS restricted to specific origins", not open_cors,
                        "CORS(app) allows any origin — restrict to your domain"))

    # 6. Auth data directory permissions
    auth_dir = Path(_project_root) / "data" / "auth"
    checks.append(("Auth data directory exists", auth_dir.exists(),
                    "Authentication data storage"))

    # 7. Users exist
    checks.append(("Users registered", len(auth.list_users()) > 0,
                    "At least one user account exists"))

    # Display results
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    st.progress(passed / total, text=f"{passed}/{total} checks passed")

    for label, ok, detail in checks:
        if ok:
            st.markdown(f"✅ **{label}**")
        else:
            st.markdown(f"❌ **{label}**")
            st.caption(f"   → {detail}")

# ====================================================================
# TAB 4: Settings
# ====================================================================

with tab_settings:
    st.subheader("Security Settings")

    st.markdown("**Session lifetime**")
    st.markdown("Current: 24 hours (configured in `auth_manager.py`)")

    st.markdown("---")
    st.markdown("**Password policy**")
    st.markdown("- Minimum 8 characters\n- At least 1 uppercase letter\n- At least 1 digit")

    st.markdown("---")
    st.markdown("**Role permissions**")
    from cognisom.auth.models import ROLE_PERMISSIONS
    for role, perms in ROLE_PERMISSIONS.items():
        with st.expander(f"**{role.value.title()}** ({len(perms)} permissions)"):
            for p in sorted(perms):
                st.markdown(f"- `{p}`")

    st.markdown("---")
    st.markdown("**Deployment checklist**")
    st.markdown("""
1. Change the default admin password
2. Set `FLASK_DEBUG=false` in environment
3. Configure CORS to your domain only
4. Use HTTPS (TLS termination via nginx/ALB)
5. Set `SECRET_KEY` environment variable
6. Rotate NVIDIA API keys if exposed
7. Enable rate limiting on API endpoints
8. Back up `data/auth/` directory regularly
""")

# Footer
from cognisom.dashboard.footer import render_footer
render_footer()
