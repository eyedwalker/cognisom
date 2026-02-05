"""Authentication middleware for Flask API and Streamlit dashboard.

Supports both:
1. Local auth (JSON file-based) - for development
2. AWS Cognito - for production (HIPAA-compliant)

The auth backend is selected based on environment variables:
- COGNITO_USER_POOL_ID + COGNITO_CLIENT_ID = Cognito
- Otherwise = Local auth
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

# ── Shared auth instances ────────────────────────────────────────────

_auth_manager = None
_cognito_provider = None
_org_manager = None


def _get_cognito_provider():
    """Lazy-load Cognito provider if configured."""
    global _cognito_provider
    if _cognito_provider is None:
        # Only import if Cognito is configured
        if os.environ.get("COGNITO_USER_POOL_ID") and os.environ.get("COGNITO_CLIENT_ID"):
            try:
                from .cognito_provider import CognitoAuthProvider
                _cognito_provider = CognitoAuthProvider()
            except ImportError:
                _cognito_provider = None
    return _cognito_provider


def _get_auth_manager():
    """Lazy-load a shared AuthManager."""
    global _auth_manager
    if _auth_manager is None:
        from .auth_manager import AuthManager

        data_dir = os.environ.get(
            "COGNISOM_AUTH_DIR",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "auth"),
        )
        _auth_manager = AuthManager(data_dir=data_dir)
    return _auth_manager


def _get_org_manager():
    """Lazy-load a shared OrgManager."""
    global _org_manager
    if _org_manager is None:
        from .org_manager import OrgManager

        data_dir = os.environ.get(
            "COGNISOM_AUTH_DIR",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "auth"),
        )
        _org_manager = OrgManager(data_dir=data_dir)
    return _org_manager


def is_cognito_enabled() -> bool:
    """Check if Cognito auth is enabled."""
    provider = _get_cognito_provider()
    return provider is not None and provider.enabled


# ── Flask decorators ─────────────────────────────────────────────────


def require_auth(f):
    """Flask route decorator: require valid session, API key, or Cognito token.

    Checks:
    1. ``Authorization: Bearer <token>`` header (Cognito JWT or session ID)
    2. ``X-API-Key: <api_key>`` header
    3. ``api_key`` query parameter

    Sets ``g.current_user`` on success, returns 401 on failure.
    """

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        from flask import request, g, jsonify

        auth = _get_auth_manager()
        cognito = _get_cognito_provider()
        user = None

        # Check Bearer token (Cognito JWT or local session)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # Try Cognito first if enabled (JWTs are longer than session IDs)
            if cognito and cognito.enabled and len(token) > 100:
                user = cognito.validate_token(token)

            # Fall back to local session
            if user is None:
                user = auth.validate_session(token)

        # Check API key header
        if user is None:
            api_key = request.headers.get("X-API-Key", "")
            if api_key:
                user = auth.validate_api_key(api_key)

        # Check query param
        if user is None:
            api_key = request.args.get("api_key", "")
            if api_key:
                user = auth.validate_api_key(api_key)

        if user is None:
            return jsonify({"error": "Authentication required"}), 401

        g.current_user = user
        return f(*args, **kwargs)

    return decorated


def require_role(role_name: str):
    """Flask route decorator: require a specific role.

    Must be used AFTER ``@require_auth``.

    Example::

        @app.route("/admin/users")
        @require_auth
        @require_role("admin")
        def list_users():
            ...
    """

    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            from flask import g, jsonify

            user = getattr(g, "current_user", None)
            if user is None:
                return jsonify({"error": "Authentication required"}), 401
            if user.role.value != role_name:
                return jsonify({"error": f"Requires {role_name} role"}), 403
            return f(*args, **kwargs)

        return decorated

    return decorator


# ── Streamlit helpers ────────────────────────────────────────────────


def get_current_user():
    """Get the currently logged-in user from Streamlit session state.

    Supports both Cognito tokens and local sessions.
    Returns None if not authenticated.
    """
    try:
        import streamlit as st

        # Check for Cognito access token first
        cognito_token = st.session_state.get("cognito_access_token")
        if cognito_token:
            cognito = _get_cognito_provider()
            if cognito and cognito.enabled:
                # Check if token is expired
                token_expires = st.session_state.get("cognito_token_expires", 0)
                import time
                if time.time() < token_expires:
                    user = cognito.validate_token(cognito_token)
                    if user:
                        return user
                # Token expired, try refresh
                refresh_token = st.session_state.get("cognito_refresh_token")
                if refresh_token:
                    new_tokens, msg = cognito.refresh_tokens(refresh_token)
                    if new_tokens:
                        st.session_state["cognito_access_token"] = new_tokens.access_token
                        st.session_state["cognito_token_expires"] = new_tokens.expires_at
                        return cognito.validate_token(new_tokens.access_token)

        # Fall back to local session
        session_id = st.session_state.get("session_id")
        if not session_id:
            return None

        auth = _get_auth_manager()
        return auth.validate_session(session_id)
    except Exception:
        return None


def _streamlit_force_password_change(user):
    """Show password change form when must_change_password is True.

    Blocks the page with st.stop() until the password is changed.
    """
    import streamlit as st

    st.warning("You must change your password before continuing.")
    with st.form("force_password_change"):
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Change Password", type="primary")

    if submitted:
        if not new_password or not confirm_password:
            st.error("Please fill in both fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        elif len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
        elif not any(c.isupper() for c in new_password):
            st.error("Password must contain at least one uppercase letter.")
        elif not any(c.isdigit() for c in new_password):
            st.error("Password must contain at least one digit.")
        else:
            auth = _get_auth_manager()
            # Use internal method to bypass old password check
            u = auth.get_user(user.username)
            if u:
                u.set_password(new_password)
                u.must_change_password = False
                auth._save_users()
                st.success("Password changed. Redirecting...")
                st.rerun()

    st.stop()


def streamlit_login_gate():
    """Show login form if user is not authenticated.

    Call this at the top of any Streamlit page that requires auth.
    Supports both Cognito and local authentication.
    Returns the User object if authenticated, None otherwise (page stops).
    """
    import streamlit as st

    user = get_current_user()
    if user is not None:
        # Check if password change is required (local auth only)
        if hasattr(user, 'must_change_password') and user.must_change_password:
            _streamlit_force_password_change(user)
            return None  # unreachable due to st.stop() above
        return user

    cognito = _get_cognito_provider()
    use_cognito = cognito is not None and cognito.enabled

    # Show login / registration tabs
    if use_cognito:
        tab_login, tab_register, tab_forgot = st.tabs([
            "Log In", "Register", "Forgot Password",
        ])
    else:
        tab_login, tab_new_org, tab_join_org = st.tabs([
            "Log In", "Create Organization", "Join Organization",
        ])

    with tab_login:
        st.markdown("### Log in to Cognisom")

        if use_cognito:
            _streamlit_cognito_login(cognito)
        else:
            _streamlit_local_login()

    if use_cognito:
        with tab_register:
            _streamlit_cognito_register(cognito)
        with tab_forgot:
            _streamlit_cognito_forgot_password(cognito)
    else:
        with tab_new_org:
            _streamlit_register_new_org()
        with tab_join_org:
            _streamlit_join_existing_org()

    st.stop()
    return None


def _streamlit_local_login():
    """Local (non-Cognito) login form."""
    import streamlit as st

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted and username and password:
        auth = _get_auth_manager()
        session, msg = auth.login(username, password)
        if session:
            st.session_state["session_id"] = session.session_id
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error(msg)


def _streamlit_cognito_login(cognito):
    """Cognito login form."""
    import streamlit as st

    with st.form("cognito_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if submitted and email and password:
        tokens, user, msg = cognito.authenticate(email, password)
        if tokens and user:
            st.session_state["cognito_access_token"] = tokens.access_token
            st.session_state["cognito_refresh_token"] = tokens.refresh_token
            st.session_state["cognito_token_expires"] = tokens.expires_at
            st.session_state["username"] = user.username
            st.rerun()
        else:
            st.error(msg)

    # Show hosted UI link for SSO
    if cognito.hosted_ui_url:
        st.divider()
        st.markdown("**Or sign in with your institution:**")
        st.link_button("University SSO", cognito.hosted_ui_url)


def _streamlit_cognito_register(cognito):
    """Cognito registration form."""
    import streamlit as st

    st.markdown("### Create an account")

    # Check if we're in confirmation step
    if st.session_state.get("cognito_confirm_email"):
        _streamlit_cognito_confirm(cognito)
        return

    with st.form("cognito_register_form"):
        email = st.text_input("Email")
        name = st.text_input("Full Name")
        organization = st.text_input("Organization (optional)")
        research_area = st.text_input("Research Area (optional)")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register", type="primary")

    st.caption("Password requirements: 12+ characters, uppercase, lowercase, number, symbol")

    if submitted:
        if not email or not password:
            st.error("Email and password are required.")
            return
        if password != password_confirm:
            st.error("Passwords do not match.")
            return

        ok, msg = cognito.register(
            email=email,
            password=password,
            display_name=name,
            organization=organization,
            research_area=research_area,
        )
        if ok:
            st.session_state["cognito_confirm_email"] = email
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def _streamlit_cognito_confirm(cognito):
    """Cognito email confirmation form."""
    import streamlit as st

    email = st.session_state.get("cognito_confirm_email", "")
    st.markdown(f"### Verify your email")
    st.info(f"A verification code was sent to **{email}**")

    with st.form("cognito_confirm_form"):
        code = st.text_input("Verification Code")
        submitted = st.form_submit_button("Verify", type="primary")

    if submitted and code:
        ok, msg = cognito.confirm_registration(email, code)
        if ok:
            st.session_state.pop("cognito_confirm_email", None)
            st.success("Account verified! You can now log in.")
            st.rerun()
        else:
            st.error(msg)

    if st.button("Resend code"):
        ok, msg = cognito.resend_confirmation_code(email)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    if st.button("Use different email"):
        st.session_state.pop("cognito_confirm_email", None)
        st.rerun()


def _streamlit_cognito_forgot_password(cognito):
    """Cognito password reset form."""
    import streamlit as st

    st.markdown("### Reset your password")

    # Check if we're in confirmation step
    if st.session_state.get("cognito_reset_email"):
        _streamlit_cognito_reset_confirm(cognito)
        return

    with st.form("cognito_forgot_form"):
        email = st.text_input("Email")
        submitted = st.form_submit_button("Send Reset Code", type="primary")

    if submitted and email:
        ok, msg = cognito.forgot_password(email)
        if ok:
            st.session_state["cognito_reset_email"] = email
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def _streamlit_cognito_reset_confirm(cognito):
    """Cognito password reset confirmation form."""
    import streamlit as st

    email = st.session_state.get("cognito_reset_email", "")
    st.info(f"A reset code was sent to **{email}**")

    with st.form("cognito_reset_confirm_form"):
        code = st.text_input("Reset Code")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Reset Password", type="primary")

    if submitted:
        if not code or not new_password:
            st.error("All fields are required.")
            return
        if new_password != confirm_password:
            st.error("Passwords do not match.")
            return

        ok, msg = cognito.confirm_forgot_password(email, code, new_password)
        if ok:
            st.session_state.pop("cognito_reset_email", None)
            st.success("Password reset! You can now log in.")
            st.rerun()
        else:
            st.error(msg)

    if st.button("Back to forgot password"):
        st.session_state.pop("cognito_reset_email", None)
        st.rerun()


def _streamlit_register_new_org():
    """Registration form: create a new organization + admin user."""
    import streamlit as st
    from .models import UserRole

    st.markdown("### Create a new organization")
    st.caption("You'll be the organization admin with a Free plan. Upgrade anytime.")

    with st.form("register_new_org"):
        org_name = st.text_input("Organization Name", placeholder="e.g. Stanford Bioengineering Lab")
        org_slug = st.text_input("Organization ID (URL-friendly)", placeholder="e.g. stanford-bio")
        st.divider()
        username = st.text_input("Your Username (min 3 chars)")
        email = st.text_input("Your Email")
        password = st.text_input("Password (min 8 chars, 1 uppercase, 1 digit)", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Organization", type="primary")

    if submitted:
        if not org_name or not org_slug:
            st.error("Organization name and ID are required.")
            return
        if not username or not email or not password:
            st.error("All user fields are required.")
            return
        if password != password_confirm:
            st.error("Passwords do not match.")
            return

        org_mgr = _get_org_manager()
        auth = _get_auth_manager()

        # Create org first
        org, msg = org_mgr.create_org(
            name=org_name,
            slug=org_slug,
            created_by=username.strip().lower(),
        )
        if org is None:
            st.error(msg)
            return

        # Create user as ORG_ADMIN
        ok, msg = auth.register(
            username=username,
            email=email,
            password=password,
            role=UserRole.ORG_ADMIN,
            org_id=org.org_id,
        )
        if not ok:
            st.error(msg)
            return

        # Auto-login
        session, login_msg = auth.login(username.strip().lower(), password)
        if session:
            st.session_state["session_id"] = session.session_id
            st.session_state["username"] = username.strip().lower()
            st.success(f"Organization **{org_name}** created! Logging you in...")
            st.rerun()
        else:
            st.success("Organization and account created. Please log in.")


def _streamlit_join_existing_org():
    """Registration form: join an existing organization by invite code."""
    import streamlit as st
    from .models import UserRole

    st.markdown("### Join an existing organization")
    st.caption("Enter the invite code from your organization admin.")

    with st.form("register_join_org"):
        invite_code = st.text_input("Invite Code")
        st.divider()
        username = st.text_input("Your Username (min 3 chars)")
        email = st.text_input("Your Email")
        password = st.text_input("Password (min 8 chars, 1 uppercase, 1 digit)", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Join Organization", type="primary")

    if submitted:
        if not invite_code:
            st.error("Invite code is required.")
            return
        if not username or not email or not password:
            st.error("All user fields are required.")
            return
        if password != password_confirm:
            st.error("Passwords do not match.")
            return

        org_mgr = _get_org_manager()
        auth = _get_auth_manager()

        # Find org by invite code
        org = org_mgr.get_org_by_invite_code(invite_code.strip())
        if org is None:
            st.error("Invalid invite code. Check with your organization admin.")
            return

        # Check user limit
        current_count = auth.get_org_user_count(org.org_id)
        if not org_mgr.check_user_limit(org.org_id, current_count):
            st.error(
                f"Organization **{org.name}** has reached its user limit "
                f"({org.max_users} users on the {org.plan.value.title()} plan). "
                f"Contact your admin to upgrade."
            )
            return

        # Create user as RESEARCHER in the org
        ok, msg = auth.register(
            username=username,
            email=email,
            password=password,
            role=UserRole.RESEARCHER,
            org_id=org.org_id,
        )
        if not ok:
            st.error(msg)
            return

        # Auto-login
        session, login_msg = auth.login(username.strip().lower(), password)
        if session:
            st.session_state["session_id"] = session.session_id
            st.session_state["username"] = username.strip().lower()
            st.success(f"Joined **{org.name}**! Logging you in...")
            st.rerun()
        else:
            st.success("Account created. Please log in.")


def streamlit_require_permission(permission: str):
    """Check if the current Streamlit user has a permission.

    Shows error and stops if not.
    """
    import streamlit as st

    user = get_current_user()
    if user is None:
        streamlit_login_gate()
        return None

    if not user.has_permission(permission):
        st.error(f"You don't have permission: **{permission}**. Contact an admin.")
        st.stop()
        return None

    return user


def streamlit_page_gate(page_name: str = ""):
    """Auth + subscription tier gate for Streamlit pages.

    Call this at the top of any page that requires auth and tier checking.
    Returns the User object if authenticated and authorized.

    Args:
        page_name: The page identifier (e.g. "3_simulation"). If empty,
                   only auth is checked, not tier access.
    """
    import streamlit as st

    # First: require authentication (includes password change check)
    user = streamlit_login_gate()

    # If no page_name specified, skip tier check
    if not page_name:
        return user

    # ADMIN users (system org) bypass tier checks
    from .models import UserRole
    if user.role == UserRole.ADMIN:
        return user

    # Check org tier for page access
    org_mgr = _get_org_manager()
    org = org_mgr.get_org(user.org_id) if user.org_id else None

    if org is None:
        st.error("Your account is not associated with an organization. Contact an admin.")
        st.stop()
        return None

    allowed_pages = org_mgr.get_page_access(user.org_id)
    if page_name not in allowed_pages:
        tier_name = org.plan.value.title()
        st.error(
            f"This page requires a higher subscription tier. "
            f"Your organization **{org.name}** is on the **{tier_name}** plan. "
            f"Contact your organization admin to upgrade."
        )
        st.stop()
        return None

    return user
