"""Organization Management — view and manage your organization, members, and plan."""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Organization | Cognisom", page_icon="🏢", layout="wide")

from cognisom.auth.middleware import streamlit_page_gate, _get_auth_manager, _get_org_manager
user = streamlit_page_gate("13_organization")

from cognisom.auth.models import UserRole
from cognisom.auth.organization import SubscriptionTier, TIER_LIMITS

auth = _get_auth_manager()
org_mgr = _get_org_manager()

# ── Determine context ───────────────────────────────────────────────

is_super_admin = user.role == UserRole.ADMIN
is_org_admin = user.role in (UserRole.ORG_ADMIN, UserRole.ADMIN)
my_org = org_mgr.get_org(user.org_id) if user.org_id else None

st.title("Organization Management")

if is_super_admin:
    tab_my_org, tab_all_orgs = st.tabs(["My Organization", "All Organizations (Admin)"])
else:
    tab_my_org = st.container()
    tab_all_orgs = None

# ══════════════════════════════════════════════════════════════════════
# MY ORGANIZATION
# ══════════════════════════════════════════════════════════════════════

with tab_my_org:
    if my_org is None:
        st.warning("You are not part of any organization.")
        st.stop()

    # ── Org details ──────────────────────────────────────────────────
    st.subheader(my_org.name)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Plan", my_org.plan.value.title())
    members = auth.get_org_users(my_org.org_id)
    limit_str = str(my_org.max_users) if my_org.max_users > 0 else "Unlimited"
    col2.metric("Members", f"{len(members)} / {limit_str}")
    col3.metric("Org ID", my_org.org_id)
    col4.metric("Created", datetime.fromtimestamp(my_org.created_at).strftime("%Y-%m-%d"))

    if my_org.description:
        st.caption(my_org.description)

    st.divider()

    # ── Plan info ────────────────────────────────────────────────────
    st.subheader("Subscription Plan")
    limits = TIER_LIMITS[my_org.plan]
    plan_cols = st.columns(4)
    plan_cols[0].markdown(f"**Plan:** {my_org.plan.value.title()}")
    plan_cols[1].markdown(f"**Max Users:** {limit_str}")
    plan_cols[2].markdown(f"**API Access:** {'Yes' if limits['api_access'] else 'No'}")
    plan_cols[3].markdown(f"**GPU Access:** {'Yes' if limits['gpu_access'] else 'No'}")
    st.caption(limits["description"])

    # Show upgrade options for non-enterprise
    if my_org.plan != SubscriptionTier.ENTERPRISE and is_org_admin:
        st.info("Contact sales@eyentelligence.com to upgrade your plan.")

    st.divider()

    # ── Members ──────────────────────────────────────────────────────
    st.subheader("Members")

    if members:
        for m in members:
            mcol1, mcol2, mcol3, mcol4 = st.columns([3, 2, 2, 1])
            mcol1.markdown(f"**{m.display_name}** (`{m.username}`)")
            mcol2.markdown(f"{m.email}")
            mcol3.markdown(f"Role: `{m.role.value}`")
            if is_org_admin and m.username != user.username:
                if mcol4.button("Remove", key=f"remove_{m.username}"):
                    auth.set_user_org(m.username, "")
                    st.success(f"Removed {m.username} from organization.")
                    st.rerun()
    else:
        st.info("No members found.")

    st.divider()

    # ── Invite code ──────────────────────────────────────────────────
    if is_org_admin:
        st.subheader("Invite New Members")
        st.markdown(
            f"Share this invite code with new team members so they can join "
            f"**{my_org.name}** during registration:"
        )
        st.code(my_org.invite_code, language=None)
        st.caption(
            f"Users who register with this code will be added to your organization "
            f"as Researchers. You can change their role after they join."
        )

    st.divider()

    # ── Org admin: change member roles ───────────────────────────────
    if is_org_admin and members:
        st.subheader("Change Member Roles")
        for m in members:
            if m.username == user.username:
                continue  # can't change own role
            role_options = [UserRole.VIEWER.value, UserRole.RESEARCHER.value, UserRole.ORG_ADMIN.value]
            current_idx = role_options.index(m.role.value) if m.role.value in role_options else 0
            new_role = st.selectbox(
                f"Role for {m.display_name}",
                role_options,
                index=current_idx,
                key=f"role_{m.username}",
            )
            if new_role != m.role.value:
                if st.button(f"Update {m.display_name}", key=f"update_role_{m.username}"):
                    auth.update_role(m.username, UserRole(new_role))
                    st.success(f"Updated {m.display_name}'s role to {new_role}.")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# ALL ORGANIZATIONS (super admin only)
# ══════════════════════════════════════════════════════════════════════

if tab_all_orgs is not None:
    with tab_all_orgs:
        st.subheader("All Organizations")

        all_orgs = org_mgr.list_orgs()
        if not all_orgs:
            st.info("No organizations found.")
        else:
            for org in all_orgs:
                with st.expander(f"{org.name} ({org.org_id}) — {org.plan.value.title()}"):
                    org_members = auth.get_org_users(org.org_id)
                    ocol1, ocol2, ocol3 = st.columns(3)
                    ocol1.markdown(f"**Members:** {len(org_members)}")
                    ocol2.markdown(f"**Created by:** {org.created_by}")
                    ocol3.markdown(f"**Active:** {'Yes' if org.is_active else 'No'}")

                    if org_members:
                        st.markdown("**Members:**")
                        for m in org_members:
                            st.markdown(f"- {m.display_name} (`{m.username}`) — {m.role.value}")

                    # Change plan
                    st.markdown("---")
                    tier_options = [t.value for t in SubscriptionTier]
                    current_idx = tier_options.index(org.plan.value)
                    new_tier = st.selectbox(
                        "Change Plan",
                        tier_options,
                        index=current_idx,
                        key=f"tier_{org.org_id}",
                    )
                    if new_tier != org.plan.value:
                        if st.button(f"Update {org.name} to {new_tier}", key=f"update_tier_{org.org_id}"):
                            ok, msg = org_mgr.update_plan(org.org_id, SubscriptionTier(new_tier))
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

st.sidebar.markdown("---")
st.sidebar.caption("eyentelligence inc.")
