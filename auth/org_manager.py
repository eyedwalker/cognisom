"""Organization manager — CRUD for multi-tenant orgs with JSON persistence."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import User, UserRole
from .organization import Organization, SubscriptionTier, TIER_LIMITS

log = logging.getLogger(__name__)


class OrgManager:
    """Manages organizations with JSON file persistence.

    Usage::

        orgs = OrgManager(data_dir="data/auth")
        org = orgs.create_org("Stanford Bio", "stanford-bio", "admin")
        orgs.add_user_to_org("alice", "stanford-bio")
    """

    def __init__(self, data_dir: str = "data/auth") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._orgs_file = self._dir / "orgs.json"
        self._orgs: Dict[str, Organization] = {}
        self._load()
        self._ensure_system_org()

    # ── CRUD ──────────────────────────────────────────────────────────

    def create_org(
        self,
        name: str,
        slug: str,
        created_by: str,
        plan: SubscriptionTier = SubscriptionTier.FREE,
        description: str = "",
    ) -> Tuple[Optional[Organization], str]:
        """Create a new organization. Returns (org, message)."""
        slug = self._normalize_slug(slug)
        if not slug or len(slug) < 2:
            return None, "Organization slug must be at least 2 characters."
        if not name.strip():
            return None, "Organization name is required."
        if slug in self._orgs:
            return None, "Organization slug already taken."

        org = Organization(
            org_id=slug,
            name=name.strip(),
            description=description,
            plan=plan,
            created_by=created_by,
            created_at=time.time(),
        )
        self._orgs[slug] = org
        self._save()
        log.info("Created organization: %s (%s)", slug, plan.value)
        return org, "Organization created."

    def get_org(self, org_id: str) -> Optional[Organization]:
        return self._orgs.get(org_id)

    def list_orgs(self) -> List[Organization]:
        return list(self._orgs.values())

    def update_plan(self, org_id: str, new_tier: SubscriptionTier) -> Tuple[bool, str]:
        """Change an org's subscription tier."""
        org = self._orgs.get(org_id)
        if org is None:
            return False, "Organization not found."
        org.plan = new_tier
        plan_limit = TIER_LIMITS[new_tier]["max_users"]
        org.max_users = plan_limit if plan_limit > 0 else -1
        self._save()
        log.info("Updated org %s to plan %s", org_id, new_tier.value)
        return True, f"Plan updated to {new_tier.value}."

    def deactivate_org(self, org_id: str) -> bool:
        org = self._orgs.get(org_id)
        if org is None:
            return False
        org.is_active = False
        self._save()
        return True

    def get_org_by_invite_code(self, invite_code: str) -> Optional[Organization]:
        """Find an org by its invite code."""
        for org in self._orgs.values():
            if org.invite_code == invite_code and org.is_active:
                return org
        return None

    # ── User membership ───────────────────────────────────────────────

    def check_user_limit(self, org_id: str, current_count: int) -> bool:
        """Check if the org can accept another user."""
        org = self._orgs.get(org_id)
        if org is None:
            return False
        if org.max_users < 0:
            return True  # unlimited
        return current_count < org.max_users

    def get_page_access(self, org_id: str) -> List[str]:
        """Get accessible page names for the org's tier."""
        from .organization import TIER_PAGE_ACCESS

        org = self._orgs.get(org_id)
        if org is None:
            return TIER_PAGE_ACCESS[SubscriptionTier.FREE]
        return TIER_PAGE_ACCESS.get(org.plan, TIER_PAGE_ACCESS[SubscriptionTier.FREE])

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if self._orgs_file.exists():
            try:
                data = json.loads(self._orgs_file.read_text())
                self._orgs = {
                    k: Organization.from_dict(v) for k, v in data.items()
                }
            except Exception as exc:
                log.error("Failed to load orgs: %s", exc)

    def _save(self) -> None:
        data = {k: v.to_dict() for k, v in self._orgs.items()}
        self._orgs_file.write_text(json.dumps(data, indent=2))

    def _ensure_system_org(self) -> None:
        """Create the 'system' org for the default admin if it doesn't exist."""
        if "system" not in self._orgs:
            self._orgs["system"] = Organization(
                org_id="system",
                name="System Administration",
                description="Default organization for platform administrators",
                plan=SubscriptionTier.ENTERPRISE,
                created_by="system",
                created_at=time.time(),
            )
            self._save()
            log.info("Created system organization (enterprise tier)")

    @staticmethod
    def _normalize_slug(slug: str) -> str:
        """Normalize an org slug to lowercase alphanumeric + hyphens."""
        slug = slug.strip().lower()
        slug = re.sub(r"[^a-z0-9-]", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug
