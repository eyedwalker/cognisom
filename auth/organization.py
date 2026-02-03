"""Organization model and subscription tiers for multi-tenant support."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class SubscriptionTier(str, Enum):
    """Platform subscription tiers."""

    FREE = "free"
    RESEARCHER = "researcher"
    INSTITUTION = "institution"
    ENTERPRISE = "enterprise"


# Page access by tier — page names match sidebar page filenames
# True = accessible, False = blocked
TIER_PAGE_ACCESS: Dict[SubscriptionTier, List[str]] = {
    SubscriptionTier.FREE: [
        "app",              # Home / Overview
        "6_research_feed",  # Research Feed
        "9_account",        # Account
    ],
    SubscriptionTier.RESEARCHER: [
        "app",
        "1_ingestion",
        "2_discovery",
        "3_simulation",
        "5_molecular_lab",
        "6_research_feed",
        "7_research_agent",
        "8_subscriptions",
        "9_account",
        "14_entity_library",
    ],
    SubscriptionTier.INSTITUTION: [
        "app",
        "1_ingestion",
        "2_discovery",
        "3_simulation",
        "4_admin",
        "5_molecular_lab",
        "6_research_feed",
        "7_research_agent",
        "8_subscriptions",
        "9_account",
        "10_security",
        "11_validation",
        "12_3d_visualization",
        "13_organization",
        "14_entity_library",
    ],
    SubscriptionTier.ENTERPRISE: [
        "app",
        "1_ingestion",
        "2_discovery",
        "3_simulation",
        "4_admin",
        "5_molecular_lab",
        "6_research_feed",
        "7_research_agent",
        "8_subscriptions",
        "9_account",
        "10_security",
        "11_validation",
        "12_3d_visualization",
        "13_organization",
        "14_entity_library",
    ],
}

# Tier limits
TIER_LIMITS: Dict[SubscriptionTier, dict] = {
    SubscriptionTier.FREE: {
        "max_users": 1,
        "api_access": False,
        "gpu_access": False,
        "description": "Basic access for individual exploration",
    },
    SubscriptionTier.RESEARCHER: {
        "max_users": 3,
        "api_access": True,
        "gpu_access": False,
        "description": "Full simulation and discovery tools for small teams",
    },
    SubscriptionTier.INSTITUTION: {
        "max_users": 25,
        "api_access": True,
        "gpu_access": False,
        "description": "All features for labs and departments",
    },
    SubscriptionTier.ENTERPRISE: {
        "max_users": -1,  # unlimited
        "api_access": True,
        "gpu_access": True,
        "description": "Unlimited users, GPU acceleration, priority support",
    },
}


@dataclass
class Organization:
    """A tenant organization on the platform."""

    org_id: str              # unique slug (e.g., "stanford-bio")
    name: str                # display name
    description: str = ""
    plan: SubscriptionTier = SubscriptionTier.FREE
    max_users: int = 1       # derived from plan, but can be overridden
    created_at: float = 0.0
    is_active: bool = True
    created_by: str = ""     # username of creator
    invite_code: str = ""    # code for joining this org

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.invite_code:
            self.invite_code = secrets.token_urlsafe(16)
        # Sync max_users with plan limits
        plan_limit = TIER_LIMITS[self.plan]["max_users"]
        if plan_limit > 0:
            self.max_users = plan_limit
        else:
            self.max_users = -1  # unlimited

    def to_dict(self) -> dict:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "plan": self.plan.value,
            "max_users": self.max_users,
            "created_at": self.created_at,
            "is_active": self.is_active,
            "created_by": self.created_by,
            "invite_code": self.invite_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Organization:
        return cls(
            org_id=data["org_id"],
            name=data["name"],
            description=data.get("description", ""),
            plan=SubscriptionTier(data.get("plan", "free")),
            max_users=data.get("max_users", 1),
            created_at=data.get("created_at", 0.0),
            is_active=data.get("is_active", True),
            created_by=data.get("created_by", ""),
            invite_code=data.get("invite_code", ""),
        )
