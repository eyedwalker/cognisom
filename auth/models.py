"""User models and data types."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class UserRole(str, Enum):
    """Platform access roles."""

    VIEWER = "viewer"           # Read-only: view dashboards, feeds
    RESEARCHER = "researcher"   # Can run simulations, agent queries, exports
    ORG_ADMIN = "org_admin"     # Manage users within their organization
    ADMIN = "admin"             # Full access: user management, API keys, settings


# Permissions per role
ROLE_PERMISSIONS: Dict[UserRole, List[str]] = {
    UserRole.VIEWER: [
        "dashboard:view",
        "feed:view",
        "subscriptions:view",
    ],
    UserRole.RESEARCHER: [
        "dashboard:view",
        "feed:view",
        "subscriptions:view",
        "subscriptions:manage",
        "simulation:run",
        "simulation:export",
        "discovery:run",
        "agent:query",
        "molecular_lab:use",
    ],
    UserRole.ORG_ADMIN: [
        "dashboard:view",
        "feed:view",
        "subscriptions:view",
        "subscriptions:manage",
        "simulation:run",
        "simulation:export",
        "discovery:run",
        "agent:query",
        "molecular_lab:use",
        "org:manage_users",
        "org:settings",
        "admin:users",  # within their org only
    ],
    UserRole.ADMIN: [
        "dashboard:view",
        "feed:view",
        "subscriptions:view",
        "subscriptions:manage",
        "simulation:run",
        "simulation:export",
        "simulation:configure",
        "discovery:run",
        "agent:query",
        "molecular_lab:use",
        "admin:users",
        "admin:api_keys",
        "admin:settings",
        "org:manage_users",
        "org:settings",
    ],
}


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 password hash with 600k iterations."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=600_000,
    )
    return dk.hex()


@dataclass
class User:
    """Platform user."""

    username: str
    email: str
    role: UserRole
    password_hash: str = ""
    salt: str = ""
    created_at: float = 0.0
    last_login: float = 0.0
    is_active: bool = True
    display_name: str = ""
    api_key: str = ""  # personal API key for programmatic access
    org_id: str = ""   # organization this user belongs to
    must_change_password: bool = False  # force password change on next login

    def set_password(self, password: str) -> None:
        """Hash and store password."""
        self.salt = secrets.token_hex(32)
        self.password_hash = _hash_password(password, self.salt)

    def check_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        if not self.password_hash or not self.salt:
            return False
        candidate = _hash_password(password, self.salt)
        return hmac.compare_digest(candidate, self.password_hash)

    def has_permission(self, permission: str) -> bool:
        """Check if user's role grants a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, [])

    def generate_api_key(self) -> str:
        """Generate a personal API key."""
        self.api_key = f"cog_{secrets.token_urlsafe(32)}"
        return self.api_key

    def to_dict(self) -> dict:
        """Serialise for storage (includes sensitive fields)."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "password_hash": self.password_hash,
            "salt": self.salt,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "display_name": self.display_name,
            "api_key": self.api_key,
            "org_id": self.org_id,
            "must_change_password": self.must_change_password,
        }

    def to_public_dict(self) -> dict:
        """Serialise for API responses (no sensitive fields)."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "display_name": self.display_name,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "org_id": self.org_id,
            "must_change_password": self.must_change_password,
        }

    @classmethod
    def from_dict(cls, data: dict) -> User:
        return cls(
            username=data["username"],
            email=data["email"],
            role=UserRole(data["role"]),
            password_hash=data.get("password_hash", ""),
            salt=data.get("salt", ""),
            created_at=data.get("created_at", 0.0),
            last_login=data.get("last_login", 0.0),
            is_active=data.get("is_active", True),
            display_name=data.get("display_name", ""),
            api_key=data.get("api_key", ""),
            org_id=data.get("org_id", ""),
            must_change_password=data.get("must_change_password", False),
        )


@dataclass
class UserSession:
    """Active login session."""

    session_id: str
    username: str
    created_at: float
    expires_at: float
    ip_address: str = ""
    user_agent: str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "username": self.username,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserSession:
        return cls(**data)
