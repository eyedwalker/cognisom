"""
Authentication Manager
======================

Handles user registration, login, session management, and persistence.
Uses a JSON file store (suitable for single-server deployments;
swap for PostgreSQL/Redis for production scale).
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import User, UserRole, UserSession

log = logging.getLogger(__name__)

# Session lifetime: 24 hours
SESSION_TTL = 24 * 3600

# Minimum password requirements
MIN_PASSWORD_LENGTH = 8


class AuthManager:
    """Manages users and sessions with JSON file persistence.

    Usage::

        auth = AuthManager(data_dir="data/auth")
        auth.register("alice", "alice@lab.org", "StrongP@ss1", role=UserRole.RESEARCHER)
        session = auth.login("alice", "StrongP@ss1")
        user = auth.validate_session(session.session_id)
    """

    def __init__(self, data_dir: str = "data/auth") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._users_file = self._dir / "users.json"
        self._sessions_file = self._dir / "sessions.json"
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, UserSession] = {}
        self._load()
        self._ensure_default_admin()

    # ── Registration ─────────────────────────────────────────────────

    def register(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.RESEARCHER,
        display_name: str = "",
        org_id: str = "",
    ) -> Tuple[bool, str]:
        """Register a new user.

        Returns (success, message).
        """
        # Validation
        username = username.strip().lower()
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters."
        if not email or "@" not in email:
            return False, "Invalid email address."
        if len(password) < MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter."
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit."
        if username in self._users:
            return False, "Username already taken."
        if any(u.email.lower() == email.lower() for u in self._users.values()):
            return False, "Email already registered."

        user = User(
            username=username,
            email=email,
            role=role,
            display_name=display_name or username,
            created_at=time.time(),
            org_id=org_id,
        )
        user.set_password(password)
        user.generate_api_key()

        self._users[username] = user
        self._save_users()
        log.info("Registered user: %s (%s) in org: %s", username, role.value, org_id or "none")
        return True, "Registration successful."

    # ── Login / Logout ───────────────────────────────────────────────

    def login(
        self,
        username: str,
        password: str,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Tuple[Optional[UserSession], str]:
        """Authenticate and create a session.

        Returns (session_or_None, message).
        """
        username = username.strip().lower()
        user = self._users.get(username)
        if user is None:
            return None, "Invalid username or password."
        if not user.is_active:
            return None, "Account is disabled."
        if not user.check_password(password):
            return None, "Invalid username or password."

        # Create session
        session = UserSession(
            session_id=secrets.token_urlsafe(48),
            username=username,
            created_at=time.time(),
            expires_at=time.time() + SESSION_TTL,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        user.last_login = time.time()
        self._sessions[session.session_id] = session
        self._save_users()
        self._save_sessions()
        log.info("User logged in: %s", username)
        return session, "Login successful."

    def logout(self, session_id: str) -> None:
        """Invalidate a session."""
        self._sessions.pop(session_id, None)
        self._save_sessions()

    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate a session and return the user, or None if invalid/expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self._sessions.pop(session_id, None)
            self._save_sessions()
            return None
        user = self._users.get(session.username)
        if user is None or not user.is_active:
            return None
        return user

    def validate_api_key(self, api_key: str) -> Optional[User]:
        """Validate a personal API key and return the user."""
        for user in self._users.values():
            if user.api_key and user.api_key == api_key and user.is_active:
                return user
        return None

    # ── User management ──────────────────────────────────────────────

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username.lower())

    def list_users(self) -> List[User]:
        return list(self._users.values())

    def update_role(self, username: str, new_role: UserRole) -> bool:
        user = self._users.get(username.lower())
        if user is None:
            return False
        user.role = new_role
        self._save_users()
        return True

    def deactivate_user(self, username: str) -> bool:
        user = self._users.get(username.lower())
        if user is None:
            return False
        user.is_active = False
        # Kill all sessions
        self._sessions = {
            k: v for k, v in self._sessions.items() if v.username != username
        }
        self._save_users()
        self._save_sessions()
        return True

    def activate_user(self, username: str) -> bool:
        user = self._users.get(username.lower())
        if user is None:
            return False
        user.is_active = True
        self._save_users()
        return True

    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        user = self._users.get(username.lower())
        if user is None:
            return False, "User not found."
        if not user.check_password(old_password):
            return False, "Current password is incorrect."
        if len(new_password) < MIN_PASSWORD_LENGTH:
            return False, f"New password must be at least {MIN_PASSWORD_LENGTH} characters."
        user.set_password(new_password)
        user.must_change_password = False
        self._save_users()
        return True, "Password changed."

    def regenerate_api_key(self, username: str) -> Optional[str]:
        user = self._users.get(username.lower())
        if user is None:
            return None
        key = user.generate_api_key()
        self._save_users()
        return key

    def active_sessions(self) -> List[UserSession]:
        """Return all non-expired sessions."""
        self._cleanup_sessions()
        return list(self._sessions.values())

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._users_file.exists():
            try:
                data = json.loads(self._users_file.read_text())
                self._users = {
                    k: User.from_dict(v) for k, v in data.items()
                }
            except Exception as exc:
                log.error("Failed to load users: %s", exc)

        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text())
                self._sessions = {
                    k: UserSession.from_dict(v) for k, v in data.items()
                }
            except Exception as exc:
                log.error("Failed to load sessions: %s", exc)

    def _save_users(self) -> None:
        data = {k: v.to_dict() for k, v in self._users.items()}
        self._users_file.write_text(json.dumps(data, indent=2))

    def _save_sessions(self) -> None:
        data = {k: v.to_dict() for k, v in self._sessions.items()}
        self._sessions_file.write_text(json.dumps(data, indent=2))

    def _cleanup_sessions(self) -> None:
        expired = [k for k, v in self._sessions.items() if v.is_expired]
        if expired:
            for k in expired:
                del self._sessions[k]
            self._save_sessions()

    def get_org_user_count(self, org_id: str) -> int:
        """Count users belonging to an organization."""
        return sum(1 for u in self._users.values() if u.org_id == org_id)

    def get_org_users(self, org_id: str) -> List[User]:
        """Get all users belonging to an organization."""
        return [u for u in self._users.values() if u.org_id == org_id]

    def set_user_org(self, username: str, org_id: str) -> bool:
        """Assign a user to an organization."""
        user = self._users.get(username.lower())
        if user is None:
            return False
        user.org_id = org_id
        self._save_users()
        return True

    def _ensure_default_admin(self) -> None:
        """Create a default admin account if no users exist."""
        if not self._users:
            self.register(
                username="admin",
                email="admin@cognisom.local",
                password="Admin1234!",
                role=UserRole.ADMIN,
                display_name="Platform Admin",
                org_id="system",
            )
            # Force password change on first login
            if "admin" in self._users:
                self._users["admin"].must_change_password = True
                self._save_users()
            log.info("Created default admin account (username: admin, password: Admin1234!)")
