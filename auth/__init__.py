"""
Authentication & Authorization
==============================

User registration, login, role-based access control, and session management
for the Cognisom HDT platform.
"""

from .models import User, UserRole, UserSession
from .auth_manager import AuthManager
from .middleware import require_auth, require_role, get_current_user

__all__ = [
    "User",
    "UserRole",
    "UserSession",
    "AuthManager",
    "require_auth",
    "require_role",
    "get_current_user",
]
