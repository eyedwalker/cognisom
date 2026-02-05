"""
AWS Cognito Authentication Provider
====================================

Integrates with AWS Cognito for research-grade authentication:
- User Pool authentication (email/password, MFA)
- Token validation (JWT)
- SAML/OIDC federation for university SSO (via Cognito hosted UI)
- HIPAA-eligible with BAA

Environment variables:
    COGNITO_USER_POOL_ID  — Cognito User Pool ID
    COGNITO_CLIENT_ID     — Cognito App Client ID
    COGNITO_CLIENT_SECRET — Cognito App Client Secret (optional)
    COGNITO_DOMAIN        — Cognito hosted UI domain
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from .models import User, UserRole, UserSession

log = logging.getLogger(__name__)


def get_cognito_config() -> dict:
    """Get Cognito configuration from environment."""
    return {
        "user_pool_id": os.environ.get("COGNITO_USER_POOL_ID", ""),
        "client_id": os.environ.get("COGNITO_CLIENT_ID", ""),
        "client_secret": os.environ.get("COGNITO_CLIENT_SECRET", ""),
        "domain": os.environ.get("COGNITO_DOMAIN", ""),
        "region": os.environ.get("AWS_REGION", "us-east-1"),
    }


def _compute_secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """Compute Cognito secret hash for authentication requests."""
    message = username + client_id
    dig = hmac.new(
        client_secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(dig).decode("utf-8")


@dataclass
class CognitoTokens:
    """Cognito authentication tokens."""
    access_token: str
    id_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"


class CognitoAuthProvider:
    """AWS Cognito authentication provider.

    Provides user authentication through AWS Cognito User Pools,
    with support for MFA, federation, and HIPAA compliance.

    Usage::

        provider = CognitoAuthProvider()
        tokens, user = provider.authenticate("user@example.com", "password")
        user = provider.validate_token(tokens.access_token)
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or get_cognito_config()
        self._client = boto3.client(
            "cognito-idp",
            region_name=self._config["region"],
        )
        self._enabled = bool(
            self._config["user_pool_id"] and self._config["client_id"]
        )
        if self._enabled:
            log.info("Cognito auth enabled (pool: %s)", self._config["user_pool_id"])
        else:
            log.warning("Cognito auth disabled (missing configuration)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def hosted_ui_url(self) -> str:
        """Return the Cognito hosted UI URL for OAuth login."""
        if not self._enabled or not self._config["domain"]:
            return ""
        return self._config["domain"]

    def _get_secret_hash(self, username: str) -> Optional[str]:
        """Get secret hash if client secret is configured."""
        if self._config["client_secret"]:
            return _compute_secret_hash(
                username,
                self._config["client_id"],
                self._config["client_secret"],
            )
        return None

    # ── Authentication ────────────────────────────────────────────────

    def authenticate(
        self,
        username: str,
        password: str,
    ) -> Tuple[Optional[CognitoTokens], Optional[User], str]:
        """Authenticate a user with username/password.

        Returns (tokens, user, message).
        """
        if not self._enabled:
            return None, None, "Cognito authentication not configured"

        try:
            auth_params = {
                "USERNAME": username,
                "PASSWORD": password,
            }

            secret_hash = self._get_secret_hash(username)
            if secret_hash:
                auth_params["SECRET_HASH"] = secret_hash

            response = self._client.initiate_auth(
                ClientId=self._config["client_id"],
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters=auth_params,
            )

            # Check for MFA challenge
            if "ChallengeName" in response:
                challenge = response["ChallengeName"]
                if challenge == "NEW_PASSWORD_REQUIRED":
                    return None, None, "Password change required"
                elif challenge in ("SMS_MFA", "SOFTWARE_TOKEN_MFA"):
                    # MFA required - return session for challenge response
                    return None, None, f"MFA required: {challenge}"
                else:
                    return None, None, f"Authentication challenge: {challenge}"

            # Extract tokens
            auth_result = response["AuthenticationResult"]
            tokens = CognitoTokens(
                access_token=auth_result["AccessToken"],
                id_token=auth_result["IdToken"],
                refresh_token=auth_result.get("RefreshToken", ""),
                expires_at=time.time() + auth_result["ExpiresIn"],
                token_type=auth_result.get("TokenType", "Bearer"),
            )

            # Get user details
            user = self.get_user(tokens.access_token)
            return tokens, user, "Authentication successful"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                return None, None, "Invalid username or password"
            elif error_code == "UserNotFoundException":
                return None, None, "Invalid username or password"
            elif error_code == "UserNotConfirmedException":
                return None, None, "Account not verified. Check your email."
            elif error_code == "PasswordResetRequiredException":
                return None, None, "Password reset required"
            else:
                log.error("Cognito auth error: %s - %s", error_code, e)
                return None, None, f"Authentication failed: {error_code}"
        except Exception as e:
            log.error("Cognito auth exception: %s", e)
            return None, None, "Authentication failed"

    def respond_to_mfa_challenge(
        self,
        username: str,
        session: str,
        mfa_code: str,
        challenge_name: str = "SOFTWARE_TOKEN_MFA",
    ) -> Tuple[Optional[CognitoTokens], Optional[User], str]:
        """Respond to MFA challenge.

        Returns (tokens, user, message).
        """
        if not self._enabled:
            return None, None, "Cognito authentication not configured"

        try:
            challenge_responses = {
                "USERNAME": username,
            }

            if challenge_name == "SOFTWARE_TOKEN_MFA":
                challenge_responses["SOFTWARE_TOKEN_MFA_CODE"] = mfa_code
            else:
                challenge_responses["SMS_MFA_CODE"] = mfa_code

            secret_hash = self._get_secret_hash(username)
            if secret_hash:
                challenge_responses["SECRET_HASH"] = secret_hash

            response = self._client.respond_to_auth_challenge(
                ClientId=self._config["client_id"],
                ChallengeName=challenge_name,
                Session=session,
                ChallengeResponses=challenge_responses,
            )

            auth_result = response["AuthenticationResult"]
            tokens = CognitoTokens(
                access_token=auth_result["AccessToken"],
                id_token=auth_result["IdToken"],
                refresh_token=auth_result.get("RefreshToken", ""),
                expires_at=time.time() + auth_result["ExpiresIn"],
            )

            user = self.get_user(tokens.access_token)
            return tokens, user, "MFA verification successful"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            return None, None, f"MFA verification failed: {error_code}"
        except Exception as e:
            log.error("MFA challenge exception: %s", e)
            return None, None, "MFA verification failed"

    def refresh_tokens(self, refresh_token: str) -> Tuple[Optional[CognitoTokens], str]:
        """Refresh authentication tokens.

        Returns (tokens, message).
        """
        if not self._enabled:
            return None, "Cognito authentication not configured"

        try:
            auth_params = {
                "REFRESH_TOKEN": refresh_token,
            }

            # Note: SECRET_HASH for refresh uses the refresh token as username
            if self._config["client_secret"]:
                # For refresh, we need to use a different approach
                # Cognito refresh doesn't always require SECRET_HASH
                pass

            response = self._client.initiate_auth(
                ClientId=self._config["client_id"],
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters=auth_params,
            )

            auth_result = response["AuthenticationResult"]
            tokens = CognitoTokens(
                access_token=auth_result["AccessToken"],
                id_token=auth_result["IdToken"],
                refresh_token=refresh_token,  # Refresh token doesn't change
                expires_at=time.time() + auth_result["ExpiresIn"],
            )

            return tokens, "Token refresh successful"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            return None, f"Token refresh failed: {error_code}"
        except Exception as e:
            log.error("Token refresh exception: %s", e)
            return None, "Token refresh failed"

    # ── User Management ───────────────────────────────────────────────

    def get_user(self, access_token: str) -> Optional[User]:
        """Get user details from access token.

        Returns a User object populated from Cognito attributes.
        """
        if not self._enabled:
            return None

        try:
            response = self._client.get_user(AccessToken=access_token)

            # Parse attributes
            attrs = {a["Name"]: a["Value"] for a in response.get("UserAttributes", [])}

            # Map Cognito groups to role
            role = self._get_user_role(response["Username"])

            return User(
                username=response["Username"],
                email=attrs.get("email", ""),
                role=role,
                display_name=attrs.get("name", attrs.get("email", response["Username"])),
                is_active=True,
                created_at=0,  # Cognito doesn't expose this easily
                org_id=attrs.get("custom:organization", ""),
            )

        except ClientError as e:
            log.warning("Failed to get user: %s", e)
            return None
        except Exception as e:
            log.error("Get user exception: %s", e)
            return None

    def _get_user_role(self, username: str) -> UserRole:
        """Get user role from Cognito groups."""
        if not self._enabled:
            return UserRole.VIEWER

        try:
            response = self._client.admin_list_groups_for_user(
                Username=username,
                UserPoolId=self._config["user_pool_id"],
            )

            groups = [g["GroupName"] for g in response.get("Groups", [])]

            # Map Cognito groups to roles (highest privilege wins)
            if "admin" in groups:
                return UserRole.ADMIN
            elif "org_admin" in groups:
                return UserRole.ORG_ADMIN
            elif "researcher" in groups:
                return UserRole.RESEARCHER
            else:
                return UserRole.VIEWER

        except ClientError:
            return UserRole.VIEWER

    def validate_token(self, access_token: str) -> Optional[User]:
        """Validate an access token and return the user.

        This calls Cognito to verify the token is valid.
        """
        return self.get_user(access_token)

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        email: str,
        password: str,
        display_name: str = "",
        organization: str = "",
        research_area: str = "",
    ) -> Tuple[bool, str]:
        """Register a new user.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            user_attributes = [
                {"Name": "email", "Value": email},
                {"Name": "name", "Value": display_name or email.split("@")[0]},
            ]

            if organization:
                user_attributes.append(
                    {"Name": "custom:organization", "Value": organization}
                )
            if research_area:
                user_attributes.append(
                    {"Name": "custom:research_area", "Value": research_area}
                )

            kwargs = {
                "ClientId": self._config["client_id"],
                "Username": email,
                "Password": password,
                "UserAttributes": user_attributes,
            }

            secret_hash = self._get_secret_hash(email)
            if secret_hash:
                kwargs["SecretHash"] = secret_hash

            self._client.sign_up(**kwargs)
            return True, "Registration successful. Check your email to verify your account."

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UsernameExistsException":
                return False, "Email already registered"
            elif error_code == "InvalidPasswordException":
                return False, "Password does not meet requirements"
            elif error_code == "InvalidParameterException":
                return False, e.response["Error"]["Message"]
            else:
                log.error("Registration error: %s - %s", error_code, e)
                return False, f"Registration failed: {error_code}"
        except Exception as e:
            log.error("Registration exception: %s", e)
            return False, "Registration failed"

    def confirm_registration(self, email: str, code: str) -> Tuple[bool, str]:
        """Confirm registration with verification code.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            kwargs = {
                "ClientId": self._config["client_id"],
                "Username": email,
                "ConfirmationCode": code,
            }

            secret_hash = self._get_secret_hash(email)
            if secret_hash:
                kwargs["SecretHash"] = secret_hash

            self._client.confirm_sign_up(**kwargs)
            return True, "Account verified successfully"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "CodeMismatchException":
                return False, "Invalid verification code"
            elif error_code == "ExpiredCodeException":
                return False, "Verification code expired"
            else:
                return False, f"Verification failed: {error_code}"
        except Exception as e:
            log.error("Confirmation exception: %s", e)
            return False, "Verification failed"

    def resend_confirmation_code(self, email: str) -> Tuple[bool, str]:
        """Resend verification code.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            kwargs = {
                "ClientId": self._config["client_id"],
                "Username": email,
            }

            secret_hash = self._get_secret_hash(email)
            if secret_hash:
                kwargs["SecretHash"] = secret_hash

            self._client.resend_confirmation_code(**kwargs)
            return True, "Verification code sent"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            return False, f"Failed to resend code: {error_code}"
        except Exception as e:
            log.error("Resend code exception: %s", e)
            return False, "Failed to resend code"

    # ── Password Management ───────────────────────────────────────────

    def forgot_password(self, email: str) -> Tuple[bool, str]:
        """Initiate password reset.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            kwargs = {
                "ClientId": self._config["client_id"],
                "Username": email,
            }

            secret_hash = self._get_secret_hash(email)
            if secret_hash:
                kwargs["SecretHash"] = secret_hash

            self._client.forgot_password(**kwargs)
            return True, "Password reset code sent to your email"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "UserNotFoundException":
                # Don't reveal if user exists
                return True, "Password reset code sent to your email"
            return False, f"Password reset failed: {error_code}"
        except Exception as e:
            log.error("Forgot password exception: %s", e)
            return False, "Password reset failed"

    def confirm_forgot_password(
        self,
        email: str,
        code: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        """Complete password reset with code.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            kwargs = {
                "ClientId": self._config["client_id"],
                "Username": email,
                "ConfirmationCode": code,
                "Password": new_password,
            }

            secret_hash = self._get_secret_hash(email)
            if secret_hash:
                kwargs["SecretHash"] = secret_hash

            self._client.confirm_forgot_password(**kwargs)
            return True, "Password reset successful"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "CodeMismatchException":
                return False, "Invalid reset code"
            elif error_code == "ExpiredCodeException":
                return False, "Reset code expired"
            elif error_code == "InvalidPasswordException":
                return False, "Password does not meet requirements"
            else:
                return False, f"Password reset failed: {error_code}"
        except Exception as e:
            log.error("Confirm forgot password exception: %s", e)
            return False, "Password reset failed"

    def change_password(
        self,
        access_token: str,
        old_password: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        """Change password for authenticated user.

        Returns (success, message).
        """
        if not self._enabled:
            return False, "Cognito authentication not configured"

        try:
            self._client.change_password(
                PreviousPassword=old_password,
                ProposedPassword=new_password,
                AccessToken=access_token,
            )
            return True, "Password changed successfully"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                return False, "Current password is incorrect"
            elif error_code == "InvalidPasswordException":
                return False, "New password does not meet requirements"
            else:
                return False, f"Password change failed: {error_code}"
        except Exception as e:
            log.error("Change password exception: %s", e)
            return False, "Password change failed"

    # ── Logout ────────────────────────────────────────────────────────

    def logout(self, access_token: str) -> bool:
        """Sign out user (invalidate tokens).

        Returns True on success.
        """
        if not self._enabled:
            return False

        try:
            self._client.global_sign_out(AccessToken=access_token)
            return True
        except ClientError:
            return False
        except Exception:
            return False
