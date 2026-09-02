"""
Cognito sign-in challenges.

Cognito does not always answer a login with tokens. An account created by
``admin-create-user`` lands in FORCE_CHANGE_PASSWORD and gets back a
NEW_PASSWORD_REQUIRED challenge; an account with MFA enabled gets an MFA
challenge. Either way the reply carries a ``Session``, and that session is
the *only* way to finish signing in.

``authenticate`` used to recognise both and then drop the session, returning
a bare "Password change required" string. Nothing could act on that, so every
admin-created account was permanently unable to log in through the dashboard
-- the hosted UI was the sole way in. It was invisible because the normal
password path, which is what anyone testing by hand would exercise, worked
fine. These tests pin the session actually reaching the caller.
"""
from __future__ import annotations

import pytest

from cognisom.auth.cognito_provider import CognitoAuthProvider, CognitoChallenge
from cognisom.auth.models import UserRole


CONFIG = {
    "user_pool_id": "us-west-2_TESTPOOL",
    "client_id": "test-client-id",
    "client_secret": "",
    "domain": "cognisom-test-auth",
    "region": "us-west-2",
}

AUTH_RESULT = {
    "AccessToken": "access-tok",
    "IdToken": "id-tok",
    "RefreshToken": "refresh-tok",
    "ExpiresIn": 3600,
    "TokenType": "Bearer",
}


class FakeCognitoClient:
    """Stands in for the boto3 cognito-idp client.

    Records what it was called with so the tests can assert on the request,
    not merely the return value -- answering a challenge with the wrong
    USERNAME is a failure mode that a return-value-only test would miss.
    """

    def __init__(self, initiate=None, respond=None, groups=("admin",)):
        self._initiate = initiate or {"AuthenticationResult": dict(AUTH_RESULT)}
        self._respond = respond or {"AuthenticationResult": dict(AUTH_RESULT)}
        self._groups = groups
        self.initiate_calls = []
        self.respond_calls = []

    def initiate_auth(self, **kwargs):
        self.initiate_calls.append(kwargs)
        if isinstance(self._initiate, Exception):
            raise self._initiate
        return self._initiate

    def respond_to_auth_challenge(self, **kwargs):
        self.respond_calls.append(kwargs)
        if isinstance(self._respond, Exception):
            raise self._respond
        return self._respond

    def get_user(self, AccessToken):  # noqa: N803 - boto3 casing
        return {
            "Username": "18d1a3e0-cognito-sub",
            "UserAttributes": [
                {"Name": "email", "Value": "researcher@lab.org"},
                {"Name": "name", "Value": "A Researcher"},
                {"Name": "custom:organization", "Value": "lab"},
            ],
        }

    def admin_list_groups_for_user(self, **kwargs):
        return {"Groups": [{"GroupName": g} for g in self._groups]}


def _provider(client: FakeCognitoClient) -> CognitoAuthProvider:
    p = CognitoAuthProvider(config=dict(CONFIG))
    p._client = client
    return p


def _client_error(code: str, message: str = "boom"):
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": code, "Message": message}}, "Op")


# ── authenticate() surfaces the challenge ────────────────────────────

def test_new_password_challenge_reaches_the_caller_with_its_session():
    """The session must survive authenticate(); without it nothing can respond."""
    client = FakeCognitoClient(initiate={
        "ChallengeName": "NEW_PASSWORD_REQUIRED",
        "Session": "session-abc",
        "ChallengeParameters": {"USER_ID_FOR_SRP": "18d1a3e0-cognito-sub"},
    })
    tokens, user, msg, challenge = _provider(client).authenticate("me@lab.org", "Temp123!")

    assert tokens is None and user is None
    assert challenge is not None, "the challenge was dropped -- the original bug"
    assert challenge.name == "NEW_PASSWORD_REQUIRED"
    assert challenge.session == "session-abc"
    assert "new password" in msg.lower()


def test_challenge_uses_the_identifier_cognito_asks_for():
    """Cognito wants USER_ID_FOR_SRP, not the typed email, for admin-made accounts."""
    client = FakeCognitoClient(initiate={
        "ChallengeName": "NEW_PASSWORD_REQUIRED",
        "Session": "s",
        "ChallengeParameters": {"USER_ID_FOR_SRP": "18d1a3e0-cognito-sub"},
    })
    *_, challenge = _provider(client).authenticate("me@lab.org", "Temp123!")
    assert challenge.username == "18d1a3e0-cognito-sub"


def test_challenge_falls_back_to_the_typed_username():
    client = FakeCognitoClient(initiate={
        "ChallengeName": "NEW_PASSWORD_REQUIRED", "Session": "s",
    })
    *_, challenge = _provider(client).authenticate("me@lab.org", "Temp123!")
    assert challenge.username == "me@lab.org"


def test_mfa_challenge_also_carries_its_session():
    """Same defect, same fix: MFA was equally unanswerable."""
    client = FakeCognitoClient(initiate={
        "ChallengeName": "SOFTWARE_TOKEN_MFA", "Session": "mfa-session",
    })
    *_, challenge = _provider(client).authenticate("me@lab.org", "pw")
    assert challenge.name == "SOFTWARE_TOKEN_MFA"
    assert challenge.session == "mfa-session"


def test_ordinary_login_returns_no_challenge():
    client = FakeCognitoClient()
    tokens, user, msg, challenge = _provider(client).authenticate("me@lab.org", "pw")
    assert challenge is None
    assert tokens is not None and tokens.access_token == "access-tok"
    assert user is not None and user.role == UserRole.ADMIN


def test_bad_password_returns_no_challenge():
    """A rejected login must not look like a pending challenge."""
    client = FakeCognitoClient(initiate=_client_error("NotAuthorizedException"))
    tokens, user, msg, challenge = _provider(client).authenticate("me@lab.org", "wrong")
    assert (tokens, user, challenge) == (None, None, None)
    assert msg == "Invalid username or password"


# ── responding to the challenge ──────────────────────────────────────

def test_responding_completes_the_sign_in():
    client = FakeCognitoClient()
    challenge = CognitoChallenge("NEW_PASSWORD_REQUIRED", "session-abc", "18d1a3e0-cognito-sub")
    tokens, user, msg, nxt = _provider(client).respond_to_new_password_challenge(
        challenge, "A-New-Passw0rd!"
    )

    assert tokens is not None and tokens.access_token == "access-tok"
    assert user is not None and user.email == "researcher@lab.org"
    assert nxt is None

    sent = client.respond_calls[0]
    assert sent["ChallengeName"] == "NEW_PASSWORD_REQUIRED"
    assert sent["Session"] == "session-abc"
    assert sent["ChallengeResponses"]["USERNAME"] == "18d1a3e0-cognito-sub"
    assert sent["ChallengeResponses"]["NEW_PASSWORD"] == "A-New-Passw0rd!"


def test_weak_password_surfaces_the_pools_own_policy_text():
    """The pool's message names the actual rule; a generic string does not."""
    client = FakeCognitoClient(respond=_client_error(
        "InvalidPasswordException", "Password must have symbol characters"
    ))
    challenge = CognitoChallenge("NEW_PASSWORD_REQUIRED", "s", "u")
    tokens, _, msg, _ = _provider(client).respond_to_new_password_challenge(challenge, "weak")
    assert tokens is None
    assert "symbol characters" in msg


def test_expired_session_tells_the_user_to_start_again():
    """Challenge sessions are single-use; retrying in place cannot work."""
    client = FakeCognitoClient(respond=_client_error("NotAuthorizedException"))
    challenge = CognitoChallenge("NEW_PASSWORD_REQUIRED", "stale", "u")
    tokens, _, msg, _ = _provider(client).respond_to_new_password_challenge(challenge, "Pw1!")
    assert tokens is None
    assert "log in again" in msg.lower()


def test_a_stacked_challenge_is_passed_back():
    """Some pools put MFA setup behind the password change."""
    client = FakeCognitoClient(respond={
        "ChallengeName": "MFA_SETUP", "Session": "next-session",
    })
    challenge = CognitoChallenge("NEW_PASSWORD_REQUIRED", "s", "u")
    tokens, _, _, nxt = _provider(client).respond_to_new_password_challenge(challenge, "Pw1!")
    assert tokens is None
    assert nxt is not None and nxt.name == "MFA_SETUP"
    assert nxt.session == "next-session"


def test_disabled_provider_reports_rather_than_crashing():
    p = CognitoAuthProvider(config={**CONFIG, "user_pool_id": "", "client_id": ""})
    challenge = CognitoChallenge("NEW_PASSWORD_REQUIRED", "s", "u")
    assert p.respond_to_new_password_challenge(challenge, "Pw1!")[0] is None
    assert p.authenticate("me@lab.org", "pw")[3] is None
