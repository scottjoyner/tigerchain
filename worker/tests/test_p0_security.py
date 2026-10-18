"""P0 authentication and query-contract regression tests (requires service deps)."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from tigerchain_app.config import Settings
from tigerchain_app.auth.security import require_admin_user, require_signing_key


def test_signing_key_missing_or_known_defaults_fail_closed():
    for value in (None, "", "change-this-secret", "short"):
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
            require_signing_key(Settings(jwt_secret_key=value))
    key = "unique-signing-key-material-with-at-least-32-bytes"
    assert require_signing_key(Settings(jwt_secret_key=key)) == key


def test_agent_administration_denied_without_explicit_allowlist():
    user = SimpleNamespace(id=42)
    with pytest.raises(HTTPException) as exc:
        require_admin_user(user, Settings(admin_user_ids=""))
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        require_admin_user(user, Settings(admin_user_ids="43, 99"))
    assert exc.value.status_code == 403
    assert require_admin_user(user, Settings(admin_user_ids="41, 42, 43")) == user



def test_password_hash_and_verify():
    from tigerchain_app.auth.security import get_password_hash, verify_password
    candidate = "nonsecret-test-password"
    assert verify_password(candidate, get_password_hash(candidate))
    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        get_password_hash("a" * 73)
