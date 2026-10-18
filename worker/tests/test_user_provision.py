"""Offline user bootstrap tests against disposable SQLite; never contact fleet DB."""
import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlmodel import Session

import tigerchain_app.auth.database as database
from tigerchain_app.auth.models import User
from tigerchain_app.auth.provision import provision_user
from tigerchain_app.config import Settings


def test_operator_can_create_normal_user_without_enabling_self_registration(tmp_path, monkeypatch):
    # get_engine() caches one instance: reset only inside this isolated fixture.
    original = database._engine
    monkeypatch.setattr(database, "_engine", None)
    try:
        settings = Settings(
            jwt_secret_key="synthetic-provision-test-secret-must-be-long-enough",
            database_url=f"sqlite:///{tmp_path / 'users.sqlite3'}",
            enable_public_registration=False,
            admin_user_ids="",
        )
        user_id = provision_user("operator@synthetic.example", "only-for-test-password", settings=settings)
        assert user_id == 1
        with Session(database.get_engine(settings)) as session:
            user = session.get(User, user_id)
            assert user.email == "operator@synthetic.example"
            assert user.is_active
        with pytest.raises(HTTPException) as duplicate:
            provision_user("operator@synthetic.example", "only-for-test-password", settings=settings)
        assert duplicate.value.status_code == 400
    finally:
        if database._engine is not None:
            database._engine.dispose()
        monkeypatch.setattr(database, "_engine", original)


def test_operator_bootstrap_rejects_weak_signing_key_before_db_mutation(tmp_path):
    settings = Settings(
        jwt_secret_key="weak",
        database_url=f"sqlite:///{tmp_path / 'must-not-exist.sqlite3'}",
    )
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        provision_user("sample@synthetic.example", "only-for-test-password", settings=settings)
    assert not (tmp_path / "must-not-exist.sqlite3").exists()
