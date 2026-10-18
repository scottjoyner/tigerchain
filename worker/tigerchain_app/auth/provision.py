"""Operator-only local user provisioning. No HTTP registration or password CLI flags.

Run on an approved host with a real TTY:
  python -m tigerchain_app.auth.provision --email user@company.example

Requires DATABASE_URL and JWT_SECRET_KEY in the local environment. Never run
against a production database without an operator-approved change window.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from .database import init_db, session_scope
from .schemas import UserCreate
from .security import require_signing_key
from .service import UserService
from ..config import Settings, get_settings


def provision_user(email: str, password: str, *, settings: Settings, full_name: str | None = None) -> int:
    """Create a normal user directly in the configured metadata database.

    Does not assign administrator rights. Authorization is controlled by a
    separate explicit ADMIN_USER_IDS deployment setting.
    """
    require_signing_key(settings)
    payload = UserCreate(email=email, password=password, full_name=full_name)
    init_db(settings)
    with session_scope(settings) as session:
        user = UserService(session, settings).create_user(payload)
        assert user.id is not None
        return int(user.id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision one TigerChain user locally")
    parser.add_argument("--email", required=True, help="Email of the new user")
    parser.add_argument("--full-name", default=None)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("A real interactive terminal is required; passwords are never command-line arguments")
    password = getpass.getpass("New password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        parser.error("Passwords do not match")
    user_id = provision_user(args.email, password, settings=get_settings(), full_name=args.full_name)
    print(f"User created with ID {user_id}; no administrator privileges assigned.")


if __name__ == "__main__":
    main()
