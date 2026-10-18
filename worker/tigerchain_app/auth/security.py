from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session

from ..config import Settings, get_settings
from .database import get_session
from .models import User
from .schemas import TokenData


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def require_signing_key(settings: Settings) -> str:
    """Refuse missing, known, or short shared signing keys."""
    secret = settings.jwt_secret_key
    if not secret or len(secret.encode("utf-8")) < 32 or secret in {"change-this-secret", "changeme", "secret"}:
        raise RuntimeError("JWT_SECRET_KEY must be a unique secret of at least 32 bytes")
    return secret


def create_access_token(*, subject: str, settings: Settings, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode = {"sub": subject, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, require_signing_key(settings), algorithm=settings.jwt_algorithm)
    return encoded_jwt


def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, require_signing_key(settings), algorithms=[settings.jwt_algorithm])
        subject: str | None = payload.get("sub")
        if subject is None:
            raise credentials_exception
        token_data = TokenData(sub=subject)
    except JWTError as exc:  # pragma: no cover - defensive path
        raise credentials_exception from exc

    try:
        user_id = int(token_data.sub) if token_data.sub is not None else None
    except (TypeError, ValueError) as exc:
        raise credentials_exception from exc

    user = session.get(User, user_id) if user_id is not None else None
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


def require_admin_user(
    current_user: User = Depends(get_current_active_user),
    settings: Settings = Depends(get_settings),
) -> User:
    allowed = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
    if str(current_user.id) not in allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return current_user
