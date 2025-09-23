from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from ..config import Settings, get_settings
from .database import get_session
from .models import User
from .schemas import OnboardingRequest, Token, UserCreate, UserRead
from .security import create_access_token, get_current_active_user
from .service import UserService


router = APIRouter()


@router.post("/register", response_model=UserRead)
def register_user(
    payload: UserCreate,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> UserRead:
    service = UserService(session, settings)
    user = service.create_user(payload)
    return UserRead.model_validate(user)


@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Token:
    service = UserService(session, settings)
    user = service.authenticate_user(form_data.username, form_data.password)
    token = create_access_token(subject=str(user.id), settings=settings)
    return Token(access_token=token)


@router.get("/me", response_model=UserRead)
def read_current_user(current_user: User = Depends(get_current_active_user)) -> UserRead:
    return UserRead.model_validate(current_user)


@router.post("/onboarding", response_model=UserRead)
def complete_onboarding(
    payload: OnboardingRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    current_user: User = Depends(get_current_active_user),
) -> UserRead:
    service = UserService(session, settings)
    user = service.update_user_preferences(current_user, payload)
    return UserRead.model_validate(user)
