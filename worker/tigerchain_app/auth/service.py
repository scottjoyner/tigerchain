from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

from fastapi import HTTPException, status
from sqlmodel import Session, select

from ..config import Settings
from ..utils.logging import get_logger
from .models import DocumentUpload, User
from .schemas import OnboardingRequest, UserCreate
from .security import get_password_hash, verify_password

logger = get_logger(__name__)


class UserService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------
    def _validate_agent(self, agent: Optional[str]) -> Optional[str]:
        if agent is None:
            return None
        if agent not in self.settings.model_registry:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown agent '{agent}'")
        return agent

    def _normalise_categories(self, categories: Optional[Iterable[str]]) -> List[str]:
        if not categories:
            return []
        unique = {c.strip() for c in categories if c and c.strip()}
        return sorted(unique)

    def create_user(self, payload: UserCreate) -> User:
        existing = self.session.exec(select(User).where(User.email == payload.email.lower())).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        preferred_agent = self._validate_agent(payload.preferred_agent) or self.settings.default_agent
        self._validate_agent(preferred_agent)

        user = User(
            email=payload.email.lower(),
            full_name=payload.full_name,
            hashed_password=get_password_hash(payload.password),
            preferred_agent=preferred_agent,
            categories=self._normalise_categories(payload.categories),
            onboarding_complete=False,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        logger.info("Created user %s", user.email)
        return user

    def authenticate_user(self, email: str, password: str) -> User:
        statement = select(User).where(User.email == email.lower())
        user = self.session.exec(statement).first()
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
        return user

    def update_user_preferences(self, user: User, payload: OnboardingRequest) -> User:
        updated = False
        if payload.full_name is not None:
            user.full_name = payload.full_name
            updated = True
        if payload.preferred_agent is not None:
            user.preferred_agent = self._validate_agent(payload.preferred_agent) or user.preferred_agent
            updated = True
        if payload.categories is not None:
            user.categories = self._normalise_categories(payload.categories)
            updated = True
        if updated:
            user.updated_at = datetime.utcnow()
        user.onboarding_complete = True
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user


class DocumentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record_upload(
        self,
        *,
        user_id: Optional[int],
        doc_id: str,
        filename: str,
        categories: Iterable[str],
        model_alias: Optional[str],
        object_uri: Optional[str],
        http_url: Optional[str],
        metadata: Optional[dict] = None,
        submission_id: Optional[str] = None,
        embedding_scope: Optional[str] = None,
        sharing_preference: Optional[str] = None,
        private_embedding_uri: Optional[str] = None,
    ) -> DocumentUpload:
        record = DocumentUpload(
            user_id=user_id,
            doc_id=doc_id,
            filename=filename,
            categories=sorted({c.strip() for c in categories if c}),
            model_alias=model_alias,
            object_uri=object_uri,
            http_url=http_url,
            metadata=metadata or {},
            submission_id=submission_id,
            embedding_scope=embedding_scope,
            sharing_preference=sharing_preference,
            private_embedding_uri=private_embedding_uri,
        )
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record
