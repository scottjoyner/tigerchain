from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    """Persistent user representation used for authentication and onboarding."""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: Optional[str] = Field(default=None)
    hashed_password: str
    is_active: bool = Field(default=True)
    onboarding_complete: bool = Field(default=False)
    preferred_agent: Optional[str] = Field(default=None, index=True)
    categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class DocumentUpload(SQLModel, table=True):
    """Tracks document ingestion activity for auditing and filtering."""

    __tablename__ = "document_uploads"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    doc_id: str = Field(index=True)
    filename: str
    categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    model_alias: Optional[str] = Field(default=None, index=True)
    object_uri: Optional[str] = Field(default=None)
    http_url: Optional[str] = Field(default=None)
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
