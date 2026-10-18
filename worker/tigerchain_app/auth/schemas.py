from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_bytes_fit_bcrypt(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value
    full_name: Optional[str] = None
    preferred_agent: Optional[str] = None
    categories: Optional[List[str]] = None


class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]
    is_active: bool
    onboarding_complete: bool
    preferred_agent: Optional[str]
    categories: List[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: Optional[str] = None


class OnboardingRequest(BaseModel):
    full_name: Optional[str] = None
    preferred_agent: Optional[str] = None
    categories: Optional[List[str]] = None


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    categories: List[str]
    model_alias: Optional[str]
    object_uri: Optional[str]
    http_url: Optional[str]
    metadata: Optional[dict] = Field(default=None, validation_alias="extra_metadata")
    submission_id: Optional[str]
    embedding_scope: Optional[str]
    sharing_preference: Optional[str]
    private_embedding_uri: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
