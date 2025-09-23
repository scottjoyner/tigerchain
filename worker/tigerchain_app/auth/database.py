from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from fastapi import Depends
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from ..config import Settings, get_settings

_engine: Engine | None = None


def get_engine(settings: Settings) -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    _engine = create_engine(settings.database_url, echo=False, connect_args=connect_args)
    return _engine


def init_db(settings: Settings) -> None:
    engine = get_engine(settings)
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(settings: Settings) -> Iterator[Session]:
    engine = get_engine(settings)
    with Session(engine) as session:
        yield session


def get_session(settings: Settings = Depends(get_settings)) -> Iterator[Session]:
    engine = get_engine(settings)
    with Session(engine) as session:
        yield session
