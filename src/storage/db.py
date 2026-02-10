from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import SETTINGS


class Base(DeclarativeBase):
    pass


engine = create_engine(SETTINGS.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from src.storage import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
