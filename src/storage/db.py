from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import SETTINGS


class Base(DeclarativeBase):
    pass


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if not database or database == ":memory:":
        return
    db_path = Path(database)
    parent = db_path.parent
    if str(parent) not in {"", "."}:
        parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir(SETTINGS.database_url)
engine = create_engine(SETTINGS.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    from src.storage import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
