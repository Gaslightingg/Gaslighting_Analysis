from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.storage.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    chat_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    type: Mapped[str] = mapped_column(String(32), default="optimize")
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    progress_json: Mapped[str] = mapped_column(Text, default="{}")
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class Optimization(Base):
    __tablename__ = "optimizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(32))
    start: Mapped[str] = mapped_column(String(16))
    end: Mapped[str] = mapped_column(String(16))
    best_config_json: Mapped[str] = mapped_column(Text, default="{}")
    best_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    best_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    equity_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    trades_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class OptimizationCheckpoint(Base):
    __tablename__ = "optimization_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"), index=True)
    checkpoint_no: Mapped[int] = mapped_column(Integer)
    trials_done: Mapped[int] = mapped_column(Integer)
    best_config_json: Mapped[str] = mapped_column(Text, default="{}")
    best_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
