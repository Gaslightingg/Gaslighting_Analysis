from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(50), default="optimization")
    status: Mapped[str] = mapped_column(String(30), default="pending")
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    progress_json: Mapped[dict] = mapped_column(JSON, default=dict)
    stop_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class Optimization(Base):
    __tablename__ = "optimizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(16))
    start: Mapped[datetime] = mapped_column(Date)
    end: Mapped[datetime] = mapped_column(Date)
    best_config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    best_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    best_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    checkpoints: Mapped[list[OptimizationCheckpoint]] = relationship(back_populates="optimization")


class OptimizationCheckpoint(Base):
    __tablename__ = "optimization_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("optimizations.job_id"), index=True)
    checkpoint_no: Mapped[int] = mapped_column(Integer)
    trials_done: Mapped[int] = mapped_column(Integer)
    best_config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    best_metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    optimization: Mapped[Optimization] = relationship(back_populates="checkpoints")


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    equity_path: Mapped[str] = mapped_column(Text)
    trades_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
