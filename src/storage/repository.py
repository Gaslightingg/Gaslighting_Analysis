from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import select

from src.storage.db import SessionLocal
from src.storage.models import Job, Optimization, OptimizationCheckpoint


class Repository:
    def create_job(
        self,
        user_id: int,
        chat_id: int,
        ticker: str,
        start: str,
        end: str,
        preset: str,
        trials_total: int,
        checkpoint_n: int,
    ) -> str:
        job_id = str(uuid.uuid4())
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "preset": preset,
            "trials_total": trials_total,
            "checkpoint_n": checkpoint_n,
        }
        progress = {
            "trials_done": 0,
            "trials_total": trials_total,
            "best_score": None,
            "last_score": None,
            "updated_at": datetime.utcnow().isoformat(),
            "state": "queued",
        }
        with SessionLocal() as session:
            session.add(
                Job(
                    id=job_id,
                    user_id=str(user_id),
                    chat_id=str(chat_id),
                    status="queued",
                    type="optimize",
                    params_json=json.dumps(params),
                    progress_json=json.dumps(progress),
                    stop_requested=False,
                )
            )
            session.add(
                Optimization(
                    job_id=job_id,
                    ticker=ticker,
                    start=start,
                    end=end,
                )
            )
            session.commit()
        return job_id

    def create_preload_job(self, user_id: int, chat_id: int, tickers: list[str], horizon: str) -> str:
        job_id = str(uuid.uuid4())
        params = {"tickers": tickers, "horizon": horizon}
        progress = {
            "done": 0,
            "total": len(tickers),
            "items": [],
            "updated_at": datetime.utcnow().isoformat(),
            "state": "queued",
        }
        with SessionLocal() as session:
            session.add(
                Job(
                    id=job_id,
                    user_id=str(user_id),
                    chat_id=str(chat_id),
                    status="queued",
                    type="preload",
                    params_json=json.dumps(params),
                    progress_json=json.dumps(progress),
                    stop_requested=False,
                )
            )
            session.commit()
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        with SessionLocal() as session:
            return session.get(Job, job_id)

    def get_job_data(self, job_id: str) -> dict | None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return None
            opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
            return {
                "job": job,
                "params": json.loads(job.params_json or "{}"),
                "progress": json.loads(job.progress_json or "{}"),
                "optimization": opt,
            }

    def set_job_status(self, job_id: str, status: str) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            job.status = status
            progress = json.loads(job.progress_json or "{}")
            progress["state"] = status
            progress["updated_at"] = datetime.utcnow().isoformat()
            job.progress_json = json.dumps(progress)
            session.commit()

    def request_stop(self, job_id: str) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            job.stop_requested = True
            job.status = "stopping"
            progress = json.loads(job.progress_json or "{}")
            progress["state"] = "stopping"
            progress["updated_at"] = datetime.utcnow().isoformat()
            job.progress_json = json.dumps(progress)
            session.commit()

    def stop_requested(self, job_id: str) -> bool:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            return bool(job and job.stop_requested)

    def update_progress(
        self,
        job_id: str,
        trials_done: int,
        trials_total: int,
        last_score: float,
        best_score: float | None,
        state: str = "running",
        reason: str | None = None,
        data_info: dict | None = None,
    ) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            progress = json.loads(job.progress_json or "{}")
            payload = {
                "trials_done": trials_done,
                "trials_total": trials_total,
                "last_score": last_score,
                "best_score": best_score,
                "updated_at": datetime.utcnow().isoformat(),
                "state": state,
            }
            if reason:
                payload["reason"] = reason
            if data_info:
                payload["data_info"] = data_info
            progress.update(payload)
            job.progress_json = json.dumps(progress)
            job.status = state
            session.commit()

    def update_preload_progress(self, job_id: str, done: int, total: int, item: dict, state: str = "running") -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            progress = json.loads(job.progress_json or "{}")
            items = list(progress.get("items", []))
            items.append(item)
            progress.update(
                {
                    "done": done,
                    "total": total,
                    "items": items,
                    "state": state,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            )
            job.progress_json = json.dumps(progress)
            job.status = state
            session.commit()

    def update_progress_last_trial(self, job_id: str, last_trial: dict) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job:
                return
            progress = json.loads(job.progress_json or "{}")
            prev_last = progress.get("last_trial") or {}
            merged_last = dict(prev_last)
            merged_last.update(last_trial)
            progress["last_trial"] = merged_last
            progress["updated_at"] = datetime.utcnow().isoformat()
            job.progress_json = json.dumps(progress)
            session.commit()

    def save_best(
        self,
        job_id: str,
        best_config: dict,
        best_metrics: dict,
        equity_path: str,
        trades_path: str,
        config_path: str,
    ) -> None:
        with SessionLocal() as session:
            opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
            if not opt:
                return
            opt.best_config_json = json.dumps(best_config)
            opt.best_metrics_json = json.dumps(best_metrics)
            opt.best_updated_at = datetime.utcnow()
            opt.equity_path = equity_path
            opt.trades_path = trades_path
            opt.config_path = config_path
            session.commit()

    def add_checkpoint(self, job_id: str, checkpoint_no: int, trials_done: int) -> None:
        with SessionLocal() as session:
            opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
            if not opt:
                return
            session.add(
                OptimizationCheckpoint(
                    job_id=job_id,
                    checkpoint_no=checkpoint_no,
                    trials_done=trials_done,
                    best_config_json=opt.best_config_json,
                    best_metrics_json=opt.best_metrics_json,
                )
            )
            session.commit()

    def get_best(self, job_id: str) -> dict | None:
        with SessionLocal() as session:
            opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
            if not opt:
                return None
            return {
                "job_id": job_id,
                "config": json.loads(opt.best_config_json or "{}"),
                "metrics": json.loads(opt.best_metrics_json or "{}"),
                "best_updated_at": opt.best_updated_at,
                "equity_path": opt.equity_path,
                "trades_path": opt.trades_path,
                "config_path": opt.config_path,
            }

    def list_jobs(self, user_id: int, limit: int = 10) -> list[Job]:
        with SessionLocal() as session:
            stmt = (
                select(Job)
                .where(Job.user_id == str(user_id))
                .order_by(Job.created_at.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    def get_last_active_job(self, user_id: int) -> Job | None:
        with SessionLocal() as session:
            stmt = (
                select(Job)
                .where(Job.user_id == str(user_id))
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            return session.scalar(stmt)
