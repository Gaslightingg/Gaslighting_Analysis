from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from src.backtester.engine import run_backtest
from src.config import settings
from src.data_provider.yfinance_provider import get_ohlcv_daily
from src.indicators.ta import add_indicators
from src.optimizer.walk_forward import run_optimization
from src.reporter.report import save_report
from src.storage.db import SessionLocal
from src.storage.models import Backtest, Job, Optimization
from src.strategy.hybrid_vote import generate_hybrid_signals
from src.worker.celery_app import celery_app


@celery_app.task(name="optimization.run")
def run_optimization_task(job_id: int) -> dict:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if not job:
            return {"error": "job not found"}
        params = job.params_json
        job.status = "running"
        session.commit()

    df = get_ohlcv_daily(params["ticker"], params["start"], params["end"])
    best_cfg, best_metrics, _ = run_optimization(job_id, df, params.get("mode", "fast"))

    if not best_cfg:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if job:
                job.status = "stopped"
                session.commit()
        return {"status": "stopped"}

    enriched = add_indicators(df, best_cfg)
    pos = generate_hybrid_signals(enriched, best_cfg)
    bt_df, metrics = run_backtest(enriched, pos)
    equity_path, trades_path, _ = save_report(job_id, bt_df, best_cfg, settings.runs_dir)

    with SessionLocal() as session:
        optimization = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
        if optimization:
            optimization.best_config_json = best_cfg
            optimization.best_metrics_json = best_metrics
            optimization.best_updated_at = datetime.utcnow()

        bt = Backtest(
            job_id=job_id,
            metrics_json=metrics,
            equity_path=equity_path,
            trades_path=trades_path,
        )
        session.add(bt)
        job = session.get(Job, job_id)
        if job:
            job.status = "completed" if not job.stop_requested else "stopped"
        session.commit()

    return {"status": "ok", "job_id": job_id, "metrics": metrics}
