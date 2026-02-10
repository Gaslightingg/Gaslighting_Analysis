from __future__ import annotations

from datetime import date

from src.data_provider.provider import get_ohlcv
from src.optimizer.optuna_runner import run_optimization_job
from src.storage.repository import Repository
from src.worker.celery_app import app


@app.task(name="optimization.run")
def optimization_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    try:
        end_date = date.fromisoformat(params["end"])
        if end_date > date.today():
            repo.set_job_status(job_id, "failed")
            repo.update_progress(
                job_id=job_id,
                trials_done=0,
                trials_total=int(params.get("trials_total", 0)),
                last_score=0.0,
                best_score=None,
                state="failed",
                reason="Дата окончания в будущем. Выберите период до сегодняшнего дня.",
            )
            return {"status": "failed", "error": "end date is in the future"}
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        return {"status": "failed", "error": "invalid date params"}

    try:
        df = get_ohlcv(params["ticker"], params["start"], params["end"])
        if df.empty:
            repo.set_job_status(job_id, "failed")
            repo.update_progress(
                job_id=job_id,
                trials_done=0,
                trials_total=int(params.get("trials_total", 0)),
                last_score=0.0,
                best_score=None,
                state="failed",
                reason="Нет исторических данных для выбранного периода.",
            )
            return {"status": "failed", "error": "empty data"}
    except Exception as exc:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        return {"status": "failed", "error": str(exc)}

    return run_optimization_job(job_id, df)
