from __future__ import annotations

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
        df = get_ohlcv(params["ticker"], params["start"], params["end"])
    except Exception as exc:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        return {"status": "failed", "error": str(exc)}

    return run_optimization_job(job_id, df)
