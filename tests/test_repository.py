from __future__ import annotations

from src.storage.db import init_db
from src.storage.repository import Repository


def test_create_job_and_get_best_empty() -> None:
    init_db()
    repo = Repository()
    job_id = repo.create_job(
        user_id=1,
        chat_id=1,
        ticker="AAPL",
        start="2024-01-01",
        end="2025-01-01",
        preset="quick",
        trials_total=10,
        checkpoint_n=2,
    )
    info = repo.get_job_data(job_id)
    assert info is not None
    assert info["params"]["ticker"] == "AAPL"
    assert repo.get_best(job_id) is not None
