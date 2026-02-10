from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./trading_lab.db")
    runs_dir: str = os.getenv("RUNS_DIR", ".runs")


settings = Settings()
