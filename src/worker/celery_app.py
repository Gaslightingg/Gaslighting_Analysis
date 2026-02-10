from __future__ import annotations

import os

from celery import Celery

from src.config import SETTINGS


def get_effective_redis_url() -> str:
    return os.getenv("REDIS_URL_EFFECTIVE") or os.getenv("REDIS_URL") or SETTINGS.redis_url


broker_url = get_effective_redis_url()
app = Celery("telegram_trading_lab", broker=broker_url, backend=broker_url)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.task_track_started = True

app.autodiscover_tasks(["src.worker"])
