from __future__ import annotations

from celery import Celery

from src.config import SETTINGS

app = Celery("telegram_trading_lab", broker=SETTINGS.redis_url, backend=SETTINGS.redis_url)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.task_track_started = True

app.autodiscover_tasks(["src.worker"])
