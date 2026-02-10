from __future__ import annotations

from celery import Celery

from src.config import settings

celery_app = Celery("trading_lab", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_track_started = True
celery_app.conf.result_expires = 3600 * 24
celery_app.conf.task_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.result_serializer = "json"
celery_app.autodiscover_tasks(["src.worker"])
