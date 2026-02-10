from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse

from src.bot.main import start_bot_sync
from src.config import SETTINGS
from src.storage.db import init_db


def run_worker_foreground() -> int:
    cmd = ["celery", "-A", "src.worker.celery_app", "worker", "-l", "INFO"]
    return subprocess.call(cmd)


def _is_redis_available(redis_url: str, timeout_s: float = 1.5) -> bool:
    parsed = urlparse(redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


def run_all() -> int:
    worker_proc: subprocess.Popen | None = None
    if _is_redis_available(SETTINGS.redis_url):
        worker_cmd = ["celery", "-A", "src.worker.celery_app", "worker", "-l", "INFO"]
        worker_proc = subprocess.Popen(worker_cmd)
    else:
        print(f"[startup-warning] Redis is unavailable at {SETTINGS.redis_url}. Worker not started.")
        print("[startup-warning] Start Redis first (e.g. `docker-compose up -d redis`) to enable optimization queue.")

    try:
        time.sleep(1)
        start_bot_sync()
        return 0
    except RuntimeError as exc:
        print(f"[startup-error] {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if worker_proc is not None:
            _stop_process(worker_proc)


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="telegram-trading-lab")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "bot", "worker"],
        help="run mode (default: all)",
    )
    args = parser.parse_args()

    init_db()

    if args.mode == "all":
        return run_all()
    if args.mode == "bot":
        try:
            start_bot_sync()
        except RuntimeError as exc:
            print(f"[startup-error] {exc}")
            return 1
        return 0
    return run_worker_foreground()


if __name__ == "__main__":
    raise SystemExit(main())
