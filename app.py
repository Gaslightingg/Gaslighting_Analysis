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


def _try_start_redis(redis_url: str) -> subprocess.Popen | None:
    if _is_redis_available(redis_url):
        return None

    startup_cmds = [
        ["docker", "compose", "up", "-d", "redis"],
        ["docker-compose", "up", "-d", "redis"],
    ]

    for cmd in startup_cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if proc.returncode == 0:
            for _ in range(12):
                if _is_redis_available(redis_url):
                    print("[startup-info] Redis started via docker compose.")
                    return None
                time.sleep(1)

    try:
        redis_proc = subprocess.Popen(["redis-server"])
    except Exception:
        return None

    for _ in range(8):
        if _is_redis_available(redis_url):
            print("[startup-info] Redis started via redis-server process.")
            return redis_proc
        time.sleep(1)

    return redis_proc


def run_all() -> int:
    redis_proc = _try_start_redis(SETTINGS.redis_url)

    if not _is_redis_available(SETTINGS.redis_url):
        print(f"[startup-warning] Redis is still unavailable at {SETTINGS.redis_url}.")
        print("[startup-warning] Worker will start anyway, but queue tasks may fail until Redis is up.")

    worker_cmd = ["celery", "-A", "src.worker.celery_app", "worker", "-l", "INFO"]
    worker_proc = subprocess.Popen(worker_cmd)

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
        _stop_process(worker_proc)
        if redis_proc is not None:
            _stop_process(redis_proc)


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
