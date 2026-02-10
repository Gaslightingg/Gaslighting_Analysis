from __future__ import annotations

import argparse
import signal
import subprocess
import time

from src.bot.main import start_bot_sync
from src.storage.db import init_db


def run_worker_foreground() -> int:
    cmd = ["celery", "-A", "src.worker.celery_app", "worker", "-l", "INFO"]
    return subprocess.call(cmd)


def run_all() -> int:
    worker_cmd = ["celery", "-A", "src.worker.celery_app", "worker", "-l", "INFO"]
    worker_proc = subprocess.Popen(worker_cmd)
    try:
        time.sleep(2)
        start_bot_sync()
        return 0
    except RuntimeError as exc:
        print(f"[startup-error] {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
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
