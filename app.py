from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import SplitResult, urlsplit, urlunsplit

from src.config import SETTINGS
from src.storage.db import init_db


def normalize_redis_url(redis_url: str) -> str:
    parsed = urlsplit(redis_url)
    scheme = parsed.scheme or "redis"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    path = parsed.path or "/0"

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    host_display = f"[{host}]" if ":" in host and not host.startswith("[") else host
    netloc = f"{userinfo}{host_display}:{port}"
    return urlunsplit(SplitResult(scheme, netloc, path, parsed.query, parsed.fragment))


def _replace_host(redis_url: str, new_host: str) -> str:
    parsed = urlsplit(redis_url)
    scheme = parsed.scheme or "redis"
    port = parsed.port or 6379

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    host_display = f"[{new_host}]" if ":" in new_host and not new_host.startswith("[") else new_host
    netloc = f"{userinfo}{host_display}:{port}"
    return urlunsplit(SplitResult(scheme, netloc, parsed.path or "/0", parsed.query, parsed.fragment))


def candidate_redis_urls(redis_url: str) -> list[str]:
    base = normalize_redis_url(redis_url)
    parsed = urlsplit(base)
    host = parsed.hostname or ""

    out: list[str] = [base]
    if host == "localhost":
        out.append(_replace_host(base, "127.0.0.1"))
        out.append(_replace_host(base, "::1"))

    unique: list[str] = []
    for item in out:
        if item not in unique:
            unique.append(item)
    return unique


def _tcp_ping_redis(redis_url: str, timeout_s: float = 1.0) -> bool:
    parsed = urlsplit(redis_url)
    host = parsed.hostname
    port = parsed.port or 6379
    if not host:
        return False

    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return False

    for family, socktype, proto, _canonname, sockaddr in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(timeout_s)
                sock.connect(sockaddr)
                return True
        except OSError:
            continue
    return False


def resolve_redis_url(redis_url: str, ping_fn=_tcp_ping_redis) -> tuple[str | None, list[str]]:
    tried = candidate_redis_urls(redis_url)
    for candidate in tried:
        if ping_fn(candidate):
            return candidate, tried
    return None, tried


def _run_compose_start() -> bool:
    commands = [
        ["docker", "compose", "up", "-d", "redis"],
        ["docker-compose", "up", "-d", "redis"],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        except Exception:
            continue
        if proc.returncode == 0:
            return True
    return False


def ensure_redis(redis_url: str) -> tuple[bool, str | None, list[str], str]:
    chosen, tried = resolve_redis_url(redis_url)
    if chosen:
        return True, chosen, tried, "already_running"

    compose_ok = _run_compose_start()
    if compose_ok:
        for _ in range(15):
            chosen, tried = resolve_redis_url(redis_url)
            if chosen:
                return True, chosen, tried, "docker_compose"
            time.sleep(1)

    return False, None, tried, "compose_failed_or_redis_unreachable"


def _set_effective_redis_url(url: str) -> None:
    os.environ["REDIS_URL"] = url
    os.environ["REDIS_URL_EFFECTIVE"] = url


def _worker_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "src.worker.celery_app",
        "worker",
        "-l",
        "INFO",
        "-P",
        "solo",
    ]


def _print_fatal_redis(redis_url: str, tried: list[str]) -> None:
    print(f"[fatal] Redis is unavailable for URL: {redis_url}")
    print("[fatal] Tried variants:")
    for item in tried:
        print(f"  - {item}")
    print("[fatal] Could not auto-start Redis via docker compose.")
    print("[fatal] Windows quick start:")
    print("  1) docker compose up -d redis")
    print("  2) python app.py doctor")
    print("  3) python app.py all")


def run_worker_foreground() -> int:
    ok, chosen, tried, _source = ensure_redis(SETTINGS.redis_url)
    if not ok or not chosen:
        _print_fatal_redis(SETTINGS.redis_url, tried)
        return 1

    _set_effective_redis_url(chosen)
    return subprocess.call(_worker_command())


def run_all() -> int:
    ok, chosen, tried, _source = ensure_redis(SETTINGS.redis_url)
    if not ok or not chosen:
        _print_fatal_redis(SETTINGS.redis_url, tried)
        return 1

    _set_effective_redis_url(chosen)
    worker_proc = subprocess.Popen(_worker_command())

    try:
        time.sleep(1)
        from src.bot.main import start_bot_sync

        start_bot_sync()
        return 0
    except RuntimeError as exc:
        print(f"[startup-error] {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        _stop_process(worker_proc)


def run_doctor() -> int:
    print(f"REDIS_URL config: {SETTINGS.redis_url}")
    chosen, tried = resolve_redis_url(SETTINGS.redis_url)
    print("Tried variants:")
    for item in tried:
        print(f"- {item}")

    if chosen:
        print(f"Selected URL: {chosen}")
        print("PING: OK")
        print("Advice: python app.py all")
        return 0

    print("Selected URL: <none>")
    print("PING: FAIL")
    print("Advice:")
    print("- docker compose up -d redis")
    print("- python app.py doctor")
    print("- python app.py all")
    return 1


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
        choices=["all", "bot", "worker", "doctor"],
        help="run mode (default: all)",
    )
    args = parser.parse_args()

    init_db()

    if args.mode == "all":
        return run_all()
    if args.mode == "bot":
        try:
            from src.bot.main import start_bot_sync

            start_bot_sync()
        except RuntimeError as exc:
            print(f"[startup-error] {exc}")
            return 1
        return 0
    if args.mode == "doctor":
        return run_doctor()
    return run_worker_foreground()


if __name__ == "__main__":
    raise SystemExit(main())
