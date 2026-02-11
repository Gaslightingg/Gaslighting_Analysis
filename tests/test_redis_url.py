from __future__ import annotations

from app import candidate_redis_urls, resolve_redis_url


def test_candidate_redis_urls_localhost_fallbacks() -> None:
    urls = candidate_redis_urls("redis://localhost:6379/0")
    assert urls[0] == "redis://localhost:6379/0"
    assert "redis://127.0.0.1:6379/0" in urls
    assert "redis://[::1]:6379/0" in urls


def test_resolve_redis_url_picks_ipv4_fallback() -> None:
    def fake_ping(url: str) -> bool:
        return url == "redis://127.0.0.1:6379/0"

    chosen, tried = resolve_redis_url("redis://localhost:6379/0", ping_fn=fake_ping)
    assert chosen == "redis://127.0.0.1:6379/0"
    assert tried[0] == "redis://localhost:6379/0"
