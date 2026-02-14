from __future__ import annotations

from src.bot.render import render_job_card, render_preload_card


def test_render_job_card_shows_checked_at() -> None:
    text = render_job_card(
        job_id="jid",
        ticker="SPY",
        start="2024-01-01",
        end="2024-12-31",
        preset="quick",
        progress={"trials_done": 1, "trials_total": 10, "state": "running", "updated_at": "u", "checked_at": "c"},
        status="running",
    )
    assert "Проверено" in text
    assert "<code>c</code>" in text


def test_render_preload_card_shows_checked_at() -> None:
    text = render_preload_card(
        job_id="jid",
        params={"horizon": "1y", "tickers": ["SPY"]},
        progress={"done": 0, "total": 1, "updated_at": "u", "checked_at": "c", "items": []},
        status="queued",
    )
    assert "Проверено" in text
    assert "<code>c</code>" in text
