from __future__ import annotations

from datetime import datetime


def render_job_card(job_id: str, ticker: str, start: str, end: str, preset: str, progress: dict) -> str:
    best_score = progress.get("best_score")
    best_score_text = "N/A" if best_score is None else f"{best_score:.6f}"
    return (
        "<b>🚀 Оптимизация запущена</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        f"<b>Тикер:</b> <code>{ticker}</code>\n"
        f"<b>Период:</b> <code>{start}</code> — <code>{end}</code>\n"
        f"<b>Режим:</b> {preset}\n"
        f"<b>Прогресс:</b> <code>{progress.get('trials_done', 0)}/{progress.get('trials_total', 0)}</code>\n"
        f"<b>Лучший score:</b> <code>{best_score_text}</code>\n"
        f"<b>Обновлено:</b> <code>{progress.get('updated_at', datetime.utcnow().isoformat())}</code>"
    )


def render_best_card(job_id: str, trials_done: int, metrics: dict, cfg: dict, updated_at: str) -> str:
    if not metrics:
        return (
            "<b>🏆 Лучший на данный момент</b>\n"
            f"<b>Job:</b> <code>{job_id}</code>\n"
            "Пока нет сохранённого best-so-far."
        )

    return (
        "<b>🏆 Лучший на данный момент</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        f"<b>Trials:</b> <code>{trials_done}</code>\n"
        f"<b>Score:</b> <code>{metrics.get('score', 0):.6f}</code>\n"
        f"<b>CAGR:</b> <code>{metrics.get('cagr', 0):.2%}</code>\n"
        f"<b>MaxDD:</b> <code>{metrics.get('max_dd', 0):.2%}</code>\n"
        f"<b>Sharpe:</b> <code>{metrics.get('sharpe', 0):.2f}</code>\n"
        f"<b>Сделок:</b> <code>{metrics.get('trades_count', 0)}</code>\n"
        f"<b>Win rate:</b> <code>{metrics.get('win_rate', 0):.1%}</code>\n"
        "<b>Параметры:</b>\n"
        "<pre>"
        f"EMA fast={cfg.get('ema_fast')}, slow={cfg.get('ema_slow')}\n"
        f"RSI p={cfg.get('rsi_period')}, buy<{cfg.get('buy_below')}, sell>{cfg.get('sell_above')}\n"
        f"BB p={cfg.get('bb_period')}, std={cfg.get('bb_std')}\n"
        f"ADX min={cfg.get('adx_min')}"
        "</pre>\n"
        f"<b>Обновлено:</b> <code>{updated_at}</code>"
    )
