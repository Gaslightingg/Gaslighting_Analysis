from __future__ import annotations

from datetime import datetime


def _note_text(note: str, reason: str | None = None) -> str:
    mapping = {
        "ok": "ok",
        "no_data": "no_data — Нет данных за выбранный период",
        "not_enough_bars": "not_enough_bars",
        "no_trades": "no_trades — стратегия не сгенерировала входов",
        "nan_score": "nan_score — score невалиден",
        "exception": "exception — ошибка при расчёте trial",
    }
    base = mapping.get(note, note or "-")
    if reason and reason not in base:
        return f"{base}. {reason}"
    return base




def _pretty_reason(reason: str) -> str:
    if reason.startswith("not_enough_bars"):
        return reason
    if reason.startswith("no_trades"):
        return reason
    return reason


def _last_trial_line(progress: dict) -> str:
    last_trial = progress.get("last_trial") or {}
    if not last_trial:
        return "<b>Последняя попытка:</b> <code>ещё не запускалась</code>"

    score = float(last_trial.get("score", 0.0))
    trades = int(last_trial.get("trades_count", 0))
    duration = float(last_trial.get("duration_sec", 0.0))
    note = _note_text(str(last_trial.get("note", "-")), _pretty_reason(str(last_trial.get("reason", ""))))
    number = int(last_trial.get("number", 0))
    return (
        f"<b>Последняя попытка #{number}:</b> "
        f"<code>score={score:.6f}</code>, "
        f"<code>trades={trades}</code>, "
        f"<code>duration={duration:.2f}s</code>, "
        f"<code>{note}</code>"
    )


def _data_line(progress: dict) -> str:
    data = progress.get("data_info") or {}
    if not data:
        return ""
    return (
        f"<b>Данные:</b> <code>{data.get('min_date', '-')} — {data.get('max_date', '-')}</code>, "
        f"<code>bars={data.get('bars_count', 0)}</code>, "
        f"<code>source={data.get('source', '-')}</code>, "
        f"<code>end_trimmed={str(bool(data.get('end_trimmed', False))).lower()}</code>"
    )


def render_job_card(
    job_id: str,
    ticker: str,
    start: str,
    end: str,
    preset: str,
    progress: dict,
    status: str | None = None,
) -> str:
    trials_done = progress.get("trials_done", 0)
    trials_total = progress.get("trials_total", 0)
    best_score = progress.get("best_score")
    state = status or progress.get("state", "queued")
    updated_at = progress.get("updated_at", datetime.utcnow().isoformat())
    last_trial_line = _last_trial_line(progress)
    reason = str(progress.get("reason") or "")
    warning_line = f"<b>Предупреждение:</b> <code>{reason}</code>" if reason.startswith("⚠️") else ""
    warning_block = f"{warning_line}\n" if warning_line else ""
    data_line = _data_line(progress)
    data_block = f"{data_line}\n" if data_line else ""

    if state == "finished_no_results":
        reason = progress.get("reason", "Не найдено ни одного валидного результата (finite score + >=1 сделка).")
        return (
            "<b>⚠️ Оптимизация завершена без результатов</b>\n"
            f"<b>Job:</b> <code>{job_id}</code>\n"
            f"<b>Тикер:</b> <code>{ticker}</code>\n"
            f"<b>Период:</b> <code>{start}</code> — <code>{end}</code>\n"
            f"<b>Режим:</b> {preset}\n"
            f"<b>Прогресс:</b> <code>{trials_done}/{trials_total}</code>\n"
            f"{warning_block}{data_block}{last_trial_line}\n"
            f"<b>Причина:</b> <code>{reason}</code>\n"
            f"<b>Обновлено:</b> <code>{updated_at}</code>"
        )

    best_score_text = "нет валидного результата" if best_score is None else f"{best_score:.6f}"
    return (
        "<b>🚀 Оптимизация</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        f"<b>Тикер:</b> <code>{ticker}</code>\n"
        f"<b>Период:</b> <code>{start}</code> — <code>{end}</code>\n"
        f"<b>Режим:</b> {preset}\n"
        f"<b>Статус:</b> <code>{state}</code>\n"
        f"<b>Прогресс:</b> <code>{trials_done}/{trials_total}</code>\n"
        f"{warning_block}{data_block}{last_trial_line}\n"
        f"<b>Лучший score:</b> <code>{best_score_text}</code>\n"
        f"<b>Обновлено:</b> <code>{updated_at}</code>"
    )


def render_preload_card(job_id: str, params: dict, progress: dict, status: str) -> str:
    done = int(progress.get("done", 0))
    total = int(progress.get("total", 0))
    items = progress.get("items", [])[-8:]
    rows = []
    for item in items:
        rows.append(
            f"• <code>{item.get('ticker')}</code> [{item.get('status')}] "
            f"bars={item.get('bars_count', 0)} "
            f"{item.get('min_date', '-')}/{item.get('max_date', '-')}"
        )
    body = "\n".join(rows) if rows else "Пока нет обработанных тикеров"
    return (
        "<b>📥 Preload данных</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        f"<b>Horizon:</b> <code>{params.get('horizon', '-')}</code>\n"
        f"<b>Тикеры:</b> <code>{', '.join(params.get('tickers', []))}</code>\n"
        f"<b>Статус:</b> <code>{status}</code>\n"
        f"<b>Прогресс:</b> <code>{done}/{total}</code>\n"
        f"<b>Детали:</b>\n{body}\n"
        f"<b>Обновлено:</b> <code>{progress.get('updated_at', datetime.utcnow().isoformat())}</code>"
    )


def render_best_card(job_id: str, trials_done: int, metrics: dict, cfg: dict, updated_at: str, status: str | None = None) -> str:
    if status == "finished_no_results":
        return (
            "<b>🏁 Результат оптимизации</b>\n"
            f"<b>Job:</b> <code>{job_id}</code>\n"
            "Оптимизация завершена без валидных результатов.\n"
            "Проверьте период данных и параметры стратегии."
        )

    if not metrics:
        return (
            "<b>🏆 Лучший на данный момент</b>\n"
            f"<b>Job:</b> <code>{job_id}</code>\n"
            "Пока нет валидного best-so-far."
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
