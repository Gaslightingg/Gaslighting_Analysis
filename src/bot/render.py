from __future__ import annotations

from datetime import datetime


def _note_text(note: str, reason: str | None = None) -> str:
    mapping = {
        "ok": "ok",
        "no_data": "no_data — Нет данных за выбранный период",
        "not_enough_bars": "not_enough_bars",
        "no_entries": "no_entries — 0 entry signals",
        "no_trades": "no_trades — сигналы входа были, но сделки не исполнились",
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




def _balance_line(payload: dict) -> str:
    start_cash = float(payload.get("start_cash", 10000.0) or 10000.0)
    final_equity = float(payload.get("final_equity", start_cash) or start_cash)
    profit_abs = float(payload.get("profit_$", final_equity - start_cash) or (final_equity - start_cash))
    profit_pct = float(payload.get("profit_%", ((final_equity / start_cash) - 1.0) * 100.0 if start_cash > 0 else 0.0) or 0.0)
    return f"<code>Баланс: ${final_equity:.2f} (profit ${profit_abs:.2f}, {profit_pct:.2f}%)</code>"

def _last_trial_line(progress: dict) -> str:
    last_trial = progress.get("last_trial") or {}
    if not last_trial:
        return "<b>Последняя попытка:</b> <code>ещё не запускалась</code>"

    score = float(last_trial.get("score", 0.0))
    trades = int(last_trial.get("trades_count", 0))
    sig_e = int(last_trial.get("entry_events_count", last_trial.get("signals_count_enter", 0)))
    sig_x = int(last_trial.get("exit_events_count", last_trial.get("signals_count_exit", 0)))
    duration = float(last_trial.get("duration_sec", 0.0))
    note = _note_text(str(last_trial.get("note", "-")), _pretty_reason(str(last_trial.get("reason", ""))))
    number = int(last_trial.get("number", 0))
    error_hint = ""
    if str(last_trial.get("note", "")) == "exception":
        error_text = str(last_trial.get("error") or "")
        if error_text:
            error_hint = f", <code>{error_text[:120]}</code>"
    balance = _balance_line(last_trial)
    return (
        f"<b>Последняя попытка #{number}:</b> "
        f"<code>score={score:.6f}</code>, "
        f"<code>trades={trades}</code>, "
        f"<code>duration={duration:.2f}s</code>, "
        f"<code>{note}</code>"
        f"{error_hint}\n"
        f"{balance}\n"
        f"<code>events: enter={sig_e}, exit={sig_x}, closed_trades={int(last_trial.get('closed_trades_count', last_trial.get('trades_count', 0)))}, forced_exit={int(last_trial.get('forced_exit_count', 0))}</code>"
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
    best_metrics: dict | None = None,
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
    best_metrics = best_metrics or {}

    if state == "finished_no_results":
        reason = progress.get("reason", "Не найдено ни одного валидного результата (finite score).")
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
    best_valid = progress.get("best_valid") or {}
    best_overall = progress.get("best_overall") or {}

    metrics_line = ""
    if best_valid:
        metrics_line += (
            f"<b>Best valid:</b> <code>trial=#{best_valid.get('number', '-')}</code>, "
            f"<code>score={float(best_valid.get('score', 0.0)):.6f}</code>, "
            f"<code>trades={int(best_valid.get('trades_count', 0))}</code>\n"
            f"{_balance_line(best_valid)}\n"
        )
    else:
        metrics_line += "<b>Best valid:</b> <code>None</code>\n"

    if best_overall:
        metrics_line += (
            f"<b>Best overall:</b> <code>trial=#{best_overall.get('number', '-')}</code>, "
            f"<code>score={float(best_overall.get('score', 0.0)):.6f}</code>, "
            f"<code>reason={best_overall.get('note', '-')}</code>, "
            f"<code>trades={int(best_overall.get('trades_count', 0))}</code>\n"
            f"{_balance_line(best_overall)}\n"
            f"<code>events: enter={int(best_overall.get('entry_events_count', best_overall.get('signals_count_enter', 0)))}, exit={int(best_overall.get('exit_events_count', best_overall.get('signals_count_exit', 0)))}, closed_trades={int(best_overall.get('closed_trades_count', best_overall.get('trades_count', 0)))}, forced_exit={int(best_overall.get('forced_exit_count', 0))}</code>\n"
        )

    if best_metrics and not best_valid:
        metrics_line += (
            f"<b>Best итог (legacy):</b> <code>trades={best_metrics.get('trades_count', 0)}</code>, "
            f"<code>final_equity={best_metrics.get('final_equity', 0.0):.2f}</code>, "
            f"<code>profit={best_metrics.get('profit_%', 0.0):.2f}%</code>\n"
        )

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
        f"{metrics_line}"
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
        f"<b>Final equity:</b> <code>{metrics.get('final_equity', 0):.2f}</code>\n"
        f"<b>Profit $:</b> <code>{metrics.get('profit_$', 0):.2f}</code>\n"
        f"<b>Profit %:</b> <code>{metrics.get('profit_%', 0):.2f}%</code>\n"
        f"<b>MaxDD %:</b> <code>{metrics.get('max_dd_%', metrics.get('max_dd', 0) * 100):.2f}%</code>\n"
        "<b>Параметры:</b>\n"
        "<pre>"
        f"EMA fast={cfg.get('ema_fast')}, slow={cfg.get('ema_slow')}\n"
        f"RSI p={cfg.get('rsi_period')}, buy<{cfg.get('buy_below')}, sell>{cfg.get('sell_above')}\n"
        f"BB p={cfg.get('bb_period')}, std={cfg.get('bb_std')}\n"
        f"ADX p={cfg.get('adx_period')}, min={cfg.get('adx_min')}, regime={cfg.get('regime_mode')}\n"
        f"enter_long_votes_required={cfg.get('enter_long')}, exit_long_votes_required={cfg.get('exit_long')}\n"
        f"sl_pct={cfg.get('sl_pct')}, tp_pct={cfg.get('tp_pct')}, exec={cfg.get('execution_mode')}"
        "</pre>\n"
        f"<b>Обновлено:</b> <code>{updated_at}</code>"
    )
