from __future__ import annotations

import logging
import traceback
import time
from datetime import date, datetime, timedelta, timezone

from src.config import SETTINGS
from src.backtester.engine import run_backtest
from src.data_provider.provider import DataProviderError, get_ohlcv_with_meta
from src.indicators.calculator import add_indicators
from src.optimizer.optuna_runner import run_optimization_job
from src.storage.repository import Repository
from src.worker.celery_app import app

MIN_BARS = SETTINGS.min_bars
_LOG = logging.getLogger("worker.tasks")


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _horizon_to_start(horizon: str, buffer_days: int = 90) -> str:
    today = _today_utc()
    days = {"1y": 365, "2y": 730, "5y": 365 * 5}.get(horizon, 365 * 5)
    return (today - timedelta(days=days + buffer_days)).isoformat()


def _guard_note_reason(bars: int, min_bars: int, auto_extended: bool) -> tuple[str, str]:
    if bars <= 0:
        return "no_data", "no_data: bars=0"
    suffix = " (auto-extend attempted)" if auto_extended else ""
    return "not_enough_bars", f"not_enough_bars: bars={bars} min_required={min_bars}{suffix}"


def _log_baselines(df, params: dict) -> dict:
    def _equity_stats(bt_df):
        if bt_df is None or bt_df.empty or "equity" not in bt_df:
            return 0.0, 0.0
        eq = bt_df["equity"].astype(float)
        total_return = float(eq.iloc[-1] - 1.0)
        max_dd = float((eq / eq.cummax() - 1.0).min())
        return total_return, max_dd

    out: dict = {}

    # Baseline #1: always in position
    always_sig = df[["Close"]].copy()
    always_sig["position"] = 1
    always_sig.iloc[-1, always_sig.columns.get_loc("position")] = 0
    always_sig["signal"] = always_sig["position"].diff().fillna(always_sig["position"]).clip(-1, 1).astype(int)
    bt_a, metrics_a, _trades_a = run_backtest(df, always_sig[["signal", "position"]], SETTINGS.commission_bps, SETTINGS.slippage_bps)
    a_ret, a_dd = _equity_stats(bt_a)
    out["always_in"] = {
        "trades_count": int(metrics_a.get("trades_count", 0)),
        "total_return": a_ret,
        "max_dd": a_dd,
    }
    _LOG.info(
        "baseline always_in trades_count=%s total_return=%.6f max_dd=%.6f",
        out["always_in"]["trades_count"],
        out["always_in"]["total_return"],
        out["always_in"]["max_dd"],
    )

    # Baseline #2: EMA-only cross
    ema_cfg = {
        "ema_fast": int(params.get("ema_fast", 20)),
        "ema_slow": int(params.get("ema_slow", 50)),
        "rsi_period": 14,
        "buy_below": 30,
        "sell_above": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "adx_min": -999,
        "enter_long": 1,
        "exit_long": 0,
    }
    ind = add_indicators(df, ema_cfg)
    ema_signal = ind[["Close"]].copy()
    ema_cross = (ind["ema_fast"] > ind["ema_slow"]).astype(int)
    ema_signal["position"] = ema_cross
    ema_signal.iloc[-1, ema_signal.columns.get_loc("position")] = 0
    ema_signal["signal"] = ema_signal["position"].diff().fillna(ema_signal["position"]).clip(-1, 1).astype(int)
    bt_e, metrics_e, _trades_e = run_backtest(ind, ema_signal[["signal", "position"]], SETTINGS.commission_bps, SETTINGS.slippage_bps)
    e_ret, e_dd = _equity_stats(bt_e)
    out["ema_only"] = {
        "trades_count": int(metrics_e.get("trades_count", 0)),
        "total_return": e_ret,
        "max_dd": e_dd,
    }
    _LOG.info(
        "baseline ema_only trades_count=%s total_return=%.6f max_dd=%.6f",
        out["ema_only"]["trades_count"],
        out["ema_only"]["total_return"],
        out["ema_only"]["max_dd"],
    )

    return out


@app.task(name="preload.run")
def preload_data_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    tickers = [str(t).strip().upper() for t in params.get("tickers", []) if str(t).strip()]
    horizon = str(params.get("horizon", "5y"))
    start = _horizon_to_start(horizon, buffer_days=90)
    end = _today_utc().isoformat()

    total = len(tickers)
    done = 0
    repo.set_job_status(job_id, "running")

    for ticker in tickers:
        status = "ok"
        reason = ""
        bars_count = 0
        min_date = ""
        max_date = ""
        source = ""
        end_trimmed = False
        try:
            df, meta = get_ohlcv_with_meta(ticker, start, end, job_id=job_id)
            bars_count = int(meta.get("bars_count", len(df)))
            min_date = str(meta.get("min_date", ""))
            max_date = str(meta.get("max_date", ""))
            source = str(meta.get("source", ""))
            end_trimmed = bool(meta.get("end_trimmed", False))
            if df.empty:
                status = "not_found"
                reason = "provider_empty"
        except DataProviderError as exc:
            status = "not_found" if exc.reason in {"provider_empty", "future_period"} else "error"
            reason = exc.reason
        except Exception as exc:  # noqa: BLE001
            status = "error"
            reason = f"error: {exc}"

        done += 1
        repo.update_preload_progress(
            job_id=job_id,
            done=done,
            total=total,
            item={
                "ticker": ticker,
                "status": status,
                "reason": reason,
                "bars_count": bars_count,
                "min_date": min_date,
                "max_date": max_date,
                "source": source,
                "end_trimmed": end_trimmed,
            },
            state="running",
        )

    repo.set_job_status(job_id, "finished")
    return {"status": "finished", "done": done, "total": total}


@app.task(name="optimization.run")
def optimization_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    trials_total = int(params.get("trials_total", 0))

    def _set_last_trial(note: str, reason: str, duration_sec: float, score: float = -9999.0) -> None:
        repo.update_progress_last_trial(
            job_id,
            {
                "number": 0,
                "score": score,
                "trades_count": 0,
                "params": {},
                "duration_sec": round(max(duration_sec, 0.01), 3),
                "note": note,
                "reason": reason,
            },
        )

    started_at = time.monotonic()

    try:
        start_date = date.fromisoformat(params["start"])
        end_date = date.fromisoformat(params["end"])
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        _set_last_trial("exception", "Некорректные параметры даты", time.monotonic() - started_at)
        return {"status": "failed", "error": "invalid date params"}

    today = _today_utc()
    effective_end = min(end_date, today)
    clipped_warning: str | None = None
    if end_date > today:
        clipped_warning = f"⚠️ Конец периода обрезан до {today.isoformat()}, потому что будущих данных нет."

    if start_date >= effective_end:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", "future_period", time.monotonic() - started_at)
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason="future_period: дата начала должна быть раньше даты окончания.",
        )
        return {"status": "finished_no_results", "reason": "future_period"}

    current_start = start_date
    expansions = 0
    max_expansions = 3
    expand_days = 60
    df = None
    meta: dict = {}

    while True:
        try:
            df, meta = get_ohlcv_with_meta(
                params["ticker"],
                current_start.isoformat(),
                end_date.isoformat(),
                job_id=job_id,
            )
            if clipped_warning:
                meta["warning"] = clipped_warning
        except DataProviderError as exc:
            repo.set_job_status(job_id, "finished_no_results")
            _set_last_trial("no_data", exc.reason, time.monotonic() - started_at)
            repo.update_progress(
                job_id=job_id,
                trials_done=0,
                trials_total=trials_total,
                last_score=0.0,
                best_score=None,
                state="finished_no_results",
                reason=f"{exc.reason}: {exc}",
            )
            return {"status": "finished_no_results", "reason": exc.reason}

        bars_before = len(df)
        if bars_before >= MIN_BARS:
            break
        if expansions >= max_expansions:
            break

        old_start = current_start
        current_start = current_start - timedelta(days=expand_days)
        expansions += 1

        try:
            df_expanded, meta_expanded = get_ohlcv_with_meta(
                params["ticker"],
                current_start.isoformat(),
                end_date.isoformat(),
                job_id=job_id,
            )
            if clipped_warning:
                meta_expanded["warning"] = clipped_warning
            bars_after = len(df_expanded)
            _LOG.info(
                "expand_start_date by 60d: %s->%s, bars %s->%s",
                old_start.isoformat(),
                current_start.isoformat(),
                bars_before,
                bars_after,
            )
            df = df_expanded
            meta = meta_expanded
            if bars_after >= MIN_BARS:
                break
        except DataProviderError as exc:
            _LOG.info("expand_start_date fetch failed reason=%s", exc.reason)
            break

    if df is None:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", "provider_empty", time.monotonic() - started_at)
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason="provider_empty: no dataframe returned",
        )
        return {"status": "finished_no_results", "reason": "provider_empty"}

    if len(df) < MIN_BARS:
        note, message = _guard_note_reason(len(df), MIN_BARS, auto_extended=expansions > 0)
        _LOG.info("trial_start bars=%s min_required=%s proceeding=false", len(df), MIN_BARS)
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial(note, message, time.monotonic() - started_at)
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=message,
            data_info=meta,
        )
        return {"status": "finished_no_results", "reason": note}

    if expansions > 0:
        meta["expanded_start_date"] = current_start.isoformat()
        meta["expanded_days_total"] = expansions * expand_days

    try:
        meta["baseline"] = _log_baselines(df, params)
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("baseline diagnostics failed: %s", exc)

    repo.update_progress(
        job_id=job_id,
        trials_done=0,
        trials_total=trials_total,
        last_score=0.0,
        best_score=None,
        state="running",
        reason=(clipped_warning or None),
        data_info=meta,
    )
    _LOG.info("trial_start bars=%s min_required=%s proceeding=true", len(df), MIN_BARS)

    try:
        _LOG.info("optimizer policy: zero-trade trials are valid if score is finite")
        return run_optimization_job(job_id, df)
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("optimization job crashed")
        tb = traceback.format_exc()[:2000]
        _set_last_trial(
            note="exception",
            reason="Критическая ошибка оптимизации",
            duration_sec=time.monotonic() - started_at,
        )
        repo.update_progress_last_trial(
            job_id,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
            },
        )
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=-9999.0,
            best_score=None,
            state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            data_info=meta,
        )
        repo.set_job_status(job_id, "failed")
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
