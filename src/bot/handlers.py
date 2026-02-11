from __future__ import annotations

import html
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message
from kombu.exceptions import OperationalError

from src.bot.keyboards import (
    POPULAR_TICKERS,
    confirm_kb,
    job_card_kb,
    jobs_list_kb,
    main_menu_kb,
    mode_kb,
    period_kb,
    preload_confirm_kb,
    preload_horizon_kb,
    preload_job_kb,
    preload_tickers_kb,
    ticker_kb,
)
from src.bot.render import render_best_card, render_job_card, render_preload_card
from src.bot.states import NewOptimizationState, PreloadState
from src.config import PRESETS, SETTINGS
from src.data_provider.provider import get_cached_range
from src.storage.repository import Repository
from src.worker.tasks import optimization_run, preload_data_run

router = Router()
repo = Repository()

if SETTINGS.telegram_allowed_user_id is not None:
    router.message.filter(F.from_user.id == SETTINGS.telegram_allowed_user_id)
    router.callback_query.filter(F.from_user.id == SETTINGS.telegram_allowed_user_id)


async def _safe_edit(callback: CallbackQuery, text: str, **kwargs) -> None:
    try:
        await callback.message.edit_text(text, **kwargs)
    except TelegramBadRequest as exc:
        if "message is not modified" in str(exc):
            return
        raise


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _period_from_code(code: str) -> tuple[str, str]:
    end = _today_utc()
    if code == "1y":
        start = end - timedelta(days=365)
    else:
        start = end - timedelta(days=365 * 2)
    return start.isoformat(), end.isoformat()


def _validate_period(start: str, end: str) -> tuple[bool, str | None]:
    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
    except ValueError:
        return False, "Неверный формат дат. Используйте YYYY-MM-DD YYYY-MM-DD."

    if end_d > _today_utc():
        return False, "Дата окончания не может быть в будущем."
    if start_d >= end_d:
        return False, "Дата начала должна быть раньше даты окончания."
    return True, None


def _reason_no_best(info: dict | None) -> str:
    if not info:
        return "Нет данных по задаче"
    progress = info.get("progress", {})
    if progress.get("state") == "finished_no_results":
        return progress.get("reason", "Оптимизация завершена без валидных результатов.")
    last_trial = progress.get("last_trial") or {}
    if last_trial.get("note") == "no_data":
        return str(last_trial.get("reason") or "Нет данных за выбранный период.")
    if progress.get("trials_done", 0) == 0:
        return "Оптимизация ещё не выполнила ни одного trial."
    return "Пока нет валидного результата (finite score)."


def _build_job_keyboard(job_id: str, info: dict | None, best: dict | None):
    if info and info.get("job") and info["job"].type == "preload":
        return preload_job_kb(job_id)
    no_results = bool(info and info.get("job") and info["job"].status == "finished_no_results")
    has_best = bool(best and best.get("metrics") and best.get("equity_path") and best.get("trades_path"))
    return job_card_kb(job_id, has_best=has_best, no_results=no_results)


@router.message(F.text == "/start")
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "<b>Telegram Trading Lab</b>\nВыберите действие:",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )


@router.callback_query(F.data == "menu:home")
async def menu_home(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await _safe_edit(
        callback,
        "<b>Telegram Trading Lab</b>\nВыберите действие:",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )
    await callback.answer()


@router.callback_query(F.data == "menu:new")
async def new_opt(callback: CallbackQuery, state: FSMContext) -> None:
    last_job = repo.get_last_active_job(callback.from_user.id)
    last_ticker = None
    if last_job:
        params = json.loads(last_job.params_json or "{}")
        last_ticker = params.get("ticker")

    await state.set_state(NewOptimizationState.choose_ticker)
    await _safe_edit(
        callback,
        "<b>Шаг 1/4:</b> Выберите тикер",
        parse_mode="HTML",
        reply_markup=ticker_kb(last_ticker=last_ticker),
    )
    await callback.answer()


@router.callback_query(F.data == "menu:preload")
async def preload_menu(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(PreloadState.choose_horizon)
    await _safe_edit(
        callback,
        "<b>Preload данных</b>\nШаг 1/3: выберите horizon",
        parse_mode="HTML",
        reply_markup=preload_horizon_kb(),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("preload:horizon:"))
async def preload_horizon(callback: CallbackQuery, state: FSMContext) -> None:
    horizon = callback.data.split(":")[-1]
    await state.update_data(horizon=horizon)

    last_job = repo.get_last_active_job(callback.from_user.id)
    last_ticker = None
    if last_job:
        params = json.loads(last_job.params_json or "{}")
        last_ticker = params.get("ticker")

    await state.set_state(PreloadState.choose_tickers)
    await _safe_edit(
        callback,
        "<b>Preload данных</b>\nШаг 2/3: выберите тикеры",
        parse_mode="HTML",
        reply_markup=preload_tickers_kb(last_ticker=last_ticker),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("preload:ticker:"))
async def preload_tickers(callback: CallbackQuery, state: FSMContext) -> None:
    raw = callback.data.split(":", maxsplit=2)[2]
    if raw == "manual":
        await state.set_state(PreloadState.manual_tickers)
        await _safe_edit(
            callback,
            "Введите список тикеров через пробел (пример: <code>SPY AAPL MSFT</code>)",
            parse_mode="HTML",
        )
        await callback.answer()
        return

    tickers = POPULAR_TICKERS if raw == "popular" else [raw.upper()]
    await state.update_data(tickers=tickers)
    await state.set_state(PreloadState.confirm)
    data = await state.get_data()
    await _safe_edit(
        callback,
        "<b>Preload данных</b>\nШаг 3/3: подтвердите запуск\n"
        f"Horizon: <code>{data.get('horizon')}</code>\n"
        f"Тикеры: <code>{' '.join(tickers)}</code>",
        parse_mode="HTML",
        reply_markup=preload_confirm_kb(),
    )
    await callback.answer()


@router.message(PreloadState.manual_tickers)
async def preload_tickers_manual(message: Message, state: FSMContext) -> None:
    tickers = [t.strip().upper() for t in (message.text or "").split() if t.strip()]
    if not tickers:
        await message.answer("Список пуст. Укажите хотя бы один тикер.")
        return
    await state.update_data(tickers=tickers)
    await state.set_state(PreloadState.confirm)
    data = await state.get_data()
    await message.answer(
        "<b>Preload данных</b>\nШаг 3/3: подтвердите запуск\n"
        f"Horizon: <code>{data.get('horizon')}</code>\n"
        f"Тикеры: <code>{' '.join(tickers)}</code>",
        parse_mode="HTML",
        reply_markup=preload_confirm_kb(),
    )


@router.callback_query(F.data == "preload:confirm")
async def preload_confirm(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()
    horizon = str(data.get("horizon", "5y"))
    tickers = [str(t).upper() for t in data.get("tickers", [])]
    if not tickers:
        await callback.answer("Не выбраны тикеры", show_alert=True)
        return

    job_id = repo.create_preload_job(
        user_id=callback.from_user.id,
        chat_id=callback.message.chat.id,
        tickers=tickers,
        horizon=horizon,
    )
    try:
        preload_data_run.delay(job_id)
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        await callback.answer("Не удалось запустить preload", show_alert=True)
        return

    info = repo.get_job_data(job_id)
    text = render_preload_card(job_id, info["params"] if info else {}, info["progress"] if info else {}, "queued")
    await _safe_edit(callback, text, parse_mode="HTML", reply_markup=preload_job_kb(job_id))
    await callback.answer("Preload запущен")
    await state.clear()


@router.callback_query(F.data.startswith("ticker:"))
async def ticker_selected(callback: CallbackQuery, state: FSMContext) -> None:
    val = callback.data.split(":", maxsplit=1)[1]
    if val == "manual":
        await state.set_state(NewOptimizationState.manual_ticker)
        await _safe_edit(
            callback,
            "Введите тикер (пример: <code>AAPL</code>)",
            parse_mode="HTML",
        )
        await callback.answer()
        return

    await state.update_data(ticker=val.upper())
    await state.set_state(NewOptimizationState.choose_period)
    await _safe_edit(
        callback,
        "<b>Шаг 2/4:</b> Выберите период",
        parse_mode="HTML",
        reply_markup=period_kb(),
    )
    await callback.answer()


@router.message(NewOptimizationState.manual_ticker)
async def ticker_manual(message: Message, state: FSMContext) -> None:
    ticker = (message.text or "").strip().upper()
    await state.update_data(ticker=ticker)
    await state.set_state(NewOptimizationState.choose_period)
    await message.answer("<b>Шаг 2/4:</b> Выберите период", parse_mode="HTML", reply_markup=period_kb())


@router.callback_query(F.data.startswith("period:"))
async def period_selected(callback: CallbackQuery, state: FSMContext) -> None:
    val = callback.data.split(":", maxsplit=1)[1]
    if val == "custom":
        await state.set_state(NewOptimizationState.manual_period)
        await _safe_edit(
            callback,
            "Введите период: <code>YYYY-MM-DD YYYY-MM-DD</code>",
            parse_mode="HTML",
        )
        await callback.answer()
        return

    if val == "cache":
        data = await state.get_data()
        ticker = str(data.get("ticker", "")).upper()
        cached = get_cached_range(ticker) if ticker else None
        if not cached:
            await callback.answer("В кэше нет данных. Сначала preload.", show_alert=True)
            return
        start = cached["min_date"]
        end = cached["max_date"]
    else:
        start, end = _period_from_code(val)

    is_ok, err = _validate_period(start, end)
    if not is_ok:
        await callback.answer(err, show_alert=True)
        return

    await state.update_data(start=start, end=end)
    await state.set_state(NewOptimizationState.choose_mode)
    await _safe_edit(
        callback,
        "<b>Шаг 3/4:</b> Выберите режим оптимизации",
        parse_mode="HTML",
        reply_markup=mode_kb(),
    )
    await callback.answer()


@router.message(NewOptimizationState.manual_period)
async def period_manual(message: Message, state: FSMContext) -> None:
    try:
        start, end = (message.text or "").strip().split()
    except Exception:  # noqa: BLE001
        await message.answer("Неверный формат. Используйте: <code>YYYY-MM-DD YYYY-MM-DD</code>", parse_mode="HTML")
        return

    is_ok, err = _validate_period(start, end)
    if not is_ok:
        await message.answer(err or "Неверный период", parse_mode="HTML")
        return

    await state.update_data(start=start, end=end)
    await state.set_state(NewOptimizationState.choose_mode)
    await message.answer("<b>Шаг 3/4:</b> Выберите режим оптимизации", parse_mode="HTML", reply_markup=mode_kb())


@router.callback_query(F.data.startswith("mode:"))
async def mode_selected(callback: CallbackQuery, state: FSMContext) -> None:
    mode = callback.data.split(":", maxsplit=1)[1]
    if mode not in PRESETS:
        await callback.answer("Неизвестный режим", show_alert=True)
        return

    data = await state.get_data()
    preset = PRESETS[mode]
    await state.update_data(mode=mode)
    await state.set_state(NewOptimizationState.confirm)
    await _safe_edit(
        callback,
        "<b>Шаг 4/4:</b> Подтвердите запуск\n"
        f"Тикер: <code>{data.get('ticker')}</code>\n"
        f"Период: <code>{data.get('start')}</code> — <code>{data.get('end')}</code>\n"
        f"Режим: <b>{preset.name}</b> ({preset.trials_total} trials)",
        parse_mode="HTML",
        reply_markup=confirm_kb(),
    )
    await callback.answer()


@router.callback_query(F.data == "confirm:start")
async def confirm_start(callback: CallbackQuery, state: FSMContext) -> None:
    data = await state.get_data()

    mode = data.get("mode")
    ticker = data.get("ticker")
    start = data.get("start")
    end = data.get("end")

    if not mode or mode not in PRESETS or not ticker or not start or not end:
        await state.clear()
        await callback.answer("Сессия устарела. Запустите мастер заново: ➕ Новая оптимизация", show_alert=True)
        await _safe_edit(
            callback,
            "<b>Telegram Trading Lab</b>\nВыберите действие:",
            parse_mode="HTML",
            reply_markup=main_menu_kb(),
        )
        return

    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
    except ValueError:
        await state.clear()
        await callback.answer("Неверный формат дат", show_alert=True)
        return

    today = _today_utc()
    clipped_notice = None
    if end_d > today:
        end_d = today
        end = end_d.isoformat()
        await state.update_data(end=end)
        clipped_notice = f"⚠️ Конец периода обрезан до {end}, потому что будущих данных нет."

    is_ok, err = _validate_period(start_d.isoformat(), end_d.isoformat())
    if not is_ok:
        await state.clear()
        await callback.answer(err or "Неверный период", show_alert=True)
        return

    preset = PRESETS[mode]

    job_id = repo.create_job(
        user_id=callback.from_user.id,
        chat_id=callback.message.chat.id,
        ticker=ticker,
        start=start,
        end=end,
        preset=mode,
        trials_total=preset.trials_total,
        checkpoint_n=preset.checkpoint_n,
    )

    try:
        optimization_run.delay(job_id)
    except OperationalError:
        repo.set_job_status(job_id, "failed")
        await callback.answer("Redis/Celery недоступен. Проверьте, что Redis запущен.", show_alert=True)
        return
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        await callback.answer("Не удалось поставить задачу в очередь", show_alert=True)
        return

    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    card = render_job_card(
        job_id=job_id,
        ticker=ticker,
        start=start,
        end=end,
        preset=preset.name,
        progress=info["progress"] if info else {},
        status=info["job"].status if info else None,
    )
    await _safe_edit(callback, card, parse_mode="HTML", reply_markup=_build_job_keyboard(job_id, info, best))
    await callback.answer(clipped_notice or "Запущено")
    await state.clear()


@router.callback_query(F.data == "menu:jobs")
async def my_jobs(callback: CallbackQuery) -> None:
    jobs = repo.list_jobs(callback.from_user.id)
    text = "<b>📌 Мои задачи</b>\n" + ("\n".join(f"• <code>{j.id}</code> [{j.status}:{j.type}]" for j in jobs) or "Нет задач")
    await _safe_edit(
        callback,
        text,
        parse_mode="HTML",
        reply_markup=jobs_list_kb([j.id for j in jobs]),
    )
    await callback.answer()


@router.callback_query(F.data == "menu:best")
async def menu_best(callback: CallbackQuery) -> None:
    job = repo.get_last_active_job(callback.from_user.id)
    if not job:
        await callback.answer("Нет задач", show_alert=True)
        return
    await _send_best(callback, job.id)


@router.callback_query(F.data.startswith("job:refresh:"))
async def refresh_job(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    if not info:
        await callback.answer("Задача не найдена", show_alert=True)
        return

    if info["job"].type == "preload":
        text = render_preload_card(job_id, info["params"], info["progress"], info["job"].status)
        await _safe_edit(callback, text, parse_mode="HTML", reply_markup=preload_job_kb(job_id))
        await callback.answer()
        return

    best = repo.get_best(job_id)
    params = info["params"]
    preset = PRESETS.get(params.get("preset", "quick"))
    card = render_job_card(
        job_id=job_id,
        ticker=params.get("ticker", ""),
        start=params.get("start", ""),
        end=params.get("end", ""),
        preset=preset.name if preset else params.get("preset", "-"),
        progress=info["progress"],
        status=info["job"].status,
        best_metrics=(best or {}).get("metrics", {}),
    )
    await _safe_edit(callback, card, parse_mode="HTML", reply_markup=_build_job_keyboard(job_id, info, best))
    await callback.answer()


@router.callback_query(F.data.startswith("job:best:"))
async def best_for_job(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    await _send_best(callback, job_id)


async def _send_best(callback: CallbackQuery, job_id: str) -> None:
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not info:
        await callback.answer("Нет данных", show_alert=True)
        return
    if info["job"].type == "preload":
        await callback.answer("Для preload нет best-карточки", show_alert=True)
        return

    status = info["job"].status
    text = render_best_card(
        job_id=job_id,
        trials_done=info["progress"].get("trials_done", 0),
        metrics=(best or {}).get("metrics", {}),
        cfg=(best or {}).get("config", {}),
        updated_at=str((best or {}).get("best_updated_at")),
        status=status,
    )
    await _safe_edit(callback, text, parse_mode="HTML", reply_markup=_build_job_keyboard(job_id, info, best))
    await callback.answer()


@router.callback_query(F.data.startswith("job:stop:"))
async def stop_job(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    repo.request_stop(job_id)
    await callback.answer("Остановка запрошена")


@router.callback_query(F.data.startswith("job:equity:"))
async def equity(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not best or not best.get("equity_path"):
        await callback.answer(_reason_no_best(info), show_alert=True)
        return
    await callback.message.answer_photo(FSInputFile(best["equity_path"]))
    await callback.answer()


@router.callback_query(F.data.startswith("job:trades:"))
async def trades(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not best or not best.get("trades_path"):
        await callback.answer(_reason_no_best(info), show_alert=True)
        return

    trades_plot = Path(SETTINGS.runs_dir) / job_id / "trades.png"
    if not trades_plot.exists():
        await callback.answer("График сделок пока не построен", show_alert=True)
        return

    await callback.message.answer_photo(FSInputFile(str(trades_plot)))
    await callback.answer()


@router.callback_query(F.data.startswith("job:values:"))
async def values(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not best or not best.get("config_path"):
        await callback.answer(_reason_no_best(info), show_alert=True)
        return

    cfg_path = Path(str(best["config_path"]))
    if not cfg_path.exists():
        await callback.answer("best_config.json не найден", show_alert=True)
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    metrics = (best or {}).get("metrics") or {}

    text = (
        "<b>🔢 Лучшие значения</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        "<b>Параметры:</b>\n"
        "<pre>"
        f"EMA fast={cfg.get('ema_fast')} slow={cfg.get('ema_slow')}\\n"
        f"RSI period={cfg.get('rsi_period')} buy_below={cfg.get('buy_below')} sell_above={cfg.get('sell_above')}\\n"
        f"BB period={cfg.get('bb_period')} std={cfg.get('bb_std')}\\n"
        f"ADX min={cfg.get('adx_min')} regime_mode={cfg.get('regime_mode')}\\n"
        f"enter_long={cfg.get('enter_long')} exit_long={cfg.get('exit_long')}"
        "</pre>\n"
        f"<b>best_score:</b> <code>{metrics.get('score', 0.0):.6f}</code>\n"
        f"<b>trades:</b> <code>{metrics.get('trades_count', 0)}</code>\n"
        f"<b>final_equity:</b> <code>{metrics.get('final_equity', 0.0):.2f}</code>\n"
        f"<b>profit_%:</b> <code>{metrics.get('profit_%', 0.0):.2f}%</code>"
    )
    await callback.message.answer(text, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("job:last_trades:"))
async def last_trades(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    if not info:
        await callback.answer("Задача не найдена", show_alert=True)
        return

    progress = info.get("progress", {})
    last_trial = progress.get("last_trial") or {}
    trades_count = int(last_trial.get("trades_count", 0) or 0)
    if trades_count <= 0:
        await callback.answer("Последняя попытка не открыла сделок", show_alert=True)
        return

    trades_path = last_trial.get("trades_path")
    if not trades_path:
        await callback.answer("Сделки последней попытки не сохранены", show_alert=True)
        return

    await callback.message.answer_document(FSInputFile(trades_path))
    await callback.answer()




@router.callback_query(F.data.startswith("job:last_error:"))
async def last_error(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    if not info:
        await callback.answer("Задача не найдена", show_alert=True)
        return

    progress = info.get("progress", {})
    last_trial = progress.get("last_trial") or {}
    error_text = str(last_trial.get("error") or "")
    traceback_text = str(last_trial.get("traceback") or "")

    if not error_text and not traceback_text:
        await callback.answer("Для последней попытки нет сохранённой ошибки", show_alert=True)
        return

    trace_lines = traceback_text.splitlines()[:30]
    trace_preview = "\n".join(trace_lines)
    text = (
        "<b>🧯 Последняя ошибка</b>\n"
        f"<b>Job:</b> <code>{job_id}</code>\n"
        f"<b>Error:</b> <code>{error_text or '-'} </code>\n"
        "<b>Traceback (preview):</b>\n"
        f"<pre>{html.escape(trace_preview[:3500])}</pre>"
    )
    await callback.message.answer(text, parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("job:export:"))
async def export_json(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not best or not best.get("config_path"):
        await callback.answer(_reason_no_best(info), show_alert=True)
        return
    await callback.message.answer_document(FSInputFile(best["config_path"]))
    await callback.answer()


@router.callback_query(F.data == "menu:settings")
async def settings(callback: CallbackQuery) -> None:
    await _safe_edit(
        callback,
        "<b>⚙️ Настройки</b>\nMVP: настройки берутся из .env",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )
    await callback.answer()


@router.callback_query(F.data == "menu:help")
async def help_menu(callback: CallbackQuery) -> None:
    await _safe_edit(
        callback,
        "<b>ℹ️ Помощь</b>\n"
        "1) Нажмите ➕ Новая оптимизация\n"
        "2) Для надёжности сначала 📥 Данные (Preload)\n"
        "3) В optimize можно выбрать диапазон из кэша",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )
    await callback.answer()
