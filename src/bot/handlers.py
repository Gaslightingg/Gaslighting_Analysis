from __future__ import annotations

import json
from datetime import date, timedelta

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message

from src.bot.keyboards import (
    confirm_kb,
    job_card_kb,
    jobs_list_kb,
    main_menu_kb,
    mode_kb,
    period_kb,
    ticker_kb,
)
from src.bot.render import render_best_card, render_job_card
from src.bot.states import NewOptimizationState
from src.config import PRESETS
from src.storage.repository import Repository
from src.worker.tasks import optimization_run

router = Router()
repo = Repository()


def _period_from_code(code: str) -> tuple[str, str]:
    end = date.today()
    if code == "1y":
        start = end - timedelta(days=365)
    else:
        start = end - timedelta(days=365 * 2)
    return start.isoformat(), end.isoformat()


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
    await callback.message.edit_text(
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
    await callback.message.edit_text(
        "<b>Шаг 1/4:</b> Выберите тикер",
        parse_mode="HTML",
        reply_markup=ticker_kb(last_ticker=last_ticker),
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ticker:"))
async def ticker_selected(callback: CallbackQuery, state: FSMContext) -> None:
    val = callback.data.split(":", maxsplit=1)[1]
    if val == "manual":
        await state.set_state(NewOptimizationState.manual_ticker)
        await callback.message.edit_text(
            "Введите тикер (пример: <code>AAPL</code>)",
            parse_mode="HTML",
        )
        await callback.answer()
        return

    await state.update_data(ticker=val.upper())
    await state.set_state(NewOptimizationState.choose_period)
    await callback.message.edit_text(
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
        await callback.message.edit_text(
            "Введите период: <code>YYYY-MM-DD YYYY-MM-DD</code>",
            parse_mode="HTML",
        )
        await callback.answer()
        return

    start, end = _period_from_code(val)
    await state.update_data(start=start, end=end)
    await state.set_state(NewOptimizationState.choose_mode)
    await callback.message.edit_text(
        "<b>Шаг 3/4:</b> Выберите режим оптимизации",
        parse_mode="HTML",
        reply_markup=mode_kb(),
    )
    await callback.answer()


@router.message(NewOptimizationState.manual_period)
async def period_manual(message: Message, state: FSMContext) -> None:
    try:
        start, end = (message.text or "").strip().split()
        _ = date.fromisoformat(start)
        _ = date.fromisoformat(end)
    except Exception:  # noqa: BLE001
        await message.answer("Неверный формат. Используйте: <code>YYYY-MM-DD YYYY-MM-DD</code>", parse_mode="HTML")
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
    await callback.message.edit_text(
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
    mode = data["mode"]
    preset = PRESETS[mode]

    job_id = repo.create_job(
        user_id=callback.from_user.id,
        chat_id=callback.message.chat.id,
        ticker=data["ticker"],
        start=data["start"],
        end=data["end"],
        preset=mode,
        trials_total=preset.trials_total,
        checkpoint_n=preset.checkpoint_n,
    )
    optimization_run.delay(job_id)

    info = repo.get_job_data(job_id)
    card = render_job_card(
        job_id=job_id,
        ticker=data["ticker"],
        start=data["start"],
        end=data["end"],
        preset=preset.name,
        progress=info["progress"] if info else {},
    )
    await callback.message.edit_text(card, parse_mode="HTML", reply_markup=job_card_kb(job_id))
    await callback.answer("Запущено")
    await state.clear()


@router.callback_query(F.data == "menu:jobs")
async def my_jobs(callback: CallbackQuery) -> None:
    jobs = repo.list_jobs(callback.from_user.id)
    text = "<b>📌 Мои задачи</b>\n" + ("\n".join(f"• <code>{j.id}</code> [{j.status}]" for j in jobs) or "Нет задач")
    await callback.message.edit_text(
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
    params = info["params"]
    preset = PRESETS.get(params.get("preset", "quick"))
    card = render_job_card(
        job_id=job_id,
        ticker=params.get("ticker", ""),
        start=params.get("start", ""),
        end=params.get("end", ""),
        preset=preset.name if preset else params.get("preset", "-"),
        progress=info["progress"],
    )
    await callback.message.edit_text(card, parse_mode="HTML", reply_markup=job_card_kb(job_id))
    await callback.answer()


@router.callback_query(F.data.startswith("job:best:"))
async def best_for_job(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    await _send_best(callback, job_id)


async def _send_best(callback: CallbackQuery, job_id: str) -> None:
    info = repo.get_job_data(job_id)
    best = repo.get_best(job_id)
    if not info or not best:
        await callback.answer("Нет данных", show_alert=True)
        return

    text = render_best_card(
        job_id=job_id,
        trials_done=info["progress"].get("trials_done", 0),
        metrics=best["metrics"],
        cfg=best["config"],
        updated_at=str(best["best_updated_at"]),
    )
    await callback.message.edit_text(text, parse_mode="HTML", reply_markup=job_card_kb(job_id))
    await callback.answer()


@router.callback_query(F.data.startswith("job:stop:"))
async def stop_job(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    repo.request_stop(job_id)
    await callback.answer("Остановка запрошена")


@router.callback_query(F.data.startswith("job:equity:"))
async def equity(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    best = repo.get_best(job_id)
    if not best or not best.get("equity_path"):
        await callback.answer("График пока недоступен", show_alert=True)
        return
    await callback.message.answer_photo(FSInputFile(best["equity_path"]))
    await callback.answer()


@router.callback_query(F.data.startswith("job:trades:"))
async def trades(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    best = repo.get_best(job_id)
    if not best or not best.get("trades_path"):
        await callback.answer("Сделки пока недоступны", show_alert=True)
        return
    await callback.message.answer_document(FSInputFile(best["trades_path"]))
    await callback.answer()


@router.callback_query(F.data.startswith("job:export:"))
async def export_json(callback: CallbackQuery) -> None:
    job_id = callback.data.split(":", maxsplit=2)[2]
    best = repo.get_best(job_id)
    if not best or not best.get("config_path"):
        await callback.answer("Файл пока недоступен", show_alert=True)
        return
    await callback.message.answer_document(FSInputFile(best["config_path"]))
    await callback.answer()


@router.callback_query(F.data == "menu:settings")
async def settings(callback: CallbackQuery) -> None:
    await callback.message.edit_text(
        "<b>⚙️ Настройки</b>\nMVP: настройки берутся из .env",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )
    await callback.answer()


@router.callback_query(F.data == "menu:help")
async def help_menu(callback: CallbackQuery) -> None:
    await callback.message.edit_text(
        "<b>ℹ️ Помощь</b>\n"
        "1) Нажмите ➕ Новая оптимизация\n"
        "2) Пройдите wizard\n"
        "3) Откройте карточку job и жмите Обновить/Текущий лучший",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )
    await callback.answer()
