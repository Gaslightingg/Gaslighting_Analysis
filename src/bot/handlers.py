from __future__ import annotations

import json
from datetime import date, timedelta

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, FSInputFile, Message
from sqlalchemy import select

from src.bot.keyboards import (
    job_keyboard,
    main_menu_keyboard,
    mode_keyboard,
    period_keyboard,
    ticker_keyboard,
)
from src.bot.states import NewOptimizationWizard
from src.storage.db import SessionLocal
from src.storage.models import Backtest, Job, Optimization
from src.worker.tasks import run_optimization_task

router = Router()


def _period_to_dates(period: str) -> tuple[str, str]:
    end = date.today()
    if period == "1y":
        start = end - timedelta(days=365)
    else:
        start = end - timedelta(days=730)
    return start.isoformat(), end.isoformat()


def _render_job_card(job: Job, optimization: Optimization | None) -> str:
    best_score = None
    if optimization and optimization.best_metrics_json:
        best_score = optimization.best_metrics_json.get("score")
    trials = job.progress_json.get("trials_done") if job.progress_json else None
    return (
        "<b>📊 Job Card</b>\n"
        f"ID: <code>{job.id}</code>\n"
        f"Статус: <b>{job.status}</b>\n"
        f"Trials: <b>{trials or 0}</b>\n"
        f"Best score: <b>{best_score if best_score is not None else 'N/A'}</b>\n"
        f"Updated: <code>{job.updated_at}</code>"
    )


@router.message(F.text == "/start")
async def start_cmd(message: Message) -> None:
    await message.answer(
        "<b>Telegram Trading Lab</b>\nВыберите действие:",
        reply_markup=main_menu_keyboard(),
        parse_mode="HTML",
    )


@router.callback_query(F.data == "new_opt")
async def new_opt_start(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(NewOptimizationWizard.waiting_ticker)
    await callback.message.edit_text(
        "<b>Шаг 1/5</b>: Выберите тикер", reply_markup=ticker_keyboard(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("ticker:"))
async def select_ticker(callback: CallbackQuery, state: FSMContext) -> None:
    ticker = callback.data.split(":", maxsplit=1)[1]
    await state.update_data(ticker=ticker)
    await state.set_state(NewOptimizationWizard.waiting_period)
    await callback.message.edit_text(
        "<b>Шаг 2/5</b>: Выберите период", reply_markup=period_keyboard(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("period:"))
async def select_period(callback: CallbackQuery, state: FSMContext) -> None:
    period = callback.data.split(":", maxsplit=1)[1]
    start, end = _period_to_dates(period)
    await state.update_data(start=start, end=end)
    await state.set_state(NewOptimizationWizard.waiting_mode)
    await callback.message.edit_text(
        "<b>Шаг 4/5</b>: Выберите режим", reply_markup=mode_keyboard(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("mode:"))
async def launch_job(callback: CallbackQuery, state: FSMContext) -> None:
    mode = callback.data.split(":", maxsplit=1)[1]
    data = await state.get_data()
    params = {
        "ticker": data["ticker"],
        "start": data["start"],
        "end": data["end"],
        "mode": mode,
        "template": "Hybrid Vote",
    }

    with SessionLocal() as session:
        job = Job(
            user_id=callback.from_user.id,
            type="optimization",
            status="queued",
            params_json=params,
        )
        session.add(job)
        session.flush()
        optimization = Optimization(
            job_id=job.id,
            ticker=params["ticker"],
            start=date.fromisoformat(params["start"]),
            end=date.fromisoformat(params["end"]),
            best_config_json={},
            best_metrics_json={},
        )
        session.add(optimization)
        session.commit()
        job_id = job.id

    run_optimization_task.delay(job_id)
    with SessionLocal() as session:
        created = session.get(Job, job_id)
        opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
        card = _render_job_card(created, opt)

    await callback.message.edit_text(card, reply_markup=job_keyboard(job_id), parse_mode="HTML")
    await callback.answer("Запущено")
    await state.clear()


@router.callback_query(F.data == "best_now")
async def best_now(callback: CallbackQuery) -> None:
    with SessionLocal() as session:
        job = session.scalar(
            select(Job).where(Job.user_id == callback.from_user.id).order_by(Job.created_at.desc())
        )
        if not job:
            await callback.answer("Нет задач", show_alert=True)
            return
        opt = session.scalar(select(Optimization).where(Optimization.job_id == job.id))
        msg = _render_job_card(job, opt)
    await callback.message.edit_text(msg, reply_markup=job_keyboard(job.id), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("refresh:"))
@router.callback_query(F.data.startswith("best:"))
async def refresh_job(callback: CallbackQuery) -> None:
    job_id = int(callback.data.split(":", maxsplit=1)[1])
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
        if not job:
            await callback.answer("Job not found", show_alert=True)
            return
        await callback.message.edit_text(
            _render_job_card(job, opt), reply_markup=job_keyboard(job_id), parse_mode="HTML"
        )
    await callback.answer()


@router.callback_query(F.data.startswith("stop:"))
async def stop_job(callback: CallbackQuery) -> None:
    job_id = int(callback.data.split(":", maxsplit=1)[1])
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job:
            job.stop_requested = True
            session.commit()
    await callback.answer("Остановка запрошена")


@router.callback_query(F.data.startswith("equity:"))
async def send_equity(callback: CallbackQuery) -> None:
    job_id = int(callback.data.split(":", maxsplit=1)[1])
    with SessionLocal() as session:
        bt = session.scalar(
            select(Backtest).where(Backtest.job_id == job_id).order_by(Backtest.created_at.desc())
        )
    if not bt:
        await callback.answer("Пока нет графика", show_alert=True)
        return
    await callback.message.answer_photo(FSInputFile(bt.equity_path), caption=f"Job {job_id}")
    await callback.answer()


@router.callback_query(F.data.startswith("trades:"))
async def send_trades(callback: CallbackQuery) -> None:
    job_id = int(callback.data.split(":", maxsplit=1)[1])
    with SessionLocal() as session:
        bt = session.scalar(
            select(Backtest).where(Backtest.job_id == job_id).order_by(Backtest.created_at.desc())
        )
    if not bt:
        await callback.answer("Сделок пока нет", show_alert=True)
        return
    await callback.message.answer_document(
        FSInputFile(bt.trades_path),
        caption=f"Trades Job {job_id}",
    )
    await callback.answer()


@router.callback_query(F.data.startswith("export:"))
async def export_config(callback: CallbackQuery) -> None:
    job_id = int(callback.data.split(":", maxsplit=1)[1])
    with SessionLocal() as session:
        opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
    if not opt or not opt.best_config_json:
        await callback.answer("Нет конфига", show_alert=True)
        return
    text = f"<b>Best config</b>\n<pre>{json.dumps(opt.best_config_json, indent=2)}</pre>"
    await callback.message.answer(text, parse_mode="HTML")
    await callback.answer()
