from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from src.bot.handlers import router
from src.config import SETTINGS


async def run_bot() -> None:
    if not SETTINGS.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN is required. Set it in .env before running bot/all mode.")

    bot = Bot(
        token=SETTINGS.telegram_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )

    me = await bot.get_me()
    if SETTINGS.telegram_bot_id is not None and me.id != SETTINGS.telegram_bot_id:
        raise RuntimeError(
            f"TELEGRAM_BOT_ID mismatch: expected {SETTINGS.telegram_bot_id}, got {me.id}."
        )

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


def start_bot_sync() -> None:
    asyncio.run(run_bot())
