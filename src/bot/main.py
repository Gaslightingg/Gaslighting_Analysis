from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from src.bot.handlers import router
from src.config import settings
from src.storage.init_db import init_db


async def main() -> None:
    init_db()
    if not settings.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN is required")
    bot = Bot(token=settings.telegram_token, parse_mode="HTML")
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
