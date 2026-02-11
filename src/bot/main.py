from __future__ import annotations

import asyncio
import logging

from aiohttp import ClientConnectorError, ClientTimeout
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
from aiogram.fsm.storage.memory import MemoryStorage

from src.bot.handlers import router
from src.config import SETTINGS

_LOG = logging.getLogger("bot.main")


def _backoff_delay(attempt: int) -> int:
    return min(SETTINGS.telegram_retry_max_sleep, max(1, 2 ** (attempt - 1)))


async def _warmup_get_me(bot: Bot) -> None:
    for attempt in range(1, SETTINGS.telegram_retry_max + 1):
        try:
            me = await bot.get_me()
            if SETTINGS.telegram_bot_id is not None and me.id != SETTINGS.telegram_bot_id:
                raise RuntimeError(
                    f"TELEGRAM_BOT_ID mismatch: expected {SETTINGS.telegram_bot_id}, got {me.id}."
                )
            _LOG.info("Telegram API warmup OK: bot_id=%s username=@%s", me.id, me.username)
            return
        except RuntimeError:
            raise
        except (TelegramNetworkError, ClientConnectorError, ConnectionResetError, OSError) as exc:
            delay = _backoff_delay(attempt)
            _LOG.warning(
                "get_me failed (%s: %s) — network retry in %ss (attempt %s/%s)",
                type(exc).__name__,
                exc,
                delay,
                attempt,
                SETTINGS.telegram_retry_max,
            )
            await asyncio.sleep(delay)

    _LOG.warning(
        "get_me warmup failed after %s attempts; continuing to polling loop",
        SETTINGS.telegram_retry_max,
    )


async def run_bot() -> None:
    if not SETTINGS.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN is required. Set it in .env before running bot/all mode.")

    session = AiohttpSession(
        timeout=ClientTimeout(
            total=None,
            connect=float(SETTINGS.telegram_connect_timeout),
            sock_connect=float(SETTINGS.telegram_connect_timeout),
            sock_read=float(SETTINGS.telegram_read_timeout),
        )
    )
    bot = Bot(
        token=SETTINGS.telegram_token,
        default=DefaultBotProperties(parse_mode="HTML"),
        session=session,
    )

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    try:
        await _warmup_get_me(bot)

        attempt = 0
        while True:
            try:
                _LOG.info("Starting Telegram polling")
                await dp.start_polling(bot)
                _LOG.info("Polling finished")
                return
            except asyncio.CancelledError:
                raise
            except (TelegramNetworkError, ClientConnectorError, ConnectionResetError, OSError) as exc:
                attempt += 1
                delay = _backoff_delay(attempt)
                _LOG.warning(
                    "polling error (%s: %s) — network retry in %ss",
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
    finally:
        await bot.session.close()
        _LOG.info("Telegram bot session closed")


def start_bot_sync() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        _LOG.info("Bot interrupted by user")
