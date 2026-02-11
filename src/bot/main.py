from __future__ import annotations

import asyncio
import logging

from aiohttp import ClientConnectorError
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
from aiogram.fsm.storage.memory import MemoryStorage

from src.bot.handlers import router
from src.config import SETTINGS

_LOG = logging.getLogger("bot.main")


def _backoff_delay(attempt: int) -> float:
    backoff_sleep_sec = min(float(SETTINGS.telegram_retry_max_sleep), float(max(1, 2 ** (attempt - 1))))
    return backoff_sleep_sec


def _validate_session_timeout(bot: Bot) -> float:
    timeout_value = bot.session.timeout
    _LOG.info("bot.session.timeout type=%s value=%s", type(timeout_value).__name__, timeout_value)
    if not isinstance(timeout_value, (int, float)):
        raise RuntimeError(
            "Invalid bot.session.timeout type: expected int/float seconds, "
            f"got {type(timeout_value).__name__}."
        )
    return float(timeout_value)


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
            backoff_sleep_sec = _backoff_delay(attempt)
            _LOG.warning(
                "get_me failed (%s: %s) — network retry in %.1fs (attempt %s/%s)",
                type(exc).__name__,
                exc,
                backoff_sleep_sec,
                attempt,
                SETTINGS.telegram_retry_max,
            )
            await asyncio.sleep(backoff_sleep_sec)

    _LOG.warning(
        "get_me warmup failed after %s attempts; continuing to polling loop",
        SETTINGS.telegram_retry_max,
    )


async def run_bot() -> None:
    if not SETTINGS.telegram_token:
        raise RuntimeError("TELEGRAM_TOKEN is required. Set it in .env before running bot/all mode.")

    request_timeout_sec = float(SETTINGS.telegram_request_timeout_sec)
    session = AiohttpSession(timeout=request_timeout_sec)
    bot = Bot(
        token=SETTINGS.telegram_token,
        default=DefaultBotProperties(parse_mode="HTML"),
        session=session,
    )

    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    try:
        request_timeout_sec = _validate_session_timeout(bot)
        await _warmup_get_me(bot)

        attempt = 0
        while True:
            try:
                _LOG.info("Starting Telegram polling")
                await dp.start_polling(
                    bot,
                    polling_timeout=int(max(1, request_timeout_sec)),
                    request_timeout=request_timeout_sec,
                )
                _LOG.info("Polling finished")
                return
            except asyncio.CancelledError:
                raise
            except (TelegramNetworkError, ClientConnectorError, ConnectionResetError, OSError) as exc:
                attempt += 1
                backoff_sleep_sec = _backoff_delay(attempt)
                _LOG.warning(
                    "polling error (%s: %s) — network retry in %.1fs",
                    type(exc).__name__,
                    exc,
                    backoff_sleep_sec,
                )
                await asyncio.sleep(backoff_sleep_sec)
    finally:
        await bot.session.close()
        _LOG.info("Telegram bot session closed")


def start_bot_sync() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        _LOG.info("Bot interrupted by user")
