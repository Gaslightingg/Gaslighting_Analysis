from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if ":" in raw:
        raw = raw.split(":", maxsplit=1)[0]
    if not raw.isdigit():
        return None
    parsed = int(raw)
    return parsed or None


@dataclass(slots=True)
class Preset:
    name: str
    trials_total: int
    checkpoint_n: int


@dataclass(slots=True)
class Settings:
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_allowed_user_id: int | None = _parse_optional_int(os.getenv("TELEGRAM_ALLOWED_USER_ID"))
    telegram_bot_id: int | None = _parse_optional_int(os.getenv("TELEGRAM_BOT_ID"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/app.db")
    commission_bps: float = float(os.getenv("COMMISSION_BPS", "2"))
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    runs_dir: str = ".runs"
    cache_dir: str = ".cache/ohlcv"
    seed: int = 42


SETTINGS = Settings()

PRESETS: dict[str, Preset] = {
    "quick": Preset(
        name="Быстро",
        trials_total=int(os.getenv("DEFAULT_TRIALS_QUICK", "200")),
        checkpoint_n=int(os.getenv("DEFAULT_CHECKPOINT_N_QUICK", "25")),
    ),
    "standard": Preset(
        name="Стандарт",
        trials_total=int(os.getenv("DEFAULT_TRIALS_STANDARD", "1000")),
        checkpoint_n=int(os.getenv("DEFAULT_CHECKPOINT_N_STANDARD", "50")),
    ),
    "deep": Preset(
        name="Глубоко",
        trials_total=int(os.getenv("DEFAULT_TRIALS_DEEP", "5000")),
        checkpoint_n=int(os.getenv("DEFAULT_CHECKPOINT_N_DEEP", "100")),
    ),
}
