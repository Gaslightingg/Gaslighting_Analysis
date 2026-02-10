# telegram-trading-lab

MVP Telegram-бот для асинхронной оптимизации торговых стратегий (Hybrid Vote + ADX regime filter) с checkpoint-обновлениями лучшего результата.

## Что реализовано
- aiogram v3 бот с inline keyboard UX (`/start`, `Новая оптимизация`, `Текущий лучший`, `Остановить`, `Обновить`).
- Celery + Redis воркер для неблокирующей оптимизации.
- SQLAlchemy-модели под SQLite (с лёгкой миграцией на Postgres через `DATABASE_URL`).
- Загрузка OHLCV через yfinance + parquet cache.
- Индикаторы EMA/RSI/MACD/ADX/ATR/Bollinger/Donchian/OBV в чистой функции `add_indicators`.
- Бэктест без look-ahead: сигнал на close `t`, исполнение позиции с open `t+1`.
- Walk-forward оптимизация с Optuna и score = `CAGR - 0.5*MaxDrawdown` + штраф за <20 сделок.
- Checkpoint каждые `checkpoint_N` trials с сохранением best-so-far в БД.
- Артефакты: equity PNG, trades JSON, best_config JSON в `.runs/{job_id}`.

## Структура
```
src/
  bot/
  worker/
  data_provider/
  indicators/
  strategy/
  backtester/
  optimizer/
  reporter/
  storage/
tests/
```

## Переменные окружения
- `TELEGRAM_TOKEN` — токен бота (обязательно).
- `REDIS_URL` — по умолчанию `redis://redis:6379/0`.
- `DATABASE_URL` — по умолчанию `sqlite:///./trading_lab.db`.
- `RUNS_DIR` — по умолчанию `.runs`.

## Локальный запуск
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m src.storage.init_db
# терминал 1
redis-server
# терминал 2
celery -A src.worker.celery_app.celery_app worker -l info
# терминал 3
python -m src.bot.main
```

## Docker Compose
```bash
export TELEGRAM_TOKEN=xxx
docker compose up --build
```

## Тесты
```bash
pytest
ruff check .
```
