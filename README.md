# telegram-trading-lab

Полностью рабочий MVP Telegram-бота для асинхронной оптимизации стратегий (Celery + Redis + SQLite + aiogram v3).

## Возможности
- Полностью кнопочный UX в Telegram (inline keyboards).
- FSM wizard: тикер, период, режим, подтверждение.
- Асинхронная оптимизация в worker (Celery), бот не блокируется.
- Walk-forward оценка по временным окнам (train/test/step months).
- No look-ahead backtest: сигнал на close t, исполнение на open t+1.
- Комиссии и slippage в bps на каждую сделку.
- Checkpoint best-so-far: запись прогресса после каждого trial и checkpoints каждые N.
- Кнопка Stop: `stop_requested=true`, worker завершает корректно.
- Артефакты в `.runs/{job_id}`: equity PNG, trades JSON, best_config JSON.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Важные переменные Telegram
- `TELEGRAM_TOKEN` — токен бота (обязателен для `all` и `bot`).
- `TELEGRAM_ALLOWED_USER_ID` — только этот user id сможет пользоваться ботом.
- `TELEGRAM_BOT_ID` — проверка, что запущен нужный бот (защита от ошибочного токена). Можно указать либо числовой id, либо токен (id возьмётся из части до `:`).

## Redis
```bash
docker-compose up -d redis
```


Режим `python app.py all` пытается сам поднять Redis (через `docker compose` / `docker-compose` / `redis-server`), затем запускает worker и bot.

## Запуск
```bash
python app.py all
```

Дополнительно:
```bash
python app.py bot
python app.py worker
```

## Где результаты
- БД: `DATABASE_URL` (по умолчанию `sqlite:///data/app.db`)
- Кэш котировок: `.cache/ohlcv/*.parquet`
- Лучшие артефакты job: `.runs/{job_id}/`

## Переменные окружения
См. `.env.example`.
