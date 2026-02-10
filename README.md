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

## Redis
```bash
docker-compose up -d redis
```

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
