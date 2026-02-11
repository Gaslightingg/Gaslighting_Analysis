# telegram-trading-lab

Полностью рабочий MVP Telegram-бота для асинхронной оптимизации стратегий (Celery + Redis + SQLite + aiogram v3).

## Установка
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Быстрый запуск (Windows/Linux)
1. Поднять Redis:
```bash
docker compose up -d redis
```
2. Проверить подключение:
```bash
python app.py doctor
```
3. Запустить всё:
```bash
python app.py all
```

## Режимы запуска
```bash
python app.py all      # ensure redis -> worker -> bot
python app.py bot      # только бот
python app.py worker   # ensure redis -> worker
python app.py doctor   # диагностика redis URL и ping
```

## Preload данных (через Telegram)
1. В главном меню нажмите **📥 Данные (Preload)**.
2. Выберите horizon: `1y`, `2y` или `5y`.
3. Выберите тикеры: последний / популярные / вручную списком через пробел.
4. Нажмите **📥 Запустить preload**.
5. Открывайте карточку job и жмите **🔄 Обновить статус** — увидите прогресс по тикерам,
   `bars_count` и диапазон дат.

## Оптимизация с кэшем
- На шаге выбора периода есть кнопка **Использовать доступный диапазон (из кэша)**.
- Если кэша нет, бот предложит сначала сделать preload.
- Если конец периода в будущем, он автоматически обрезается до сегодняшнего дня и это явно показывается в UI.

## Что делает doctor
- показывает исходный `REDIS_URL`
- показывает пробованные варианты (`localhost`, `127.0.0.1`, `::1` при необходимости)
- показывает выбранный URL
- показывает `PING: OK/FAIL`
- печатает конкретные шаги запуска Redis

## Важные переменные Telegram
- `TELEGRAM_TOKEN` — токен бота (обязателен для `all` и `bot`).
- `TELEGRAM_ALLOWED_USER_ID` — только этот user id сможет пользоваться ботом.
- `TELEGRAM_BOT_ID` — проверка, что запущен нужный бот (защита от ошибочного токена). Можно указать либо числовой id, либо токен (id возьмётся из части до `:`).

## Где результаты и кэш
- БД: `DATABASE_URL` (по умолчанию `sqlite:///data/app.db`)
- Кэш котировок: `.cache/ohlcv/{TICKER}.parquet`
- Лучшие артефакты job: `.runs/{job_id}/`
