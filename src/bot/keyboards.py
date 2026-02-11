from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


POPULAR_TICKERS = ["SPY", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG", "META"]


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➕ Новая оптимизация", callback_data="menu:new")],
            [InlineKeyboardButton(text="📥 Данные (Preload)", callback_data="menu:preload")],
            [InlineKeyboardButton(text="🏆 Текущий лучший", callback_data="menu:best")],
            [InlineKeyboardButton(text="📌 Мои задачи", callback_data="menu:jobs")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="menu:settings")],
            [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="menu:help")],
        ]
    )


def ticker_kb(last_ticker: str | None = None) -> InlineKeyboardMarkup:
    rows = []
    if last_ticker:
        rows.append([InlineKeyboardButton(text=f"Последний: {last_ticker}", callback_data=f"ticker:{last_ticker}")])
    rows.extend(
        [
            [InlineKeyboardButton(text="AAPL", callback_data="ticker:AAPL")],
            [InlineKeyboardButton(text="MSFT", callback_data="ticker:MSFT")],
            [InlineKeyboardButton(text="SPY", callback_data="ticker:SPY")],
            [InlineKeyboardButton(text="Ввести вручную", callback_data="ticker:manual")],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def period_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1 год", callback_data="period:1y")],
            [InlineKeyboardButton(text="2 года", callback_data="period:2y")],
            [InlineKeyboardButton(text="Кастом", callback_data="period:custom")],
            [InlineKeyboardButton(text="Использовать доступный диапазон (из кэша)", callback_data="period:cache")],
        ]
    )


def mode_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Быстро", callback_data="mode:quick")],
            [InlineKeyboardButton(text="Стандарт", callback_data="mode:standard")],
            [InlineKeyboardButton(text="Глубоко", callback_data="mode:deep")],
        ]
    )


def confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="🚀 Запустить", callback_data="confirm:start")]]
    )


def preload_horizon_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1y", callback_data="preload:horizon:1y")],
            [InlineKeyboardButton(text="2y", callback_data="preload:horizon:2y")],
            [InlineKeyboardButton(text="5y", callback_data="preload:horizon:5y")],
        ]
    )


def preload_tickers_kb(last_ticker: str | None = None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if last_ticker:
        rows.append([InlineKeyboardButton(text=f"Последний: {last_ticker}", callback_data=f"preload:ticker:{last_ticker}")])
    rows.append([InlineKeyboardButton(text="Популярные (SPY AAPL MSFT TSLA NVDA AMZN GOOG META)", callback_data="preload:ticker:popular")])
    rows.append([InlineKeyboardButton(text="Ввести список вручную", callback_data="preload:ticker:manual")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def preload_confirm_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="📥 Запустить preload", callback_data="preload:confirm")]])


def job_card_kb(job_id: str, has_best: bool = True, no_results: bool = False) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"job:refresh:{job_id}")],
        [InlineKeyboardButton(text="🏆 Текущий лучший", callback_data=f"job:best:{job_id}")],
    ]

    if has_best and not no_results:
        rows.append([InlineKeyboardButton(text="📈 График equity", callback_data=f"job:equity:{job_id}")])
        rows.append([InlineKeyboardButton(text="🧾 Сделки", callback_data=f"job:trades:{job_id}")])
        rows.append([InlineKeyboardButton(text="📦 Экспорт JSON", callback_data=f"job:export:{job_id}")])

    rows.append([InlineKeyboardButton(text="🧾 Сделки (последняя попытка)", callback_data=f"job:last_trades:{job_id}")])
    rows.append([InlineKeyboardButton(text="🧯 Последняя ошибка", callback_data=f"job:last_error:{job_id}")])

    rows.append([InlineKeyboardButton(text="🛑 Остановить", callback_data=f"job:stop:{job_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def preload_job_kb(job_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"job:refresh:{job_id}")],
            [InlineKeyboardButton(text="🏠 Главное меню", callback_data="menu:home")],
        ]
    )


def jobs_list_kb(job_ids: list[str]) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=f"Открыть {jid[:8]}", callback_data=f"job:refresh:{jid}")] for jid in job_ids]
    if not rows:
        rows = [[InlineKeyboardButton(text="Главное меню", callback_data="menu:home")]]
    return InlineKeyboardMarkup(inline_keyboard=rows)
