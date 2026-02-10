from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➕ Новая оптимизация", callback_data="new_opt")],
            [InlineKeyboardButton(text="🏆 Текущий лучший", callback_data="best_now")],
            [InlineKeyboardButton(text="📌 Мои задачи", callback_data="my_jobs")],
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
            [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help")],
        ]
    )


def ticker_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="AAPL", callback_data="ticker:AAPL")],
            [InlineKeyboardButton(text="MSFT", callback_data="ticker:MSFT")],
            [InlineKeyboardButton(text="SPY", callback_data="ticker:SPY")],
        ]
    )


def period_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1 год", callback_data="period:1y")],
            [InlineKeyboardButton(text="2 года", callback_data="period:2y")],
        ]
    )


def mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Быстро", callback_data="mode:fast")],
            [InlineKeyboardButton(text="Стандарт", callback_data="mode:standard")],
            [InlineKeyboardButton(text="Глубоко", callback_data="mode:deep")],
        ]
    )


def job_keyboard(job_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"refresh:{job_id}")],
            [InlineKeyboardButton(text="🏆 Текущий лучший", callback_data=f"best:{job_id}")],
            [InlineKeyboardButton(text="📈 График equity", callback_data=f"equity:{job_id}")],
            [InlineKeyboardButton(text="🧾 Сделки", callback_data=f"trades:{job_id}")],
            [InlineKeyboardButton(text="📦 Экспорт JSON", callback_data=f"export:{job_id}")],
            [InlineKeyboardButton(text="🛑 Остановить", callback_data=f"stop:{job_id}")],
        ]
    )
