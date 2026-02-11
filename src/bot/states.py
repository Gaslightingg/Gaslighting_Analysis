from aiogram.fsm.state import State, StatesGroup


class NewOptimizationState(StatesGroup):
    choose_ticker = State()
    manual_ticker = State()
    choose_period = State()
    manual_period = State()
    choose_mode = State()
    confirm = State()


class PreloadState(StatesGroup):
    choose_horizon = State()
    choose_tickers = State()
    manual_tickers = State()
    confirm = State()
