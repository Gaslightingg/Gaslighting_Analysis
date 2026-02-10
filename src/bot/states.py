from aiogram.fsm.state import State, StatesGroup


class NewOptimizationWizard(StatesGroup):
    waiting_ticker = State()
    waiting_period = State()
    waiting_mode = State()
