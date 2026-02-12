# Root cause: why logs showed `enter>0, exit=0` while trades existed

## What actually happened
The backtester executed and closed trades, but optimization diagnostics counted signals from columns that were not reliably propagated into backtest/walk-forward frames.
In particular, signal counting could read fallback zeros when expected exit-related columns were missing in the combined frame.

## Why this produced misleading logs
- Trade lifecycle (open/close) was handled in engine state via `position` transitions.
- Signal diagnostics (enter/exit counters) were computed from separate columns and could be out of sync.
- Result: `trades_count>0` with `signals_count_exit=0` in logs.

## Fix implemented
1. Introduced explicit FLAT/LONG/SHORT state machine in backtester and symmetric close/open transitions.
2. Propagated explicit event columns (`enter_long`, `exit_long`, `enter_short`, `exit_short`) into backtest frame.
3. Ensured forced end-of-test close is a real exit event and increments exit counters.
4. Reworked walk-forward signal counters to aggregate explicit event columns.
5. Added tests for long/short exits, forced exit accounting, flip behavior, no double-entry, PnL correctness and symmetry.
