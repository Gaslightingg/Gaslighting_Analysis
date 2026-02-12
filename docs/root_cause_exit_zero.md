# Root cause: why summary sometimes showed `signals: enter=..., exit=0`

## Confirmed cause
There were two different counting paths:
1. `trial_diag_end` used trial metrics produced during optimization.
2. Telegram summary rendered counters from persisted progress payload fields that could be derived from incomplete/misaligned signal columns.

When explicit exit event columns were missing or not consistently propagated, fallback zeros were used for `signals_count_exit`, while trades were still closed by execution state transitions (including forced close). This produced the mismatch: exits happened, but summary displayed `exit=0`.

## Fix
- Made event accounting explicit and consistent:
  - `entry_events_count`, `exit_events_count`, `closed_trades_count`, `forced_exit_count`
  - exit breakdown: `exits_by_rule`, `exits_by_sl_tp`, `exits_forced_end`, `exits_flip`
- Unified source of truth through trial metrics snapshot persisted in progress and rendered by Telegram summary.
- Added FLAT/LONG/SHORT state machine with symmetric long/short lifecycle and forced-end close as a real exit event.
- Added invariants:
  - `exit_events_count <= entry_events_count`
  - forced end close increments both `exit_events_count` and `forced_exit_count`.

## Expected log consistency after fix
`trial_diag_end` and Telegram last-trial summary now use the same persisted event counters, so enter/exit values match (except explicitly explainable components like forced exits and flip exits, which are broken out separately).
