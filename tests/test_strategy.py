from __future__ import annotations

import pandas as pd

from src.strategy.hybrid_vote import generate_hybrid_signals


def test_generate_hybrid_signals_shape() -> None:
    idx = pd.RangeIndex(10)
    df = pd.DataFrame(
        {
            "Close": [10, 11, 12, 13, 14, 13, 12, 11, 10, 9],
            "ema_fast": [10, 11, 12, 13, 14, 13, 12, 11, 10, 9],
            "ema_slow": [10, 10, 10, 11, 12, 12, 12, 12, 12, 12],
            "rsi": [50, 40, 30, 20, 25, 70, 80, 60, 50, 40],
            "bb_lower": [9] * 10,
            "bb_upper": [15] * 10,
            "adx": [25] * 10,
        },
        index=idx,
    )
    pos = generate_hybrid_signals(df, {"enter_long": 1, "exit_long": 0, "adx_min": 20})
    assert len(pos) == len(df)
    assert set(pos.unique()).issubset({0, 1})
