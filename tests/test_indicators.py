from __future__ import annotations

import pandas as pd

from src.indicators.ta import add_indicators


def test_add_indicators_no_mutation() -> None:
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5, 6],
            "High": [2, 3, 4, 5, 6, 7],
            "Low": [1, 1, 2, 3, 4, 5],
            "Close": [1.5, 2.2, 3.1, 3.7, 4.8, 5.5],
            "Volume": [100, 120, 110, 130, 125, 140],
        }
    )
    before = df.copy()
    out = add_indicators(df, {"ema_fast": 2, "ema_slow": 3})

    assert "ema_fast" in out.columns
    assert "rsi" in out.columns
    assert list(df.columns) == list(before.columns)
