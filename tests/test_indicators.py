from __future__ import annotations

import pandas as pd

from src.indicators.calculator import add_indicators


def test_add_indicators_non_mutating() -> None:
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [1, 1, 2, 3, 4],
            "Close": [1.2, 2.2, 3.1, 4.1, 5.2],
            "Volume": [100, 110, 120, 130, 140],
        }
    )
    cols_before = list(df.columns)
    out = add_indicators(df, {"ema_fast": 5, "ema_slow": 20})
    assert list(df.columns) == cols_before
    assert "adx" in out.columns
    assert "obv" in out.columns


def test_add_indicators_handles_multiindex_ohlcv_columns() -> None:
    base = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [1, 1, 2, 3, 4],
            "Close": [1.2, 2.2, 3.1, 4.1, 5.2],
            "Volume": [100, 110, 120, 130, 140],
        }
    )
    multi = pd.concat({"SPY": base}, axis=1)

    out = add_indicators(multi, {})

    assert "adx" in out.columns
    assert out["adx"].notna().any()
