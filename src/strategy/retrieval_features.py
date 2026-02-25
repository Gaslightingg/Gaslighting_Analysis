from __future__ import annotations

import numpy as np
import pandas as pd


_FEATURE_WINDOWS = (5, 10, 20, 40, 80)


def _rolling_slope_log(close: pd.Series, window: int) -> pd.Series:
    logp = np.log(close.clip(lower=1e-12))
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum()) if window > 1 else 1.0

    def _fit(vals: np.ndarray) -> float:
        y = vals - vals.mean()
        return float((x * y).sum() / denom)

    return logp.rolling(window=window, min_periods=window).apply(_fit, raw=True)


def build_retrieval_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")

    ret1 = close.pct_change()
    log_ret = np.log(close / close.shift(1).replace(0, np.nan))
    hl_spread = (high - low) / close.replace(0, np.nan)

    out["ret_1"] = ret1
    out["ret_2"] = close.pct_change(2)
    out["ret_5"] = close.pct_change(5)
    out["range_1"] = hl_spread

    for w in _FEATURE_WINDOWS:
        ma = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std(ddof=0)
        ret_std = ret1.rolling(w, min_periods=w).std(ddof=0)

        out[f"dist_ma_{w}"] = (close / ma) - 1.0
        out[f"breakout_hi_{w}"] = (close / high.rolling(w, min_periods=w).max()) - 1.0
        out[f"breakout_lo_{w}"] = (close / low.rolling(w, min_periods=w).min()) - 1.0
        out[f"realized_vol_{w}"] = log_ret.rolling(w, min_periods=w).std(ddof=0) * np.sqrt(252)
        out[f"atr_proxy_{w}"] = hl_spread.rolling(w, min_periods=w).mean()
        out[f"ret_std_{w}"] = ret_std
        out[f"ret_skew_{w}"] = ret1.rolling(w, min_periods=w).skew()
        out[f"ret_kurt_{w}"] = ret1.rolling(w, min_periods=w).kurt()
        out[f"vol_z_{w}"] = (volume - volume.rolling(w, min_periods=w).mean()) / volume.rolling(w, min_periods=w).std(ddof=0)
        dv = close * volume
        out[f"dollar_vol_z_{w}"] = (dv - dv.rolling(w, min_periods=w).mean()) / dv.rolling(w, min_periods=w).std(ddof=0)
        out[f"slope_log_{w}"] = _rolling_slope_log(close, w)
        out[f"vol_of_vol_{w}"] = ret_std.rolling(max(3, w // 2), min_periods=max(3, w // 2)).std(ddof=0)

    for lag in (1, 2, 3, 5):
        out[f"autocorr_{lag}_20"] = ret1.rolling(20, min_periods=20).corr(ret1.shift(lag))
        out[f"autocorr_{lag}_40"] = ret1.rolling(40, min_periods=40).corr(ret1.shift(lag))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
