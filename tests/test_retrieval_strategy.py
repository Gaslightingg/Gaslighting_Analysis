from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer.walk_forward import WalkForwardConfig, evaluate_config_walk_forward
from src.strategy.retrieval_strategy import generate_retrieval_positions


def _sample_ohlcv(n: int = 420) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    base = 100 + np.cumsum(np.random.default_rng(7).normal(0.05, 1.0, size=n))
    close = pd.Series(base, index=idx).clip(lower=1.0)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998
    vol = pd.Series(np.random.default_rng(9).integers(1000, 5000, size=n), index=idx)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def test_retrieval_positions_no_pre_test_entries() -> None:
    df = _sample_ohlcv(360)
    test_start = df.index[300]
    cfg = {
        "retrieval_horizon": 5,
        "retrieval_k": 30,
        "retrieval_min_neighbors": 10,
        "entry_mean_threshold": -1e-4,
        "entry_score_threshold": -1.0,
        "risk_limit": 0.1,
        "max_dist_percentile": 0.95,
        "hold_bars": 4,
        "embargo_bars": 2,
    }
    out = generate_retrieval_positions(df, cfg, test_start=test_start)
    assert int(out.signal_df.loc[out.signal_df.index < test_start, "enter_long"].sum()) == 0


def test_walk_forward_retrieval_outputs_baselines() -> None:
    df = _sample_ohlcv(500)
    cfg = {
        "strategy_family": "retrieval",
        "sl_pct": 0.01,
        "tp_pct": 0.03,
        "position_size_pct": 0.1,
        "execution_mode": "next_open",
        "allow_short": False,
        "retrieval_horizon": 5,
        "retrieval_k": 25,
        "retrieval_min_neighbors": 8,
        "entry_mean_threshold": -1e-4,
        "entry_score_threshold": -0.5,
        "exit_score_threshold": -2.0,
        "risk_limit": 0.2,
        "max_dist_percentile": 0.95,
        "hold_bars": 5,
        "embargo_bars": 2,
    }
    wf = WalkForwardConfig(train_size=252, test_size=84, step_size=84, window_type="rolling")
    score, metrics, combined, _trades = evaluate_config_walk_forward(df, cfg, 0, 0, wf)

    assert np.isfinite(score)
    assert "buyhold_equity" in combined.columns
    assert "ma200_equity" in combined.columns
    assert "buyhold_final_equity" in metrics
    assert "ma200_final_equity" in metrics
