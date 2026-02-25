from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.strategy.hybrid_vote import (
    SIGNAL_ENTER_LONG,
    SIGNAL_EXIT_LONG,
    SIGNAL_HOLD,
)
from src.strategy.retrieval_features import build_retrieval_features


@dataclass(slots=True)
class RetrievalArtifacts:
    signal_df: pd.DataFrame
    features: pd.DataFrame

class _SimpleScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "_SimpleScaler":
        self.mean_ = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def _make_scaler() -> object:
    try:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler()
    except Exception:
        return _SimpleScaler()


def _compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    return (close.shift(-horizon) / close.replace(0, np.nan)) - 1.0


def _faiss_knn(x_train: np.ndarray, x_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss

        index = faiss.IndexFlatL2(x_train.shape[1])
        index.add(np.ascontiguousarray(x_train.astype(np.float32)))
        dists, idxs = index.search(np.ascontiguousarray(x_query.astype(np.float32)), int(k))
        return dists, idxs
    except Exception:
        # Safe fallback in environments where faiss cannot be imported.
        all_dists = ((x_train[None, :, :] - x_query[:, None, :]) ** 2).sum(axis=2)
        ord_idx = np.argsort(all_dists, axis=1)[:, :k]
        row = np.arange(len(x_query))[:, None]
        return all_dists[row, ord_idx], ord_idx


def generate_retrieval_positions(df: pd.DataFrame, config: dict, test_start: pd.Timestamp) -> RetrievalArtifacts:
    horizon = int(config.get("retrieval_horizon", 5))
    k = int(config.get("retrieval_k", 50))
    min_neighbors = int(config.get("retrieval_min_neighbors", max(10, k // 3)))
    entry_mean_threshold = float(config.get("entry_mean_threshold", 0.0005))
    entry_score_threshold = float(config.get("entry_score_threshold", 0.2))
    risk_limit = float(config.get("risk_limit", 0.02))
    max_dist_percentile = float(config.get("max_dist_percentile", 0.8))
    exit_score_threshold = float(config.get("exit_score_threshold", -0.1))
    hold_bars = int(config.get("hold_bars", horizon))
    embargo_bars = int(config.get("embargo_bars", max(1, horizon // 2)))

    features = build_retrieval_features(df)
    close = pd.to_numeric(df["Close"], errors="coerce")
    y_fwd = _compute_forward_returns(close, horizon)

    train_mask = features.index < pd.Timestamp(test_start)
    valid_train_mask = train_mask & features.notna().all(axis=1) & y_fwd.notna()
    if int(valid_train_mask.sum()) < max(min_neighbors, 20):
        sig = pd.DataFrame(index=df.index)
        sig["signal"] = SIGNAL_HOLD
        sig["position"] = 0
        sig["enter_long"] = 0
        sig["exit_long"] = 0
        sig["enter_short"] = 0
        sig["exit_short"] = 0
        sig["entry_ok"] = 0
        sig["exit_ok"] = 0
        sig["entry_votes"] = 0
        sig["exit_votes"] = 0
        sig["edge_score"] = 0.0
        sig["edge_mean"] = 0.0
        sig["edge_std"] = 0.0
        sig["edge_prob_pos"] = 0.0
        sig["edge_downside_q05"] = 0.0
        sig["n_eff"] = 0
        sig["dist_rank"] = 1.0
        sig["entry_vote_components"] = "retrieval:no_train"
        sig["exit_vote_components"] = "retrieval:no_train"
        return RetrievalArtifacts(signal_df=sig, features=features)

    scaler = _make_scaler()
    x_train = scaler.fit_transform(features.loc[valid_train_mask].to_numpy())
    y_train = y_fwd.loc[valid_train_mask].to_numpy()

    dist_ref, _ = _faiss_knn(x_train, x_train, min(k + 1, len(x_train)))
    if dist_ref.shape[1] > 1:
        ref = np.sqrt(dist_ref[:, 1:]).reshape(-1)
    else:
        ref = np.sqrt(dist_ref.reshape(-1))
    ref = ref[np.isfinite(ref)]
    max_dist = float(np.quantile(ref, min(max(max_dist_percentile, 0.05), 0.99))) if len(ref) else float("inf")

    test_mask = features.index >= pd.Timestamp(test_start)
    valid_test_mask = test_mask & features.notna().all(axis=1)

    q_idx = features.index[valid_test_mask]
    x_query = scaler.transform(features.loc[valid_test_mask].to_numpy()) if len(q_idx) else np.empty((0, x_train.shape[1]))
    q_d, q_i = _faiss_knn(x_train, x_query, min(k, len(x_train))) if len(q_idx) else (np.empty((0, 0)), np.empty((0, 0), dtype=int))

    sig = pd.DataFrame(index=df.index)
    sig["signal"] = SIGNAL_HOLD
    sig["position"] = 0
    sig["enter_long"] = 0
    sig["exit_long"] = 0
    sig["enter_short"] = 0
    sig["exit_short"] = 0
    sig["entry_ok"] = 0
    sig["exit_ok"] = 0
    sig["entry_votes"] = 0
    sig["exit_votes"] = 0
    sig["edge_score"] = 0.0
    sig["edge_mean"] = 0.0
    sig["edge_std"] = 0.0
    sig["edge_prob_pos"] = 0.0
    sig["edge_downside_q05"] = 0.0
    sig["n_eff"] = 0
    sig["dist_rank"] = 1.0
    sig["entry_vote_components"] = ""
    sig["exit_vote_components"] = ""

    query_map = {ts: j for j, ts in enumerate(q_idx)}
    in_pos = 0
    bars_in_pos = 0

    for i, ts in enumerate(sig.index):
        if ts not in query_map:
            sig.at[ts, "position"] = in_pos
            continue

        j = query_map[ts]
        nb_i = q_i[j]
        nb_d = np.sqrt(q_d[j]) if len(q_d) else np.array([])

        cur_loc = sig.index.get_loc(ts)
        cutoff_loc = max(0, cur_loc - horizon - embargo_bars)
        cutoff_ts = sig.index[cutoff_loc]
        train_idx = features.index[valid_train_mask]
        valid_neighbor_mask = train_idx[nb_i] <= cutoff_ts

        nb_i = nb_i[valid_neighbor_mask]
        nb_d = nb_d[valid_neighbor_mask]
        nb_y = y_train[nb_i] if len(nb_i) else np.array([], dtype=float)

        n_eff = int(len(nb_y))
        mean_y = float(np.mean(nb_y)) if n_eff else 0.0
        std_y = float(np.std(nb_y)) if n_eff else 0.0
        prob_pos = float(np.mean(nb_y > 0)) if n_eff else 0.0
        downside_q = float(np.quantile(nb_y, 0.05)) if n_eff else -1.0
        edge_score = float(mean_y / (std_y + 1e-8) * np.sqrt(n_eff)) if n_eff else 0.0
        dist_rank = float(np.nanmean(nb_d) / (max_dist + 1e-12)) if n_eff else 1.0

        entry_ok = bool(
            n_eff >= min_neighbors
            and mean_y > entry_mean_threshold
            and edge_score > entry_score_threshold
            and downside_q > -abs(risk_limit)
            and (not np.isfinite(max_dist) or float(np.nanmean(nb_d)) <= max_dist)
        )

        exit_ok = bool((in_pos == 1) and (bars_in_pos >= hold_bars or edge_score < exit_score_threshold))

        if in_pos == 0 and entry_ok:
            in_pos = 1
            bars_in_pos = 0
            sig.at[ts, "signal"] = SIGNAL_ENTER_LONG
            sig.at[ts, "enter_long"] = 1
            sig.at[ts, "entry_ok"] = 1
        elif in_pos == 1 and exit_ok:
            in_pos = 0
            bars_in_pos = 0
            sig.at[ts, "signal"] = SIGNAL_EXIT_LONG
            sig.at[ts, "exit_long"] = 1
            sig.at[ts, "exit_ok"] = 1

        if in_pos == 1:
            bars_in_pos += 1

        sig.at[ts, "position"] = in_pos
        sig.at[ts, "entry_votes"] = int(entry_ok)
        sig.at[ts, "exit_votes"] = int(exit_ok)
        sig.at[ts, "edge_score"] = edge_score
        sig.at[ts, "edge_mean"] = mean_y
        sig.at[ts, "edge_std"] = std_y
        sig.at[ts, "edge_prob_pos"] = prob_pos
        sig.at[ts, "edge_downside_q05"] = downside_q
        sig.at[ts, "n_eff"] = n_eff
        sig.at[ts, "dist_rank"] = dist_rank
        sig.at[ts, "entry_vote_components"] = (
            f"n_eff={n_eff},mean={mean_y:.5f},score={edge_score:.3f},q05={downside_q:.5f},dist={float(np.nanmean(nb_d)) if n_eff else 0.0:.5f}"
        )
        sig.at[ts, "exit_vote_components"] = f"hold={bars_in_pos},score={edge_score:.3f}"

    return RetrievalArtifacts(signal_df=sig, features=features)
