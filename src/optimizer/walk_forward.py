from __future__ import annotations

from dataclasses import dataclass

import optuna
import pandas as pd
from sqlalchemy import select

from src.backtester.engine import run_backtest
from src.indicators.ta import add_indicators
from src.storage.db import SessionLocal
from src.storage.models import Job, Optimization, OptimizationCheckpoint
from src.strategy.hybrid_vote import generate_hybrid_signals


@dataclass(slots=True)
class ModeConfig:
    trials_total: int
    checkpoint_n: int


MODE_PRESETS = {
    "fast": ModeConfig(trials_total=20, checkpoint_n=5),
    "standard": ModeConfig(trials_total=60, checkpoint_n=10),
    "deep": ModeConfig(trials_total=120, checkpoint_n=20),
}


def sample_params(trial: optuna.trial.Trial) -> dict:
    return {
        "ema_fast": trial.suggest_int("ema_fast", 5, 20),
        "ema_slow": trial.suggest_int("ema_slow", 21, 60),
        "rsi_period": trial.suggest_int("rsi_period", 8, 21),
        "rsi_buy_below": trial.suggest_int("rsi_buy_below", 20, 40),
        "rsi_sell_above": trial.suggest_int("rsi_sell_above", 60, 85),
        "adx_min": trial.suggest_int("adx_min", 10, 35),
        "enter_long": trial.suggest_int("enter_long", 1, 3),
        "exit_long": trial.suggest_int("exit_long", -1, 1),
        "bb_period": trial.suggest_int("bb_period", 10, 30),
        "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
    }


def walk_forward_score(df: pd.DataFrame, config: dict) -> tuple[float, dict]:
    splits = 3
    fold_size = len(df) // (splits + 1)
    scores: list[float] = []
    last_metrics: dict = {}
    for i in range(splits):
        train_end = (i + 1) * fold_size
        test_end = (i + 2) * fold_size
        test_df = df.iloc[train_end:test_end].copy()
        if len(test_df) < 30:
            continue
        with_ind = add_indicators(test_df, config)
        pos = generate_hybrid_signals(with_ind, config)
        _, metrics = run_backtest(with_ind, pos)
        scores.append(metrics["score"])
        last_metrics = metrics
    score = float(sum(scores) / len(scores)) if scores else -999.0
    return score, last_metrics


def should_stop(job_id: int) -> bool:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        return bool(job and job.stop_requested)


def save_best(
    job_id: int, trials_done: int, best_cfg: dict, best_metrics: dict, checkpoint_n: int
) -> None:
    with SessionLocal() as session:
        opt = session.scalar(select(Optimization).where(Optimization.job_id == job_id))
        if opt:
            opt.best_config_json = best_cfg
            opt.best_metrics_json = best_metrics
        checkpoint = OptimizationCheckpoint(
            job_id=job_id,
            checkpoint_no=max(1, trials_done // checkpoint_n),
            trials_done=trials_done,
            best_config_json=best_cfg,
            best_metrics_json=best_metrics,
        )
        job = session.get(Job, job_id)
        if job:
            job.progress_json = {
                "trials_done": trials_done,
                "best_score": best_metrics.get("score", None),
            }
        session.add(checkpoint)
        session.commit()


def run_optimization(job_id: int, df: pd.DataFrame, mode: str) -> tuple[dict, dict, int]:
    mode_cfg = MODE_PRESETS[mode]
    study = optuna.create_study(direction="maximize")
    best_cfg: dict = {}
    best_metrics: dict = {}

    for i in range(1, mode_cfg.trials_total + 1):
        if should_stop(job_id):
            break
        trial = study.ask()
        params = sample_params(trial)
        score, metrics = walk_forward_score(df, params)
        study.tell(trial, score)

        if study.best_trial.number == trial.number:
            best_cfg = params
            best_metrics = {**metrics, "score": score}

        if i % mode_cfg.checkpoint_n == 0 and best_cfg:
            save_best(job_id, i, best_cfg, best_metrics, mode_cfg.checkpoint_n)

    if best_cfg:
        save_best(job_id, mode_cfg.trials_total, best_cfg, best_metrics, mode_cfg.checkpoint_n)
    return best_cfg, best_metrics, mode_cfg.trials_total
