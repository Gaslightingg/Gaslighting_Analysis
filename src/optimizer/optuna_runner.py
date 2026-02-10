from __future__ import annotations

import json
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from src.config import SETTINGS
from src.optimizer.walk_forward import WalkForwardConfig, evaluate_config_walk_forward
from src.reporter.report import save_best_artifacts
from src.storage.repository import Repository


def run_optimization_job(job_id: str, df) -> dict:
    repo = Repository()
    job_data = repo.get_job_data(job_id)
    if not job_data:
        return {"status": "failed", "error": "job not found"}

    params = job_data["params"]
    trials_total = int(params["trials_total"])
    checkpoint_n = int(params["checkpoint_n"])

    repo.set_job_status(job_id, "running")

    best_score = float("-inf")
    best_metrics: dict = {}
    best_config: dict = {}
    last_score = 0.0
    trials_done = 0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SETTINGS.seed))
    wf_cfg = WalkForwardConfig(train_months=12, test_months=3, step_months=3)

    for i in range(1, trials_total + 1):
        trials_done = i
        if repo.stop_requested(job_id):
            repo.update_progress(
                job_id=job_id,
                trials_done=trials_done,
                trials_total=trials_total,
                last_score=last_score,
                best_score=best_score if best_score != float("-inf") else None,
                state="stopped",
            )
            return {"status": "stopped", "best_score": best_score}

        trial = study.ask()
        cfg = _sample(trial)
        score, metrics, eq_df, trades = evaluate_config_walk_forward(
            df,
            cfg,
            SETTINGS.commission_bps,
            SETTINGS.slippage_bps,
            wf_cfg,
        )

        last_score = score
        study.tell(trial, score)

        if score > best_score and metrics:
            best_score = score
            best_metrics = metrics
            best_config = cfg
            equity_path, trades_path, config_path, _summary_path = save_best_artifacts(
                job_id=job_id,
                equity_df=eq_df,
                trades=trades,
                best_config=best_config,
                runs_dir=SETTINGS.runs_dir,
            )
            repo.save_best(job_id, best_config, best_metrics, equity_path, trades_path, config_path)

        repo.update_progress(
            job_id=job_id,
            trials_done=i,
            trials_total=trials_total,
            last_score=score,
            best_score=best_score if best_score != float("-inf") else None,
            state="running",
        )

        if i % checkpoint_n == 0:
            repo.add_checkpoint(job_id, checkpoint_no=i // checkpoint_n, trials_done=i)

    repo.update_progress(
        job_id=job_id,
        trials_done=trials_done,
        trials_total=trials_total,
        last_score=last_score,
        best_score=best_score if best_score != float("-inf") else None,
        state="finished",
    )
    repo.set_job_status(job_id, "finished")

    run_dir = Path(SETTINGS.runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "summary.txt", "a", encoding="utf-8") as f:
        f.write("Optimization finished\n")
        f.write(json.dumps(best_metrics, indent=2))

    return {"status": "finished", "best_score": best_score, "metrics": best_metrics}


def _sample(trial: optuna.trial.Trial) -> dict:
    ema_fast = trial.suggest_int("ema_fast", 5, 50)
    ema_slow = trial.suggest_int("ema_slow", 20, 200)
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 1
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": trial.suggest_int("rsi_period", 5, 30),
        "buy_below": trial.suggest_int("buy_below", 10, 40),
        "sell_above": trial.suggest_int("sell_above", 60, 90),
        "bb_period": trial.suggest_int("bb_period", 10, 40),
        "bb_std": trial.suggest_float("bb_std", 1.5, 3.5),
        "adx_min": trial.suggest_int("adx_min", 10, 30),
        "enter_long": trial.suggest_int("enter_long", 1, 3),
        "exit_long": trial.suggest_int("exit_long", -1, 1),
    }
