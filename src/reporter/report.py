from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_best_artifacts(
    job_id: str,
    equity_df: pd.DataFrame,
    trades: list[dict],
    best_config: dict,
    runs_dir: str,
) -> tuple[str, str, str, str]:
    run_dir = Path(runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    equity_path = run_dir / "equity_best.png"
    trades_path = run_dir / "trades_best.json"
    config_path = run_dir / "best_config.json"
    summary_path = run_dir / "summary.txt"

    fig, ax = plt.subplots(figsize=(10, 4))
    equity_df["equity"].plot(ax=ax, title=f"Equity curve job={job_id}")
    fig.tight_layout()
    fig.savefig(equity_path)
    plt.close(fig)

    trades_path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    summary_path.write_text(
        f"Job {job_id}\n"
        f"Bars: {len(equity_df)}\n"
        f"Trades: {len(trades)}\n"
        f"Updated: {pd.Timestamp.utcnow().isoformat()}\n",
        encoding="utf-8",
    )

    return str(equity_path), str(trades_path), str(config_path), str(summary_path)
