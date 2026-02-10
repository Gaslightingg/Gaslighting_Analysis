from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_report(
    job_id: int, bt_df: pd.DataFrame, best_config: dict, runs_dir: str
) -> tuple[str, str, str]:
    run_dir = Path(runs_dir) / str(job_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    equity_path = run_dir / "equity_best.png"
    trades_path = run_dir / "trades_best.json"
    config_path = run_dir / "best_config.json"

    plt.figure(figsize=(10, 4))
    bt_df["equity"].plot(title=f"Equity curve job={job_id}")
    plt.tight_layout()
    plt.savefig(equity_path)
    plt.close()

    turns = bt_df[(bt_df["position"] != bt_df["position"].shift(1).fillna(0))][
        ["Open", "Close", "position"]
    ]
    trades_path.write_text(turns.to_json(orient="table"), encoding="utf-8")
    config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    return str(equity_path), str(trades_path), str(config_path)
