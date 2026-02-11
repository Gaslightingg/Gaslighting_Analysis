from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd


def normalize_equity_df(equity_data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(equity_data, pd.Series):
        equity_df = equity_data.to_frame(name="equity")
    elif isinstance(equity_data, pd.DataFrame):
        equity_df = equity_data.copy()
    else:
        raise ValueError(f"Unsupported equity data type: {type(equity_data).__name__}")

    if "equity" not in equity_df.columns:
        numeric_cols = [c for c in equity_df.columns if pd.api.types.is_numeric_dtype(equity_df[c])]
        if len(numeric_cols) == 1:
            equity_df = equity_df.rename(columns={numeric_cols[0]: "equity"})
        else:
            raise ValueError(
                "Cannot normalize equity data: missing 'equity' column. "
                f"type={type(equity_data).__name__}, columns={list(equity_df.columns)}"
            )

    equity_df = equity_df.copy()
    equity_df.index = pd.to_datetime(equity_df.index, errors="coerce")
    equity_df = equity_df[~equity_df.index.isna()].sort_index()
    if equity_df.empty:
        raise ValueError("Cannot normalize equity data: empty index after datetime conversion")

    equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce").ffill().bfill().fillna(1.0)
    return equity_df


def _save_trades_plot(trades_plot_path: Path, equity_df: pd.DataFrame, trades: list[dict], job_id: str) -> None:
    if "Close" not in equity_df.columns:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    equity_df["Close"].plot(ax=ax, color="steelblue", linewidth=1.2, title=f"Trades chart job={job_id}")

    if trades:
        entry_x = pd.to_datetime([t.get("entry_date") for t in trades], errors="coerce")
        exit_x = pd.to_datetime([t.get("exit_date") for t in trades], errors="coerce")
        entry_y = [float(t.get("entry_price", 0.0) or 0.0) for t in trades]
        exit_y = [float(t.get("exit_price", 0.0) or 0.0) for t in trades]

        ax.scatter(entry_x, entry_y, color="green", marker="^", s=40, label="entry", zorder=3)
        ax.scatter(exit_x, exit_y, color="red", marker="v", s=40, label="exit", zorder=3)
        ax.legend(loc="best")

    ax.set_ylabel("Price ($)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(trades_plot_path)
    plt.close(fig)


def save_best_artifacts(
    job_id: str,
    equity_df: pd.DataFrame | pd.Series,
    trades: list[dict],
    best_config: dict,
    runs_dir: str,
    best_metrics: dict | None = None,
) -> tuple[str, str, str, str]:
    run_dir = Path(runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    equity_path = run_dir / "equity_best.png"
    trades_path = run_dir / "trades_best.json"
    config_path = run_dir / "best_config.json"
    summary_path = run_dir / "summary.txt"
    trades_plot_path = run_dir / "trades.png"

    normalized = normalize_equity_df(equity_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    normalized["equity"].plot(ax=ax, title=f"Equity curve job={job_id}")
    ax.set_ylabel("Equity ($)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(equity_path)
    plt.close(fig)

    _save_trades_plot(trades_plot_path, normalized, trades, job_id)

    trades_path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    metrics = best_metrics or {}
    summary_path.write_text(
        f"Job {job_id}\n"
        f"Bars: {len(normalized)}\n"
        f"Trades: {len(trades)}\n"
        f"Final equity: {metrics.get('final_equity', '-')}\n"
        f"Profit $: {metrics.get('profit_$', '-')}\n"
        f"Profit %: {metrics.get('profit_%', '-')}\n"
        f"Max DD %: {metrics.get('max_dd_%', '-')}\n"
        f"Updated: {pd.Timestamp.utcnow().isoformat()}\n",
        encoding="utf-8",
    )

    return str(equity_path), str(trades_path), str(config_path), str(summary_path)
