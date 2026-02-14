from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

from src.reporter.diagnostics import format_diagnostic_summary, save_diagnostic_artifacts


def normalize_equity_df(equity_data: pd.Series | pd.DataFrame, start_cash: float = 10000.0) -> pd.DataFrame:
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
            return pd.DataFrame({"equity": [float(start_cash)]}, index=[pd.Timestamp.utcnow()])

    equity_df = equity_df.copy()
    equity_df.index = pd.to_datetime(equity_df.index, errors="coerce")
    equity_df = equity_df[~equity_df.index.isna()].sort_index()
    if equity_df.empty:
        return pd.DataFrame({"equity": [float(start_cash)]}, index=[pd.Timestamp.utcnow()])

    equity_df["equity"] = pd.to_numeric(equity_df["equity"], errors="coerce").ffill().bfill().fillna(float(start_cash))
    return equity_df


def build_trades_plot(price_df: pd.DataFrame, trades: list[dict], out_path: str | Path, title: str) -> None:
    src = price_df.copy()
    if "Close" not in src.columns:
        raise ValueError(f"Cannot build trades plot: missing Close column. columns={list(src.columns)}")
    src.index = pd.to_datetime(src.index, errors="coerce")
    src = src[~src.index.isna()].sort_index()
    if src.empty:
        raise ValueError("Cannot build trades plot: empty price frame after index normalization")

    fig, ax = plt.subplots(figsize=(11, 5))
    src["Close"].plot(ax=ax, color="steelblue", linewidth=1.2, title=title)

    if trades:
        entry_x = pd.to_datetime([t.get("entry_date") for t in trades], errors="coerce")
        exit_x = pd.to_datetime([t.get("exit_date") for t in trades], errors="coerce")
        entry_y = [float(t.get("entry_price", 0.0) or 0.0) for t in trades]
        exit_y = [float(t.get("exit_price", 0.0) or 0.0) for t in trades]

        ax.scatter(entry_x, entry_y, color="green", marker="^", s=40, label="entry", zorder=3)
        ax.scatter(exit_x, exit_y, color="red", marker="v", s=40, label="exit", zorder=3)

        for x1, y1, x2, y2 in zip(entry_x, entry_y, exit_x, exit_y):
            if pd.isna(x1) or pd.isna(x2):
                continue
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.7, alpha=0.7)
        ax.legend(loc="best")

    ax.set_ylabel("Price ($)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_exposure_plot(bt: pd.DataFrame, out_path: str | Path, title: str) -> None:
    if "qty" not in bt.columns:
        return
    exp = (pd.to_numeric(bt["qty"], errors="coerce").fillna(0.0) != 0).astype(int)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    exp.plot(ax=ax, color="purple", linewidth=1.0, title=title)
    ax.set_ylabel("Exposure")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_equity_comparison_plot(bt: pd.DataFrame, out_path: str | Path, title: str) -> None:
    if "equity" not in bt.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.to_numeric(bt["equity"], errors="coerce").ffill().bfill().plot(ax=ax, label="strategy")
    if "buyhold_equity" in bt.columns:
        pd.to_numeric(bt["buyhold_equity"], errors="coerce").ffill().bfill().plot(ax=ax, label="buy&hold", linestyle="--")
    if "ma200_equity" in bt.columns:
        pd.to_numeric(bt["ma200_equity"], errors="coerce").ffill().bfill().plot(ax=ax, label="MA200", linestyle=":")
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_edge_plot(bt: pd.DataFrame, out_path: str | Path, title: str) -> None:
    if "edge_score" not in bt.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 2.5))
    pd.to_numeric(bt["edge_score"], errors="coerce").fillna(0.0).plot(ax=ax, color="teal", linewidth=1.0, title=title)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("edge_score")
    fig.tight_layout()
    fig.savefig(out_path)
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
    exposure_plot_path = run_dir / "exposure.png"
    comparison_plot_path = run_dir / "equity_comparison.png"
    edge_plot_path = run_dir / "edge_score.png"

    start_cash = float((best_metrics or {}).get("start_cash", 10000.0))
    normalized = normalize_equity_df(equity_df, start_cash=start_cash)

    fig, ax = plt.subplots(figsize=(10, 4))
    normalized["equity"].plot(ax=ax, title=f"Equity curve job={job_id}")
    ax.set_ylabel("Equity ($)")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    fig.savefig(equity_path)
    plt.close(fig)

    try:
        build_trades_plot(normalized, trades, trades_plot_path, title=f"Trades chart job={job_id}")
    except Exception:
        pass

    try:
        build_exposure_plot(normalized, exposure_plot_path, title=f"Exposure job={job_id}")
    except Exception:
        pass

    try:
        build_equity_comparison_plot(normalized, comparison_plot_path, title=f"Equity comparison job={job_id}")
    except Exception:
        pass

    try:
        build_edge_plot(normalized, edge_plot_path, title=f"Edge score job={job_id}")
    except Exception:
        pass

    trades_path.write_text(json.dumps(trades or [], indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")

    metrics = best_metrics or {}
    summary_path.write_text(
        f"Job {job_id}\n"
        f"Bars: {len(normalized)}\n"
        f"Trades: {len(trades)}\n"
        f"Final equity: {metrics.get('final_equity', '-') }\n"
        f"Profit $: {metrics.get('profit_$', '-')}\n"
        f"Profit %: {metrics.get('profit_%', '-')}\n"
        f"Max DD %: {metrics.get('max_dd_%', '-')}\n"
        f"Buy&Hold equity: {metrics.get('buyhold_final_equity', '-')}\n"
        f"MA200 equity: {metrics.get('ma200_final_equity', '-')}\n"
        f"Updated: {pd.Timestamp.utcnow().isoformat()}\n",
        encoding="utf-8",
    )


    diag_summary, diag_artifacts = save_diagnostic_artifacts(
        run_dir=run_dir,
        bt=normalized if "Close" in normalized.columns else pd.concat([normalized, normalized.rename(columns={"equity": "Close"})], axis=1),
        trades=trades or [],
        initial_cash=float((best_metrics or {}).get("start_cash", 10000.0)),
        allow_short=bool(best_config.get("allow_short", True)),
        metrics=best_metrics or {},
    )
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + format_diagnostic_summary(diag_summary) + "\n")
        fh.write(f"diag_trades_csv: {diag_artifacts.trades_csv}\n")
        fh.write(f"diag_equity_csv: {diag_artifacts.equity_csv}\n")
        fh.write(f"diag_summary_json: {diag_artifacts.summary_json}\n")

    return str(equity_path), str(trades_path), str(config_path), str(summary_path)
