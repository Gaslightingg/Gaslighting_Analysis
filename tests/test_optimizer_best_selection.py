from src.optimizer.optuna_runner import _is_better_score, _select_leaders


def test_best_by_score_and_equity_with_maximize() -> None:
    rows = [
        {"trial": 1, "final_equity": 10500.0, "base_score": 0.10, "diag_score": 0.20, "PF": 1.2, "maxDD": -0.08, "n_trades": 40, "flags": "ok", "total_return": 0.05},
        {"trial": 2, "final_equity": 11000.0, "base_score": 0.30, "diag_score": 0.15, "PF": 1.1, "maxDD": -0.09, "n_trades": 45, "flags": "ok", "total_return": 0.10},
        {"trial": 3, "final_equity": 10800.0, "base_score": 0.25, "diag_score": 0.35, "PF": 1.3, "maxDD": -0.07, "n_trades": 38, "flags": "ok", "total_return": 0.08},
    ]
    leaders = _select_leaders(rows, maximize=True)
    assert leaders["best_by_equity"]["trial"] == 2
    assert leaders["best_by_base_score"]["trial"] == 2
    assert leaders["best_by_diag_score"]["trial"] == 3


def test_best_by_score_and_equity_with_minimize() -> None:
    rows = [
        {"trial": 10, "final_equity": 12000.0, "base_score": -0.40, "diag_score": -0.10, "PF": 1.5, "maxDD": -0.06, "n_trades": 55, "flags": "ok", "total_return": 0.20},
        {"trial": 11, "final_equity": 11800.0, "base_score": -0.60, "diag_score": -0.50, "PF": 1.4, "maxDD": -0.05, "n_trades": 52, "flags": "ok", "total_return": 0.18},
    ]
    leaders = _select_leaders(rows, maximize=False)
    assert leaders["best_by_equity"]["trial"] == 10
    assert leaders["best_by_base_score"]["trial"] == 11
    assert leaders["best_by_diag_score"]["trial"] == 11


def test_is_better_score_respects_direction() -> None:
    assert _is_better_score(2.0, 1.0, True) is True
    assert _is_better_score(0.5, 1.0, True) is False
    assert _is_better_score(0.5, 1.0, False) is True
    assert _is_better_score(2.0, 1.0, False) is False
