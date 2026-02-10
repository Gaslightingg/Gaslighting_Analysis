from src.strategy.hybrid_vote import generate_positions


def test_strategy_callable() -> None:
    assert callable(generate_positions)
