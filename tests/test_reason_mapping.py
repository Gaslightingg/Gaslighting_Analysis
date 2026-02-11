from src.worker.tasks import _guard_note_reason


def test_guard_reason_bars_positive_is_not_no_data() -> None:
    note, reason = _guard_note_reason(bars=251, min_bars=252, auto_extended=True)
    assert note == "not_enough_bars"
    assert "bars=251" in reason


def test_guard_reason_zero_bars_is_no_data() -> None:
    note, reason = _guard_note_reason(bars=0, min_bars=252, auto_extended=False)
    assert note == "no_data"
    assert "bars=0" in reason
