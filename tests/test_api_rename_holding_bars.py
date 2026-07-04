"""hold_bar → get_holding_bars（硬改）."""

from akquant.strategy import Strategy


def test_holding_bars_renamed() -> None:
    """hold_bar 方法应已移除，get_holding_bars 应存在."""
    assert not hasattr(Strategy, "hold_bar")
    assert hasattr(Strategy, "get_holding_bars")
