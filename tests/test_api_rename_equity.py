"""get_portfolio_value() → equity 属性（硬改）."""

from akquant.strategy import Strategy


def test_equity_property_and_method_removed() -> None:
    """get_portfolio_value() 方法应已移除，equity 应为只读属性."""
    assert not hasattr(Strategy, "get_portfolio_value")
    assert isinstance(getattr(Strategy, "equity"), property)
