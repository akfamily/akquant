"""get_cash() 方法 → cash 属性（硬改）."""

from akquant.strategy import Strategy


def test_cash_is_property_and_get_cash_removed() -> None:
    """get_cash() 方法应已移除，cash 应为只读属性."""
    assert not hasattr(Strategy, "get_cash")
    assert isinstance(getattr(Strategy, "cash"), property)
