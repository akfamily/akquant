"""place_bracket_order/create_oco_order_group 改名 + register_indicator 别名去除."""

from akquant.strategy import Strategy


def test_composite_renames() -> None:
    """旧名应已移除，新名应存在."""
    assert not hasattr(Strategy, "place_bracket_order")
    assert not hasattr(Strategy, "create_oco_order_group")
    assert not hasattr(Strategy, "register_indicator")
    assert hasattr(Strategy, "place_bracket")
    assert hasattr(Strategy, "place_oco")
    assert hasattr(Strategy, "register_precomputed_indicator")
