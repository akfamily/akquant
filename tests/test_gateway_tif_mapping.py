"""_to_tif：字符串映射真实 TimeInForce 枚举（回归 GTD 误引用不存在成员的 bug）."""

from akquant.akquant import TimeInForce
from akquant.gateway import broker_event_adapter as mod


def test_day_and_gtd_map_to_real_day_member() -> None:
    """GTD(good-till-date) 无对应精确枚举成员，规范化为 Day；DAY 本就是 Day."""
    assert mod._to_tif("DAY") is TimeInForce.Day
    assert mod._to_tif("GTD") is TimeInForce.Day


def test_gtc_maps_to_gtc() -> None:
    """GTC 正常映射，不受 GTD 修正影响."""
    assert mod._to_tif("GTC") is TimeInForce.GTC


def test_none_input_returns_none() -> None:
    """None 输入直接返回 None，不查表."""
    assert mod._to_tif(None) is None


def test_unknown_value_returns_none() -> None:
    """未知字符串返回 None（dict.get 无默认值）."""
    assert mod._to_tif("nonsense") is None
