"""stop_buy/stop_sell/buy_all 已硬删（防回潮）."""

from akquant.strategy import Strategy


def test_thin_wrappers_removed() -> None:
    """stop_buy/stop_sell/buy_all 三个薄封装应已从 Strategy 上硬删，无回潮."""
    for name in ("stop_buy", "stop_sell", "buy_all"):
        assert not hasattr(Strategy, name), f"{name} 应已删除"
