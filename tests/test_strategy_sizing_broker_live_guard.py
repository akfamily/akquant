"""组合目标类下单(buy_all/order_target*) 在 broker_live 下清晰报错."""

import pytest
from akquant import strategy_trading_api as api


class _Strategy:
    """Strategy stub reporting broker_live capabilities."""

    def __init__(self, broker_live: bool = True) -> None:
        """Inject execution capabilities reporting the given broker_live flag."""
        self.ctx = object()
        self.__dict__["get_execution_capabilities"] = lambda: {
            "broker_live": broker_live
        }


def test_buy_all_rejected_in_broker_live() -> None:
    """buy_all raises clearly in broker_live (sizes off sim otherwise)."""
    with pytest.raises(RuntimeError, match="broker_live|柜台|不支持"):
        api.buy_all(_Strategy(), symbol="600000.SH")


def test_order_target_rejected_in_broker_live() -> None:
    """order_target raises clearly in broker_live."""
    with pytest.raises(RuntimeError, match="broker_live|柜台|不支持"):
        api.order_target(_Strategy(), symbol="600000.SH", target=100)


def test_order_target_value_rejected_in_broker_live() -> None:
    """order_target_value raises clearly in broker_live."""
    with pytest.raises(RuntimeError, match="broker_live|柜台|不支持"):
        api.order_target_value(_Strategy(), symbol="600000.SH", target_value=1000)


def test_order_target_percent_rejected_in_broker_live() -> None:
    """order_target_percent raises clearly in broker_live."""
    with pytest.raises(RuntimeError, match="broker_live|柜台|不支持"):
        api.order_target_percent(_Strategy(), symbol="600000.SH", target_percent=0.5)


def test_order_target_weights_rejected_in_broker_live() -> None:
    """order_target_weights raises clearly in broker_live."""
    with pytest.raises(RuntimeError, match="broker_live|柜台|不支持"):
        api.order_target_weights(_Strategy(), target_weights={"600000.SH": 0.5})


def test_buy_all_not_rejected_when_broker_live_false() -> None:
    """buy_all does not raise the broker_live guard when capability is False."""
    strategy = _Strategy(broker_live=False)
    strategy.current_bar = None
    strategy.current_tick = None
    # Should proceed past the guard; price resolves to 0 -> no-op, no exception.
    api.buy_all(strategy, symbol="600000.SH")


def test_order_target_not_rejected_when_broker_live_absent() -> None:
    """order_target does not raise the broker_live guard for a plain strategy."""

    class _PlainCtx:
        def get_position(self, symbol: str) -> float:
            return 0.0

    class _PlainStrategy:
        def __init__(self) -> None:
            self.ctx = _PlainCtx()
            self.submit_order_calls: list = []

        def submit_order(self, **kwargs):
            self.submit_order_calls.append(kwargs)
            return "order-1"

    strategy = _PlainStrategy()
    order_id = api.order_target(strategy, symbol="600000.SH", target=100)
    assert order_id == "order-1"
