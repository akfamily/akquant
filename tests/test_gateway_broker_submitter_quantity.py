"""broker_live 下 symbol/quantity 缺省解析与卖空拦截(对齐回测口径)."""

from typing import Any

import pytest
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest
from akquant.gateway.order_submitter import BrokerOrderSubmitter
from akquant.sizer import FixedSize, PercentSizer


class _Execution:
    """最小执行后端桩:只提供 submitter 会读的资金/持仓口径."""

    def __init__(self, cash: float = 0.0, available: float = 0.0) -> None:
        self._cash = cash
        self._available = available

    def get_cash(self) -> float:
        return self._cash

    def get_available_position(self, symbol: str | None = None) -> float:
        _ = symbol
        return self._available


class _Bar:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol


class _Strategy:
    """带 sizer / execution / 最新价的策略桩."""

    _owner_strategy_id = "_default"
    broker_ready = True
    ctx = None

    def __init__(
        self,
        sizer: Any = None,
        execution: Any = None,
        last_prices: dict[str, float] | None = None,
        current_bar: Any = None,
    ) -> None:
        self.sizer = sizer
        self.execution = execution
        self._last_prices = last_prices or {}
        self.current_bar = current_bar
        self.current_tick = None


def _submitter(
    strategy: _Strategy,
    captured: list[UnifiedOrderRequest],
    capability: BrokerCapability | None = None,
) -> BrokerOrderSubmitter:
    """Build a submitter recording every placed request."""

    class _Gw:
        def place_order(self, req: UnifiedOrderRequest) -> str:
            captured.append(req)
            return f"b{len(captured)}"

    cap = capability or BrokerCapability(broker_name="qmf")
    return BrokerOrderSubmitter(
        trader_gateway=_Gw(),
        strategy=strategy,
        resolve_trader_capabilities=lambda _gw: cap,
        next_client_order_id=lambda: "c1",
        can_submit_client_order=lambda _cid: True,
        sync_order_id_mapping=lambda _c, _b: None,
        bind_order_owner=lambda _c, _b, _o: None,
        notify_strategy_error=lambda *_a, **_k: None,
        payload_field=lambda obj, name: getattr(obj, name, None),
        get_execution_capabilities=lambda: cap.as_execution_capabilities(),
        record_order_request=lambda *_a: None,
    )


def test_buy_without_quantity_uses_sizer() -> None:
    """买入不传 quantity 时走 strategy.sizer(与回测同口径)."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(sizer=FixedSize(300), execution=_Execution(cash=100_000.0))
    _submitter(strategy, captured).submit_order(
        symbol="600000.SH", side="Buy", price=10.0
    )
    assert [r.quantity for r in captured] == [300.0]


def test_buy_without_quantity_and_price_uses_last_price() -> None:
    """未传 price 时参考价取 _last_prices(PercentSizer 才能算出量)."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(
        sizer=PercentSizer(10.0),
        execution=_Execution(cash=100_000.0),
        last_prices={"600000.SH": 10.0},
    )
    _submitter(strategy, captured).submit_order(symbol="600000.SH", side="Buy")
    # 100000 * 10% / 10.0 = 1000
    assert [r.quantity for r in captured] == [1000.0]


def test_sell_without_quantity_closes_available_position() -> None:
    """卖出不传 quantity 时全平**可用**持仓(T+1 冻结部分不报单)."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(execution=_Execution(available=400.0))
    _submitter(strategy, captured).submit_order(
        symbol="600000.SH", side="Sell", price=10.0
    )
    assert [r.quantity for r in captured] == [400.0]


def test_sell_without_position_places_nothing() -> None:
    """无可用持仓时卖出返回空回执,不向柜台报单."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(execution=_Execution(available=0.0))
    receipt = _submitter(strategy, captured).submit_order(
        symbol="600000.SH", side="Sell"
    )
    assert captured == []
    assert len(receipt) == 0
    assert receipt.primary == ""


def test_zero_quantity_places_nothing() -> None:
    """quantity<=0 不报单(此前会把 0 手单发给柜台)."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(execution=_Execution())
    receipt = _submitter(strategy, captured).submit_order(
        symbol="600000.SH", side="Buy", quantity=0
    )
    assert captured == []
    assert len(receipt) == 0


def test_symbol_defaults_to_current_bar() -> None:
    """缺省 symbol 取当前 bar(与回测 resolve_symbol 同口径)."""
    captured: list[UnifiedOrderRequest] = []
    strategy = _Strategy(
        sizer=FixedSize(100),
        execution=_Execution(cash=10_000.0),
        current_bar=_Bar("000001.SZ"),
    )
    _submitter(strategy, captured).submit_order(side="Buy", price=10.0)
    assert [r.symbol for r in captured] == ["000001.SZ"]


def test_sizer_returning_none_reports_clearly() -> None:
    """自定义 sizer 漏 return 时报错点明原因,而非 NoneType 比较崩溃."""

    class _BadSizer:
        def get_size(self, *_a: Any, **_k: Any) -> Any:
            return None

    strategy = _Strategy(sizer=_BadSizer(), execution=_Execution(cash=1.0))
    with pytest.raises(RuntimeError, match="get_size"):
        _submitter(strategy, []).submit_order(
            symbol="600000.SH", side="Buy", price=10.0
        )


def test_buy_without_sizer_reports_clearly() -> None:
    """未配置 sizer 且不传 quantity 时报错点明原因."""
    strategy = _Strategy(sizer=None, execution=_Execution(cash=1.0))
    with pytest.raises(RuntimeError, match="sizer"):
        _submitter(strategy, []).submit_order(
            symbol="600000.SH", side="Buy", price=10.0
        )


def test_short_sell_rejected_when_capability_denies() -> None:
    """capability.supports_short_sell=False 时本地拦截卖空,不打到柜台."""
    captured: list[UnifiedOrderRequest] = []
    cap = BrokerCapability(
        broker_name="middleware",
        position_effect=True,
        supports_short_sell=False,
        supported_position_effects=("auto", "open", "close"),
    )
    strategy = _Strategy(execution=_Execution(available=0.0))
    with pytest.raises(RuntimeError, match="short sell"):
        _submitter(strategy, captured, capability=cap).submit_order(
            symbol="600000.SH",
            side="Sell",
            quantity=100,
            position_effect="open",
        )
    assert captured == []


def test_short_sell_allowed_when_capability_declares() -> None:
    """声明 supports_short_sell=True 的 broker(如 CTP)不受拦截影响."""
    captured: list[UnifiedOrderRequest] = []
    cap = BrokerCapability(
        broker_name="ctp",
        position_effect=True,
        supports_short_sell=True,
        supported_position_effects=("auto", "open", "close"),
    )
    strategy = _Strategy(execution=_Execution())
    _submitter(strategy, captured, capability=cap).submit_order(
        symbol="IF2601", side="Sell", quantity=2, position_effect="open"
    )
    assert [r.quantity for r in captured] == [2.0]
