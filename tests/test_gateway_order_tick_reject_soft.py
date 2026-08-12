"""实盘 tick 拒单必须走软拒单契约(record_reject + on_reject), 不能抛异常炸穿 on_bar.

**修复前的缺陷**: `BrokerOrderSubmitter.submit_order` 里 `_validate_price_tick`
校验失败直接 `raise ValueError`, 从 `self.buy()`/`self.sell()` 里原样炸出
`on_bar`。而紧接着几十行外的前置风控拒单(`_check_risk`)是
`record_reject(...)` + `_emit_risk_reject(...)` + 返回空 `OrderReceipt`——
显式对齐回测的 `Rejected` `ExecutionReport` -> `on_reject` 语义。

两条本该同构的拒单路径行为不一致: tick 拒单既不进订单审计流水, 也不触发
`on_reject`, 还会把异常炸穿用户的 `on_bar` 回调, 与回测里同一笔单收到软
`Rejected` 回报的体验完全不同。

修复方式: `submit_order` 捕获 `_validate_price_tick` 抛出的 `ValueError`,
复用风控拒单同一套 `record_reject` + `_emit_risk_reject` + 空回执路径。

覆盖范围: 端到端 `run_live` 会话(与 `test_broker_live_risk_enforcement.py`
同构), 断言非对齐价格的单既不到柜台也不让 `run_live` 抛异常, 而是触发
`on_reject`。
"""

from typing import Any, List, Optional

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest

SYMBOL = "TICKREJ_A"
BAR_CLOSE = 100.0
# tick_size=0.01, 100.003 不是其整数倍。
MISALIGNED_PRICE = 100.003
ORDER_QUANTITY = 10.0


def _ts(text: str) -> int:
    """本地时间字符串 → 纳秒时间戳."""
    return int(pd.Timestamp(text, tz="Asia/Shanghai").value)


def _instrument(symbol: str) -> Instrument:
    """构造股票标的(字段口径同 tests/test_replay_live_end_to_end.py)."""
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bar(ts: int, symbol: str, close: float) -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


class _RecordingTraderGateway:
    """记录每一笔到达柜台的报单的 trader 网关桩."""

    def __init__(self) -> None:
        self.placed: List[UnifiedOrderRequest] = []

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def start(self) -> None:
        """no-op: 无推送线程."""

    def place_order(self, req: UnifiedOrderRequest) -> str:
        """记录报单并返回柜台单号."""
        self.placed.append(req)
        return f"broker-{len(self.placed)}"

    def cancel_order(self, broker_order_id: str) -> None:
        """no-op."""

    def get_capabilities(self) -> BrokerCapability:
        """最小能力集: 只声明 broker_live 与 client_order_id."""
        return BrokerCapability(
            broker_name="tickreject-fake",
            broker_live=True,
            client_order_id=True,
            order_type=True,
            time_in_force_str=True,
        )

    def query_order(self, broker_order_id: str) -> None:
        """柜台无快照."""
        return None

    def query_trades(self, since: Optional[int] = None) -> List[Any]:
        """无成交."""
        return []

    def query_account(self) -> None:
        """无账户快照(submitter 不依赖)."""
        return None

    def query_positions(self) -> List[Any]:
        """无持仓."""
        return []

    def on_order(self, callback: Any) -> None:
        """不推送委托回报."""

    def on_trade(self, callback: Any) -> None:
        """不推送成交回报."""

    def on_execution_report(self, callback: Any) -> None:
        """不推送执行报告."""

    def sync_open_orders(self) -> List[Any]:
        """无挂单."""
        return []

    def sync_today_trades(self) -> List[Any]:
        """无当日成交."""
        return []

    def heartbeat(self) -> bool:
        """始终就绪, 让 _await_broker_ready 立即通过."""
        return True


class _MisalignedPriceStrategy(Strategy):
    """在首根 bar 上报一笔 tick 未对齐的限价单, 并记录 on_reject."""

    def __init__(self) -> None:
        """初始化拒单记录容器."""
        self.rejected: List[Any] = []
        self.submitted = False

    def on_bar(self, bar: Bar) -> None:
        """只在第一根 bar 报单一次."""
        if self.submitted:
            return
        self.submitted = True
        self.buy(bar.symbol, quantity=ORDER_QUANTITY, price=MISALIGNED_PRICE)

    def on_reject(self, order: Any) -> None:
        """记录风控/tick 拒单."""
        self.rejected.append(order)


def _bundle(gateway: _RecordingTraderGateway) -> Any:
    """把桩网关包成只有交易侧的 GatewayBundle."""
    from akquant.gateway.protocols import GatewayBundle

    return GatewayBundle(
        market_gateway=None,
        trader_gateway=gateway,
        trader_capabilities=gateway.get_capabilities(),
        metadata={"broker": "tickreject-fake"},
    )


def _run_session(gateway: _RecordingTraderGateway) -> _MisalignedPriceStrategy:
    """跑一轮 broker_live 会话: replay 供行情, 桩网关供交易."""
    strategy = _MisalignedPriceStrategy()
    register_broker("tickreject-fake", lambda **_: _bundle(gateway))
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument(SYMBOL)],
            market_broker="replay",
            trader_broker="tickreject-fake",
            trading_mode="broker_live",
            gateway_options={
                "bars": [
                    _bar(_ts("2024-01-02 09:31:00"), SYMBOL, BAR_CLOSE),
                    _bar(_ts("2024-01-02 09:32:00"), SYMBOL, BAR_CLOSE),
                ]
            },
            cash=1_000_000.0,
            show_progress=False,
            duration="60s",
        )
    finally:
        unregister_broker("tickreject-fake")
    return strategy


def test_broker_live_tick_reject_is_soft_not_an_exception() -> None:
    """Tick 未对齐的单走软拒单(on_reject), run_live 不因此抛异常, 单不到柜台."""
    gateway = _RecordingTraderGateway()
    strategy = _run_session(gateway)  # 修复前: ValueError 会从这里炸出来

    assert gateway.placed == [], (
        f"tick 未对齐 {MISALIGNED_PRICE} 不应到达柜台, 但有 "
        f"{len(gateway.placed)} 笔到达"
    )
    assert strategy.rejected, "tick 拒单应触发 on_reject, 与回测软拒单同口径"
    reason = str(strategy.rejected[0].reject_reason)
    assert "0.01" in reason and str(MISALIGNED_PRICE) in reason
