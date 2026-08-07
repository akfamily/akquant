"""broker_live 下策略级限额必须真实拦单(回归锚).

设计依据: docs/zh/meta/signal-ingestion-rfc.md 第 3.3 节。

**修复前的缺陷**: `strategy_max_*` 经 `engine.set_strategy_max_*_limits()` 下发到 Rust
`risk_manager`, 但 broker_live 的订单走 Python `BrokerOrderSubmitter` 直接
`place_order()` 送柜台, 既不经 `ChannelProcessor` 也不碰 `risk_manager` —— 这组风控
在实盘完全不生效, 用户按文档配了限额却不拦单。

**修复方式**: `BrokerOrderSubmitter._check_risk` 在报单前调 Rust 的
`check_strategy_limits`(无状态自由函数, 与 `Engine::check_strategy_*_limit` 共用同一批
判定逻辑)。之所以不直接调 engine: 报单发生在策略回调内, 那一刻
`Engine::run(&mut self)` 正独占借用引擎, 任何经 Python 触达 `Engine` 的调用都会
`RuntimeError: Already borrowed`。

**覆盖范围**: order_value / order_size / position_size 三项。daily_loss / drawdown /
risk_budget 依赖引擎累计盈亏与预算用量, 在 broker_live 下仍未生效(已知缺口)。

这条通路此前无任何测试覆盖: 其余 live 测试都 mock 掉 `Engine`,
`test_live_runner_broker_bridge.py` 用 `_DummyEngine` 只断言"配置被下发", 从不验证
下发之后风控是否真的执行 —— 这正是缺陷能长期存在的原因。
"""

from typing import Any, List, Optional

import pandas as pd
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.broker_models import BrokerCapability, UnifiedOrderRequest

SYMBOL = "RISKGUARD_A"
BAR_CLOSE = 100.0
ORDER_QUANTITY = 100.0
# 单笔名义 = 100 股 * 100 元 = 10000 元, 远超下面设的 500 元上限。
MAX_ORDER_VALUE = 500.0


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
            broker_name="riskguard-fake",
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


class _OversizedBuyStrategy(Strategy):
    """在首根 bar 上报一笔超限单, 并记录被拒回调."""

    def __init__(self) -> None:
        """初始化拒单记录容器."""
        self.rejected: List[Any] = []
        self.submitted = False

    def on_bar(self, bar: Bar) -> None:
        """只在第一根 bar 报单一次."""
        if self.submitted:
            return
        self.submitted = True
        self.buy(bar.symbol, quantity=ORDER_QUANTITY, price=BAR_CLOSE)

    def on_reject(self, order: Any) -> None:
        """记录风控/柜台拒单."""
        self.rejected.append(order)


def _run_session(
    gateway: _RecordingTraderGateway,
    *,
    max_order_value: float | None = MAX_ORDER_VALUE,
    max_order_size: float | None = None,
) -> _OversizedBuyStrategy:
    """跑一轮 broker_live 会话: replay 供行情, 桩网关供交易.

    `market_broker`/`trader_broker` 分离模式让 replay(只有行情)与交易桩拼起来;
    replay 的 `bounded_event_total` 会在事件放完后终止会话, `duration` 兜底。
    """
    strategy = _OversizedBuyStrategy()
    register_broker("riskguard-fake", lambda **_: _bundle(gateway))
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument(SYMBOL)],
            market_broker="replay",
            trader_broker="riskguard-fake",
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
            strategy_max_order_value=(
                None if max_order_value is None else {"_default": max_order_value}
            ),
            strategy_max_order_size=(
                None if max_order_size is None else {"_default": max_order_size}
            ),
        )
    finally:
        unregister_broker("riskguard-fake")
    return strategy


def _bundle(gateway: _RecordingTraderGateway) -> Any:
    """把桩网关包成只有交易侧的 GatewayBundle."""
    from akquant.gateway.protocols import GatewayBundle

    return GatewayBundle(
        market_gateway=None,
        trader_gateway=gateway,
        trader_capabilities=gateway.get_capabilities(),
        metadata={"broker": "riskguard-fake"},
    )


def test_broker_live_risk_blocks_oversized_order() -> None:
    """超过 strategy_max_order_value 的单不得到达柜台(回归锚: issue 3.3)."""
    gateway = _RecordingTraderGateway()
    strategy = _run_session(gateway)

    assert gateway.placed == [], (
        f"风控应拦下超限单(名义 {ORDER_QUANTITY * BAR_CLOSE} > 上限 "
        f"{MAX_ORDER_VALUE}), 但仍有 {len(gateway.placed)} 笔到达柜台"
    )
    assert strategy.rejected, "被风控拦下的单应触发 on_reject"


def test_broker_live_risk_allows_within_limit_order() -> None:
    """限额内的单必须照常到达柜台(防止前置风控误杀)."""
    gateway = _RecordingTraderGateway()
    strategy = _run_session(gateway, max_order_value=1_000_000.0)

    assert len(gateway.placed) == 1, "限额内的单应正常报到柜台"
    assert gateway.placed[0].symbol == SYMBOL
    assert strategy.rejected == [], "限额内不应产生拒单"


def test_broker_live_risk_blocks_oversized_quantity() -> None:
    """strategy_max_order_size(数量维度)同样必须在实盘生效."""
    gateway = _RecordingTraderGateway()
    strategy = _run_session(gateway, max_order_size=10.0)

    assert gateway.placed == [], "超过数量上限的单不应到达柜台"
    assert strategy.rejected, "应触发 on_reject"
    assert "quantity" in str(strategy.rejected[0].reject_reason)
