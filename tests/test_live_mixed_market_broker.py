"""``run_live(market_broker=...)``: 行情源与交易源分开指定的端到端通路.

对应 ``broker='qmf'`` 形态的缺口: 交易通道齐备但 ``market_gateway=None``, 此前
无法借用别的 broker 的行情。本模块用"只有交易的 broker + replay 行情"跑通整条
链路 facade → runner → factory → 行情网关 → feed → 引擎 → 策略回调。
"""

from typing import Any, List, Sequence, cast

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway.protocols import GatewayBundle, TraderGateway
from akquant.gateway.registry import register_broker, unregister_broker

SYMBOL = "MIXED_A"
TRADER_ONLY = "test_live_trader_only"


def _instrument(symbol: str) -> Instrument:
    """构造一个股票标的."""
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


def _ts(text: str) -> int:
    """把本地时间字符串转成纳秒时间戳."""
    return int(pd.Timestamp(text, tz="Asia/Shanghai").value)


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


class _TraderOnlyGateway:
    """只提供交易的网关替身(模拟 qmf: 无行情通道)."""

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def start(self) -> None:
        """no-op."""

    def heartbeat(self) -> bool:
        """始终就绪."""
        return True


@pytest.fixture(autouse=True)
def _register_trader_only() -> Any:
    """注册一个纯交易 broker, 用后注销以免污染全局 registry."""

    def _build(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, symbols, use_aggregator, kwargs)
        return GatewayBundle(
            market_gateway=None,
            trader_gateway=cast(TraderGateway, _TraderOnlyGateway()),
            trader_capabilities=None,
            metadata={"broker": TRADER_ONLY},
        )

    register_broker(TRADER_ONLY, _build)
    yield
    unregister_broker(TRADER_ONLY)


class _Recorder(Strategy):
    """记录收到的 bar."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.bars: List[Any] = []

    def on_bar(self, bar: Bar) -> None:
        """记录 bar."""
        self.bars.append((int(bar.timestamp), str(bar.symbol)))


def test_split_brokers_deliver_bars_to_trade_only_broker() -> None:
    """纯交易 broker 配 replay 行情后, 策略应真正收到 bar.

    这是本能力的核心用例: 交易侧只有交易通道(``market_gateway=None``), 行情侧
    借用 replay。未接线时 ``run_live`` 不认识这两个参数, 行情网关仍为 None →
    收到 0 根 bar。
    """
    strategy = _Recorder()
    stamps = [_ts("2023-01-03 09:30:00"), _ts("2023-01-03 09:31:00")]

    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        market_broker="replay",
        trader_broker=TRADER_ONLY,
        trading_mode="paper",
        gateway_options={"bars": [_bar(ts, SYMBOL, 10.0) for ts in stamps]},
        cash=100_000.0,
        show_progress=False,
        duration="30s",
    )

    assert [ts for ts, _ in strategy.bars] == stamps


def test_run_live_rejects_one_sided_override() -> None:
    """只给一侧应报错, 且错误要能穿透 run_live 传到用户面前.

    ``run_live`` 的 ``broker`` 默认值是 ``'ctp'``, 若校验缺失或被吞掉, 用户只写
    ``market_broker`` 时会收到一句与本次配置无关的 "md_front is required"。
    """
    with pytest.raises(ValueError, match="trader_broker"):
        run_live(
            strategy_cls=_Recorder(),
            instruments=[_instrument(SYMBOL)],
            market_broker="replay",
            trading_mode="paper",
            gateway_options={"bars": []},
            cash=100_000.0,
            show_progress=False,
            duration="30s",
        )


def test_mixed_session_ends_on_bounded_event_total() -> None:
    """混搭时行情 broker 声明的 bounded_event_total 不得丢失.

    replay 通过 metadata 声明事件总数, runner 据此在放完后立即结束会话。若
    混搭只保留交易 broker 的 metadata, 该声明丢失, 会话就只能等 duration 墙钟
    超时——本测试会从"秒级结束"退化成"卡满 duration"。
    """
    strategy = _Recorder()
    stamps = [_ts("2023-01-03 09:30:00"), _ts("2023-01-03 09:31:00")]

    started = pd.Timestamp.now()
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument(SYMBOL)],
        market_broker="replay",
        trader_broker=TRADER_ONLY,
        trading_mode="paper",
        gateway_options={"bars": [_bar(ts, SYMBOL, 10.0) for ts in stamps]},
        cash=100_000.0,
        show_progress=False,
        # 墙钟上限远大于实际所需(放完 2 根 bar 只需数秒): 只有 bounded_event_total
        # 生效才能提前结束。取 30s 而非更大值, 是为了让这条断言**失败得快**——
        # 丢失该声明时本测试撑满 duration 即返回, 而不是把整个套件拖死。
        duration="30s",
    )
    elapsed = (pd.Timestamp.now() - started).total_seconds()

    assert [ts for ts, _ in strategy.bars] == stamps
    assert elapsed < 20, f"会话未按 bounded_event_total 结束, 耗时 {elapsed}s"
