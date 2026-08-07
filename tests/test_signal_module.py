"""akquant.signal 端到端: 外部信号 → 幂等 → 下单 → 回执.

设计依据: docs/zh/meta/signal-ingestion-rfc.md 第 4.3 节。

测试时序约定同 test_signal_port_injection.py: 策略与信号源双向握手, 不赌时间窗
(注入必须落在最后一根 bar 之前才有撮合机会)。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Sequence

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar
from akquant.gateway import register_broker, unregister_broker
from akquant.gateway.protocols import GatewayBundle
from akquant.signal import (
    QueueSignalSource,
    Signal,
    SignalDedup,
    SignalDispatcher,
    SignalStatus,
)

SYMBOL = "SIGMOD_A"
BAR_CLOSE = 10.0
BROKER = "sigmod-paced"


def _instrument() -> Instrument:
    """构造股票标的."""
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(count: int = 6) -> List[Bar]:
    """当前墙钟时间戳起的连续 bar."""
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    return [
        Bar(
            timestamp=int((now + pd.Timedelta(seconds=i)).value),
            open=BAR_CLOSE,
            high=BAR_CLOSE + 0.5,
            low=BAR_CLOSE - 0.5,
            close=BAR_CLOSE,
            volume=100_000.0,
            symbol=SYMBOL,
        )
        for i in range(count)
    ]


class _PacedGateway:
    """按间隔逐根推 bar, 可被停止(避免跨测试污染)."""

    def __init__(self, feed: Any, symbols: Sequence[str], bars: List[Bar]) -> None:
        """记录 feed、订阅集与待推事件."""
        self._feed = feed
        self._symbols = [str(s) for s in symbols]
        self._bars = bars
        self.stop = threading.Event()
        self.finished = threading.Event()

    def connect(self) -> None:
        """no-op."""

    def disconnect(self) -> None:
        """no-op."""

    def subscribe(self, symbols: Sequence[str]) -> None:
        """替换订阅集."""
        self._symbols = [str(s) for s in symbols]

    def unsubscribe(self, symbols: Sequence[str]) -> None:
        """移除订阅."""

    def on_tick(self, callback: Callable[[dict], None]) -> None:
        """仅存引用."""

    def on_bar(self, callback: Callable[[dict], None]) -> None:
        """仅存引用."""

    def start(self) -> None:
        """按间隔推送, 可提前中断."""
        try:
            allowed = set(self._symbols)
            for bar in self._bars:
                if self.stop.is_set():
                    return
                if str(bar.symbol) in allowed:
                    self._feed.add_bar(bar)
                    if self.stop.wait(timeout=0.25):
                        return
        finally:
            self.finished.set()


class _HandshakeStrategy(Strategy):
    """自身不下单; 与信号源双向握手, 消除时序竞态."""

    def __init__(self) -> None:
        """初始化记录容器与握手闸门."""
        self.trades: List[Any] = []
        self.rejected: List[Any] = []
        self.bars_seen = 0
        self.gate = threading.Event()
        self.settled = threading.Event()

    def on_bar(self, bar: Bar) -> None:
        """首根放行信号投递; 第二根等其落地(不下单)."""
        self.bars_seen += 1
        if self.bars_seen == 1:
            self.gate.set()
        elif self.bars_seen == 2:
            self.settled.wait(timeout=10.0)

    def on_trade(self, trade: Any) -> None:
        """记录成交."""
        self.trades.append(trade)

    def on_reject(self, order: Any) -> None:
        """记录拒单."""
        self.rejected.append(order)


def _run(
    strategy: _HandshakeStrategy,
    source: QueueSignalSource,
    feeder: Callable[[QueueSignalSource], None],
    *,
    max_order_value: float | None = None,
) -> None:
    """跑一轮 paper 会话; ``feeder`` 在独立线程里投信号."""
    gateways: List[_PacedGateway] = []
    bars = _bars()

    def _build(**kwargs: Any) -> GatewayBundle:
        gateway = _PacedGateway(kwargs["feed"], kwargs["symbols"], bars)
        gateways.append(gateway)
        return GatewayBundle(
            market_gateway=gateway,
            trader_gateway=None,
            trader_capabilities=None,
            metadata={"broker": BROKER, "bounded_event_total": len(bars)},
        )

    register_broker(BROKER, _build)
    running = threading.Event()

    def worker() -> None:
        running.set()
        if not strategy.gate.wait(timeout=10.0):
            strategy.settled.set()
            return
        try:
            feeder(source)
            # 给消费线程一点时间把队列排空(dispatch 完成)后再放行会话。
            # 这不是赌时间窗: settled 只控制"会话何时继续", 信号是否被处理由
            # 队列消费保证; 稍等只是让成交发生在会话结束之前。
            time.sleep(0.4)
        finally:
            strategy.settled.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    running.wait(timeout=5.0)
    try:
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument()],
            broker=BROKER,
            trading_mode="paper",
            cash=1_000_000.0,
            show_progress=False,
            duration="15s",
            signal_source=source,
            strategy_max_order_value=(
                None if max_order_value is None else {"_default": max_order_value}
            ),
        )
    finally:
        unregister_broker(BROKER)
        for gateway in gateways:
            gateway.stop.set()
            gateway.finished.wait(timeout=2.0)


def test_signal_reaches_engine_and_fills() -> None:
    """一条信号经 QueueSignalSource → dispatcher → 引擎, 应成交."""
    strategy = _HandshakeStrategy()
    source = QueueSignalSource()

    def feed(src: QueueSignalSource) -> None:
        src.put(
            Signal(
                signal_id="sig-1",
                symbol=SYMBOL,
                action="buy",
                quantity=100.0,
                price=BAR_CLOSE,
            )
        )

    _run(strategy, source, feed)

    assert source.results, "信号源未收到任何回执"
    accepted = [r for r in source.results if r.status is SignalStatus.ACCEPTED]
    assert len(accepted) == 1, f"应有 1 条被接受, 实际回执: {source.results}"
    assert accepted[0].signal_id == "sig-1"
    assert accepted[0].order_id, "回执应带订单 id"
    assert strategy.trades, "信号单未成交"


def test_duplicate_signal_id_is_dropped_once() -> None:
    """同一 signal_id 投三次, 只应下一次单."""
    strategy = _HandshakeStrategy()
    source = QueueSignalSource()

    def feed(src: QueueSignalSource) -> None:
        for _ in range(3):
            src.put(
                Signal(
                    signal_id="dup-1",
                    symbol=SYMBOL,
                    action="buy",
                    quantity=100.0,
                    price=BAR_CLOSE,
                )
            )

    _run(strategy, source, feed)

    accepted = [r for r in source.results if r.status is SignalStatus.ACCEPTED]
    duplicates = [r for r in source.results if r.status is SignalStatus.DUPLICATE]
    assert len(accepted) == 1, f"只应接受 1 条, 实际 {len(accepted)}"
    assert len(duplicates) == 2, f"应有 2 条判重, 实际 {len(duplicates)}"
    filled = sum(float(getattr(t, "quantity", 0.0)) for t in strategy.trades)
    assert filled == 100.0, f"只应成交 100 股, 实际 {filled}"


def test_dedup_is_atomic_under_concurrency() -> None:
    """并发投递同一 id, 只有一个线程能放行(检查与标记必须原子)."""
    dedup = SignalDedup()
    passed: List[int] = []
    barrier = threading.Barrier(16)

    def worker(index: int) -> None:
        barrier.wait()  # 尽量让 16 个线程同时冲同一个 id
        if dedup.check_and_mark("same-id"):
            passed.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(passed) == 1, f"应只有 1 个线程放行, 实际 {len(passed)}"
    assert len(dedup) == 1


def test_dedup_evicts_and_warns_when_capacity_reached() -> None:
    """达容量后淘汰最老 id, 并计数(静默淘汰=静默重复下单)."""
    dedup = SignalDedup(capacity=3)
    for i in range(5):
        assert dedup.check_and_mark(f"id-{i}") is True
    assert len(dedup) == 3
    assert dedup.evicted == 2
    # id-0/id-1 已被淘汰: 重推会被当作新信号
    assert dedup.check_and_mark("id-0") is True


def test_dispatcher_forgets_id_when_submit_raises() -> None:
    """投递抛异常时须放开去重标记, 允许平台重推."""

    class _FailingSink:
        mode = "paper"
        calls = 0

        def submit(self, signal: Signal) -> str:
            _FailingSink.calls += 1
            raise RuntimeError("gateway down")

    dispatcher = SignalDispatcher(_FailingSink())
    signal = Signal(
        signal_id="retry-1", symbol=SYMBOL, action="buy", quantity=1.0, price=1.0
    )
    first = dispatcher.dispatch(signal)
    assert first.status is SignalStatus.ERROR
    # 同一 id 应能再次进入(因为上一次并未真正下单)
    second = dispatcher.dispatch(signal)
    assert second.status is SignalStatus.ERROR
    assert _FailingSink.calls == 2, "投递失败后重推应再次尝试, 而非被判重丢弃"


def test_signal_model_rejects_invalid_payload() -> None:
    """契约校验: 非正数量 / 空 symbol / 未知 order_type 立即报错."""
    with pytest.raises(ValueError):
        Signal(signal_id="a", symbol=SYMBOL, action="buy", quantity=0.0)
    with pytest.raises(ValueError):
        Signal(signal_id="a", symbol="  ", action="buy", quantity=1.0)
    with pytest.raises(ValueError):
        Signal(
            signal_id="a",
            symbol=SYMBOL,
            action="buy",
            quantity=1.0,
            order_type="Iceberg",
        )
    # order_type 大小写归一
    assert (
        Signal(
            signal_id="a",
            symbol=SYMBOL,
            action="buy",
            quantity=1.0,
            order_type="limit",
        ).order_type
        == "Limit"
    )
    # action → 网关方向字符串
    assert (
        Signal(signal_id="a", symbol=SYMBOL, action="sell", quantity=1.0).side == "Sell"
    )


def test_risk_rejection_is_reported_back_to_source() -> None:
    """被风控拒的信号必须回执给来源, 否则平台以为下成功了."""
    strategy = _HandshakeStrategy()
    source = QueueSignalSource()

    def feed(src: QueueSignalSource) -> None:
        src.put(
            Signal(
                signal_id="over-1",
                symbol=SYMBOL,
                action="buy",
                quantity=100.0,
                price=BAR_CLOSE,
            )
        )

    # 名义 1000 > 上限 500
    _run(strategy, source, feed, max_order_value=500.0)

    assert strategy.rejected, "超限信号单应被风控拒绝"
    rejected = [r for r in source.results if r.status is SignalStatus.REJECTED]
    assert rejected, f"拒单必须回执给信号源, 实际回执: {source.results}"
    assert rejected[0].signal_id == "over-1"
    assert strategy.trades == [], "被拒的信号不应成交"


def test_signal_source_registry_builtins_and_custom() -> None:
    """内置源应已注册, 自定义源可注册/构造/注销."""
    from akquant.signal import (
        create_signal_source,
        list_signal_sources,
        register_signal_source,
        unregister_signal_source,
    )

    names = list_signal_sources()
    assert {"queue", "http", "redis"} <= set(names), names

    built = create_signal_source("queue")
    assert isinstance(built, QueueSignalSource)

    class _Custom(QueueSignalSource):
        pass

    register_signal_source("my-platform", _Custom)
    try:
        assert isinstance(create_signal_source("my-platform"), _Custom)
    finally:
        unregister_signal_source("my-platform")
    with pytest.raises(ValueError, match="未知信号源"):
        create_signal_source("my-platform")


def test_registry_rejects_empty_name() -> None:
    """空名称注册应报错."""
    from akquant.signal import register_signal_source

    with pytest.raises(ValueError, match="不能为空"):
        register_signal_source("  ", QueueSignalSource)
