"""行情源推来的未收盘 K 线(``DataFeed.add_forming_bar``).

设计见 ``docs/superpowers/specs/2026-11-24-forming-bar-preview-design.md``。
用真实 ``Engine`` 跑 ``replay`` 的有界会话; 另注册一个包装 replay 的测试 broker,
把 feed 换成代理: 推某根收盘 bar 之前(或之后)先推脚本里给它安排的未收盘 bar。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar, DataFeed
from akquant.gateway import GatewayBundle
from akquant.gateway.brokers.replay.gateway import build_replay_bundle
from akquant.gateway.registry import register_broker, unregister_broker

_SYMBOL = "600000"
_BROKER = "replay_forming_test"
_Key = Tuple[str, int]


def _instrument(symbol: str = _SYMBOL) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(count: int = 8, symbol: str = _SYMBOL, base: float = 10.0) -> List[Bar]:
    """Count 根 1 分钟 bar: 第 i 根收盘价 base+i, 最高 +0.5, 最低 -0.5."""
    stamps = pd.date_range(
        "2024-01-02 09:31:00", periods=count, freq="1min", tz="Asia/Shanghai"
    )
    return [
        Bar(
            timestamp=int(ts.value),
            open=base + i - 0.1,
            high=base + i + 0.5,
            low=base + i - 0.5,
            close=base + i,
            volume=100.0,
            symbol=symbol,
        )
        for i, ts in enumerate(stamps)
    ]


def _forming(bar: Bar, close: Optional[float] = None) -> Bar:
    """与 ``bar`` 同一桶的未收盘 bar; 不给 close 时 OHLCV 与收盘 bar 完全相同."""
    price = float(bar.close) if close is None else float(close)
    return Bar(
        timestamp=int(bar.timestamp),
        open=float(bar.open),
        high=max(float(bar.high), price),
        low=min(float(bar.low), price),
        close=price,
        volume=float(bar.volume),
        symbol=str(bar.symbol),
    )


def _key(bar: Bar) -> _Key:
    return str(bar.symbol), int(bar.timestamp)


@dataclass
class _Script:
    """收盘 bar → 推它之前 / 之后要推的未收盘 bar."""

    before: Dict[_Key, List[Bar]] = field(default_factory=dict)
    after: Dict[_Key, List[Bar]] = field(default_factory=dict)


class _FormingFeed:
    """包住真实 feed: 只在 add_bar 前后插入脚本里的未收盘 bar."""

    def __init__(self, feed: Any, script: _Script) -> None:
        self._feed = feed
        self._script = script

    def add_bar(self, bar: Bar) -> None:
        for forming in self._script.before.get(_key(bar), []):
            self._feed.add_forming_bar(forming)
        self._feed.add_bar(bar)
        for forming in self._script.after.get(_key(bar), []):
            self._feed.add_forming_bar(forming)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._feed, name)


@pytest.fixture
def script() -> Any:
    """注册包装 replay 的测试 broker, 用后注销以免污染全局 registry."""
    current = _Script()

    def _build(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        return build_replay_bundle(
            _FormingFeed(feed, current), symbols, use_aggregator, **kwargs
        )

    register_broker(_BROKER, _build)
    yield current
    unregister_broker(_BROKER)


def _run(
    strategy: Strategy,
    bars: List[Bar],
    *,
    on_event: Any = None,
    instruments: Optional[List[Instrument]] = None,
    **kwargs: Any,
) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=instruments or [_instrument()],
        broker=_BROKER,
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="30s",
        on_event=on_event,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 引擎: 入队、派发、与行情隔离(F1、F2)
# ---------------------------------------------------------------------------


def test_add_forming_bar_rejected_on_backtest_feed() -> None:
    """回测数据源没有未收盘的 K 线, 调用即报错."""
    with pytest.raises(RuntimeError, match="add_forming_bar"):
        DataFeed().add_forming_bar(_bars(1)[0])


class _HookProbe(Strategy):
    """把引擎派发的未收盘 bar 与 on_bar 记进同一条日志."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[str, int, float]] = []

    def on_bar(self, bar: Bar) -> None:
        self.records.append(("bar", int(bar.timestamp), float(bar.close)))

    def _on_forming_bar_event(self, bar: Bar) -> None:
        self.records.append(("forming", int(bar.timestamp), float(bar.close)))


def test_forming_bar_reaches_strategy_in_feed_order_without_on_bar(
    script: _Script,
) -> None:
    """未收盘 bar 按入队顺序交给策略内部入口, 不触发 on_bar."""
    bars = _bars()
    script.before[_key(bars[3])] = [_forming(bars[3], 99.0), _forming(bars[3], 98.0)]
    probe = _HookProbe()

    _run(probe, bars)

    ts3 = int(bars[3].timestamp)
    expected: List[Tuple[str, int, float]] = []
    for i, bar in enumerate(bars):
        if i == 3:
            expected += [("forming", ts3, 99.0), ("forming", ts3, 98.0)]
        expected.append(("bar", int(bar.timestamp), float(bar.close)))
    assert probe.records == expected


class _Trader(Strategy):
    """第 2 根买入、第 5 根卖出, 逐根记下引擎状态."""

    def __init__(self) -> None:
        super().__init__()
        self.set_history_depth(5)
        self.seen = 0
        self.records: List[Tuple[int, float, float, float, List[float]]] = []

    def on_bar(self, bar: Bar) -> None:
        self.seen += 1
        if self.seen == 2:
            self.buy(bar.symbol, 100)
        if self.seen == 5:
            self.sell(bar.symbol, 100)
        history = self.get_history(min(self.seen, 3), bar.symbol)
        self.records.append(
            (
                int(bar.timestamp),
                float(bar.close),
                float(self.get_position(bar.symbol)),
                round(float(self.cash), 6),
                [float(value) for value in history],
            )
        )


def test_forming_bars_do_not_change_engine_state(script: _Script) -> None:
    """穿插价格离谱的未收盘 bar, 逐根的历史、持仓、现金与不穿插时完全相同."""
    bars = _bars()
    baseline = _Trader()
    _run(baseline, bars)

    for bar in bars:
        script.before[_key(bar)] = [_forming(bar, 99.0), _forming(bar, 1.0)]
    with_forming = _Trader()
    _run(with_forming, bars)

    assert len(baseline.records) == len(bars)
    assert with_forming.records == baseline.records
    # 买单确实成交过, 否则上面的相等只是两边都没交易。
    assert any(position == 100.0 for _, _, position, _, _ in baseline.records)
