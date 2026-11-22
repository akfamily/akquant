"""Tick 一档盘口(买一/卖一价与量): 构造、pickle、CTP 网关填充、策略缓存."""

import pickle
import sys
from typing import Any, List, cast

import pytest
from akquant import DataFeed, Strategy, run_backtest
from akquant.akquant import Tick
from akquant.gateway.brokers.ctp import native as ctp_native

DBL_MAX = sys.float_info.max
NS = 1_700_000_000_000_000_000


def test_tick_quotes_default_to_none() -> None:
    """不传报价时四个字段均为 None, 原有构造方式不变."""
    tick = Tick(timestamp=NS, price=10.0, volume=1.0, symbol="X")
    assert (tick.bid_price, tick.ask_price, tick.bid_volume, tick.ask_volume) == (
        None,
        None,
        None,
        None,
    )


def test_tick_quotes_construct_set_and_pickle() -> None:
    """构造、修改(含置回 None)与 pickle 往返保留报价."""
    tick = Tick(
        timestamp=NS,
        price=10.0,
        volume=1.0,
        symbol="X",
        bid_price=9.5,
        ask_price=10.5,
        bid_volume=3,
        ask_volume=4,
    )
    assert (tick.bid_price, tick.ask_price, tick.bid_volume, tick.ask_volume) == (
        9.5,
        10.5,
        3.0,
        4.0,
    )
    tick.ask_price = None
    tick.bid_volume = 7
    restored = pickle.loads(pickle.dumps(tick))
    assert (restored.bid_price, restored.ask_price, restored.bid_volume) == (
        9.5,
        None,
        7.0,
    )


class _FakeFeed:
    def __init__(self) -> None:
        self.ticks: List[Any] = []

    def add_tick(self, tick: Any) -> None:
        self.ticks.append(tick)


def _md(bid: float, bid_volume: int, ask: float, ask_volume: int) -> Any:
    class _MarketData:
        InstrumentID = "m2401-C-3000"
        LastPrice = 100.0
        Volume = 10
        BidPrice1 = bid
        BidVolume1 = bid_volume
        AskPrice1 = ask
        AskVolume1 = ask_volume

    return _MarketData()


@pytest.mark.parametrize(
    "bid, ask, expected",
    [
        (99.5, 100.5, (99.5, 100.5, 2.0, 3.0)),
        (99.5, DBL_MAX, (99.5, None, 2.0, None)),  # CTP 以 DBL_MAX 表示无卖一
        (0.0, 100.5, (None, 100.5, None, 3.0)),  # 价格为 0 表示无买一
    ],
)
def test_ctp_gateway_fills_top_of_book(
    monkeypatch: pytest.MonkeyPatch, bid: float, ask: float, expected: tuple
) -> None:
    """CTP 网关把 BidPrice1/AskPrice1 填进 Tick, 无报价的一侧为 None."""
    monkeypatch.setattr(ctp_native, "HAS_OPENCTP", True)
    feed = _FakeFeed()
    gw = ctp_native.CTPMarketGateway(
        feed=cast(Any, feed),
        symbols=["m2401-C-3000"],
        front_url="tcp://test",
        emit_ticks=True,
        emit_bars=False,
    )
    gw.OnRtnDepthMarketData(_md(bid, 2, ask, 3))
    tick = feed.ticks[-1]
    assert (tick.bid_price, tick.ask_price, tick.bid_volume, tick.ask_volume) == (
        expected
    )


class _QuoteRecorder(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.quotes: List[Any] = []

    def on_tick(self, tick: Any) -> None:
        self.quotes.append(self._last_quotes[tick.symbol])


def test_backtest_ticks_update_strategy_quote_cache() -> None:
    """回测中每笔 tick 的报价写入策略的 _last_quotes, 无报价的 tick 覆盖为 None."""
    ticks = [
        Tick(NS, 10.0, 1.0, "X", bid_price=9.9, ask_price=10.1),
        Tick(NS + 60_000_000_000, 10.0, 1.0, "X", bid_price=9.8),
        Tick(NS + 120_000_000_000, 10.0, 1.0, "X"),
    ]
    strategy = _QuoteRecorder()
    run_backtest(
        data=ticks,
        strategy=strategy,
        symbols=["X"],
        initial_cash=100_000.0,
        show_progress=False,
    )
    assert strategy.quotes[0] == pytest.approx((9.9, 10.1))
    assert strategy.quotes[1][0] == pytest.approx(9.8)
    assert strategy.quotes[1][1] is None
    assert strategy.quotes[2] == (None, None)


def test_data_feed_accepts_quoted_ticks() -> None:
    """带报价的 Tick 可正常推入 DataFeed."""
    feed = DataFeed()
    feed.add_tick(Tick(NS, 10.0, 1.0, "X", bid_price=9.9, ask_price=10.1))
