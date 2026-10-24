"""同刻 bar(E) 不撮合在它之前(同一时刻的 tick/定时器上)所下的单.

右闭分桶后, 恰在整分 E 的 tick 与以 E 结尾的合成 bar 落在同一时间片, 处理顺序为
tick(E) -> bar(E)。在 tick(E) 上下的单若被 bar(E) 按区间 (S, E] 的 OHLC 撮合,
就用到了下单之前的价格。带区间的 bar 因此不撮合 ``created_at == bar.timestamp``
且在该 bar 之前已存在的订单, 留给后续事件; ``on_bar`` 里下的单在该 bar 撮合之后
才创建, 不受影响。
"""

from __future__ import annotations

import datetime as dt
from typing import Any, List, Tuple

import pandas as pd
from akquant import Bar, Strategy, Tick, run_backtest
from akquant.backtest.fill_mode import CurrentClose, NextOpen

SYM = "600000.SH"
TZ = "Asia/Shanghai"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text, tz=TZ).value)


PRICES = [
    ("2024-01-02 09:31:10", 9.0),
    ("2024-01-02 09:31:30", 10.0),
    ("2024-01-02 09:32:00", 11.0),
    ("2024-01-02 09:32:30", 11.0),
    ("2024-01-02 09:33:30", 11.0),
    ("2024-01-02 09:34:30", 11.0),
]


def _ticks() -> List[Tick]:
    return [
        Tick(timestamp=_ns(t), price=p, volume=100.0, symbol=SYM) for t, p in PRICES
    ]


class _TickOrder(Strategy):
    """在指定时刻的 tick 上下一张买单, 记录成交."""

    def __init__(self, at: str, price: Any) -> None:
        super().__init__()
        self.at = _ns(at)
        self.price = price
        self.done = False
        self.fills: List[Tuple[int, float]] = []

    def on_tick(self, tick: Any) -> None:
        if int(tick.timestamp) == self.at and not self.done:
            self.done = True
            self.buy(SYM, 100, price=self.price)

    def on_trade(self, trade: Any) -> None:
        self.fills.append((int(trade.timestamp), float(trade.price)))


def _run_tick(at: str, price: Any, policy: Any) -> List[Tuple[int, float]]:
    strategy = _TickOrder(at, price)
    run_backtest(
        data=_ticks(),
        strategy=strategy,
        symbols=[SYM],
        initial_cash=1e6,
        show_progress=False,
        fill_policy=policy,
        freq="1min",
    )
    return strategy.fills


def test_limit_on_boundary_tick_not_filled_by_same_time_bar_history() -> None:
    """审查场景: 09:32:00 的 tick 上挂 9.5 限价买, 此后价格一直是 11, 不得成交.

    bar(09:32) 的区间 (09:31, 09:32] 低点 9.0 发生在下单之前。
    """
    assert _run_tick("2024-01-02 09:32:00", 9.5, CurrentClose()) == []


def test_market_on_boundary_tick_fills_at_tick_price() -> None:
    """同刻 bar 跳过后, 市价单仍在本时间片按 tick(E) 的价格成交."""
    fills = _run_tick("2024-01-02 09:32:00", None, CurrentClose())
    assert fills == [(_ns("2024-01-02 09:32:00"), 11.0)]


class _BarOrder(Strategy):
    """在首根合成 bar 的 on_bar 里下单, 记录成交."""

    def __init__(self, price: Any) -> None:
        super().__init__()
        self.price = price
        self.done = False
        self.fills: List[Tuple[int, float]] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.done:
            self.done = True
            self.buy(SYM, 100, price=self.price)

    def on_trade(self, trade: Any) -> None:
        self.fills.append((int(trade.timestamp), float(trade.price)))


def _bars() -> List[Bar]:
    """带区间的 1 分钟 bar (终点戳 E, interval_start=S), 开盘价逐根可区分."""
    out = []
    for k, (o, c) in enumerate([(10.0, 10.2), (11.0, 11.2), (12.0, 12.2)]):
        end = _ns("2024-01-02 09:32:00") + k * 60_000_000_000
        out.append(
            Bar(
                timestamp=end,
                open=o,
                high=c + 0.1,
                low=o - 0.1,
                close=c,
                volume=1e4,
                symbol=SYM,
                interval_start=end - 60_000_000_000,
            )
        )
    return out


def test_on_bar_next_open_order_still_fills_on_next_bar() -> None:
    """on_bar 下的 NextOpen 单仍由下一根 bar 的开盘价撮合."""
    strategy = _BarOrder(None)
    run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=[SYM],
        initial_cash=1e6,
        show_progress=False,
        fill_policy=NextOpen(),
    )
    # warmup 推断可能吞掉首根 on_bar: 只断言成交在"下单那根之后一根"的开盘价
    assert len(strategy.fills) == 1
    ts, price = strategy.fills[0]
    bars = {int(b.timestamp): float(b.open) for b in _bars()}
    assert price == bars[ts]
    assert ts > min(bars)


def test_on_bar_current_close_order_still_fills_on_its_bar() -> None:
    """on_bar 下的 CurrentClose 单仍按该 bar 收盘价成交(在该 bar 撮合之后创建)."""
    strategy = _BarOrder(None)
    run_backtest(
        data=_bars(),
        strategy=strategy,
        symbols=[SYM],
        initial_cash=1e6,
        show_progress=False,
        fill_policy=CurrentClose(),
    )
    assert len(strategy.fills) == 1
    ts, price = strategy.fills[0]
    closes = {int(b.timestamp): float(b.close) for b in _bars()}
    assert price == closes[ts]


def test_daily_before_trading_order_with_interval_bars_fills_same_day_open() -> None:
    """日线带区间(上一收盘, 15:00]: on_before_trading 下单仍按当日开盘价成交."""
    bars = []
    for k, day in enumerate(["2024-01-02", "2024-01-03", "2024-01-04"]):
        end = _ns(f"{day} 15:00:00")
        bars.append(
            Bar(
                timestamp=end,
                open=10.0 + k,
                high=10.5 + k,
                low=9.5 + k,
                close=10.2 + k,
                volume=1e6,
                symbol=SYM,
                interval_start=end - 24 * 3600 * 1_000_000_000,
            )
        )

    class P(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.fills: List[Tuple[int, float]] = []

        def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
            if trading_date == dt.date(2024, 1, 3):
                self.buy(SYM, 100)

        def on_trade(self, trade: Any) -> None:
            self.fills.append((int(trade.timestamp), float(trade.price)))

    strategy = P()
    run_backtest(
        data=bars,
        strategy=strategy,
        symbols=[SYM],
        initial_cash=1e6,
        show_progress=False,
    )
    assert strategy.fills == [(_ns("2024-01-03 15:00:00"), 11.0)]
