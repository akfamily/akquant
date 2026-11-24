"""回测用 K 线模拟 tick: ``run_backtest(simulate_ticks=True)``."""

from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, Tick, run_backtest

_SYMBOL = "600000"
_TZ = "Asia/Shanghai"
_SEC = 1_000_000_000


def _frame(
    count: int = 8,
    *,
    symbol: str = _SYMBOL,
    base: float = 10.0,
    high_first: bool = False,
    start: str = "2024-01-02 09:31:00",
) -> pd.DataFrame:
    """生成 count 根 1 分钟线.

    缺省: 第 i 根 open=base+i, high=open+0.6, low=open-0.2, close=open+0.4,
    开盘价离最低价更近, 模拟路径为 开、低、高、收。``high_first=True`` 时
    high=open+0.1, low=open-0.5, close=open-0.3, 路径为 开、高、低、收。
    """
    stamps = pd.date_range(start, periods=count, freq="1min", tz=_TZ)
    opens = [base + i for i in range(count)]
    if high_first:
        highs = [o + 0.1 for o in opens]
        lows = [o - 0.5 for o in opens]
        closes = [o - 0.3 for o in opens]
    else:
        highs = [o + 0.6 for o in opens]
        lows = [o - 0.2 for o in opens]
        closes = [o + 0.4 for o in opens]
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": 1000.0,
        }
    )


def _run(strategy: Any, data: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("freq", "1min")
    kwargs.setdefault("initial_cash", 1_000_000.0)
    kwargs.setdefault("show_progress", False)
    kwargs.setdefault("commission_rate", 0.0)
    kwargs.setdefault("t_plus_one", False)
    kwargs.setdefault("lot_size", 100)
    return run_backtest(strategy=strategy, data=data, **kwargs)


class _Recorder(Strategy):
    """把 on_tick 与 on_bar 按到达顺序记下来."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[str, str, int, float, float]] = []

    def on_tick(self, tick: Tick) -> None:
        self.events.append(
            (
                "tick",
                str(tick.symbol),
                int(tick.timestamp),
                round(float(tick.price), 6),
                float(tick.volume),
            )
        )

    def on_bar(self, bar: Bar) -> None:
        self.events.append(
            (
                "bar",
                str(bar.symbol),
                int(bar.timestamp),
                round(float(bar.close), 6),
                float(bar.volume),
            )
        )


def _expected(frame: pd.DataFrame, *, high_first: bool = False) -> List[Any]:
    out: List[Any] = []
    for row in frame.to_dict("records"):
        end = int(pd.Timestamp(row["timestamp"]).value)
        symbol = str(row["symbol"])
        high, low = float(row["high"]), float(row["low"])
        close = float(row["close"])
        middle = (high, low) if high_first else (low, high)
        prices = [float(row["open"]), middle[0], middle[1], close]
        for step, price in zip((4, 3, 2, 1), prices):
            out.append(("tick", symbol, end - step * _SEC, round(price, 6), 250.0))
        out.append(("bar", symbol, end, round(close, 6), 1000.0))
    return out


# ---------------------------------------------------------------------------
# 生成(S3～S6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("high_first", [False, True])
def test_each_bar_is_preceded_by_four_simulated_ticks(high_first: bool) -> None:
    """每根 K 线之前依次收到 开、极值、极值、收 四个 tick, 时间戳为 E-4s..E-1s."""
    frame = _frame(high_first=high_first)
    recorder = _Recorder()

    _run(recorder, frame, simulate_ticks=True)

    assert recorder.events == _expected(frame, high_first=high_first)


def test_switch_off_delivers_no_ticks() -> None:
    """缺省关闭: on_tick 不触发, on_bar 照常."""
    frame = _frame()
    recorder = _Recorder()

    _run(recorder, frame)

    assert [e[0] for e in recorder.events] == ["bar"] * len(frame)


def test_same_time_bars_of_two_symbols_expand_together() -> None:
    """多标的同刻的 K 线一起展开: 先按(时间戳, 标的)排出全部模拟 tick, 再排出 K 线."""
    other = "600001"
    frame = pd.concat(
        [_frame(2), _frame(2, symbol=other, base=20.0)], ignore_index=True
    )
    recorder = _Recorder()

    _run(recorder, frame, simulate_ticks=True, symbols=[_SYMBOL, other])

    first_end = int(pd.Timestamp("2024-01-02 09:31:00", tz=_TZ).value)
    first_group = [e[:3] for e in recorder.events[:10]]
    expected = []
    for step in (4, 3, 2, 1):
        expected.append(("tick", _SYMBOL, first_end - step * _SEC))
        expected.append(("tick", other, first_end - step * _SEC))
    expected += [("bar", _SYMBOL, first_end), ("bar", other, first_end)]
    assert first_group == expected


def test_zero_volume_bar_gets_no_simulated_ticks() -> None:
    """成交量为 0 的 K 线(停牌占位)不生成模拟 tick."""
    frame = _frame(3)
    frame.loc[1, "volume"] = 0.0
    recorder = _Recorder()

    _run(recorder, frame, simulate_ticks=True)

    ticks_per_bar = []
    count = 0
    for event in recorder.events:
        if event[0] == "tick":
            count += 1
        else:
            ticks_per_bar.append(count)
            count = 0
    assert ticks_per_bar == [4, 0, 4]


def test_real_ticks_are_rejected_when_simulating() -> None:
    """数据里已有真实 tick 时不能再模拟."""
    stamp = int(pd.Timestamp("2024-01-02 09:31:00", tz=_TZ).value)
    data = [Tick(timestamp=stamp, price=10.0, volume=1.0, symbol=_SYMBOL)]

    with pytest.raises(ValueError, match="simulate_ticks"):
        _run(_Recorder(), data, simulate_ticks=True)


class _TimerProbe(_Recorder):
    """在第 2 根 K 线的第 3 个模拟 tick 的时刻登记一个定时器."""

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-02 09:31:58", tz=_TZ), "probe")

    def on_timer(self, payload: str) -> None:
        self.events.append(("timer", payload, 0, 0.0, 0.0))


def test_timer_fires_in_time_order_between_simulated_ticks() -> None:
    """定时器与模拟 tick 按时间先后处理; 同一时刻定时器在前(与现有回测规则一致)."""
    frame = _frame(3)
    probe = _TimerProbe()

    _run(probe, frame, simulate_ticks=True)

    second_end = int(pd.Timestamp("2024-01-02 09:32:00", tz=_TZ).value)
    kinds = [
        (e[0], e[2])
        for e in probe.events
        if e[0] == "timer" or second_end - 4 * _SEC <= e[2] <= second_end
    ]
    assert kinds == [
        ("tick", second_end - 4 * _SEC),
        ("tick", second_end - 3 * _SEC),
        ("timer", 0),
        ("tick", second_end - 2 * _SEC),
        ("tick", second_end - 1 * _SEC),
        ("bar", second_end),
    ]


# ---------------------------------------------------------------------------
# 对统计、计数、历史不可见(S13、S14)
# ---------------------------------------------------------------------------


class _BarTrader(Strategy):
    """只在 on_bar 里交易; on_tick 为空操作."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0
        self.history_in_tick: List[List[float]] = []
        self.equity_in_bar: List[float] = []

    def on_start(self) -> None:
        self.set_history_depth(5)
        self.sma = self.I(aq.SMA(3), name="sma3")

    def on_tick(self, tick: Tick) -> None:
        if self.seen >= 2:
            self.history_in_tick.append(
                [float(v) for v in self.get_history(2, tick.symbol)]
            )

    def on_bar(self, bar: Bar) -> None:
        self.seen += 1
        if self.seen == 2:
            self.buy(bar.symbol, 1000)
        if self.seen == 6:
            self.sell(bar.symbol, 1000)
        self.equity_in_bar.append(round(float(self.equity), 6))


def _fingerprint(result: Any) -> Tuple[Any, ...]:
    curve = result.equity_curve
    trades = result.trades_df
    return (
        [int(ts.value) for ts in curve.index],
        [round(float(v), 6) for v in curve.to_numpy()],
        trades[["entry_price", "exit_price", "quantity", "pnl"]]
        .round(6)
        .values.tolist(),
        round(float(result.metrics.total_return_pct), 9),
        round(float(result.metrics.max_drawdown_pct), 9),
    )


def _progress(events: List[Any]) -> List[int]:
    """进度流事件里的已处理 K 线数(即引擎的 bar_count)."""
    return [
        int(event.get("payload", {}).get("processed", "0"))
        for event in events
        if event.get("event_type") == "progress"
    ]


# Task 2(撮合通道)完成后, 这里补上 aq.NextClose 与 aq.NextAverage。
@pytest.mark.parametrize("fill_policy", [aq.NextOpen, aq.CurrentClose])
def test_noop_on_tick_leaves_results_unchanged(fill_policy: Any) -> None:
    """不变量: on_tick 为空时, 开关开与关的权益曲线、成交、指标逐项相同."""
    # SMA(3) 使推断的 warmup 为 3, on_bar 从下标 2 起触发: 卖单在下标 7 的 K 线上
    # 发出, NextOpen 要到下标 8 才成交, 故取 10 根。
    frame = _frame(10)
    off = _BarTrader()
    events_off: List[Any] = []
    result_off = _run(off, frame, fill_policy=fill_policy(), on_event=events_off.append)
    on = _BarTrader()
    events_on: List[Any] = []
    result_on = _run(
        on,
        frame,
        fill_policy=fill_policy(),
        on_event=events_on.append,
        simulate_ticks=True,
    )

    assert len(result_off.trades_df) == 1, "基线应完成一笔交易"
    assert _fingerprint(result_on) == _fingerprint(result_off)
    # 模拟 tick 不计 bar_count、不发进度与 tick 流事件。
    assert _progress(events_off), "基线应有进度事件"
    assert _progress(events_on) == _progress(events_off)
    assert not [e for e in events_on if e.get("event_type") == "tick"]
    assert on.equity_in_bar == off.equity_in_bar
    assert on.history_in_tick, "开关打开时 on_tick 应被调用"


def test_get_history_in_on_tick_holds_closed_bars_only() -> None:
    """模拟 tick 不写历史: on_tick 里不带 freq 取到的是已收盘 K 线的收盘价."""
    frame = _frame(6)
    trader = _BarTrader()

    _run(trader, frame, simulate_ticks=True)

    closes = [round(float(c), 6) for c in frame["close"]]
    # SMA(3) 使推断的 warmup 为 3: on_bar 从下标 2 起触发, seen 在下标 3 的 K 线上
    # 到 2, 于是从下标 4 的 K 线起记录。它的四个 tick 上, 最近两根已收盘的是下标
    # 2、3; 下标 5 的四个 tick 上是 3、4。
    assert trader.warmup_period == 3
    assert trader.history_in_tick[:4] == [closes[2:4]] * 4
    assert trader.history_in_tick[4:8] == [closes[3:5]] * 4
