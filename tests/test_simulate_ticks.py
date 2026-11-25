"""回测用 K 线模拟 tick: ``run_backtest(simulate_ticks=True)``."""

from typing import Any, List, Optional, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, Tick, run_backtest
from akquant.backtest.fill_mode import CurrentClose

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


def test_set_simulate_ticks_rejects_live_feed() -> None:
    """行情源是实盘数据源时不能打开开关(会绕过实盘的等待逻辑); 关闭照常."""
    engine = aq.Engine()
    engine.add_data(aq.DataFeed.create_live())

    with pytest.raises(RuntimeError, match="simulate_ticks"):
        engine.set_simulate_ticks(True)
    engine.set_simulate_ticks(False)


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
        [int(pd.Timestamp(t).value) for t in trades["entry_time"]],
        [int(pd.Timestamp(t).value) for t in trades["exit_time"]],
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


@pytest.mark.parametrize(
    "fill_policy", [aq.NextOpen, aq.CurrentClose, aq.NextClose, aq.NextAverage]
)
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


# ---------------------------------------------------------------------------
# 两条撮合通道(S8～S11)
# ---------------------------------------------------------------------------


class _TickLane(Strategy):
    """第 2 根 K 线买入; 之后按 ``action`` 在某个模拟 tick 上下单."""

    #: (第几根 K 线(从 1 起)的模拟 tick, 该根的第几个 tick(从 0 起))
    at: Tuple[int, int] = (4, 1)

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.tick_index = 0
        self.trigger: Optional[Tuple[int, float]] = None
        self.fills: List[Tuple[str, int, float]] = []

    def act(self, tick: Tick) -> None:
        raise NotImplementedError

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        self.tick_index = 0
        if self.bars_seen == 2:
            self.buy(bar.symbol, 1000)

    def on_tick(self, tick: Tick) -> None:
        if (self.bars_seen + 1, self.tick_index) == self.at:
            self.trigger = (int(tick.timestamp), round(float(tick.price), 6))
            self.act(tick)
        self.tick_index += 1

    def on_trade(self, trade: Any) -> None:
        side = "buy" if str(trade.side).endswith("Buy") else "sell"
        self.fills.append((side, int(trade.timestamp), round(float(trade.price), 6)))


class _MarketStop(_TickLane):
    def act(self, tick: Tick) -> None:
        self.sell(tick.symbol, 1000)


@pytest.mark.parametrize(
    "fill_policy", [aq.NextOpen, aq.CurrentClose, aq.NextClose, aq.NextAverage]
)
def test_market_order_from_on_tick_fills_on_that_tick(fill_policy: Any) -> None:
    """on_tick 里的市价单按触发它的模拟 tick 的时刻与价格成交, 不看运行级成交模式."""
    strategy = _MarketStop()

    _run(strategy, _frame(8), fill_policy=fill_policy(), simulate_ticks=True)

    assert strategy.trigger is not None
    sells = [f for f in strategy.fills if f[0] == "sell"]
    assert sells == [("sell", strategy.trigger[0], strategy.trigger[1])]
    # 第 4 根 K 线的第 2 个 tick 是最低价: open=13, low=12.8。
    assert strategy.trigger[1] == pytest.approx(12.8)


class _RestingLimit(_TickLane):
    """在第 4 根的开盘 tick 上挂低于现价的买入限价单(限价 12.9, 该根最低 12.8)."""

    at = (4, 0)

    def act(self, tick: Tick) -> None:
        self.buy(tick.symbol, 100, price=12.9)


def test_resting_limit_from_on_tick_fills_at_limit_price_on_a_later_tick() -> None:
    """挂单被之后的模拟 tick 越过时按限价成交(不给价格改善), 时刻是越过它的那个 tick."""
    strategy = _RestingLimit()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    buys = [f for f in strategy.fills if f[0] == "buy"]
    # 第一笔是 on_bar 里的买入; 第二笔是限价单: 下一个 tick(最低价 12.8)越过 12.9。
    assert len(buys) == 2
    assert buys[1] == ("buy", strategy.trigger[0] + _SEC, pytest.approx(12.9))


class _RestingStop(_TickLane):
    """在第 4 根的开盘 tick 上挂卖出止损单(触发价 12.9, 该根最低 12.8)."""

    at = (4, 0)

    def act(self, tick: Tick) -> None:
        self.sell(tick.symbol, 1000, trigger_price=12.9)


def test_resting_stop_from_on_tick_fills_at_the_triggering_tick_price() -> None:
    """止损单被模拟 tick 触发后按该 tick 的价格成交."""
    strategy = _RestingStop()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    sells = [f for f in strategy.fills if f[0] == "sell"]
    assert sells == [("sell", strategy.trigger[0] + _SEC, pytest.approx(12.8))]


class _LimitOnCloseTick(_TickLane):
    """在第 4 根的收盘 tick 上挂买入限价单(限价 12.85).

    限价高于该根最低 12.8, 但之后各根的最低价都高于它。
    """

    at = (4, 3)

    def act(self, tick: Tick) -> None:
        self.buy(tick.symbol, 100, price=12.85)


def test_order_from_on_tick_is_never_matched_by_a_bar() -> None:
    """模拟 tick 上下的单不被 K 线撮合: 否则会被紧随其后的本根 K 线(最低 12.8)成交."""
    strategy = _LimitOnCloseTick()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    # 只有 on_bar 里那一笔买入; 限价单始终没有成交。
    assert len([f for f in strategy.fills if f[0] == "buy"]) == 1


class _MarketableLimit(_TickLane):
    """在第 4 根的最低价 tick(12.8)上下买入限价 13.0: 当场可成交."""

    def act(self, tick: Tick) -> None:
        self.buy(tick.symbol, 100, price=13.0)


def test_marketable_limit_from_on_tick_fills_at_that_tick_price() -> None:
    """当场可成交的限价单按该 tick 的价格成交(不是限价)."""
    strategy = _MarketableLimit()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    buys = [f for f in strategy.fills if f[0] == "buy"]
    assert len(buys) == 2
    assert buys[1] == ("buy", strategy.trigger[0], pytest.approx(12.8))


class _StopLimit(_TickLane):
    """在第 4 根的开盘 tick 上挂卖出止损限价单(触发价 12.9)."""

    at = (4, 0)
    limit = 12.7
    trigger_at = 12.9

    def act(self, tick: Tick) -> None:
        self.sell(tick.symbol, 1000, price=self.limit, trigger_price=self.trigger_at)


def test_stop_limit_fills_at_the_triggering_tick_price() -> None:
    """止损限价单被触发的那个 tick 上按 tick 价成交(满足限价时), 不按限价.

    按限价 12.7 成交会低于这根 K 线的最低价 12.8。
    """
    strategy = _StopLimit()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    sells = [f for f in strategy.fills if f[0] == "sell"]
    assert sells == [("sell", strategy.trigger[0] + _SEC, pytest.approx(12.8))]


class _StopLimitRests(_StopLimit):
    limit = 12.85


def test_triggered_stop_limit_not_marketable_rests_as_limit() -> None:
    """触发的 tick(12.8)不满足限价 12.85 时不成交, 之后按挂着的限价单撮合."""
    strategy = _StopLimitRests()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    sells = [f for f in strategy.fills if f[0] == "sell"]
    # 下一个 tick 是最高价 13.6, 越过限价 12.85, 按限价成交。
    assert sells == [("sell", strategy.trigger[0] + 2 * _SEC, pytest.approx(12.85))]


# ---------------------------------------------------------------------------
# 订单回调的派发步(S8): 模拟 tick 步只在 tick 通道有未了结订单时派发
# ---------------------------------------------------------------------------


class _TradeReactor(Strategy):
    """on_tick 只计数; 第 2 根 on_bar 买入, 对买入成交在 on_trade 里卖出 100 股."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.ticks = 0
        self.sold = False
        self.trade_times: List[int] = []
        self.order_log: List[Tuple[str, str, int]] = []

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        if self.bars_seen == 2:
            self.buy(bar.symbol, 1000)

    def on_order(self, order: Any) -> None:
        assert self.ctx is not None
        self.order_log.append(
            (str(order.side), str(order.status), int(self.ctx.current_time))
        )

    def on_trade(self, trade: Any) -> None:
        assert self.ctx is not None
        self.trade_times.append(int(self.ctx.current_time))
        if not self.sold and str(trade.side).endswith("Buy"):
            self.sold = True
            self.sell(trade.symbol, 100)


@pytest.mark.parametrize(
    "fill_policy", [aq.NextOpen, aq.CurrentClose, aq.NextClose, aq.NextAverage]
)
def test_order_callbacks_without_tick_orders_match_switch_off(
    fill_policy: Any,
) -> None:
    """不在模拟 tick 上下单的策略: 订单回调的时刻与关闭开关时相同.

    回调里的单属 K 线通道。
    """
    frame = _frame(8)
    off = _TradeReactor()
    result_off = _run(off, frame, fill_policy=fill_policy())
    on = _TradeReactor()
    result_on = _run(on, frame, fill_policy=fill_policy(), simulate_ticks=True)

    assert off.sold, "基线应在 on_trade 里卖出"
    assert on.ticks > 0
    assert _fingerprint(result_on) == _fingerprint(result_off)
    assert on.trade_times == off.trade_times
    assert on.order_log == off.order_log
    assert list(result_on.orders_df["created_at"]) == list(
        result_off.orders_df["created_at"]
    )


class _BuyBackOnTrade(_MarketStop):
    """on_tick 在最低价 tick 上市价卖出; 对这笔卖出成交在 on_trade 里市价买回 100 股."""

    def __init__(self) -> None:
        super().__init__()
        self.bought_back = False

    def on_trade(self, trade: Any) -> None:
        super().on_trade(trade)
        if self.fills[-1][0] == "sell" and not self.bought_back:
            self.bought_back = True
            self.buy(trade.symbol, 100)


def test_callback_of_tick_order_is_dispatched_on_the_next_tick() -> None:
    """模拟 tick 上下的单, 回调在下一个 tick 步派发.

    回调里下的单在那个 tick 上当场成交。
    """
    strategy = _BuyBackOnTrade()

    _run(strategy, _frame(8), simulate_ticks=True)

    assert strategy.trigger is not None
    buys = [f for f in strategy.fills if f[0] == "buy"]
    # 第一笔是 on_bar 里的买入; 买回在最低价之后的 tick(最高价 13.6)上成交。
    assert len(buys) == 2
    assert buys[1] == ("buy", strategy.trigger[0] + _SEC, pytest.approx(13.6))


class _T1SellOnTick(_TickLane):
    """T+1 下当日买入后, 在模拟 tick 上卖出当日买入的股票."""

    def __init__(self) -> None:
        super().__init__()
        self.rejects: List[str] = []

    def act(self, tick: Tick) -> None:
        self.sell(tick.symbol, 1000)

    def on_order(self, order: Any) -> None:
        if str(order.status).endswith("Rejected"):
            self.rejects.append(str(order.reject_reason))


@pytest.mark.parametrize(
    "fill_policy", [aq.NextOpen, aq.CurrentClose, aq.NextClose, aq.NextAverage]
)
def test_tick_order_is_risk_checked_with_its_own_fill_policy(fill_policy: Any) -> None:
    """模拟 tick 上的单在提交时按改写后的成交模式过风控: T+1 卖单当场被拒."""
    strategy = _T1SellOnTick()

    _run(
        strategy,
        _frame(8),
        fill_policy=fill_policy(),
        simulate_ticks=True,
        t_plus_one=True,
    )

    assert strategy.trigger is not None
    assert len(strategy.rejects) == 1
    assert "Insufficient available position" in strategy.rejects[0]
    assert "at execution" not in strategy.rejects[0]


# ---------------------------------------------------------------------------
# 策略侧: 指标(S16)与未触发告警(S19)
# ---------------------------------------------------------------------------


def _indicator_points(events: List[Any]) -> List[Tuple[int, float, bool]]:
    out = []
    for event in events:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None and message["type"] == "point":
                point = message["indicator"]
                out.append((point["timestamp"], point["value"], point["confirmed"]))
    return out


class _Indicators(Strategy):
    intrabar = False

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma", intrabar=self.intrabar, plot=True)

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        pass


def test_base_indicator_is_advanced_by_bars_only() -> None:
    """基础周期指标只由 K 线推进: 确认点与关闭开关时逐点相同, 没有临时点."""
    frame = _frame(6)
    off: List[Any] = []
    _run(_Indicators(), frame, on_event=off.append)
    on: List[Any] = []
    strategy_on = _Indicators()
    _run(strategy_on, frame, on_event=on.append, simulate_ticks=True)

    assert strategy_on.ticks > 0
    assert _indicator_points(off), "基线应有指标点"
    assert _indicator_points(on) == _indicator_points(off)


def test_intrabar_indicator_previews_on_simulated_ticks() -> None:
    """Intrabar 指标在模拟 tick 上出临时点; 第四个 tick 上的临时值等于确认值."""

    class _Intrabar(_Indicators):
        intrabar = True

    frame = _frame(6)
    events: List[Any] = []
    _run(_Intrabar(), frame, on_event=events.append, simulate_ticks=True)

    points = _indicator_points(events)
    confirmed = {ts: value for ts, value, ok in points if ok}
    provisional = [(ts, value) for ts, value, ok in points if not ok]
    assert provisional, "应有临时点"
    assert all(ts in confirmed for ts, _ in provisional)
    last_preview = {ts: value for ts, value in provisional}
    for ts, value in last_preview.items():
        assert value == pytest.approx(confirmed[ts])


class _OverridesOnTick(Strategy):
    def on_tick(self, tick: Tick) -> None:
        pass

    def on_bar(self, bar: Bar) -> None:
        pass


class _BarOnly(Strategy):
    def on_bar(self, bar: Bar) -> None:
        pass


def _never_fired_warnings(caplog: Any) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "on_tick" in r.getMessage()
        and "simulate_ticks" in r.getMessage()
        and r.levelname == "WARNING"
    ]


def test_warns_when_on_tick_is_overridden_but_never_fires(caplog: Any) -> None:
    """覆写了 on_tick、整场回测一个 tick 都没收到: 告警一次并提示开关."""
    with caplog.at_level("WARNING"):
        _run(_OverridesOnTick(), _frame(4))

    assert len(_never_fired_warnings(caplog)) == 1


@pytest.mark.parametrize(
    ("strategy_cls", "simulate"),
    [(_OverridesOnTick, True), (_BarOnly, False)],
)
def test_no_warning_when_ticks_arrive_or_on_tick_is_not_overridden(
    caplog: Any, strategy_cls: Any, simulate: bool
) -> None:
    """开关打开时有 tick、未覆写 on_tick 时: 都不告警."""
    with caplog.at_level("WARNING"):
        _run(strategy_cls(), _frame(4), simulate_ticks=simulate)

    assert _never_fired_warnings(caplog) == []


def test_no_warning_when_real_ticks_are_fed(caplog: Any) -> None:
    """喂了真实 tick(聚合成 K 线)时 on_tick 会触发, 不告警."""
    stamps = pd.date_range("2024-01-02 09:31:00", periods=30, freq="10s", tz=_TZ)
    ticks = [
        Tick(timestamp=int(ts.value), price=10.0 + i * 0.01, volume=1.0, symbol=_SYMBOL)
        for i, ts in enumerate(stamps)
    ]
    with caplog.at_level("WARNING"):
        _run(_OverridesOnTick(), ticks)

    assert _never_fired_warnings(caplog) == []


# ---------------------------------------------------------------------------
# 开关打开却没有生成任何模拟 tick(S19 第二条)、标记复位、定时器、S6 与 S2
# ---------------------------------------------------------------------------

_NO_TICK_TEXT = "没有生成任何模拟 tick"


def _no_tick_warnings(caplog: Any) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if _NO_TICK_TEXT in r.getMessage() and r.levelname == "WARNING"
    ]


def _closing_stamped_daily(count: int = 3) -> pd.DataFrame:
    """收盘时刻(15:00)打戳的日线."""
    stamps = pd.date_range("2024-01-02 15:00:00", periods=count, freq="1D", tz=_TZ)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": _SYMBOL,
            "open": [10.0 + i for i in range(count)],
            "high": [10.6 + i for i in range(count)],
            "low": [9.8 + i for i in range(count)],
            "close": [10.4 + i for i in range(count)],
            "volume": 1000.0,
        }
    )


def test_warns_when_switch_on_but_no_tick_is_generated(caplog: Any) -> None:
    """15:00 打戳的日线不传 freq: 整根落在收盘集合竞价, 没有模拟 tick, 告警一次."""
    recorder = _Recorder()
    with caplog.at_level("WARNING"):
        _run(recorder, _closing_stamped_daily(), freq=None, simulate_ticks=True)

    assert [e for e in recorder.events if e[0] == "tick"] == []
    assert len(_no_tick_warnings(caplog)) == 1


def test_daily_bars_with_freq_get_ticks_and_no_warning(caplog: Any) -> None:
    """同样的日线传 freq='1d': 每根 K 线之前四个模拟 tick, 没有该告警."""
    recorder = _Recorder()
    with caplog.at_level("WARNING"):
        _run(recorder, _closing_stamped_daily(), freq="1d", simulate_ticks=True)

    kinds = [e[0] for e in recorder.events]
    assert kinds == ["tick", "tick", "tick", "tick", "bar"] * 3
    assert _no_tick_warnings(caplog) == []


def test_simulated_flag_is_reset_when_instance_is_reused() -> None:
    """同一策略实例先开后关: 第二次回测后标记复位为 False."""
    instance = _Recorder()
    _run(instance, _frame(3), simulate_ticks=True)
    assert getattr(instance, "_framework_simulated_ticks") is True

    _run(instance, _frame(3))
    assert getattr(instance, "_framework_simulated_ticks") is False


def test_simulated_flag_is_reset_when_instance_goes_live() -> None:
    """跑过模拟 tick 回测的实例再交给 run_live: 标记复位."""
    from akquant import run_live
    from akquant.akquant import AssetType, Instrument

    instance = _Recorder()
    _run(instance, _frame(3), simulate_ticks=True)
    assert getattr(instance, "_framework_simulated_ticks") is True

    idx = pd.date_range(
        "2024-01-02 09:31:00", periods=3, freq="1min", tz="Asia/Shanghai"
    )
    bars = [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, _SYMBOL)
        for i, ts in enumerate(idx)
    ]
    run_live(
        strategy_cls=instance,
        instruments=[
            Instrument(
                symbol=_SYMBOL,
                asset_type=AssetType.Stock,
                multiplier=1.0,
                margin_ratio=1.0,
                tick_size=0.01,
                lot_size=100,
                option_type=None,
                strike_price=None,
                expiry_date=None,
            )
        ],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": bars, "freq": "1min"},
        cash=1e6,
        show_progress=False,
        duration="30s",
    )

    assert getattr(instance, "_framework_simulated_ticks") is False


def test_simulated_flag_is_pickled() -> None:
    """存档带上模拟 tick 标记: 续跑据此识别并拒绝 simulate_ticks 回测的存档.

    回测跑完的实例仍持有引擎句柄, 整体不可 pickle(与本功能无关), 所以直接断言
    ``__getstate__`` 的结果, 再用它重建一个实例验证标记。
    """
    instance = _Recorder()
    _run(instance, _frame(3), simulate_ticks=True)
    assert getattr(instance, "_framework_simulated_ticks") is True

    state = instance.__getstate__()
    assert state.get("_framework_simulated_ticks") is True
    restored = _Recorder.__new__(_Recorder)
    restored.__setstate__(state)
    assert getattr(restored, "_framework_simulated_ticks") is True


class _TimerEquityProbe(Strategy):
    """第 1 根 K 线买入; 定时器登记在第 2 根 K 线的时间戳上, 记下权益."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[str, int]] = []
        self.timer_equity: Optional[float] = None
        self.timer_bar_is_none: Optional[bool] = None
        self.bought = False

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-02 09:32:00", tz=_TZ), "probe")

    def on_tick(self, tick: Tick) -> None:
        self.events.append(("tick", int(tick.timestamp)))

    def on_bar(self, bar: Bar) -> None:
        self.events.append(("bar", int(bar.timestamp)))
        if not self.bought:
            self.bought = True
            self.buy(bar.symbol, 100)

    def on_timer(self, payload: str) -> None:
        self.events.append(("timer", 0))
        self.timer_equity = round(float(self.equity), 6)
        self.timer_bar_is_none = self.current_bar is None


def test_timer_on_bar_timestamp_reads_prices_updated_by_simulated_ticks() -> None:
    """定时器恰在第 2 根 K 线时间戳上: 开关打开时读到本根收盘价, 关闭时读到上一根的."""
    frame = _frame(3)
    second_end = int(pd.Timestamp("2024-01-02 09:32:00", tz=_TZ).value)

    on = _TimerEquityProbe()
    _run(on, frame, simulate_ticks=True, fill_policy=CurrentClose())
    off = _TimerEquityProbe()
    _run(off, frame, fill_policy=CurrentClose())

    kinds = [e[0] for e in on.events]
    timer_pos = kinds.index("timer")
    assert kinds[timer_pos - 4 : timer_pos + 2] == ["tick"] * 4 + ["timer", "bar"]
    assert [e for e in on.events if e[0] == "bar"][1][1] == second_end
    # 买入在第 1 根 on_bar 下单, 按当根收盘价 10.4 成交;
    # 权益 = 现金 + 100 * (最新价 - 10.4)。
    # 第 1 根收盘 10.4、第 2 根收盘 11.4。
    assert on.timer_equity is not None and off.timer_equity is not None
    assert on.timer_equity == pytest.approx(1_000_000.0 + 100 * (11.4 - 10.4))
    assert off.timer_equity == pytest.approx(1_000_000.0 + 100 * (10.4 - 10.4))
    # 落在模拟 tick 之后、K 线之前: 打开时 current_bar 为空(文档写明)。
    assert on.timer_bar_is_none is True


def test_symbols_outside_whitelist_get_no_simulated_ticks() -> None:
    """白名单之外的标的不生成模拟 tick."""
    other = "600001"
    frame = pd.concat(
        [_frame(2), _frame(2, symbol=other, base=20.0)], ignore_index=True
    )
    recorder = _Recorder()

    _run(recorder, frame, simulate_ticks=True, symbols=[_SYMBOL])

    ticks = [e for e in recorder.events if e[0] == "tick"]
    assert len(ticks) == 8
    assert {e[1] for e in ticks} == {_SYMBOL}


def test_bar_outside_continuous_auction_gets_no_simulated_ticks() -> None:
    """09:30 打戳的 1 分钟线区间在开盘前: 没有 tick; 之后连续竞价的 K 线有四个."""
    frame = _frame(3, start="2024-01-02 09:30:00")
    recorder = _Recorder()

    _run(recorder, frame, simulate_ticks=True)

    first_end = int(pd.Timestamp("2024-01-02 09:30:00", tz=_TZ).value)
    before_first = [e for e in recorder.events if e[0] == "tick" and e[2] <= first_end]
    assert before_first == []
    assert [e[0] for e in recorder.events] == ["bar"] + (["tick"] * 4 + ["bar"]) * 2


def test_datafeed_with_real_tick_raises_in_engine() -> None:
    """DataFeed 输入里混有真实 tick: 引擎取到 tick 事件时报错."""
    feed = aq.DataFeed()
    first = int(pd.Timestamp("2024-01-02 09:31:00", tz=_TZ).value)
    for i in range(3):
        feed.add_bar(
            Bar(
                timestamp=first + i * 60 * _SEC,
                open=10.0,
                high=10.5,
                low=9.8,
                close=10.2,
                volume=1000.0,
                symbol=_SYMBOL,
            )
        )
    feed.add_tick(
        Tick(timestamp=first + 150 * _SEC, price=10.0, volume=1.0, symbol=_SYMBOL)
    )
    feed.sort()

    with pytest.raises(ValueError, match="simulate_ticks"):
        _run(_Recorder(), feed, simulate_ticks=True, symbols=[_SYMBOL])


class _TimerEquityNextOpen(_TimerEquityProbe):
    """同上, 缺省成交模式(NextOpen): 第 2 根开盘成交, 定时器登记在第 3 根的时间戳上."""

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-02 09:33:00", tz=_TZ), "probe")


def test_timer_on_bar_timestamp_reads_tick_prices_under_next_open() -> None:
    """缺省 NextOpen 下同样区分得出: 打开时按第 3 根收盘价计, 关闭时按第 2 根的."""
    frame = _frame(4)

    on = _TimerEquityNextOpen()
    _run(on, frame, simulate_ticks=True)
    off = _TimerEquityNextOpen()
    _run(off, frame)

    # 第 1 根 on_bar 买入 100 股, 第 2 根开盘价 11.0 成交; 权益 = 现金 + 100 * (最新价
    # - 11.0)。第 2 根收盘 11.4、第 3 根收盘 12.4。
    assert on.timer_equity == pytest.approx(1_000_000.0 + 100 * (12.4 - 11.0))
    assert off.timer_equity == pytest.approx(1_000_000.0 + 100 * (11.4 - 11.0))


def test_tick_count_is_reset_when_instance_is_reused(caplog: Any) -> None:
    """同一实例先开开关收到 tick, 再关开关跑一次: 第二次照样告警 on_tick 从未触发."""
    instance = _OverridesOnTick()
    _run(instance, _frame(3), simulate_ticks=True)
    assert getattr(instance, "_framework_tick_seen") is True

    with caplog.at_level("WARNING"):
        _run(instance, _frame(3))

    assert len(_never_fired_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# 不变量(S15): golden 场景之外, 跨交易日、多标的、成交回调与 on_start 定时器
# ---------------------------------------------------------------------------

_OTHER = "600001"


def _two_days(count: int = 4) -> pd.DataFrame:
    """两个交易日各 count 根 1 分钟线."""
    return pd.concat(
        [_frame(count), _frame(count, base=20.0, start="2024-01-03 09:31:00")],
        ignore_index=True,
    )


def _date_only_daily(count: int = 3) -> pd.DataFrame:
    """只有日期(本地 0 点)的日线: 引擎改写到收盘时刻, 区间为前 24 小时."""
    frame = _frame(count)
    frame["timestamp"] = pd.date_range("2024-01-02", periods=count, freq="1D", tz=_TZ)
    return frame


class _LifecycleProbe(Strategy):
    """记下生命周期回调与 on_bar 的时刻、交易日与权益; 第 2 根 K 线买入."""

    def __init__(self) -> None:
        super().__init__()
        self.records: List[Tuple[Any, ...]] = []
        self.bars = 0

    def _now(self) -> int:
        assert self.ctx is not None
        return int(self.ctx.current_time)

    def on_pre_open(self, event: Any) -> None:
        self.records.append(("pre_open", self._now(), str(self.trading_day)))

    def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
        self.records.append(("before", str(trading_date), int(timestamp), self._now()))

    def on_after_trading(self, trading_date: Any, timestamp: int) -> None:
        self.records.append(
            (
                "after",
                str(trading_date),
                int(timestamp),
                self._now(),
                round(float(self.equity), 6),
            )
        )

    def on_bar(self, bar: Bar) -> None:
        self.bars += 1
        if self.bars == 2:
            self.buy(bar.symbol, 100)
        self.records.append(("bar", int(bar.timestamp), str(self.trading_day)))


class _LifecycleProbeWithTick(_LifecycleProbe):
    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1


@pytest.mark.parametrize(
    ("frame", "freq"),
    [(_two_days(), "1min"), (_date_only_daily(), None)],
    ids=["minute_two_days", "date_only_daily"],
)
def test_lifecycle_hooks_across_trading_days_match_switch_off(
    frame: pd.DataFrame, freq: Optional[str]
) -> None:
    """跨交易日: 盘前、开始交易、收盘后回调的时刻与参数, 开与关逐项相同."""
    off = _LifecycleProbe()
    result_off = _run(off, frame, freq=freq)
    on = _LifecycleProbeWithTick()
    result_on = _run(on, frame, freq=freq, simulate_ticks=True)

    assert on.ticks == 4 * len(frame)
    kinds = [entry[0] for entry in off.records]
    assert kinds.count("before") >= 2, "基线应跨至少两个交易日"
    assert kinds.count("after") >= 1
    assert on.records == off.records
    assert _order_rows(result_on) == _order_rows(result_off)
    assert list(result_on.equity_curve.items()) == list(result_off.equity_curve.items())
    assert result_on.trading_day_ends == result_off.trading_day_ends


class _TwoSymbolRebalancer(Strategy):
    """多标的: 在第一个标的的 on_bar 里按目标市值调第二个标的, 记下当时的权益."""

    def __init__(self) -> None:
        super().__init__()
        self.equity_in_bar: List[Tuple[str, int, float]] = []
        self.seen = 0

    def on_bar(self, bar: Bar) -> None:
        self.equity_in_bar.append(
            (str(bar.symbol), int(bar.timestamp), round(float(self.equity), 6))
        )
        if bar.symbol != _SYMBOL:
            return
        self.seen += 1
        if self.seen == 2:
            self.order_target_value(target_value=100_000.0, symbol=_OTHER)
        if self.seen == 3:
            # 下一根(第二个标的停牌)撮合不了, 按停牌拒单
            self.order_target_value(target_value=50_000.0, symbol=_OTHER)
        if self.seen == 5:
            self.order_target_value(target_value=0.0, symbol=_OTHER)


class _TwoSymbolRebalancerWithTick(_TwoSymbolRebalancer):
    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1


def _two_symbols_with_suspension() -> pd.DataFrame:
    """两个标的同刻的 1 分钟线; 第二个标的第 4 根停牌(成交量为 0)."""
    other = _frame(6, symbol=_OTHER, base=20.0)
    other.loc[3, "volume"] = 0.0
    return pd.concat([_frame(6), other], ignore_index=True)


def _order_rows(result: Any) -> List[Tuple[Any, ...]]:
    orders = result.orders_df.sort_values("created_at")
    return [
        (
            int(pd.Timestamp(row["created_at"]).value),
            str(row["status"]),
            float(row["quantity"]),
            float(row["filled_quantity"]),
        )
        for row in orders.to_dict("records")
    ]


def test_two_symbols_with_suspension_match_switch_off() -> None:
    """多标的(含停牌): on_bar 里读到的权益、按最新价定的下单数量与关闭时相同.

    同刻全部标的的模拟 tick 排在 K 线之前; K 线到来时其余标的的最新价不能停在
    它们自己的模拟 tick 价上(关闭开关时那里还是上一根的收盘价)。
    """
    frame = _two_symbols_with_suspension()
    symbols = [_SYMBOL, _OTHER]
    off = _TwoSymbolRebalancer()
    result_off = _run(off, frame, symbols=symbols)
    on = _TwoSymbolRebalancerWithTick()
    result_on = _run(on, frame, symbols=symbols, simulate_ticks=True)

    rows_off = _order_rows(result_off)
    assert on.ticks > 0
    assert len(rows_off) == 3
    assert "rejected" in rows_off[1][1].lower(), "停牌那根应拒单"
    assert _order_rows(result_on) == rows_off
    assert on.equity_in_bar == off.equity_in_bar
    assert _fingerprint(result_on) == _fingerprint(result_off)


class _TradeAccountProbe(Strategy):
    """第 1 根 K 线买入(缺省 NextOpen, 第 2 根开盘成交); on_trade 里记账户权益."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.in_trade: List[Tuple[int, float, float]] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.bought:
            self.bought = True
            self.buy(bar.symbol, 1000)

    def on_trade(self, trade: Any) -> None:
        assert self.ctx is not None
        self.in_trade.append(
            (
                int(self.ctx.current_time),
                round(float(self.equity), 6),
                round(float(self.get_account()["equity"]), 6),
            )
        )


class _TradeAccountProbeWithTick(_TradeAccountProbe):
    def on_tick(self, tick: Tick) -> None:
        pass


def test_account_seen_in_on_trade_matches_switch_off() -> None:
    """on_trade 里的账户口径(更新本根价格之前的快照)与关闭开关时相同."""
    frame = _frame(4)
    off = _TradeAccountProbe()
    _run(off, frame)
    on = _TradeAccountProbeWithTick()
    _run(on, frame, simulate_ticks=True)

    second_end = int(pd.Timestamp("2024-01-02 09:32:00", tz=_TZ).value)
    # 第 2 根开盘价 11.0 买入 1000 股, 按上一根收盘价 10.4 计价。
    expected = round(1_000_000.0 + 1000 * (10.4 - 11.0), 6)
    assert off.in_trade == [(second_end, expected, expected)]
    assert on.in_trade == off.in_trade


class _EarlyDailyTimer(Strategy):
    """on_start 里登记每日 09:30:58 的定时器(首根 K 线 09:31 之前 2 秒)."""

    def __init__(self) -> None:
        super().__init__()
        self.fired: List[int] = []

    def on_start(self) -> None:
        self.schedule_daily("09:30:58", "early")

    def on_timer(self, payload: str) -> None:
        assert self.ctx is not None
        self.fired.append(int(self.ctx.current_time))

    def on_bar(self, bar: Bar) -> None:
        pass


class _EarlyDailyTimerWithTick(_EarlyDailyTimer):
    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1


def test_on_start_timers_are_registered_against_the_first_bar() -> None:
    """on_start 里登记的定时任务按首根 K 线跳过已过的时点, 不按它前面的模拟 tick."""
    frame = _two_days(3)
    off = _EarlyDailyTimer()
    _run(off, frame)
    on = _EarlyDailyTimerWithTick()
    _run(on, frame, simulate_ticks=True)

    assert on.ticks > 0
    # 首日 09:30:58 早于首根 K 线(09:31), 跳过; 次日照常触发。
    assert off.fired == [int(pd.Timestamp("2024-01-03 09:30:58", tz=_TZ).value)]
    assert on.fired == off.fired


def _midnight_boundary() -> pd.DataFrame:
    """全天连续时段下, 后一交易日的第一根 K 线打在前一交易日收盘(0 点)后 2 秒."""
    stamps = [
        "2024-01-02 10:00:00",
        "2024-01-02 23:58:00",
        "2024-01-03 00:00:02",
        "2024-01-03 10:00:00",
        "2024-01-04 10:00:00",
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps).tz_localize(_TZ),
            "symbol": _SYMBOL,
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 100.0,
        }
    )


class _BoundaryProbe(Strategy):
    """第 1 根买入 1000 股; 收盘后回调里记下交易日与权益."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.after: List[Tuple[str, float]] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.bought:
            self.bought = True
            self.buy(bar.symbol, 1000)

    def on_after_trading(self, trading_date: Any, timestamp: int) -> None:
        self.after.append((str(trading_date), round(float(self.equity), 6)))


class _BoundaryProbeWithTick(_BoundaryProbe):
    def __init__(self) -> None:
        super().__init__()
        self.tick_times: List[int] = []

    def on_tick(self, tick: Tick) -> None:
        self.tick_times.append(int(tick.timestamp))


def test_simulated_ticks_never_reach_back_into_the_previous_trading_day() -> None:
    """K 线晚于当前交易日收盘时, 它的四个模拟 tick 都严格晚于该收盘(spec S5)."""
    frame = _midnight_boundary()
    kwargs: Any = {
        "freq": None,
        "sessions": {"default": "ALWAYS_OPEN"},
        "initial_cash": 100_000.0,
    }
    off = _BoundaryProbe()
    _run(off, frame, **kwargs)
    on = _BoundaryProbeWithTick()
    _run(on, frame, simulate_ticks=True, **kwargs)

    # 23:58 开盘价 10.0 买入 1000 股, 01-02 收盘按 10.2 计价。
    assert off.after[0] == ("2024-01-02", 100_200.0)
    assert on.after == off.after
    close = int(pd.Timestamp("2024-01-03 00:00:00", tz=_TZ).value)
    bar_end = int(pd.Timestamp("2024-01-03 00:00:02", tz=_TZ).value)
    previous_bar = int(pd.Timestamp("2024-01-02 23:58:00", tz=_TZ).value)
    boundary_ticks = [t for t in on.tick_times if previous_bar < t < bar_end]
    assert len(boundary_ticks) == 4
    assert all(close < t < bar_end for t in boundary_ticks)


class _PortfolioProbe(Strategy):
    """两个标的, 第 1 根买入第一个标的; on_portfolio_update 里记下回调所在的步与快照."""

    def __init__(self) -> None:
        super().__init__()
        self.bought = False
        self.updates: List[Tuple[Any, ...]] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.bought and bar.symbol == _SYMBOL:
            self.bought = True
            self.buy(bar.symbol, 1000)

    def on_portfolio_update(self, snapshot: Any) -> None:
        self.updates.append(
            (
                self.current_bar is not None,
                int(snapshot["timestamp"]),
                round(float(snapshot["cash"]), 6),
                round(float(snapshot["equity"]), 6),
                tuple(sorted(snapshot["positions"].items())),
            )
        )


class _PortfolioProbeWithTick(_PortfolioProbe):
    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1


def test_portfolio_updates_on_bar_steps_match_switch_off() -> None:
    """K 线步上的 on_portfolio_update(次数、时刻、快照)与关闭开关时相同.

    模拟 tick 步上多出的回调属 spec S17, 不在比较之列。
    """
    frame = pd.concat(
        [_frame(6), _frame(6, symbol=_OTHER, base=20.0)], ignore_index=True
    )
    symbols = [_SYMBOL, _OTHER]
    off = _PortfolioProbe()
    _run(off, frame, symbols=symbols)
    on = _PortfolioProbeWithTick()
    _run(on, frame, symbols=symbols, simulate_ticks=True)

    assert on.ticks > 0
    assert off.updates, "基线应有组合更新回调"
    assert all(entry[0] for entry in off.updates)
    assert [entry for entry in on.updates if entry[0]] == off.updates


# ---------------------------------------------------------------------------
# 函数式策略的 on_tick 未触发告警(S19)
# ---------------------------------------------------------------------------


def test_functional_strategy_without_on_tick_does_not_warn(caplog: Any) -> None:
    """函数式写法只给 on_bar 函数、开关关闭: 不告警 on_tick 从未触发."""

    def on_bar(ctx: Any, bar: Bar) -> None:
        pass

    with caplog.at_level("WARNING"):
        _run(on_bar, _frame(4))

    assert _never_fired_warnings(caplog) == []


def test_functional_strategy_with_on_tick_warns_once(caplog: Any) -> None:
    """函数式写法给了 on_tick 函数、数据只有 K 线、开关关闭: 告警一次."""
    ticks: List[int] = []

    def on_bar(ctx: Any, bar: Bar) -> None:
        pass

    def on_tick(ctx: Any, tick: Tick) -> None:
        ticks.append(int(tick.timestamp))

    with caplog.at_level("WARNING"):
        _run(on_bar, _frame(4), on_tick=on_tick)

    assert ticks == []
    assert len(_never_fired_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# K 线步的订单回调与定时器里, current_bar / current_tick 与关闭开关时相同(S15)
# ---------------------------------------------------------------------------

_FILL_POLICIES = [aq.NextOpen, aq.CurrentClose, aq.NextClose, aq.NextAverage]


def _bars(rows: List[Tuple[float, float, float, float]]) -> pd.DataFrame:
    """按 (开, 高, 低, 收) 逐根生成 1 分钟线, 首根 09:31."""
    stamps = pd.date_range(
        "2024-01-02 09:31:00", periods=len(rows), freq="1min", tz=_TZ
    )
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "symbol": _SYMBOL,
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1000.0,
        }
    )


def _stop_from_trade_frame() -> pd.DataFrame:
    """第 3 根收盘大涨到 12.0, 第 4 根最低 11.2, 第 6 根跌到 9.0."""
    return _bars(
        [
            (10.0, 10.2, 9.9, 10.0),
            (10.0, 10.2, 9.9, 10.0),
            (10.0, 12.1, 9.9, 12.0),
            (12.0, 12.1, 11.2, 11.5),
            (11.5, 11.6, 11.4, 11.5),
            (11.5, 11.6, 9.0, 9.2),
            (9.2, 9.3, 9.1, 9.2),
            (9.2, 9.3, 9.1, 9.2),
        ]
    )


class _StopFromTrade(Strategy):
    """第 2 根 on_bar 买入; on_trade 里按 ``self.close * 0.95`` 挂止损."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.ticks = 0
        self.stop_set = False

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        if self.bars_seen == 2:
            self.buy(bar.symbol, 1000)

    def on_trade(self, trade: Any) -> None:
        if not self.stop_set and str(trade.side).endswith("Buy"):
            self.stop_set = True
            self.sell(trade.symbol, 1000, trigger_price=round(self.close * 0.95, 4))


@pytest.mark.parametrize("fill_policy", _FILL_POLICIES)
def test_stop_priced_from_self_close_in_on_trade_matches_switch_off(
    fill_policy: Any,
) -> None:
    """on_trade 里按 self.close 定止损价: 开与关的成交、止损价逐项相同."""
    frame = _stop_from_trade_frame()
    off = _StopFromTrade()
    result_off = _run(off, frame, fill_policy=fill_policy())
    on = _StopFromTrade()
    result_on = _run(on, frame, fill_policy=fill_policy(), simulate_ticks=True)

    assert on.ticks > 0
    assert off.stop_set and on.stop_set
    assert len(result_off.trades_df) == 1, "基线的止损应成交"
    assert _fingerprint(result_on) == _fingerprint(result_off)

    def stops(result: Any) -> List[float]:
        orders = result.orders_df.sort_values("created_at")
        return [float(v) for v in orders["stop_price"].dropna()]

    # 关闭开关时 on_trade 里的 self.close 是第 2 根的收盘价 10.0。
    assert stops(result_off) == [pytest.approx(9.5)]
    assert stops(result_on) == stops(result_off)


class _CallbackStateProbe(Strategy):
    """第 2 根买入、第 4 根卖出; 两根 K 线中间登记一个定时器.

    在 on_order、on_trade、on_timer 里记下当时可见的行情状态。
    """

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.ticks = 0
        self.records: List[Tuple[Any, ...]] = []

    def on_start(self) -> None:
        self.schedule(pd.Timestamp("2024-01-02 09:33:30", tz=_TZ), "mid")

    def _record(self, name: str) -> None:
        assert self.ctx is not None
        bar = self.current_bar
        self.records.append(
            (
                name,
                int(self.ctx.current_time),
                bar is None,
                None if bar is None else int(bar.timestamp),
                round(float(self.close), 6),
                self.current_tick is None,
            )
        )

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        if self.bars_seen == 2:
            self.buy(bar.symbol, 1000)
        if self.bars_seen == 4:
            self.sell(bar.symbol, 1000)

    def on_order(self, order: Any) -> None:
        self._record("on_order")

    def on_trade(self, trade: Any) -> None:
        self._record("on_trade")

    def on_timer(self, payload: str) -> None:
        self._record("on_timer")


@pytest.mark.parametrize("fill_policy", _FILL_POLICIES)
def test_bar_and_tick_seen_in_callbacks_match_switch_off(fill_policy: Any) -> None:
    """on_order、on_trade 与两根 K 线中间的 on_timer: 可见的行情状态与关闭时相同."""
    frame = _frame(8)
    off = _CallbackStateProbe()
    _run(off, frame, fill_policy=fill_policy())
    on = _CallbackStateProbe()
    _run(on, frame, fill_policy=fill_policy(), simulate_ticks=True)

    assert on.ticks > 0
    names = [entry[0] for entry in off.records]
    assert {"on_order", "on_trade", "on_timer"} <= set(names)
    assert on.records == off.records


# ---------------------------------------------------------------------------
# on_tick 里的 place_bracket: 进场在该 tick 成交后, 止损止盈子单照常挂出
# ---------------------------------------------------------------------------


class _BracketOnTick(Strategy):
    """首个模拟 tick(第 1 根开盘 10.0)上 place_bracket: 止损 9.9, 止盈 10.85."""

    def __init__(self) -> None:
        super().__init__()
        self.placed = False
        self.ticks = 0

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1
        if not self.placed:
            self.placed = True
            self.place_bracket(
                tick.symbol, 1000, stop_trigger_price=9.9, take_profit_price=10.85
            )

    def on_bar(self, bar: Bar) -> None:
        pass


def _order_table(result: Any) -> List[Tuple[Any, ...]]:
    """(方向, 类型, 状态, 成交量, 均价), 按前三项排序(子单与进场单同一创建时刻)."""
    rows = [
        (
            str(row["side"]),
            str(row["order_type"]),
            str(row["status"]),
            float(row["filled_quantity"]),
            round(float(row["avg_price"]), 6) if row["filled_quantity"] else None,
        )
        for row in result.orders_df.to_dict("records")
    ]
    return sorted(rows, key=lambda row: row[:3])


def test_bracket_from_on_tick_places_its_exit_orders() -> None:
    """进场在该 tick 成交后两张子单都被接受; 最低价 tick 9.8 触发止损, 止盈被撤."""
    strategy = _BracketOnTick()

    result = _run(strategy, _frame(4), simulate_ticks=True)

    assert strategy.ticks > 0
    rows = _order_table(result)
    assert ("buy", "market", "filled", 1000.0, 10.0) in rows
    assert ("sell", "stopmarket", "filled", 1000.0, 9.8) in rows
    assert ("sell", "limit", "cancelled", 0.0, None) in rows
    assert len(rows) == 3
    assert not [r for r in rows if r[2] == "rejected"]
    assert strategy.get_position(_SYMBOL) == 0


def test_bracket_from_on_tick_under_t1_rejects_exits_at_submission() -> None:
    """T+1: 模拟 tick 上的子单照旧在提交时按可用持仓校验, 当日买入不可卖而被拒."""
    strategy = _BracketOnTick()

    result = _run(strategy, _frame(4), simulate_ticks=True, t_plus_one=True)

    rows = _order_table(result)
    assert rows == [
        ("buy", "market", "filled", 1000.0, 10.0),
        ("sell", "limit", "rejected", 0.0, None),
        ("sell", "stopmarket", "rejected", 0.0, None),
    ]
    reasons = [
        str(row["reject_reason"])
        for row in result.orders_df.to_dict("records")
        if str(row["status"]) == "rejected"
    ]
    assert all(
        "Insufficient available position" in reason and "at execution" not in reason
        for reason in reasons
    )
    assert strategy.get_position(_SYMBOL) == 1000


class _BracketOnBar(Strategy):
    """关闭开关: 第 1 根 on_bar 里 place_bracket(止损 10.3, 止盈 11.5)."""

    def __init__(self) -> None:
        super().__init__()
        self.placed = False
        self.rejects: List[str] = []

    def on_bar(self, bar: Bar) -> None:
        if not self.placed:
            self.placed = True
            self.place_bracket(
                bar.symbol, 1000, stop_trigger_price=10.3, take_profit_price=11.5
            )

    def on_order(self, order: Any) -> None:
        if str(order.status).endswith("Rejected"):
            self.rejects.append(str(order.reject_reason))


def test_bracket_on_bar_under_current_close_exits_from_next_bar() -> None:
    """关闭开关、CurrentClose: 进场当根成交后两张子单被接受, 从下一根 K 线起撮合.

    第 1 根最低价 9.8 低于止损 10.3, 但那是进场(收盘 10.4)之前的价格, 不触发;
    第 2 根最高价 11.6 触及止盈 11.5, 止盈成交、止损被撤。
    """
    strategy = _BracketOnBar()

    result = _run(strategy, _frame(4), fill_policy=CurrentClose())

    assert _order_table(result) == [
        ("buy", "market", "filled", 1000.0, 10.4),
        ("sell", "limit", "filled", 1000.0, 11.5),
        ("sell", "stopmarket", "cancelled", 0.0, None),
    ]
    assert strategy.rejects == []
    fills = result.executions_df.to_dict("records")
    assert [str(row["side"]) for row in fills] == ["buy", "sell"]
    assert (
        int(pd.Timestamp(fills[1]["timestamp"]).value)
        - int(pd.Timestamp(fills[0]["timestamp"]).value)
        == 60 * _SEC
    )
    assert strategy.get_position(_SYMBOL) == 0


# ---------------------------------------------------------------------------
# 存档续跑不支持模拟 tick: 显式报错, 不静默停掉 on_tick
# ---------------------------------------------------------------------------


def _checkpoint(tmp_path: Any, strategy: Any, *, simulate: bool) -> str:
    from akquant import save_checkpoint

    result = _run(strategy, _frame(3), simulate_ticks=simulate)
    assert result.engine is not None
    path = str(tmp_path / "part1.ckpt")
    save_checkpoint(result.engine, result.strategy, path)
    return path


def _resume(path: str, **kwargs: Any) -> Any:
    from akquant import run_from_checkpoint

    later = _frame(3, start="2024-01-02 09:41:00")
    return run_from_checkpoint(
        path, data=later, symbols=[_SYMBOL], show_progress=False, **kwargs
    )


@pytest.mark.parametrize("value", [True, False])
def test_resume_rejects_simulate_ticks_argument(tmp_path: Any, value: bool) -> None:
    """续跑时传 simulate_ticks(无论真假): ValueError."""
    path = _checkpoint(tmp_path, _Recorder(), simulate=False)

    with pytest.raises(ValueError, match="simulate_ticks"):
        _resume(path, simulate_ticks=value)


def test_resume_rejects_checkpoint_of_a_simulated_tick_backtest(tmp_path: Any) -> None:
    """第一段 simulate_ticks=True 的存档拿去续跑: ValueError."""
    strategy = _Recorder()
    path = _checkpoint(tmp_path, strategy, simulate=True)
    assert [e for e in strategy.events if e[0] == "tick"], "第一段应收到模拟 tick"

    with pytest.raises(ValueError, match="simulate_ticks"):
        _resume(path)


def test_resume_of_plain_checkpoint_still_runs(tmp_path: Any) -> None:
    """关闭开关的存档照常续跑, 续跑段不产生 tick."""
    strategy = _Recorder()
    path = _checkpoint(tmp_path, strategy, simulate=False)

    result = _resume(path)

    resumed = result.strategy
    assert resumed is not None
    assert getattr(resumed, "_framework_simulated_ticks") is False
    kinds = [e[0] for e in resumed.events]
    assert kinds == ["bar"] * 6


def test_resume_warns_when_on_tick_never_fires(tmp_path: Any, caplog: Any) -> None:
    """续跑段覆写了 on_tick 却一个 tick 都没收到: 与 run_backtest 一样告警一次."""
    path = _checkpoint(tmp_path, _OverridesOnTick(), simulate=False)
    caplog.clear()

    with caplog.at_level("WARNING"):
        _resume(path)

    assert len(_never_fired_warnings(caplog)) == 1


# ---------------------------------------------------------------------------
# 文档里的 tick 通道口径: 不受成交量上限约束、跳空时限价单按限价成交
# ---------------------------------------------------------------------------


class _BuyOnce(Strategy):
    """在首个模拟 tick 上市价买入 1000 股.

    ``on_tick_side=False`` 时改在第 1 根 on_bar 里买。
    """

    on_tick_side = True

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0
        self.placed = False
        self.fills: List[Tuple[int, float]] = []

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1
        if self.on_tick_side and not self.placed:
            self.placed = True
            self.buy(tick.symbol, 1000)

    def on_bar(self, bar: Bar) -> None:
        if not self.on_tick_side and not self.placed:
            self.placed = True
            self.buy(bar.symbol, 1000)

    def on_trade(self, trade: Any) -> None:
        self.fills.append((int(trade.timestamp), float(trade.quantity)))


class _BuyOnceOnBar(_BuyOnce):
    on_tick_side = False


def test_tick_lane_is_not_capped_by_volume_limit() -> None:
    """量限 10%: on_tick 的市价单当场全部成交; 同样的单放在 on_bar 里被截断."""
    frame = _frame(4)
    on_tick = _BuyOnce()
    _run(on_tick, frame, simulate_ticks=True, volume_limit_pct=0.1)
    on_bar = _BuyOnceOnBar()
    _run(on_bar, frame, simulate_ticks=True, volume_limit_pct=0.1)

    assert on_tick.ticks > 0
    first_tick = int(pd.Timestamp("2024-01-02 09:31:00", tz=_TZ).value) - 4 * _SEC
    assert on_tick.fills == [(first_tick, 1000.0)]
    # K 线通道每根最多成交 1000 * 10% = 100 股。
    assert on_bar.fills, "on_bar 的单应部分成交"
    assert all(quantity <= 100.0 for _, quantity in on_bar.fills)
    assert sum(quantity for _, quantity in on_bar.fills) < 1000.0


class _LimitAcrossGap(_TickLane):
    """在第 3 根的收盘 tick(12.4)上挂买入限价 11.0; 第 4 根开盘跳空到 10.5."""

    at = (3, 3)

    def act(self, tick: Tick) -> None:
        self.buy(tick.symbol, 100, price=11.0)


def test_resting_limit_fills_at_limit_price_across_a_gap() -> None:
    """K 线之间跳空越过限价: 挂着的限价单在下一根的开盘 tick 上按限价成交."""
    frame = _frame(6)
    frame.loc[3, ["open", "high", "low", "close"]] = [10.5, 11.2, 10.3, 10.9]
    strategy = _LimitAcrossGap()

    _run(strategy, frame, simulate_ticks=True)

    assert strategy.trigger is not None
    assert strategy.trigger[1] == pytest.approx(12.4)
    buys = [f for f in strategy.fills if f[0] == "buy"]
    fourth_open = int(pd.Timestamp("2024-01-02 09:34:00", tz=_TZ).value) - 4 * _SEC
    # 第一笔是 on_bar 里的买入; 第二笔按限价 11.0 成交, 不按开盘 tick 价 10.5。
    assert len(buys) == 2
    assert buys[1] == ("buy", fourth_open, pytest.approx(11.0))


# ---------------------------------------------------------------------------
# intrabar 基础周期指标: K 线步的订单回调里读到上一根的确认值(S15)
# ---------------------------------------------------------------------------


class _IntrabarStopFromTrade(Strategy):
    """第 3 根 on_bar 买入; on_trade 里按 ``sma[0] * 0.97`` 挂止损并记下读数."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.ticks = 0
        self.stop_set = False
        self.records: List[Tuple[Any, ...]] = []

    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma", intrabar=True)

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        if self.bars_seen == 3:
            self.buy(bar.symbol, 1000)

    def on_trade(self, trade: Any) -> None:
        assert self.ctx is not None
        self.records.append(
            (
                int(self.ctx.current_time),
                round(float(self.sma[0]), 6),
                bool(self.sma.confirmed),
                round(float(self.close), 6),
            )
        )
        if not self.stop_set and str(trade.side).endswith("Buy"):
            self.stop_set = True
            self.sell(
                trade.symbol, 1000, trigger_price=round(float(self.sma[0]) * 0.97, 2)
            )


@pytest.mark.parametrize("fill_policy", [aq.NextOpen, aq.CurrentClose])
def test_intrabar_base_indicator_in_order_callbacks_matches_switch_off(
    fill_policy: Any,
) -> None:
    """on_trade 里读 intrabar 基础周期指标: 读数与按它定的止损价与关闭时相同."""
    frame = _frame(10)
    # 第 8 根起大幅下跌, 让止损单成交(成交后 _fingerprint 才有成交行可比)。
    frame.loc[7:, ["open", "high", "low", "close"]] = [5.0, 5.1, 4.9, 5.0]
    off = _IntrabarStopFromTrade()
    result_off = _run(off, frame, fill_policy=fill_policy())
    on = _IntrabarStopFromTrade()
    result_on = _run(on, frame, fill_policy=fill_policy(), simulate_ticks=True)

    assert on.ticks > 0
    assert off.stop_set and on.stop_set
    assert off.records, "基线的 on_trade 应被调用"
    assert on.records == off.records
    assert _fingerprint(result_on) == _fingerprint(result_off)

    def stops(result: Any) -> List[float]:
        orders = result.orders_df.sort_values("created_at")
        return [float(v) for v in orders["stop_price"].dropna()]

    assert stops(result_off)
    assert stops(result_on) == stops(result_off)


class _TwoSymbolIntrabar(Strategy):
    """标的 A 的 on_bar 里读标的 B 的 intrabar 指标: B 的 K 线此刻还没处理."""

    def __init__(self) -> None:
        super().__init__()
        self.ticks = 0
        self.records: List[Tuple[Any, ...]] = []

    def on_start(self) -> None:
        self.sma = self.I(factory=lambda: aq.SMA(2), name="sma", intrabar=True)

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        if str(bar.symbol) != _SYMBOL:
            return
        other = self.sma.of(_OTHER)
        value = other[0]
        self.records.append(
            (
                int(bar.timestamp),
                None if value is None else round(float(value), 6),
                bool(other.confirmed),
            )
        )


def test_intrabar_other_symbol_value_in_on_bar_matches_switch_off() -> None:
    """多标的: A 的 on_bar 里读 B 的指标, 读到 B 上一根的确认值, 与关闭时相同."""
    frame = pd.concat([_frame(8), _frame(8, symbol=_OTHER, base=20.0)])
    symbols = [_SYMBOL, _OTHER]
    off = _TwoSymbolIntrabar()
    _run(off, frame, symbols=symbols)
    on = _TwoSymbolIntrabar()
    _run(on, frame, symbols=symbols, simulate_ticks=True)

    assert on.ticks > 0
    assert any(entry[1] is not None for entry in off.records)
    assert all(entry[2] for entry in off.records)
    assert on.records == off.records


class _WindowIntrabar(Strategy):
    """窗口周期 intrabar 指标: 在 on_bar 里记 ``[0]`` 与 confirmed."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.ticks = 0
        self.records: List[Tuple[Any, ...]] = []

    def on_start(self) -> None:
        self.sma5 = self.I(aq.SMA(2), name="sma5", freq="5min", intrabar=True)

    def on_tick(self, tick: Tick) -> None:
        self.ticks += 1

    def on_bar(self, bar: Bar) -> None:
        self.records.append(
            (int(bar.timestamp), self.sma5[0], bool(self.sma5.confirmed))
        )


def test_window_intrabar_indicator_in_on_bar_matches_switch_off() -> None:
    """窗口周期指标的临时值不受影响: 读数与关闭开关时相同, 且确有临时值."""
    frame = _frame(24)
    off = _WindowIntrabar()
    _run(off, frame)
    on = _WindowIntrabar()
    _run(on, frame, simulate_ticks=True)

    assert on.ticks > 0
    assert any(not entry[2] for entry in off.records), "基线应读到窗口临时值"
    assert on.records == off.records
