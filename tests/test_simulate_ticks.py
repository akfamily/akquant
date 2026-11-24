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
    """没有 on_tick; 第 2 根 on_bar 买入, 对买入成交在 on_trade 里卖出 100 股."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_seen = 0
        self.sold = False
        self.trade_times: List[int] = []
        self.order_log: List[Tuple[str, str, int]] = []

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

    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma", intrabar=self.intrabar, plot=True)

    def on_tick(self, tick: Tick) -> None:
        pass

    def on_bar(self, bar: Bar) -> None:
        pass


def test_base_indicator_is_advanced_by_bars_only() -> None:
    """基础周期指标只由 K 线推进: 确认点与关闭开关时逐点相同, 没有临时点."""
    frame = _frame(6)
    off: List[Any] = []
    _run(_Indicators(), frame, on_event=off.append)
    on: List[Any] = []
    _run(_Indicators(), frame, on_event=on.append, simulate_ticks=True)

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


class _TimerEquityProbe(Strategy):
    """第 1 根 K 线买入; 定时器登记在第 2 根 K 线的时间戳上, 记下权益."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Tuple[str, int]] = []
        self.timer_equity: Optional[float] = None
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
