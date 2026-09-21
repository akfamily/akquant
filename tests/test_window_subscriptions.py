"""引擎原生多周期: subscribe_bars / on_window_bar / get_history(freq=) / 指标按周期."""

from typing import Optional

import akquant
import numpy as np
import pandas as pd
import pytest
from akquant import SMA, Strategy, run_backtest
from akquant.akquant import Bar, StrategyContext


def test_bar_freq_defaults_to_none_and_is_settable() -> None:
    """Bar.freq 默认 None, 可读写, 构造函数支持 freq 关键字参数."""
    bar = Bar(1_700_000_000_000_000_000, 1.0, 2.0, 0.5, 1.5, 100.0, "X")
    assert bar.freq is None
    bar.freq = "5min"
    assert bar.freq == "5min"
    tagged = Bar(1, 1.0, 1.0, 1.0, 1.0, 1.0, "X", freq="1d")
    assert tagged.freq == "1d"
    assert "freq=1d" in repr(tagged)


def _minute_bars(symbol: str, start: str, n: int) -> list[Bar]:
    idx = pd.date_range(start, periods=n, freq="1min", tz="Asia/Shanghai")
    return [
        Bar(int(ts.value), 10.0 + i, 10.5 + i, 9.5 + i, 10.1 + i, 100.0, symbol)
        for i, ts in enumerate(idx)
    ]


class _EngineDispatchProbe(Strategy):
    """绕过 Task 5 的 Python API, 直接验证引擎会调 _on_window_bar_event_and_flush."""

    def __init__(self) -> None:
        super().__init__()
        self.window_calls: list[
            tuple[int, str, int]
        ] = []  # (ts, freq, 触发时基础 bar 数)
        self.base_seen = 0

    def on_bar(self, bar: Bar) -> None:
        self.base_seen += 1

    def _on_window_bar_event_and_flush(self, bar: Bar, ctx: object) -> None:
        self.window_calls.append((int(bar.timestamp), str(bar.freq), self.base_seen))
        return None


def test_engine_dispatches_closed_window_bars_after_base_on_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """绕过 Task 5 的 Python API, 直接验证引擎会同步派发闭合的窗口 bar.

    Task 5 之前 Python 侧还没有 subscribe_bars, 用工厂替换 engine.py 里的 Engine 类,
    在引擎创建后、run() 之前直接配置订阅。不能在 on_start 里配置: run() 期间再入
    Engine pymethod 会 'Already borrowed'。
    """
    import akquant.backtest.engine as engine_module

    real_engine_cls = engine_module.Engine

    def _factory() -> object:
        eng = real_engine_cls()
        eng.configure_window_subscriptions([(None, "5min", None)], 1)
        return eng

    monkeypatch.setattr(engine_module, "Engine", _factory)
    bars = _minute_bars("X", "2024-01-02 09:31:00", 7)  # 09:31..09:37
    strategy = _EngineDispatchProbe()
    run_backtest(data=bars, strategy=strategy, symbols="X", show_progress=False)
    # 09:35 闭合一根(基础周期已知 → 即时), 尾部 09:36-09:37 在结束 flush 时闭合一根
    assert [c[1] for c in strategy.window_calls] == ["5min", "5min"]
    assert strategy.window_calls[0][0] == int(
        pd.Timestamp("2024-01-02 09:35", tz="Asia/Shanghai").value
    )
    assert strategy.window_calls[0][2] == 5, "应在第 5 根基础 bar 的 on_bar 之后派发"
    assert strategy.window_calls[1][2] == 7, "尾部窗口在全部 bar 处理完后 flush"


class _WindowBuyerSlotStrategy(Strategy):
    """多 slot 累加测试用 slot A: 每根闭合窗口 bar 上都下一笔买单."""

    def on_bar(self, bar: Bar) -> None:
        return None

    def _on_window_bar_event_and_flush(self, bar: Bar, ctx: StrategyContext) -> None:
        ctx.buy(bar.symbol, 100)
        return None


class _WindowObserverSlotStrategy(Strategy):
    """多 slot 累加测试用 slot B: 记录派发时看到的 active_orders 长度."""

    def __init__(self) -> None:
        super().__init__()
        self.active_orders_seen: list[int] = []

    def on_bar(self, bar: Bar) -> None:
        return None

    def _on_window_bar_event_and_flush(self, bar: Bar, ctx: StrategyContext) -> None:
        self.active_orders_seen.append(len(ctx.active_orders))
        return None


def test_window_dispatch_accumulates_orders_across_slots() -> None:
    """窗口 bar 派发须与 on_bar 路径(StrategyProcessor)同构, 在 slot 间累积当步订单.

    同一根窗口 bar 上, 后面的 slot 应看到前面 slot 刚下的单, 而不是仅本步开始时的
    active_orders 快照。
    """
    engine = akquant.Engine()
    if not hasattr(engine, "set_strategy_slots") or not hasattr(
        engine, "set_strategy_for_slot"
    ):
        pytest.skip("Engine 未暴露多 slot 配置方法")

    symbol = "WINDOW_SLOT"
    engine.use_simple_market(0.0)
    engine.set_force_session_continuous(True)
    engine.set_fill_mode(akquant.ExecutionMode.CurrentClose, "same_cycle")
    engine.set_cash(1_000_000.0)
    engine.set_stock_fee_rules(0.0, 0.0, 0.0, 0.0)

    instr = akquant.Instrument(
        symbol=symbol,
        asset_type=akquant.AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1.0,
    )
    engine.add_instrument(instr)
    engine.add_bars(_minute_bars(symbol, "2024-01-02 09:31:00", 7))
    engine.configure_window_subscriptions([(None, "5min", None)], 1)

    engine.set_strategy_slots(["buyer", "observer"])
    observer = _WindowObserverSlotStrategy()
    engine.set_strategy_for_slot(1, observer)
    buyer = _WindowBuyerSlotStrategy()
    engine.run(buyer, show_progress=False)

    assert observer.active_orders_seen, "observer 应至少见过一次窗口 bar 派发"
    assert observer.active_orders_seen[0] >= 1, (
        "slot B 应在同一根窗口 bar 上看到 slot A(buyer) 刚下的单"
    )


def _pandas_windows(bars: list[Bar], freq: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        },
        index=pd.to_datetime([b.timestamp for b in bars], utc=True).tz_convert(
            "Asia/Shanghai"
        ),
    )
    out = df.resample(freq, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])  # type: ignore[no-any-return]


class _MultiWindow(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.subscribe_bars("15min", callback=self.on_15m)
        self.got5: list[Bar] = []
        self.got15: list[Bar] = []
        self.order_of_events: list[str] = []
        self.hist_in_window: Optional[np.ndarray] = None
        self.inferred_in_window: Optional[np.ndarray] = None

    def on_start(self) -> None:
        self.set_history_depth(50)

    def on_bar(self, bar: Bar) -> None:
        self.order_of_events.append(f"bar:{bar.timestamp}")

    def on_window_bar(self, bar: Bar) -> None:
        assert bar.freq == "5min"
        self.got5.append(bar)
        self.order_of_events.append(f"5min:{bar.timestamp}")
        if len(self.got5) == 2:
            self.hist_in_window = self.get_history(2, bar.symbol, "close", freq="5min")
            # 回调内定档
            self.inferred_in_window = self.get_history(2, bar.symbol, "close")

    def on_15m(self, bar: Bar) -> None:
        assert bar.freq == "15min"
        self.got15.append(bar)
        self.order_of_events.append(f"15min:{bar.timestamp}")


def test_window_bars_match_pandas_resample_and_dispatch_order() -> None:
    """窗口 bar 的 OHLCV 与 pandas resample 结果一致, 多周期同步内 5min 先于 15min."""
    bars = _minute_bars("X", "2024-01-02 09:31:00", 30)  # 09:31..10:00
    strategy = _MultiWindow()
    run_backtest(data=bars, strategy=strategy, symbols="X", show_progress=False)
    # 注意: 纯 bar 输入不能传 freq (既有校验), 基础周期未知 → 延迟闭合;
    # 09:35 窗口在 09:36 那根 bar 之后派发, 尾部 10:00 窗口在结束 flush 派发。
    exp5 = _pandas_windows(bars, "5min")
    assert len(strategy.got5) == len(exp5) == 6
    for got, (ts, row) in zip(strategy.got5, exp5.iterrows()):
        assert got.timestamp == int(ts.value)  # type: ignore[attr-defined]
        assert got.open == pytest.approx(row["open"])
        assert got.high == pytest.approx(row["high"])
        assert got.low == pytest.approx(row["low"])
        assert got.close == pytest.approx(row["close"])
        assert got.volume == pytest.approx(row["volume"])
    exp15 = _pandas_windows(bars, "15min")
    assert [b.timestamp for b in strategy.got15] == [int(t.value) for t in exp15.index]
    # 同一步多周期闭合: 5min 先于 15min
    ev = strategy.order_of_events
    i5 = ev.index(f"5min:{strategy.got15[0].timestamp}")
    i15 = ev.index(f"15min:{strategy.got15[0].timestamp}")
    assert i5 < i15
    # 窗口 bar 在触发它的基础 bar 之后
    first_base_after = ev.index(f"bar:{bars[5].timestamp}")  # 09:36
    assert ev.index(f"5min:{strategy.got5[0].timestamp}") > first_base_after
    # 历史: 显式与回调内定档一致
    assert strategy.hist_in_window is not None
    assert strategy.hist_in_window.tolist() == pytest.approx(
        [strategy.got5[0].close, strategy.got5[1].close]
    )
    assert strategy.inferred_in_window is not None
    assert strategy.inferred_in_window.tolist() == strategy.hist_in_window.tolist()


class _IndicatorByFreq(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.sma_base = self.I(SMA(2), name="sma_base", source="close")
        self.sma_5m = self.I(SMA(2), name="sma_5m", source="close", freq="5min")
        self.updates_5m: list[float] = []

    def on_window_bar(self, bar: Bar) -> None:
        v = self.sma_5m.value  # type: ignore[attr-defined]
        if v is not None:
            self.updates_5m.append(float(v))


def test_incremental_indicator_driven_only_by_its_freq() -> None:
    """freq='5min' 的增量指标只被窗口 bar 驱动, 不被基础 bar 驱动, 反之亦然."""
    bars = _minute_bars("X", "2024-01-02 09:31:00", 12)  # close = 10.1 + i, i=0..11
    s = _IndicatorByFreq()
    run_backtest(data=bars, strategy=s, symbols="X", show_progress=False)
    # 5min 窗口(基础周期未知 → 延迟闭合): 09:35 在 09:36 派发, 09:40 在 09:41 派发,
    # 尾部 09:41-09:42 在结束 flush → 共 3 根; SMA(2) 从第 2 根起有值 → 2 次记录
    assert len(s.updates_5m) == 2
    base_inst = s.sma_base.get_instance("X")  # type: ignore[attr-defined]
    win_inst = s.sma_5m.get_instance("X")  # type: ignore[attr-defined]
    # 窗口序列 SMA(2) = mean(09:40 窗口收盘 19.1, 尾部窗口收盘 21.1) = 20.1
    assert float(win_inst.value) == pytest.approx(20.1)
    # 基础序列 SMA(2) = mean(09:41 收盘 20.1, 09:42 收盘 21.1) = 20.6
    # —— 二者若互相驱动数值会变
    assert float(base_inst.value) == pytest.approx(20.6)


def test_subscribe_bars_validation_errors() -> None:
    """subscribe_bars/I(freq=) 的各类非法输入报错."""
    s = Strategy()
    with pytest.raises(ValueError, match="30s"):
        s.subscribe_bars("30s")
    with pytest.raises(ValueError, match="1d"):
        s.subscribe_bars("2d")
    with pytest.raises(ValueError, match="1w"):
        s.subscribe_bars("1w")
    with pytest.raises(ValueError, match="session_windows"):
        s.subscribe_bars("5min", session_windows=[("11:30", "09:30")])
    s.subscribe_bars("5min")
    with pytest.raises(ValueError, match="subscribe_bars"):
        s.I(SMA(2), name="x", freq="15min")  # 未订阅


def test_subscribe_bars_after_engine_configured_raises() -> None:
    """在 __init__ 之外(如 on_start)调用 subscribe_bars 必须报错, 不能悄悄生效."""

    class _Late(Strategy):
        def on_start(self) -> None:
            self.subscribe_bars("5min")

    with pytest.raises(RuntimeError, match="__init__"):
        run_backtest(
            data=_minute_bars("X", "2024-01-02 09:31:00", 3),
            strategy=_Late(),
            symbols="X",
            show_progress=False,
        )


def test_get_history_unsubscribed_freq_raises() -> None:
    """get_history(freq=) 未订阅该周期时报 ValueError 并指向 subscribe_bars.

    on_bar 内的异常会经 call_user_callback 转发 on_error(默认可能吞掉), 故在
    回调里自己捕获并记录, 不依赖异常是否冒出 run_backtest。
    """

    class _Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.err: Optional[str] = None

        def on_start(self) -> None:
            self.set_history_depth(10)

        def on_bar(self, bar: Bar) -> None:
            if self.err is not None:
                return
            try:
                self.get_history(2, bar.symbol, "close", freq="5min")
            except ValueError as exc:
                self.err = str(exc)

    s = _Probe()
    run_backtest(
        data=_minute_bars("X", "2024-01-02 09:31:00", 3),
        strategy=s,
        symbols="X",
        show_progress=False,
    )
    assert s.err is not None and "subscribe_bars" in s.err


def test_window_freq_below_or_equal_base_rejected_when_base_known() -> None:
    """含 Tick 输入配 freq='5min' 时基础周期已知; 再订阅 5min/1min 应报错."""
    from akquant.akquant import Tick

    class _Bad(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_bars("5min")

    ticks = [
        Tick(
            timestamp=int(
                pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai").value
            ),
            price=10.0,
            volume=1.0,
            symbol="X",
        )
    ]
    with pytest.raises(ValueError, match="基础周期"):
        run_backtest(
            data=ticks, strategy=_Bad(), symbols="X", freq="5min", show_progress=False
        )


def test_current_window_returns_partial_and_symbols_filter() -> None:
    """current_window 返回正在形成的窗口快照, 且只对已订阅的 symbols 生效."""

    class _Cur(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_bars("5min", symbols=["X"])
            self.partials: list[Optional[Bar]] = []
            self.y_windows = 0

        def on_bar(self, bar: Bar) -> None:
            if bar.symbol == "X":
                self.partials.append(self.current_window("X", "5min"))

        def on_window_bar(self, bar: Bar) -> None:
            if bar.symbol == "Y":
                self.y_windows += 1

    bars = _minute_bars("X", "2024-01-02 09:31:00", 3) + _minute_bars(
        "Y", "2024-01-02 09:31:00", 3
    )
    s = _Cur()
    run_backtest(data=bars, strategy=s, symbols=["X", "Y"], show_progress=False)
    assert all(p is not None and p.freq == "5min" for p in s.partials)
    last_partial = s.partials[-1]
    assert last_partial is not None
    assert last_partial.close == pytest.approx(bars[2].close)
    assert s.y_windows == 0


class _GenericAndSpecific(Strategy):
    """generic(全量) + 同周期同 session 的 symbol 专属订阅须合并为一条 Rust spec."""

    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min")
        self.subscribe_bars("5min", callback=self.on_x, symbols=["X"])
        self.generic_seen: list[Bar] = []
        self.x_seen: list[Bar] = []

    def on_window_bar(self, bar: Bar) -> None:
        self.generic_seen.append(bar)

    def on_x(self, bar: Bar) -> None:
        self.x_seen.append(bar)


def test_generic_and_symbol_specific_subscriptions_share_one_window() -> None:
    """全量 subscribe_bars 与按 symbols 的 subscribe_bars 重叠时须合并成一条 Rust spec.

    若 Python 侧未合并直接把两条重叠 scope 下发给 Rust, `WindowAggregator` 会对
    同一根窗口重复聚合(volume 翻倍); 这里用 volume 校验没有发生双重聚合, 同时
    校验两个回调各自按 scope 精确触发且每根窗口只触发一次。

    刻意只喂 5 根 bar(而非 6 根): 基础周期未知时窗口延迟闭合, 若再多喂一根
    第 6 根会额外触发一根只含它自己的尾部 partial 窗口(结束 flush 补发), 使
    "每根窗口只触发一次"的断言与"这一根窗口没被重复聚合"的断言混在一起、
    难以区分是合并 bug 还是正常的尾部 partial 窗口。5 根不多不少刚好撑满一个
    窗口, 只在结束 flush 时整根一次性派发。
    """
    bars = _minute_bars("X", "2024-01-02 09:31:00", 5) + _minute_bars(
        "Y", "2024-01-02 09:31:00", 5
    )
    s = _GenericAndSpecific()
    run_backtest(data=bars, strategy=s, symbols=["X", "Y"], show_progress=False)

    generic_x = [b for b in s.generic_seen if b.symbol == "X"]
    generic_y = [b for b in s.generic_seen if b.symbol == "Y"]
    assert len(generic_x) == 1
    assert len(generic_y) == 1
    # on_x 只对 X 触发, 且每根窗口只触发一次(未被合并前会因两条重叠 spec 各触发一次)
    assert len(s.x_seen) == 1
    assert s.x_seen[0].symbol == "X"
    assert s.x_seen[0].timestamp == generic_x[0].timestamp
    # 未发生双重聚合: 5 根基础 bar、每根 volume=100 → 窗口 volume 应为 500, 不是 1000
    assert generic_x[0].volume == pytest.approx(500.0)
    assert s.x_seen[0].volume == pytest.approx(500.0)
