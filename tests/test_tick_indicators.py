"""tick 路径下的增量指标.

两处此前的静默失效:
1. ``on_tick_event`` 不调 ``_update_incremental_indicators``, 指标不推进
2. ``_build_incremental_indicator_args`` 用 ``getattr(payload, source)``, 而
   ``Tick`` 无 ``open``/``high``/``low``/``close`` 属性——实测
   ``getattr(tick, "close")`` 抛 ``AttributeError``
"""

from typing import Any, List, Optional

import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Tick
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "TKIND"
_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int) -> int:
    """构造纳秒级时间戳."""
    return _BASE_NS + minutes * _MINUTE_NS


def _ticks(count: int) -> List[Tick]:
    """构造递增价格的 tick 序列."""
    return [
        Tick(
            timestamp=_ns(i),
            price=10.0 + i * 0.1,
            volume=100.0,
            symbol=SYMBOL,
        )
        for i in range(1, count + 1)
    ]


def test_tick_source_close_maps_to_price() -> None:
    """source='close' 在 tick 上返回 price, 而非抛 AttributeError.

    Tick 只有 price/volume。直接 getattr(tick, "close") 会崩, 必须走名映射。
    """
    from akquant.strategy import Strategy as StrategyClass

    probe = StrategyClass.__new__(StrategyClass)
    tick = Tick(timestamp=_ns(1), price=12.5, volume=300.0, symbol=SYMBOL)

    args = probe._build_incremental_indicator_args(tick, "close", "source")

    assert args == (12.5,)


def test_tick_source_ohlc_names_all_map_to_price() -> None:
    """open/high/low 在 tick 上同样返回 price(退化后相等)."""
    from akquant.strategy import Strategy as StrategyClass

    probe = StrategyClass.__new__(StrategyClass)
    tick = Tick(timestamp=_ns(1), price=12.5, volume=300.0, symbol=SYMBOL)

    for name in ("open", "high", "low"):
        assert probe._build_incremental_indicator_args(tick, name, "source") == (12.5,)


def test_tick_source_volume_returns_tick_volume() -> None:
    """source='volume' 返回单笔量."""
    from akquant.strategy import Strategy as StrategyClass

    probe = StrategyClass.__new__(StrategyClass)
    tick = Tick(timestamp=_ns(1), price=12.5, volume=300.0, symbol=SYMBOL)

    assert probe._build_incremental_indicator_args(tick, "volume", "source") == (300.0,)


def test_tick_close_volume_mode_returns_price_and_volume() -> None:
    """close_volume 模式在 tick 上返回 (price, volume), 不抛 AttributeError."""
    from akquant.strategy import Strategy as StrategyClass

    probe = StrategyClass.__new__(StrategyClass)
    tick = Tick(timestamp=_ns(1), price=12.5, volume=300.0, symbol=SYMBOL)

    assert probe._build_incremental_indicator_args(tick, "close", "close_volume") == (
        12.5,
        300.0,
    )


@pytest.mark.parametrize("mode", ["hl", "hlc", "ohlc"])
def test_tick_high_low_modes_raise(mode: str) -> None:
    """需要真实 H/L 的模式在 tick 上必须报错.

    tick 的 OHLC 恒等, ATR/振幅类指标会恒为 0——静默返回 0 比报错危险。
    """
    from akquant.strategy import Strategy as StrategyClass

    probe = StrategyClass.__new__(StrategyClass)
    tick = Tick(timestamp=_ns(1), price=12.5, volume=300.0, symbol=SYMBOL)

    with pytest.raises(ValueError, match="freq"):
        probe._build_incremental_indicator_args(tick, "close", mode)


class _PairRecorder:
    """记录 (high, low) 二元组的最简增量指标."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.pairs: List[Any] = []

    def update(self, high: float, low: float) -> None:
        """接收一对高低价."""
        self.pairs.append((float(high), float(low)))

    @property
    def value(self) -> Any:
        """最近一对; 无数据时为 None."""
        return self.pairs[-1] if self.pairs else None


class _SmaProbe(Strategy):
    """在 tick 路径上注册单值增量指标.

    ``self.I()`` 的第一个位置参数是**指标对象**(需有 ``update()``), 不是名字;
    名字走 ``name=`` 关键字, 省略时按指标类名生成。

    返回的 ``IndicatorBinding`` 用 ``.value`` 或 ``[0]`` 取当前值。
    """

    def __init__(self) -> None:
        """初始化观测容器."""
        self.observed: Optional[float] = None
        self.tick_count = 0
        self.recorder = _MeanOfLastThree()

    def on_start(self) -> None:
        """订阅并注册增量指标."""
        self.subscribe(SYMBOL)
        self.sma3 = self.I(self.recorder, name="sma3", source="close")

    def on_tick(self, tick: Any) -> None:
        """在第 5 个 tick 上读取指标值."""
        self.tick_count += 1
        if self.tick_count == 5:
            self.observed = self.recorder.value


class _MeanOfLastThree:
    """最简增量指标: 保留最近 3 个值并给出均值.

    只需 ``update(value)`` 与 ``value`` 属性——这正是框架对增量指标的全部要求
    (见 tests/test_strategy_timers_indicators.py 的既有写法)。
    """

    def __init__(self) -> None:
        """初始化滑动窗口."""
        self.values: List[float] = []

    def update(self, value: float) -> None:
        """接收一个新值."""
        self.values.append(float(value))

    @property
    def value(self) -> Optional[float]:
        """最近 3 个值的均值; 不足 3 个时为 None."""
        if len(self.values) < 3:
            return None
        return sum(self.values[-3:]) / 3.0


def test_single_value_indicator_advances_on_tick() -> None:
    """单值指标在 tick 路径上正确推进.

    此前 ``on_tick_event`` 不调 ``_update_incremental_indicators``, 指标恒不推进
    ——且是静默的。
    """
    strategy = _SmaProbe()
    run_backtest(
        data=_ticks(6),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.tick_count == 6
    assert strategy.observed is not None, "指标未在 tick 路径推进"
    # 第 5 个 tick 时, 最近 3 个价格是 10.3 / 10.4 / 10.5, 均值 10.4。
    assert strategy.observed == pytest.approx(10.4, abs=1e-9)


def test_mixed_input_with_precomputed_indicator_raises() -> None:
    """混合 [Bar, Tick] + precompute 必须显式报错, 而非静默丢失指标.

    归一后走 DataFeed 分支, 它不构建 data_map_for_indicators; 纯 bar 列表分支会构建。
    不报错的话, 同一批 bar 单独传有指标、加一个 tick 就没了, 用户无从察觉。
    """
    from akquant.akquant import Bar
    from akquant.indicator import Indicator

    def _make_precomputed() -> Indicator:
        """最简向量化预计算指标."""
        return Indicator("dummy", lambda df: df["close"])

    class _Precompute(Strategy):
        """真正注册一个预计算指标的最小策略.

        判据是 ``_precomputed_indicators`` 非空: 只有 ``self.I(Indicator(...))``
        这条路才需要完整 DataFrame, 增量指标的 tick 路径不受影响。所以这里必须
        真的声明一个向量化预计算指标。

        注册放在 ``on_start`` 里是安全的: 引擎先调 ``on_start``(``engine.py`` 内
        ``strategy_instance.on_start()``), 之后才走到归一块, 已核实此时序。
        """

        def on_start(self) -> None:
            """声明一个预计算指标."""
            self.dummy = self.I(_make_precomputed(), name="dummy")

        def on_bar(self, bar: Any) -> None:
            """不做任何事."""

    bar = Bar(
        timestamp=_ns(1),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=1000.0,
        symbol=SYMBOL,
    )
    with pytest.raises(ValueError, match="预计算指标"):
        run_backtest(
            data=[bar, _ticks(1)[0]],
            strategy=_Precompute(),
            symbols=[SYMBOL],
            initial_cash=100_000.0,
            show_progress=False,
            fill_policy=CurrentClose(),
        )


def test_hl_indicator_works_with_freq_aggregation() -> None:
    """文档承诺 freq 聚合后 H/L 类指标可用, 必须真的可用.

    此前守卫按 ``isinstance(payload, Tick)`` 触发, 而 freq 按设计仍投递原始 tick,
    于是 tick payload 照样到达指标参数构建器、照样报错——错误信息叫用户去做他刚做
    过的事(传 freq)。
    """

    class _HLProbe(Strategy):
        """在 freq 聚合的 bar 上用 H/L 指标."""

        def __init__(self) -> None:
            """初始化观测容器."""
            self.recorder = _PairRecorder()

        def on_start(self) -> None:
            """订阅并注册 H/L 指标."""
            self.subscribe(SYMBOL)
            self.hl = self.I(self.recorder, name="hl", input_mode="hl")

        def on_bar(self, bar: Any) -> None:
            """不做任何事."""

        def on_tick(self, tick: Any) -> None:
            """不做任何事."""

    strategy = _HLProbe()
    run_backtest(
        data=_ticks(8),
        freq="1min",
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.recorder.pairs, "H/L 指标在 freq 聚合下未收到任何更新"


def test_bar_hl_indicator_survives_added_tick() -> None:
    """混合输入下, 为 bar 注册的 H/L 指标不得因为多了一个 tick 就报错.

    给一份 bar 数据加一个 tick, 原本能用的 bar 指标坏掉 = 守卫过宽。
    """
    from akquant.akquant import Bar

    class _HLOnBars(Strategy):
        """在 bar 上用 H/L 指标."""

        def __init__(self) -> None:
            """初始化观测容器."""
            self.recorder = _PairRecorder()

        def on_start(self) -> None:
            """订阅并注册."""
            self.subscribe(SYMBOL)
            self.hl = self.I(self.recorder, name="hl", input_mode="hl")

        def on_bar(self, bar: Any) -> None:
            """不做任何事."""

        def on_tick(self, tick: Any) -> None:
            """不做任何事."""

    bars = [
        Bar(
            timestamp=_ns(i),
            open=10.0,
            high=10.5 + i * 0.1,
            low=9.5 - i * 0.1,
            close=10.0,
            volume=1000.0,
            symbol=SYMBOL,
        )
        for i in (1, 2, 3)
    ]
    strategy = _HLOnBars()
    run_backtest(
        data=[*bars, _ticks(1)[0]],
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.recorder.pairs, "bar 的 H/L 指标因混入 tick 而失效"


def test_pure_tick_hl_indicator_raises_to_caller() -> None:
    """纯 tick 会话注册 H/L 指标, 异常必须到达 run_backtest 的调用方.

    此前该校验放在 ``_on_stop_internal``, 而 ``engine.py`` 的四个调用点都把异常吞成
    一行 ERROR 日志, 于是 ``run_backtest`` 正常返回、指标一次没更新, 用户拿到全 0 的
    ATR 而不知情——把响亮的失败换成了安静的失败。

    本测试走完整的 ``run_backtest`` 路径(不是直接调私有方法), 因为吞异常发生在
    engine 层: 直接调私有方法的测试无法发现它。
    """

    class _HLOnTicksOnly(Strategy):
        """纯 tick 会话里注册 H/L 指标."""

        def __init__(self) -> None:
            """初始化观测容器."""
            self.recorder = _PairRecorder()

        def on_start(self) -> None:
            """订阅并注册 H/L 指标."""
            self.subscribe(SYMBOL)
            self.hl = self.I(self.recorder, name="hl", input_mode="hl")

        def on_bar(self, bar: Any) -> None:
            """不做任何事."""

        def on_tick(self, tick: Any) -> None:
            """不做任何事."""

    with pytest.raises(ValueError, match="hl"):
        run_backtest(
            data=_ticks(5),
            strategy=_HLOnTicksOnly(),
            symbols=[SYMBOL],
            initial_cash=100_000.0,
            show_progress=False,
            fill_policy=CurrentClose(),
        )


def test_user_on_stop_error_still_tolerated() -> None:
    """用户 on_stop 里抛异常仍只记日志, 不毁掉回测结果.

    engine 的四处 except 是有意的宽容。本轮只让**框架校验**异常穿透, 不得顺手把这
    份宽容也拆掉。
    """
    from akquant.akquant import Bar

    class _BadOnStop(Strategy):
        """on_stop 里故意抛错的策略."""

        def on_start(self) -> None:
            """订阅标的."""
            self.subscribe(SYMBOL)

        def on_bar(self, bar: Any) -> None:
            """不做任何事."""

        def on_stop(self) -> None:
            """故意抛一个用户级异常."""
            raise RuntimeError("用户 on_stop 的 bug")

    bars = [
        Bar(
            timestamp=_ns(i),
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0,
            volume=1000.0,
            symbol=SYMBOL,
        )
        for i in (1, 2, 3)
    ]
    result = run_backtest(
        data=bars,
        strategy=_BadOnStop(),
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert result is not None, "用户 on_stop 的异常不应毁掉回测结果"
