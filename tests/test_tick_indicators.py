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


class _SmaProbe(Strategy):
    """在 tick 路径上注册单值增量指标.

    注意 ``register_incremental_indicator`` 的真实签名是
    ``(name, indicator=None, source="close", symbols=None, *, warmup_bars=0,
    indicator_factory=None, input_mode="source")``——第二个位置参数是**指标对象**,
    不是指标名字符串, 也没有 ``period`` 关键字。且必须先设
    ``indicator_mode = "incremental"``, 否则注册时抛 ValueError。

    读值经 ``IncrementalIndicatorBinding``: 注册后 ``self.<name>`` 是一个
    binding 对象, 用 ``.value`` 取当前值(``strategy.py:255``)。
    """

    def __init__(self) -> None:
        """初始化观测容器."""
        self.observed: Optional[float] = None
        self.tick_count = 0
        self.recorder = _MeanOfLastThree()

    def on_start(self) -> None:
        """订阅并注册增量指标."""
        self.subscribe(SYMBOL)
        self.indicator_mode = "incremental"
        self.register_incremental_indicator("sma3", self.recorder, source="close")

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


def test_mixed_input_with_precompute_mode_raises() -> None:
    """混合 [Bar, Tick] + precompute 必须显式报错, 而非静默丢失指标.

    归一后走 DataFeed 分支, 它不构建 data_map_for_indicators; 纯 bar 列表分支会构建。
    不报错的话, 同一批 bar 单独传有指标、加一个 tick 就没了, 用户无从察觉。
    """
    from akquant.akquant import Bar

    class _DummyPrecomputed:
        """最简预计算指标: 只需能被 register_precomputed_indicator 收下."""

        def update(self, value: float) -> None:
            """接收一个值(本测试不校验它被调用)."""

    class _Precompute(Strategy):
        """真正注册一个预计算指标的最小策略.

        判据是 ``_precomputed_indicators`` 非空, **不是** ``indicator_mode``——
        后者默认就是 ``"precompute"``(``strategy.py``), 用它做判据会误伤所有未显式
        改模式的 tick 用户。所以这里必须真的注册一个指标。

        注册放在 ``on_start`` 里是安全的: 引擎先调 ``on_start``(``engine.py`` 内
        ``strategy_instance.on_start()``), 之后才走到归一块, 已核实此时序。
        """

        def on_start(self) -> None:
            """注册一个预计算指标."""
            self.register_precomputed_indicator(
                "dummy",
                _DummyPrecomputed(),  # type: ignore[arg-type]
            )

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
    with pytest.raises(ValueError, match="precompute"):
        run_backtest(
            data=[bar, _ticks(1)[0]],
            strategy=_Precompute(),
            symbols=[SYMBOL],
            initial_cash=100_000.0,
            show_progress=False,
            fill_policy=CurrentClose(),
        )
