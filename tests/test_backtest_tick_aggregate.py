"""freq 参数: 把 tick 聚合成 bar.

没有框架直接在裸 tick 上算 bar 型指标(NautilusTrader / QuantConnect Lean 都
要求先经聚合器), 故 AKQuant 用 ``freq`` 扮演这一角色。
"""

from typing import Any, List

import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Bar, Tick
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "TKAGG"
_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int, seconds: int = 0) -> int:
    """构造纳秒级时间戳."""
    return _BASE_NS + minutes * _MINUTE_NS + seconds * 1_000_000_000


def _tick(minutes: int, seconds: int, price: float, volume: float) -> Tick:
    """构造一个 tick."""
    return Tick(
        timestamp=_ns(minutes, seconds),
        price=price,
        volume=volume,
        symbol=SYMBOL,
    )


class _BothRecorder(Strategy):
    """同时记录 bar 与 tick."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.bars: List[Bar] = []
        self.ticks: List[Tick] = []

    def on_start(self) -> None:
        """订阅标的."""
        self.subscribe(SYMBOL)

    def on_bar(self, bar: Any) -> None:
        """记录合成 bar."""
        self.bars.append(bar)

    def on_tick(self, tick: Any) -> None:
        """记录原始 tick."""
        self.ticks.append(tick)


def _minute_one_ticks() -> List[Tick]:
    """第 1 分钟内的 4 个 tick, 第 2 分钟 1 个(用于封闭第 1 分钟的 bar).

    BarAggregator 无 flush: 最后一个未满周期不会发出, 故第 2 分钟的 tick 只用来
    触发第 1 分钟 bar 的封闭, 本身不产生 bar。
    """
    return [
        _tick(1, 0, 10.0, 100.0),
        _tick(1, 15, 10.6, 200.0),
        _tick(1, 30, 9.8, 150.0),
        _tick(1, 45, 10.2, 50.0),
        _tick(2, 0, 11.0, 300.0),
    ]


def _run(data: List[Any], freq: Any = None) -> _BothRecorder:
    """跑一次回测并返回策略实例."""
    strategy = _BothRecorder()
    run_backtest(
        data=data,
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
        freq=freq,
    )
    return strategy


def test_freq_delivers_both_ticks_and_synthesized_bars() -> None:
    """Freq 下原始 tick 仍投递, 且额外收到合成 bar."""
    strategy = _run(_minute_one_ticks(), freq="1min")

    assert len(strategy.ticks) == 5, "原始 tick 必须照常投递"
    assert len(strategy.bars) >= 1, "未收到任何合成 bar"


def test_synthesized_bar_has_real_ohlc() -> None:
    """合成 bar 的 OHLC 非恒等——这正是聚合的价值."""
    strategy = _run(_minute_one_ticks(), freq="1min")

    first = strategy.bars[0]
    assert float(first.open) == pytest.approx(10.0)
    assert float(first.high) == pytest.approx(10.6)
    assert float(first.low) == pytest.approx(9.8)
    assert float(first.close) == pytest.approx(10.2)
    assert float(first.high) != float(first.low), "OHLC 恒等说明聚合没生效"


def test_synthesized_bar_volume_sums_tick_volumes() -> None:
    """合成 bar 的成交量等于区间内 tick 量之和, 一笔不落.

    ``BarAggregator`` **默认**按累计口径解释 ``volume``(内部算差分), 那是 CTP 日累计
    Volume 的语义; 该口径下每个 symbol 的**首笔**量会被丢弃——首次调用会把
    ``last_cumulative_volumes[symbol]`` 播种为本次自己的 volume, 差分恒为 0。
    回测传的是**单笔量**, 故适配层构造聚合器时传 ``volume_is_cumulative=False``,
    让它直接计入每一笔, **不是**在 Python 侧做累加。
    本测试守住这条: 100+200+150+50 必须得到 500; 得到 400 说明口径又回退成累计了。
    """
    strategy = _run(_minute_one_ticks(), freq="1min")

    first = strategy.bars[0]
    assert float(first.volume) == pytest.approx(500.0), (
        "第 1 分钟 4 个 tick 量为 100+200+150+50=500; "
        "若得到 0 或接近 0, 说明 volume 未做累计适配"
    )


def test_freq_without_ticks_raises() -> None:
    """Freq 给了但数据里没有 tick: 参数无意义, 早失败."""
    bar = Bar(
        timestamp=_ns(1),
        open=10.0,
        high=10.5,
        low=9.5,
        close=10.0,
        volume=1000.0,
        symbol=SYMBOL,
    )

    with pytest.raises(ValueError, match="freq"):
        _run([bar], freq="1min")


def test_sub_minute_freq_raises() -> None:
    """秒级 freq 明确报错, 不静默取整."""
    with pytest.raises(ValueError, match="整数分钟"):
        _run(_minute_one_ticks(), freq="30s")


def test_no_freq_yields_no_bars() -> None:
    """回归防护: 不传 freq 时不得凭空产生 bar."""
    strategy = _run(_minute_one_ticks())

    assert len(strategy.ticks) == 5
    assert strategy.bars == []


def test_synthesized_bar_never_precedes_its_source_ticks() -> None:
    """合成 bar 的时间戳必须 >= 形成它的最后一个 tick, 否则策略读到未来.

    ``BarAggregator`` 给封闭的 bar 打**区间起点**时间戳。live 下无害: bar 是在下一
    区间的 tick 到达时才发出的, 墙钟顺序保护了消费者。但回测里 ``run_backtest`` 会
    ``feed.sort()``, 按该起点时间戳重排后 bar 落到形成它的 tick **之前**, 于是
    ``on_bar`` 拿到的 high/low/close 来自尚未发生的 tick。

    实测的错误顺序(第 1 分钟 60s=10.0, 90s=20.0 尖峰, 105s=10.5):
        TICK off= 60s  10.0
        BAR  off= 60s  high=20.0   <- 在尖峰发生前就知道它
        TICK off= 90s  20.0
    """
    strategy = _BothRecorder()
    run_backtest(
        data=_minute_one_ticks(),
        freq="1min",
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    assert strategy.bars, "未产生合成 bar"
    first_bar_ts = int(strategy.bars[0].timestamp)
    contributing = [
        int(t.timestamp)
        for t in strategy.ticks
        if _ns(1, 0) <= int(t.timestamp) < _ns(2, 0)
    ]
    assert contributing, "第 1 分钟没有 tick"
    assert first_bar_ts >= max(contributing), (
        f"合成 bar 时间戳 {first_bar_ts} 早于其最后贡献 tick {max(contributing)}: "
        "策略会读到未来数据"
    )
