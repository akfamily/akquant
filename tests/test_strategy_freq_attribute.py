"""策略只读属性 self.freq: 数据周期的对外暴露.

对应反馈「挂载合约、周期等无法全局调用」的周期部分: run_backtest(freq=) 此前只用于
tick→bar 聚合、用完即弃, klinedata 的 period 是纯网关内部参数, 策略没有任何途径
知道自己跑在什么周期上。
"""

from datetime import datetime, timedelta

import pytest
from akquant import Bar, Tick
from akquant.backtest import run_backtest
from akquant.strategy import Strategy


class FreqReader(Strategy):
    """把首个回调里读到的 self.freq 记下来."""

    #: 首个 on_bar 里读到的周期; 未触发过回调时保持 False 以区分"没跑"与"跑了但是 None"
    seen_freq: object = False

    def on_start(self) -> None:
        """初始化采集位."""
        self.seen_freq = False

    def on_bar(self, bar: Bar) -> None:
        """首个 bar 上采集一次."""
        if self.seen_freq is False:
            self.seen_freq = self.freq


def _make_ticks(n: int = 300) -> list[Tick]:
    base = datetime(2024, 1, 2, 9, 30)
    return [
        Tick(
            symbol="600000.SH",
            timestamp=int((base + timedelta(seconds=i * 10)).timestamp()),
            price=10.0 + i * 0.01,
            volume=100,
        )
        for i in range(n)
    ]


def _make_bars(n: int = 30) -> list[Bar]:
    base = datetime(2024, 1, 2, 9, 30)
    return [
        Bar(
            symbol="600000.SH",
            timestamp=int((base + timedelta(minutes=i)).timestamp()),
            open=10.0,
            high=10.5,
            low=9.5,
            close=10.0 + i * 0.01,
            volume=1000,
        )
        for i in range(n)
    ]


def test_freq_from_run_backtest_tick_aggregation() -> None:
    """run_backtest(freq=) 会注入 self.freq."""
    strategy = FreqReader()
    run_backtest(
        data=_make_ticks(), freq="1min", strategy=strategy, initial_cash=1_000_000
    )
    assert strategy.seen_freq == "1min"


def test_freq_is_none_for_pure_bar_without_freq() -> None:
    """纯 bar 数据未传 freq 时 self.freq 为 None, 不做推断.

    相邻 bar 的时间戳差在停牌/跨日/午休时会给出错误答案, 而错误的周期比未知的
    周期更危险(按周期折年化会静默错一个数量级), 故这里刻意不推断。
    """
    strategy = FreqReader()
    run_backtest(data=_make_bars(), strategy=strategy, initial_cash=1_000_000)
    assert strategy.seen_freq is None, "on_bar 必须触发过, 且读到的是 None"


def test_freq_defaults_to_none_before_run() -> None:
    """未经引擎注入时 self.freq 为 None(直接构造也不会 AttributeError)."""
    assert FreqReader().freq is None


def test_freq_is_read_only() -> None:
    """写 self.freq 报错并指向正确的配置入口."""
    strategy = FreqReader()
    with pytest.raises(AttributeError, match="只读"):
        strategy.freq = "5min"  # type: ignore[misc]


def test_freq_injected_into_slot_strategies() -> None:
    """多策略槽位下每个实例都拿到同一个周期(共用同一份数据)."""

    class SlotReader(FreqReader):
        """槽位策略, 复用 FreqReader 的采集逻辑."""

    main = FreqReader()
    slot = SlotReader()
    run_backtest(
        data=_make_ticks(),
        freq="1min",
        strategy=main,
        strategies_by_slot={"slot_a": slot},
        initial_cash=1_000_000,
    )
    assert main.seen_freq == "1min"
    assert slot.freq == "1min"
