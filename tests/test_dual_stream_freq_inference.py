"""测试双流下 freq 自动推断: on_bar 里取 bar, on_tick 里取 tick, 其余位置仍报错.

对应反馈 vendor/docs/报错.txt: 只写 on_bar 的策略在 tick 流存在时(实盘
emit_ticks / 回测 run_backtest(data=[Tick,...], freq=...)) 撞双流歧义 ValueError
并中止会话 —— 而它从不读 tick, 调用点意图并无歧义。
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest
from akquant import Bar, Tick
from akquant.backtest import run_backtest
from akquant.strategy import Strategy


class OnlyOnBar(Strategy):
    """只写 on_bar, 不写 on_tick, 双流下 get_history 不传 freq 应自动定档到 bar."""

    #: on_bar 是否已完成断言(供测试确认回调真的触发过)
    checked: bool = False

    def on_start(self) -> None:
        """开启历史并重置采集标记."""
        self.set_history_depth(20)
        self.checked = False

    def on_bar(self, bar: Bar) -> None:
        """首个 bar 上验证五个历史入口都能不传 freq 正常取数."""
        if self.checked:
            return
        # 不传 freq, 应自动推断为 'bar'
        close = self.get_history(5, bar.symbol, "close")
        assert len(close) == 5
        assert close[-1] > 0
        # 五个公开入口全覆盖: 上次加 freq 参数时漏改过其中三个
        multi = self.get_history_multi(3, bar.symbol, ("close", "volume"))
        assert "close" in multi
        df = self.get_history_df(3, bar.symbol)
        assert df.shape[0] == 3
        hmap = self.get_history_map(3, [bar.symbol], "close")
        assert bar.symbol in hmap
        rolling = self.get_rolling_data(length=3, symbol=bar.symbol)
        assert rolling[0].shape[0] == 3
        self.checked = True


class OnlyOnTick(Strategy):
    """只写 on_tick, 双流下 get_history 不传 freq 应自动定档到 tick.

    必须等 bar 序列也建立起来(freq='1min' 聚合出第一根 bar 之后)才构成双流,
    否则首个 tick 时 bar 容器还空, 压根走不到歧义判断。
    """

    #: on_tick 是否已完成断言(供测试确认回调真的触发过)
    checked: bool = False

    def on_start(self) -> None:
        """开启历史并重置采集标记."""
        self.set_history_depth(20)
        self.checked = False

    def on_tick(self, tick: Tick) -> None:
        """等双流成立后验证不传 freq 会取到 tick 序列."""
        if self.checked:
            return
        # 等 bar 容器非空(双流真正成立)后再验证
        bar_close = self.get_history(2, tick.symbol, "close", freq="bar")
        if np.isnan(bar_close).all():
            return
        # 不传 freq, 应自动推断为 'tick'
        close = self.get_history(5, tick.symbol, "close")
        assert len(close) == 5
        assert close[-1] > 0
        self.checked = True


class BothCallbacks(Strategy):
    """同时写 on_bar 和 on_tick, 各自应自动定档到对应粒度."""

    #: 两个行情回调各自是否已完成断言
    bar_checked: bool = False
    tick_checked: bool = False

    def on_start(self) -> None:
        """开启历史并重置两个采集标记."""
        self.set_history_depth(20)
        self.bar_checked = False
        self.tick_checked = False

    def on_bar(self, bar: Bar) -> None:
        """on_bar 里不传 freq 应取到 bar 序列."""
        if self.bar_checked:
            return
        close = self.get_history(5, bar.symbol, "close")
        assert len(close) == 5
        self.bar_checked = True

    def on_tick(self, tick: Tick) -> None:
        """on_tick 里不传 freq 应取到 tick 序列."""
        if self.tick_checked:
            return
        close = self.get_history(5, tick.symbol, "close")
        assert len(close) == 5
        self.tick_checked = True


class BeforeTradingAmbiguity(Strategy):
    """on_before_trading 里双流下不传 freq 应仍然报歧义错误."""

    def on_start(self) -> None:
        """只开启历史, 取数放到 on_before_trading."""
        self.set_history_depth(20)

    def on_bar(self, bar: Bar) -> None:
        """让 bar 序列建立起来, 使次日 on_before_trading 时双流成立."""

    def on_before_trading(self, ctx: Any, timestamp: int) -> None:
        """行情回调之外取历史: 推断不介入, 应抛歧义错误."""
        self.get_history(5, "600000.SH", "close")


def _make_ticks(n: int = 600) -> list[Tick]:
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


def _make_two_day_ticks(per_day: int = 120) -> list[Tick]:
    """两个交易日的 tick, 用于触发跨日的 on_before_trading."""
    ticks: list[Tick] = []
    for day in (2, 3):
        base = datetime(2024, 1, day, 9, 30)
        ticks.extend(
            Tick(
                symbol="600000.SH",
                timestamp=int((base + timedelta(seconds=i * 10)).timestamp()),
                price=10.0 + i * 0.01,
                volume=100,
            )
            for i in range(per_day)
        )
    return ticks


def test_only_on_bar_infers_bar() -> None:
    """只写 on_bar, 双流下自动定档 freq='bar'."""
    strategy = OnlyOnBar()
    run_backtest(
        data=_make_ticks(), freq="1min", strategy=strategy, initial_cash=1_000_000
    )
    assert strategy.checked, "on_bar 应该触发过"


def test_only_on_tick_infers_tick() -> None:
    """只写 on_tick, 双流下自动定档 freq='tick'."""
    strategy = OnlyOnTick()
    run_backtest(
        data=_make_ticks(), freq="1min", strategy=strategy, initial_cash=1_000_000
    )
    assert strategy.checked, "on_tick 应该触发过"


def test_both_callbacks_each_infers_own_freq() -> None:
    """同时写两个回调, 各自定档到对应粒度."""
    strategy = BothCallbacks()
    run_backtest(
        data=_make_ticks(), freq="1min", strategy=strategy, initial_cash=1_000_000
    )
    assert strategy.bar_checked
    assert strategy.tick_checked


def test_non_market_callback_still_raises_ambiguity_in_dual_stream() -> None:
    """行情回调之外(on_before_trading)双流下不传 freq 仍报错.

    锁住推断的边界: 只有 on_bar / on_tick 能定档, 其余位置维持既有歧义报错,
    不要把这个放宽成"全局默认取 bar"。
    """
    ticks = _make_two_day_ticks()
    with pytest.raises(ValueError, match="同时存在 bar 与 tick 两条历史序列"):
        run_backtest(
            data=ticks,
            freq="1min",
            strategy=BeforeTradingAmbiguity(),
            initial_cash=1_000_000,
        )
