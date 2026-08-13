"""双流(tick+bar)下 get_history 的粒度选择与歧义报错."""

from typing import Optional

import numpy as np
import pandas as pd
import pytest
from akquant import Bar, BarAggregator, DataFeed, Strategy, Tick, run_backtest

T0 = pd.Timestamp("2024-01-02 09:30:00", tz="Asia/Shanghai").value


def _dual_stream_feed() -> DataFeed:
    """构造双流 feed: 每分钟 4 个 tick + 1 根合成 bar.

    价格设计成可区分段: 分钟 n 的 tick 为 n*10+0..3, 故该分钟 bar 的 close 为 n*10+3。
    """
    feed = DataFeed()
    aggregator = BarAggregator(
        feed, 1, volume_is_cumulative=False, stamp_bar_at_interval_end=True
    )
    for i in range(40):
        timestamp = T0 + 15_000_000_000 * i
        price = float((i // 4 + 1) * 10 + (i % 4))
        aggregator.on_tick("X", price, 10.0, timestamp)
        feed.add_tick(Tick(timestamp=timestamp, price=price, volume=10.0, symbol="X"))
    return feed


class _Collector(Strategy):
    """在第一次 on_bar 时按指定 freq 取历史."""

    warmup_period = 3
    freq_arg: Optional[str] = None
    captured: list = []

    def on_start(self) -> None:
        """重置采集状态."""
        self.done = False
        type(self).captured = []

    def on_bar(self, bar: Bar) -> None:
        """首个非预热 bar 上取一次历史."""
        if self.done:
            return
        self.done = True
        type(self).captured = list(
            self.get_history(
                count=3, symbol=bar.symbol, field="close", freq=self.freq_arg
            )
        )


def test_dual_stream_bar_history_is_not_polluted_by_ticks() -> None:
    """freq='bar' 必须拿到三根 bar 的收盘价, 而不是混入 tick 价格."""

    class BarStrategy(_Collector):
        freq_arg = "bar"

    run_backtest(
        strategy=BarStrategy, data=_dual_stream_feed(), symbols=["X"], initial_cash=1e5
    )
    assert BarStrategy.captured == [13.0, 23.0, 33.0]


def test_dual_stream_tick_history_returns_tick_series() -> None:
    """freq='tick' 必须拿到 tick 序列(相邻 tick 价差为 1)."""

    class TickStrategy(_Collector):
        freq_arg = "tick"

    run_backtest(
        strategy=TickStrategy, data=_dual_stream_feed(), symbols=["X"], initial_cash=1e5
    )
    values = TickStrategy.captured
    assert len(values) == 3
    assert values[-1] - values[-2] == 1.0


def test_dual_stream_without_freq_raises_with_actionable_message() -> None:
    """双流下省略 freq 必须报错并指出怎么办, 而不是静默选一条."""

    class AmbiguousStrategy(_Collector):
        freq_arg = None

    with pytest.raises(Exception) as excinfo:
        run_backtest(
            strategy=AmbiguousStrategy,
            data=_dual_stream_feed(),
            symbols=["X"],
            initial_cash=1e5,
        )
    message = str(excinfo.value)
    assert "freq='bar'" in message
    assert "freq='tick'" in message


def test_tick_freq_rejects_ohlc_only_field() -> None:
    """Tick 没有 open/high/low, 取这些字段必须报错而非返回退化 OHLC."""

    class HighStrategy(_Collector):
        freq_arg = "tick"

        def on_bar(self, bar: Bar) -> None:
            """故意用 high 字段取 tick 历史."""
            if self.done:
                return
            self.done = True
            self.get_history(count=3, symbol=bar.symbol, field="high", freq="tick")

    with pytest.raises(ValueError, match="tick"):
        run_backtest(
            strategy=HighStrategy,
            data=_dual_stream_feed(),
            symbols=["X"],
            initial_cash=1e5,
        )


class _DFCollector(Strategy):
    """在第一次 on_bar 时按指定 freq 取 get_history_df 结果."""

    warmup_period = 3
    freq_arg: Optional[str] = None
    captured: pd.DataFrame = pd.DataFrame()

    def on_start(self) -> None:
        """重置采集状态."""
        self.done = False
        type(self).captured = pd.DataFrame()

    def on_bar(self, bar: Bar) -> None:
        """首个非预热 bar 上取一次 get_history_df."""
        if self.done:
            return
        self.done = True
        type(self).captured = self.get_history_df(
            count=3, symbol=bar.symbol, freq=self.freq_arg
        )


def test_dual_stream_get_history_df_bar_freq_is_not_polluted_by_ticks() -> None:
    """get_history_df 在双流 + freq='bar' 下必须返回未被 tick 污染的 bar OHLCV."""

    class DFStrategy(_DFCollector):
        freq_arg = "bar"

    run_backtest(
        strategy=DFStrategy, data=_dual_stream_feed(), symbols=["X"], initial_cash=1e5
    )
    df = DFStrategy.captured
    assert df["open"].tolist() == [10.0, 20.0, 30.0]
    assert df["high"].tolist() == [13.0, 23.0, 33.0]
    assert df["low"].tolist() == [10.0, 20.0, 30.0]
    assert df["close"].tolist() == [13.0, 23.0, 33.0]


def test_dual_stream_get_history_multi_tick_freq_rejects_ohlc_field() -> None:
    """get_history_multi 在 freq='tick' 下混入 open/high/low 字段必须报错并指路."""

    class MultiFieldStrategy(_Collector):
        freq_arg = "tick"

        def on_bar(self, bar: Bar) -> None:
            """故意在 fields 里混入 high 字段取 tick 历史."""
            if self.done:
                return
            self.done = True
            self.get_history_multi(
                count=3, symbol=bar.symbol, fields=("close", "high"), freq="tick"
            )

    with pytest.raises(ValueError, match="get_history"):
        run_backtest(
            strategy=MultiFieldStrategy,
            data=_dual_stream_feed(),
            symbols=["X"],
            initial_cash=1e5,
        )


def test_pure_bar_path_behavior_unchanged() -> None:
    """纯 bar 路径省略 freq 行为不变(回归底线)."""
    index = pd.date_range("2024-01-02", periods=6, freq="D")
    data = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100},
        index=index,
    )

    class PureBar(_Collector):
        warmup_period = 2
        freq_arg = None

    run_backtest(strategy=PureBar, data=data, symbols=["X"], initial_cash=1e5)
    assert PureBar.captured and not np.isnan(PureBar.captured[-1])
