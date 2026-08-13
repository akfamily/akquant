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
    """在第一次 on_bar 时按指定 freq/field 取历史."""

    warmup_period = 3
    freq_arg: Optional[str] = None
    field_arg: str = "close"
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
                count=3, symbol=bar.symbol, field=self.field_arg, freq=self.freq_arg
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


class _MapCollector(Strategy):
    """在第一次 on_bar 时按指定 freq 取 get_history_map 结果.

    用于验证 freq 能在 get_history_map 内部 "逐 symbol 调用 get_history" 的
    循环中被正确传递(该循环是 self 派发, 曾在自查阶段出过回归)。
    """

    warmup_period = 3
    freq_arg: Optional[str] = None
    captured: dict = {}

    def on_start(self) -> None:
        """重置采集状态."""
        self.done = False
        type(self).captured = {}

    def on_bar(self, bar: Bar) -> None:
        """首个非预热 bar 上取一次 get_history_map."""
        if self.done:
            return
        self.done = True
        type(self).captured = self.get_history_map(
            count=3, symbols=["X"], field="close", freq=self.freq_arg
        )


def test_dual_stream_get_history_map_bar_freq_is_not_polluted_by_ticks() -> None:
    """get_history_map 在双流 + freq='bar' 下必须拿到未被 tick 污染的 bar 收盘价.

    本用例验证 freq 是否穿透 get_history_map 内部按 symbol 逐个调用
    self.get_history(...) 的循环, 用单 symbol 构造即可覆盖(循环体本身对每个
    symbol 的处理是一致的, 不需要靠多 symbol 才能暴露"漏转发"这类问题)。

    历史上多 symbol 双流构造曾触发过一个与本任务(暴露 freq 参数)无关的
    既有引擎缺陷: fill_missing_bars 会给"本批次未见"的 symbol 用 last_prices
    (成交价)合成退化假 bar 塞进 bar 序列, 双流下 tick 与聚合 bar 的时间戳
    天然错开, 于是每个 bar 事件都会让其余 symbol 显得"本批次缺失", 导致假
    bar 混入、bar 序列被 tick 价污染。该缺陷已在别的修复轮次修复(见
    src/pipeline/stages/data.rs 里对 `buffer.has_tick_history(symbol)` 的
    判断: 跑 tick 流的 symbol 不再走 fill_missing_bars 补合成 bar), 并由
    test_multi_symbol_dual_stream_bar_history_is_not_synthesized_from_tick_price
    专门覆盖多 symbol 场景下"不被 tick 价污染"这件事。本用例的职责仍然只是
    验证 get_history_map 的 freq 转发本身没有在按 symbol 循环里丢失, 用单
    symbol 构造更聚焦、也更省样板代码, 与那个已修复的引擎缺陷无关。
    """

    class BarMapStrategy(_MapCollector):
        freq_arg = "bar"

    run_backtest(
        strategy=BarMapStrategy,
        data=_dual_stream_feed(),
        symbols=["X"],
        initial_cash=1e5,
    )
    assert list(BarMapStrategy.captured["X"]) == [13.0, 23.0, 33.0]


def test_dual_stream_get_history_map_tick_freq_returns_tick_series_per_symbol() -> None:
    """get_history_map 在双流 + freq='tick' 下必须拿到逐 tick 序列(而非 bar).

    同 test_dual_stream_get_history_map_bar_freq_is_not_polluted_by_ticks 的
    多 symbol 说明: 这里同样退回单 symbol, 原因见该用例的 docstring。
    """

    class TickMapStrategy(_MapCollector):
        freq_arg = "tick"

    run_backtest(
        strategy=TickMapStrategy,
        data=_dual_stream_feed(),
        symbols=["X"],
        initial_cash=1e5,
    )
    values = list(TickMapStrategy.captured["X"])
    assert len(values) == 3
    assert values[-1] - values[-2] == 1.0


def _dual_stream_feed_multi() -> DataFeed:
    """两个 symbol 各自双流: X 的分钟 n 价格为 n*10+k, Y 为 n*100+k(k=0..3).

    故分钟 n 的 bar close: X = n*10+3, Y = n*100+3。
    """
    feed = DataFeed()
    aggregator = BarAggregator(
        feed, 1, volume_is_cumulative=False, stamp_bar_at_interval_end=True
    )
    for i in range(40):
        timestamp = T0 + 15_000_000_000 * i
        for symbol, mult in (("X", 10), ("Y", 100)):
            price = float((i // 4 + 1) * mult + (i % 4))
            aggregator.on_tick(symbol, price, 10.0, timestamp)
            feed.add_tick(
                Tick(timestamp=timestamp, price=price, volume=10.0, symbol=symbol)
            )
    return feed


def test_multi_symbol_dual_stream_bar_history_is_not_synthesized_from_tick_price() -> (
    None
):
    """多 symbol 双流下, bar 序列不得混入 fill_missing_bars 用 tick 价合成的假 bar.

    修复前 X 会拿到 [13.0, 20.0, 23.0]（20.0 是 X 自己的 tick 价），
    Y 会拿到 [103.0, 203.0, 203.0]（203 重复）。

    warmup_period 取 5 而非直觉上的 3: `strategy._bar_count`
    (strategy_events.py:109) 是跨 symbol 的全局计数器, 每收到一个真实
    Bar 事件(不分 symbol)就 +1。双 symbol 下每个真实分钟会产生 2 个 Bar
    事件, 计数器按 symbol 数缩放, warmup_period=3 会在每个 symbol 各自
    只攒够 2 根真 bar 时就跨过门槛, 导致 get_history(count=3) 里混入 1 个
    nan 占位。这是纯 bar 路径也存在的既有缺陷(与本用例要验证的"tick 价
    污染 bar 序列"正交, 已确认与本任务无关, 另行跟踪), 此前恰被本任务修的
    污染 bug 掩盖——污染多塞一条(错误的)记录, 把长度顶到了 3, 掩盖了"实际
    只有 2 条真数据"这个事实。此处取 5 以绕开这个既有缺陷, 让断言精确落在
    "bar 序列是否被 tick 价污染"上。
    """
    captured: dict[str, list[float]] = {}

    class MultiSymbolStrategy(Strategy):
        warmup_period = 5

        def on_start(self) -> None:
            """重置采集状态."""
            self.seen: set[str] = set()

        def on_bar(self, bar: Bar) -> None:
            """每个 symbol 首次收到 bar 时取一次 bar 历史."""
            if bar.symbol in self.seen:
                return
            self.seen.add(bar.symbol)
            captured[bar.symbol] = [
                float(x)
                for x in self.get_history(
                    count=3, symbol=bar.symbol, field="close", freq="bar"
                )
            ]

    run_backtest(
        strategy=MultiSymbolStrategy,
        data=_dual_stream_feed_multi(),
        symbols=["X", "Y"],
        initial_cash=1e5,
    )
    assert captured["X"] == [13.0, 23.0, 33.0]
    assert captured["Y"] == [103.0, 203.0, 303.0]


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


def test_tick_freq_field_price_returns_tick_price_series() -> None:
    """field='price' 是 tick 的唯一迁移出口, 语义 = tick 容器的成交价(close).

    修复前 Rust 侧字段解析只认 open/high/low/close/volume, 'price' 落到
    "Invalid field" 分支 -- 尽管 Python 侧白名单
    (``strategy_history._TICK_ALLOWED_FIELDS``)与文档/报错文案都已把它当作
    合法字段推荐给用户。这里必须拿到与 field='close' 完全一致的 tick 序列
    (相邻价差为 1, 与 test_dual_stream_tick_history_returns_tick_series 同源)。
    """

    class PriceStrategy(_Collector):
        freq_arg = "tick"
        field_arg = "price"

    run_backtest(
        strategy=PriceStrategy,
        data=_dual_stream_feed(),
        symbols=["X"],
        initial_cash=1e5,
    )
    values = PriceStrategy.captured
    assert len(values) == 3
    assert values[-1] - values[-2] == 1.0


def test_bar_freq_field_price_still_rejected() -> None:
    """freq='bar' 下 field='price' 必须继续报错: bar 没有 price 字段.

    回归防护: 修 field='price' 的 tick 语义时, 不能顺带让 bar 路径也接受它
    -- bar 放行 'price' 会让用户以为拿到了什么有意义的东西, 而 bar 容器
    根本没有这个概念。
    """

    class BarPriceStrategy(_Collector):
        freq_arg = "bar"
        field_arg = "price"

    with pytest.raises(ValueError, match="price"):
        run_backtest(
            strategy=BarPriceStrategy,
            data=_dual_stream_feed(),
            symbols=["X"],
            initial_cash=1e5,
        )
