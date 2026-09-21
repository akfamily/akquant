"""声明式指标层 ``Strategy.I()``.

对标 TradingView Pine Script: 一次声明即自动增量更新 + 自动上报绘图, 并支持
``ind[0]`` / ``ind[1]`` 序列回溯。取代旧的 ``indicator_mode`` 开关与
``register_incremental_indicator`` / ``register_precomputed_indicator``。

设计契约见 docs/zh/meta/indicator-tradingview-rfc.md。
"""

from typing import Any, List, Optional, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest

SYMBOL = "DECL"
_BASE_NS = 1_672_707_000_000_000_000
_MINUTE_NS = 60_000_000_000


def _bars(closes: List[float]) -> pd.DataFrame:
    """按给定收盘价构造单标的分钟 bar 数据."""
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(
                    _BASE_NS + i * _MINUTE_NS, unit="ns", tz="UTC"
                ),
                "symbol": SYMBOL,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100.0,
            }
            for i, close in enumerate(closes)
        ]
    )


def _run(strategy_cls: type, closes: List[float], **kwargs: Any) -> Any:
    """跑一段最小回测."""
    return run_backtest(
        strategy=strategy_cls,
        data=_bars(closes),
        symbols=[SYMBOL],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
        **kwargs,
    )


def test_series_lookback_returns_current_and_previous_value() -> None:
    """``ind[0]`` 是当前 bar 的值, ``ind[1]`` 是上一根 —— 与 Pine 一致.

    这是声明式指标层存在的根本理由: 判断金叉这类最基本的形态, 用户不该自己
    维护 deque。
    """
    seen: List[Tuple[Optional[float], Optional[float]]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            seen.append((self.sma[0], self.sma[1]))

    _run(Probe, [10.0, 20.0, 30.0, 40.0])

    # SMA(2) 逐根: NaN(未满窗) -> 15.0 -> 25.0 -> 35.0
    assert seen[0] == (None, None)
    assert seen[1] == (15.0, None)
    assert seen[2] == (25.0, 15.0)
    assert seen[3] == (35.0, 25.0)


def test_lookback_out_of_range_returns_none_instead_of_raising() -> None:
    """越界回溯返回 None 而非抛异常 —— 预热期天然越界, 抛错会逼每个策略写 try."""
    seen: List[Any] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), lookback=2)

        def on_bar(self, bar: Bar) -> None:
            seen.append(self.sma[99])

    _run(Probe, [10.0, 20.0, 30.0])

    assert seen == [None, None, None]


def test_value_property_is_alias_of_index_zero() -> None:
    """``.value`` 是 ``[0]`` 的别名, 既有写法不破."""
    seen: List[Tuple[Any, Any]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            seen.append((self.sma.value, self.sma[0]))

    _run(Probe, [10.0, 20.0, 30.0])

    for value, item0 in seen:
        assert value == item0


def test_plot_metadata_triggers_automatic_reporting() -> None:
    """传了绘图参数就自动上报, 无需在 on_bar 里手写 record_indicator.

    这是声明式指标层的第二个核心收益: 计算与绘图在 Pine 里是一件事
    (``plot(sma)``), 在旧 API 里是毫不相干的两件事。
    """

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), pane=0, color="#e91e63", label="SMA2")

        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(Probe, [10.0, 20.0, 30.0, 40.0])
    frame = result.indicator_df()

    assert not frame.empty
    assert set(frame["indicator_key"]) == {"sma"}
    # 首根 bar 未满窗, 不上报; 其余三根各一个点
    assert len(frame) == 3
    assert list(frame["value"]) == [15.0, 25.0, 35.0]

    definitions = result.indicator_definitions.set_index("indicator_key")
    assert definitions.loc["sma", "display_name"] == "SMA2"
    assert definitions.loc["sma", "color"] == "#e91e63"
    assert definitions.loc["sma", "pane"] == 0


def test_declaration_without_plot_metadata_reports_nothing() -> None:
    """没给绘图参数就只算不画 —— 与 Pine 的 ta.sma() 不画一致.

    实盘 sink 是每点一个事件, 多标的 x 多指标默认全开会淹没前端链路。
    """

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(Probe, [10.0, 20.0, 30.0, 40.0])

    assert result.indicator_df().empty


def test_plot_true_reports_without_other_metadata() -> None:
    """显式 plot=True 即可上报, 不必为了画图硬凑一个 pane/color."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(Probe, [10.0, 20.0, 30.0])

    assert len(result.indicator_df()) == 2


def test_warmup_values_are_not_reported() -> None:
    """未就绪(值为 None/NaN)的点不上报 —— Pine 预热期是 na, 不画."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(5), plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(Probe, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    frame = result.indicator_df()

    # 6 根 bar, SMA(5) 只在第 5、6 根就绪
    assert len(frame) == 2
    assert not frame["value"].isna().any()


def test_multi_output_indicator_splits_into_separate_lines() -> None:
    """多值指标按 outputs 展开成多条独立的线, 而不是把元组丢给前端去拆.

    前端的渲染单元是"一条线", 让它解析元组等于把契约复杂度外推给每个消费者。
    """

    class Probe(Strategy):
        def on_start(self) -> None:
            self.macd = self.I(aq.MACD(2, 3, 2), pane=2, outputs=("dif", "dea", "hist"))

        def on_bar(self, bar: Bar) -> None:
            pass

    result = _run(Probe, [10.0, 12.0, 11.0, 15.0, 14.0, 18.0])
    keys = set(result.indicator_df()["indicator_key"])

    assert keys == {"macd.dif", "macd.dea", "macd.hist"}


def test_multi_output_component_access_by_name() -> None:
    """``self.macd.dif[0]`` 取分量, ``self.macd[0]`` 取整个元组.

    三个分量都要断言: 只验第 0 个的话, "分量下标恒取 0" 这种实现缺陷抓不出来。
    """
    seen: List[Tuple[Any, Any, Any, Any]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.macd = self.I(aq.MACD(2, 3, 2), outputs=("dif", "dea", "hist"))

        def on_bar(self, bar: Bar) -> None:
            seen.append(
                (self.macd[0], self.macd.dif[0], self.macd.dea[0], self.macd.hist[0])
            )

    _run(Probe, [10.0, 12.0, 11.0, 15.0, 14.0, 18.0])

    ready = [row for row in seen if row[0] is not None]
    assert ready, "MACD 应在若干根后就绪"
    for whole, dif, dea, hist in ready:
        assert isinstance(whole, tuple) and len(whole) == 3
        assert (dif, dea, hist) == whole
    # 三个分量不能恒等 —— 否则下标映射错了也看不出来
    assert any(row[1] != row[2] for row in ready)
    assert any(row[2] != row[3] for row in ready)


def test_multi_output_component_supports_lookback() -> None:
    """分量同样支持 ``[1]`` 回溯."""
    seen: List[Any] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.macd = self.I(aq.MACD(2, 3, 2), outputs=("dif", "dea", "hist"))

        def on_bar(self, bar: Bar) -> None:
            seen.append((self.macd.hist[0], self.macd.hist[1]))

    _run(Probe, [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 20.0])

    # 后一根的 [1] 应等于前一根的 [0]
    for i in range(1, len(seen)):
        assert seen[i][1] == seen[i - 1][0]


def test_outputs_length_must_match_indicator_arity() -> None:
    """Outputs 个数与指标实际分量数不符时 fail-fast, 不静默丢分量."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.macd = self.I(aq.MACD(2, 3, 2), outputs=("dif", "dea"))

        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(ValueError, match="outputs"):
        _run(Probe, [10.0, 12.0, 11.0, 15.0, 14.0, 18.0])


def _multi_symbol_bars(rows: List[Tuple[str, float]]) -> pd.DataFrame:
    """构造多标的 bar 数据, 每个 (symbol, close) 占一根."""
    per_symbol: dict = {}
    records = []
    for symbol, close in rows:
        i = per_symbol.get(symbol, 0)
        per_symbol[symbol] = i + 1
        records.append(
            {
                "timestamp": pd.Timestamp(
                    _BASE_NS + i * _MINUTE_NS, unit="ns", tz="UTC"
                ),
                "symbol": symbol,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(records)


def test_factory_gives_each_symbol_an_isolated_indicator_state() -> None:
    """多标的下每个 symbol 有独立的指标状态, 互不污染."""
    seen: dict = {}

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(factory=lambda: aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            seen.setdefault(bar.symbol, []).append(self.sma[0])

    run_backtest(
        strategy=Probe,
        data=_multi_symbol_bars(
            [("AAA", 10.0), ("BBB", 100.0), ("AAA", 20.0), ("BBB", 200.0)]
        ),
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        show_progress=False,
        timezone="UTC",
    )

    # 若两个 symbol 共用一个 SMA(2) 实例, AAA 的第二个值会被 BBB 的 100 污染
    assert seen["AAA"] == [None, 15.0]
    assert seen["BBB"] == [None, 150.0]


def test_shared_instance_across_symbols_fails_fast() -> None:
    """传实例(而非 factory)却跨多标的使用时报错, 不静默共用状态."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(ValueError, match="factory"):
        run_backtest(
            strategy=Probe,
            data=_multi_symbol_bars([("AAA", 10.0), ("BBB", 100.0)]),
            symbols=["AAA", "BBB"],
            initial_cash=100000.0,
            show_progress=False,
            timezone="UTC",
        )


def test_freq_binds_indicator_to_subscribed_window() -> None:
    """``freq=`` 让指标只被该周期的窗口 bar 驱动, 不被基础 bar 推进."""
    base_counts: List[int] = []

    class Probe(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.subscribe_bars("3min")

        def on_start(self) -> None:
            self.w = self.I(aq.SMA(2), freq="3min")

        def on_bar(self, bar: Bar) -> None:
            base_counts.append(len(self.w))

    _run(Probe, [float(10 + i) for i in range(9)])

    # 9 根 1min bar -> 3 根 3min 窗口。回溯长度最多 3, 远少于 9
    assert max(base_counts) <= 3


def test_freq_requires_prior_subscribe_bars() -> None:
    """``freq=`` 指向未订阅的周期时报错并指向 subscribe_bars."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.w = self.I(aq.SMA(2), freq="5min")

        def on_bar(self, bar: Bar) -> None:
            pass

    with pytest.raises(ValueError, match="subscribe_bars"):
        _run(Probe, [10.0, 20.0, 30.0])


def test_lookback_buffer_is_bounded() -> None:
    """回溯缓冲有界 —— 实盘是无限流, 无界缓冲必然泄漏."""
    lengths: List[int] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), lookback=3)

        def on_bar(self, bar: Bar) -> None:
            lengths.append(len(self.sma))

    _run(Probe, [float(10 + i) for i in range(20)])

    assert max(lengths) == 3


def test_precomputed_indicator_declared_through_same_entry() -> None:
    """向量化预计算指标走同一个 ``self.I()`` 入口, 且同样支持 ``[n]`` 回溯.

    删掉 indicator_mode 互斥开关的直接收益: 预计算与增量可以在同一个策略里
    共存, 不必二选一。
    """
    seen: List[Tuple[Any, Any]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(aq.Indicator("mom", lambda df: df["close"].diff()))
            self.sma = self.I(aq.SMA(2))

        def on_bar(self, bar: Bar) -> None:
            seen.append((self.mom[0], self.sma[0]))

    _run(Probe, [10.0, 20.0, 35.0, 40.0])

    # diff: NaN, 10, 15, 5
    assert seen[0][0] is None
    assert [row[0] for row in seen[1:]] == [10.0, 15.0, 5.0]
    # 同一策略里增量指标照常工作
    assert seen[-1][1] == 37.5


def test_precomputed_indicator_supports_lookback() -> None:
    """预计算指标的 ``[1]`` 回溯与增量指标语义一致."""
    seen: List[Tuple[Any, Any]] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(aq.Indicator("mom", lambda df: df["close"].diff()))

        def on_bar(self, bar: Bar) -> None:
            seen.append((self.mom[0], self.mom[1]))

    _run(Probe, [10.0, 20.0, 35.0, 40.0])

    for i in range(1, len(seen)):
        assert seen[i][1] == seen[i - 1][0]


def test_indicator_point_carries_confirmed_flag() -> None:
    """指标点位带 ``confirmed`` 字段, 当前恒为 True(bar 闭合后才计算).

    这是为 L2(未闭合 bar 的实时指标)预留的字段位: 届时未确认的临时点发
    ``False``, 前端按同 ``(indicator_key, symbol, timestamp)`` 后来者覆盖。
    现在就占住字段位, 是为了避免 L2 落地时被迫升 MAJOR —— 前端一旦按
    "无此字段 = 永远确认" 写死, 再加就是破坏性的。
    """
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None:
                messages.append(message)

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    _run(Probe, [10.0, 20.0, 30.0], on_event=on_event, stream_batch_size=1)

    points = [m for m in messages if m["type"] == "point"]
    assert points, "应至少有一个指标点事件"
    assert all(m["indicator"]["confirmed"] is True for m in points)


def test_indicator_snapshot_carries_confirmed_flag() -> None:
    """Snapshot 同样带 ``confirmed``, 与 point 口径一致."""
    messages: List[Any] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None:
                messages.append(message)

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), plot=True)

        def on_bar(self, bar: Bar) -> None:
            pass

    _run(Probe, [10.0, 20.0, 30.0], on_event=on_event, stream_batch_size=1)

    snapshots = [m for m in messages if m["type"] == "snapshot"]
    assert snapshots, "应至少有一个指标快照事件"
    assert all(m["snapshot"]["confirmed"] is True for m in snapshots)


def test_stream_schema_version_bumped_for_confirmed() -> None:
    """新增 ``confirmed`` 是向后兼容的加字段, 按约定升 MINOR."""
    major, _, minor = aq.STREAM_SCHEMA_VERSION.partition(".")

    assert (int(major), int(minor)) >= (1, 3), aq.STREAM_SCHEMA_VERSION


def test_precomputed_indicator_fails_fast_in_live_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘下声明向量化预计算指标必须当场中止会话, 不能静默返回 NaN.

    实盘没有完整的历史 DataFrame, ``_prepare_indicators`` 永远不会跑, ``get_value``
    恒 NaN —— 用户会以为策略在正常运行。``run_live`` 装配时已给策略打上
    ``_live_market_data_owner`` 标记, ``self.I`` 据此抛 ``ValueError``; runner 的
    既有语义是**不重抛**运行期异常, 而是记 CRITICAL 并以 ``ABORTED ON ERROR`` 收尾
    (见 ``_runner.py`` 与 ``test_live_runner_lifecycle``), 所以这里断言那条日志。
    """
    import logging

    from akquant import Indicator, run_live
    from akquant.akquant import AssetType, Instrument

    class Probe(Strategy):
        def on_start(self) -> None:
            self.mom = self.I(Indicator("mom", lambda df: df["close"].diff()))

        def on_bar(self, bar: Bar) -> None:
            pass

    bars = [
        Bar(
            int(pd.Timestamp("2024-03-01 09:31", tz="Asia/Shanghai").value)
            + i * 60_000_000_000,
            10.0,
            10.5,
            9.5,
            10.0 + i,
            100.0,
            "600000",
        )
        for i in range(3)
    ]
    with caplog.at_level(logging.CRITICAL):
        run_live(
            strategy_cls=Probe,
            instruments=[
                Instrument(
                    symbol="600000",
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
            gateway_options={"bars": bars},
            cash=1e6,
            show_progress=False,
            duration="30s",
        )

    critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert critical, "实盘会话应因预计算指标而中止并记 CRITICAL"
    assert any("实盘不支持向量化预计算指标" in r.getMessage() for r in critical)
