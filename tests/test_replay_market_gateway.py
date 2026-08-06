"""回放行情网关: 覆盖此前零测试的实盘数据通路."""

from typing import Any, List

import pandas as pd
import pytest
from akquant.akquant import Bar, Tick
from akquant.gateway.brokers.replay.gateway import ReplayMarketGateway


class _FakeFeed:
    """记录推送顺序的假 feed（替代 DataFeed.create_live）."""

    def __init__(self) -> None:
        """初始化推送记录."""
        self.pushed: List[Any] = []

    def add_bar(self, bar: Bar) -> None:
        """记录 bar 推送."""
        self.pushed.append(bar)

    def add_tick(self, tick: Tick) -> None:
        """记录 tick 推送."""
        self.pushed.append(tick)


# Bar/Tick 的构造器带"秒级→纳秒"自动修正: 时间戳 < 1e10 会被乘 1e9。因此测试
# 必须用真实的纳秒级时间戳, 否则断言里的小数字(100/200/300)会被静默改写成
# 1e11/2e11/3e11, 测试对不上。
_BASE_NS = 1_672_707_000_000_000_000  # 2023-01-03 09:30 前后
_MINUTE_NS = 60_000_000_000


def _ns(minutes: int) -> int:
    """构造纳秒级时间戳: 基准时刻 + ``minutes`` 分钟."""
    return _BASE_NS + minutes * _MINUTE_NS


def _bar(ts: int, symbol: str, close: float = 10.0) -> Bar:
    """构造一根 bar."""
    return Bar(
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


def _tick(ts: int, symbol: str, price: float = 10.0) -> Tick:
    """构造一个 tick."""
    return Tick(timestamp=ts, price=price, volume=100.0, symbol=symbol)


def test_start_pushes_bars_in_timestamp_order() -> None:
    """乱序输入必须按时间戳升序推送: live feed 不能排序, 推送顺序即引擎所见顺序."""
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A"],
        bars=[_bar(_ns(30), "A"), _bar(_ns(10), "A"), _bar(_ns(20), "A")],
    )

    gateway.start()

    assert [b.timestamp for b in feed.pushed] == [_ns(10), _ns(20), _ns(30)]


def test_multi_symbol_events_are_globally_interleaved() -> None:
    """多品种必须按时间戳全局交错, 而非按品种分组."""
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A", "B", "C"],
        bars=[
            _bar(_ns(10), "A"),
            _bar(_ns(20), "A"),
            _bar(_ns(15), "B"),
            _bar(_ns(25), "B"),
            _bar(_ns(12), "C"),
        ],
    )

    gateway.start()

    assert [(e.timestamp, e.symbol) for e in feed.pushed] == [
        (_ns(10), "A"),
        (_ns(12), "C"),
        (_ns(15), "B"),
        (_ns(20), "A"),
        (_ns(25), "B"),
    ]


def test_bars_and_ticks_are_merged_by_timestamp() -> None:
    """Bar 与 tick 混合时按时间戳统一排序."""
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A"],
        bars=[_bar(_ns(10), "A"), _bar(_ns(30), "A")],
        ticks=[_tick(_ns(20), "A")],
    )

    gateway.start()

    assert [e.timestamp for e in feed.pushed] == [_ns(10), _ns(20), _ns(30)]
    assert isinstance(feed.pushed[1], Tick)


def test_subscribe_filters_pending_events() -> None:
    """subscribe() 替换过滤集, 只推送订阅内的标的."""
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A", "B"],
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "B")],
    )

    gateway.subscribe(["B"])
    gateway.start()

    assert [e.symbol for e in feed.pushed] == ["B"]


def test_unsubscribe_removes_symbol() -> None:
    """unsubscribe() 从过滤集移除标的."""
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A", "B"],
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "B")],
    )

    gateway.unsubscribe(["A"])
    gateway.start()

    assert [e.symbol for e in feed.pushed] == ["B"]


def test_unsubscribe_all_pushes_nothing() -> None:
    """退订全部后不得推送任何事件.

    回归防护: 若 pending_events 把"过滤集为空"当成"无过滤", 退订反而会推送
    **全部**品种(含从未订阅过的), 与订阅语义完全相反。
    """
    feed = _FakeFeed()
    gateway = ReplayMarketGateway(
        feed=feed,
        symbols=["A"],
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "B"), _bar(_ns(30), "C")],
    )

    gateway.unsubscribe(["A"])

    assert gateway.pending_events == []
    gateway.start()
    assert feed.pushed == []


def test_dataframe_input_matches_bar_list_input() -> None:
    """DataFrame 入参与 list[Bar] 入参产生相同事件序列.

    注意列名必须是 ``date`` 而非 ``timestamp``: ``dataframe_to_bars`` 的
    必需字段映射为 ``{"date": "timestamp", ...}``。
    """
    rows = [(_ns(10), 10.0), (_ns(20), 11.0), (_ns(30), 12.0)]
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(ts, unit="ns", tz="UTC"),
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1000.0,
            }
            for ts, close in rows
        ]
    )

    feed_df = _FakeFeed()
    ReplayMarketGateway(feed=feed_df, symbols=["A"], bars=df).start()

    feed_list = _FakeFeed()
    ReplayMarketGateway(
        feed=feed_list,
        symbols=["A"],
        bars=[_bar(ts, "A", close) for ts, close in rows],
    ).start()

    assert [e.timestamp for e in feed_df.pushed] == [
        e.timestamp for e in feed_list.pushed
    ]
    assert [e.close for e in feed_df.pushed] == [e.close for e in feed_list.pushed]


def test_pending_events_reports_count() -> None:
    """pending_events 暴露待推送事件, builder 据此声明 bounded_event_total."""
    gateway = ReplayMarketGateway(
        feed=_FakeFeed(),
        symbols=["A"],
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "A")],
        ticks=[_tick(_ns(15), "A")],
    )

    assert len(gateway.pending_events) == 3


def test_connect_and_disconnect_are_noops() -> None:
    """无外部连接, connect/disconnect 不应有副作用."""
    gateway = ReplayMarketGateway(
        feed=_FakeFeed(), symbols=["A"], bars=[_bar(_ns(0), "A")]
    )

    gateway.connect()
    gateway.disconnect()

    assert len(gateway.pending_events) == 1


def test_callbacks_are_recorded() -> None:
    """on_bar/on_tick 仅记录回调引用（与 CTPMarketAdapter 一致）."""
    gateway = ReplayMarketGateway(
        feed=_FakeFeed(), symbols=["A"], bars=[_bar(_ns(0), "A")]
    )

    def cb(payload: dict) -> None:
        return None

    gateway.on_bar(cb)
    gateway.on_tick(cb)

    assert gateway.bar_callback is cb
    assert gateway.tick_callback is cb


def test_push_error_does_not_propagate(caplog: pytest.LogCaptureFixture) -> None:
    """推送线程内异常须记日志后退出, 不向上抛（否则 daemon 线程静默死掉无痕迹）."""

    class _BoomFeed(_FakeFeed):
        def add_bar(self, bar: Bar) -> None:
            raise RuntimeError("feed closed")

    gateway = ReplayMarketGateway(
        feed=_BoomFeed(), symbols=["A"], bars=[_bar(_ns(10), "A")]
    )

    gateway.start()

    assert any("replay" in r.getMessage().lower() for r in caplog.records)


def test_build_bundle_declares_bounded_event_total() -> None:
    """Builder 必须声明预期事件总数, runner 据此终止有界会话."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bundle = build_replay_bundle(
        feed=_FakeFeed(),
        symbols=["A"],
        use_aggregator=True,
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "A")],
    )

    assert bundle.metadata is not None
    assert bundle.metadata["bounded_event_total"] == 2
    assert bundle.metadata["broker"] == "replay"


def test_build_bundle_has_no_trader_gateway() -> None:
    """回放只提供行情, 不模拟成交."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bundle = build_replay_bundle(
        feed=_FakeFeed(), symbols=["A"], use_aggregator=True, bars=[_bar(_ns(0), "A")]
    )

    assert bundle.trader_gateway is None
    assert bundle.trader_capabilities is None
    assert bundle.market_gateway is not None


def test_build_bundle_counts_only_subscribed_symbols() -> None:
    """bounded_event_total 只计入订阅内的事件, 否则计数永远达不到."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bundle = build_replay_bundle(
        feed=_FakeFeed(),
        symbols=["A"],
        use_aggregator=True,
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "B")],
    )

    assert bundle.metadata is not None
    assert bundle.metadata["bounded_event_total"] == 1


def test_build_bundle_rejects_empty_data() -> None:
    """空数据必须早失败: 否则 live 循环会静默空跑挂死."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    with pytest.raises(ValueError, match="replay"):
        build_replay_bundle(feed=_FakeFeed(), symbols=["A"], use_aggregator=True)


def test_build_bundle_rejects_all_filtered_out() -> None:
    """数据非空但全被订阅集过滤掉, 同样是空跑, 须报错."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    with pytest.raises(ValueError, match="replay"):
        build_replay_bundle(
            feed=_FakeFeed(),
            symbols=["Z"],
            use_aggregator=True,
            bars=[_bar(_ns(10), "A")],
        )


def test_build_bundle_rejects_non_positive_timestamp_in_bar_list() -> None:
    """list[Bar] 中出现非正时间戳必须早失败, 否则引擎静默丢弃导致计数永远达不到.

    ``Bar`` 构造器的秒级->纳秒自动修正保留符号: 传入 ``-1`` 得到的是
    ``-1_000_000_000``, 不是 ``-1`` 本身——断言要对真实值而非假设值。
    """
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bad_bar = _bar(-1, "A")
    assert bad_bar.timestamp == -1_000_000_000

    with pytest.raises(ValueError, match="replay") as exc_info:
        build_replay_bundle(
            feed=_FakeFeed(),
            symbols=["A"],
            use_aggregator=True,
            bars=[_bar(_ns(10), "A"), bad_bar],
        )

    message = str(exc_info.value)
    assert "A" in message
    assert "-1000000000" in message


def test_build_bundle_rejects_unparseable_date_in_dataframe() -> None:
    """DataFrame 中一行日期无法解析(NaT)会产生非正时间戳, 必须在 build 期报错.

    这是最贴近真实用户的路径: ``pd.to_datetime(..., errors="coerce")`` 把无法
    解析的日期变成 ``NaT``, ``_timestamps_to_utc_ns`` 把 ``NaT`` 填成 Epoch 0
    再按时区本地化, 产出一个负的纳秒时间戳——引擎会把它当"早于快照时间"静默丢弃。
    """
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    df = pd.DataFrame(
        [
            {
                "date": "2023-01-03 09:30:00",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "volume": 1000.0,
                "股票代码": "A",
            },
            {
                "date": "not-a-date",
                "open": 11.0,
                "high": 11.5,
                "low": 10.5,
                "close": 11.0,
                "volume": 1000.0,
                "股票代码": "A",
            },
        ]
    )

    with pytest.raises(ValueError, match="replay"):
        build_replay_bundle(
            feed=_FakeFeed(),
            symbols=["A"],
            use_aggregator=True,
            bars=df,
        )


def test_build_bundle_accepts_valid_data_without_over_rejecting() -> None:
    """回归防护: 新增的非正时间戳校验不能误伤合法数据."""
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bundle = build_replay_bundle(
        feed=_FakeFeed(),
        symbols=["A"],
        use_aggregator=True,
        bars=[_bar(_ns(10), "A"), _bar(_ns(20), "A")],
    )

    assert bundle.metadata is not None
    assert bundle.metadata["bounded_event_total"] == 2
