"""run_live 自动预加载历史: 行情源经 ``GatewayBundle.history_provider`` 提供历史.

设计见 ``docs/superpowers/specs/2026-10-21-auto-history-preload-design.md``
(H1–H6)。用真实 ``Engine`` 跑 ``replay`` 的有界会话, 另注册一个包装 replay 的
测试 broker, 往 bundle 上挂假 provider 并可声明 ``metadata["drive"]``。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import AssetType, Instrument, Strategy, run_live
from akquant.akquant import Bar, DataFeed
from akquant.backtest import BacktestStreamEvent
from akquant.gateway import GatewayBundle
from akquant.gateway.brokers.replay.gateway import build_replay_bundle
from akquant.gateway.factory import create_gateway_bundle
from akquant.gateway.registry import register_broker, unregister_broker
from akquant.indicator_stream import to_indicator_message
from akquant.live._runner import LiveRunner

_SYMBOL = "600000.SH"
_BROKER = "replay_hp_test"
_TRADER_ONLY = "trader_only_hp_test"


def _instrument(symbol: str = _SYMBOL) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=1,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bar(close: float, stamp: str, symbol: str = _SYMBOL) -> Bar:
    return Bar(
        timestamp=int(pd.Timestamp(stamp, tz="Asia/Shanghai").value),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        symbol=symbol,
    )


def _history_bars(symbol: str = _SYMBOL) -> List[Bar]:
    """三根已收盘的历史日线, 收盘价可辨认."""
    return [
        _bar(10.0, "2026-08-20 15:00", symbol),
        _bar(11.0, "2026-08-21 15:00", symbol),
        _bar(12.0, "2026-08-22 15:00", symbol),
    ]


@dataclass
class _FakeProvider:
    """记录调用参数的假 history_provider."""

    result: Optional[List[Bar]] = None
    error: Optional[Exception] = None
    calls: List[Tuple[List[str], int]] = field(default_factory=list)

    def __call__(self, symbols: Sequence[str], count: int) -> List[Bar]:
        self.calls.append((list(symbols), count))
        if self.error is not None:
            raise self.error
        return list(self.result if self.result is not None else _history_bars())


@dataclass
class _BrokerConfig:
    provider: Optional[_FakeProvider] = None
    drive: Optional[str] = None


@pytest.fixture
def broker_config() -> Any:
    """注册包装 replay 的测试 broker, 用后注销以免污染全局 registry."""
    config = _BrokerConfig(provider=_FakeProvider())

    def _build(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        bundle = build_replay_bundle(feed, symbols, use_aggregator, **kwargs)
        bundle.history_provider = config.provider
        if config.drive is not None:
            assert bundle.metadata is not None
            bundle.metadata["drive"] = config.drive
        return bundle

    register_broker(_BROKER, _build)
    yield config
    unregister_broker(_BROKER)


def _run(
    strategy: Strategy,
    *,
    freq: str = "1d",
    on_event: Any = None,
    strategies_by_slot: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker=_BROKER,
        trading_mode="paper",
        gateway_options={
            "bars": [_bar(20.0, "2026-08-25 15:00")],
            "freq": freq,
        },
        cash=1_000_000.0,
        duration="60s",
        on_event=on_event,
        strategies_by_slot=strategies_by_slot,
        **kwargs,
    )


class _HistoryProbe(Strategy):
    """首个 on_bar 里抓一次历史; 刻意不调 set_history_depth."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: List[Any] = []
        self.error: Optional[Exception] = None
        self.on_bar_calls = 0

    def on_bar(self, bar: Any) -> None:
        self.on_bar_calls += 1
        if self.captured or self.error is not None:
            return
        try:
            self.captured.append(list(self.get_history(3, field="close")))
        except Exception as exc:  # noqa: BLE001 - 要把异常带出来断言
            self.error = exc


class _InferredSma300(_HistoryProbe):
    """源码里出现 SMA(300), 推断 warmup=300(AST 静态分析, 不必真调用)."""

    def _declared(self) -> Any:
        return aq.SMA(300)


class _InferredSma5000(_HistoryProbe):
    def _declared(self) -> Any:
        return aq.SMA(5000)


class _ExplicitWarmup(_HistoryProbe):
    warmup_period = 700


class _ExplicitDepth(_HistoryProbe):
    def __init__(self) -> None:
        super().__init__()
        self.set_history_depth(800)


# ---------------------------------------------------------------------------
# 默认开启与根数(H2/H3)
# ---------------------------------------------------------------------------


def test_auto_preload_calls_provider_with_floor_and_fills_history(
    broker_config: _BrokerConfig,
) -> None:
    """默认开启: 无指标的策略按下限 500 根取, 首根实盘 bar 就有历史."""
    strategy = _HistoryProbe()
    _run(strategy)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], 500)]
    assert strategy.error is None, f"get_history 抛了异常: {strategy.error!r}"
    assert strategy.captured == [[11.0, 12.0, 20.0]]
    assert strategy.on_bar_calls == 1, "历史只预热, 不驱动 on_bar"


@pytest.mark.parametrize(
    ("strategy_cls", "expected"),
    [
        (_InferredSma300, 900),
        (_InferredSma5000, 10000),
        (_ExplicitWarmup, 700),
        (_ExplicitDepth, 800),
    ],
)
def test_auto_preload_count_rules(
    broker_config: _BrokerConfig, strategy_cls: Any, expected: int
) -> None:
    """根数 = min(10000, max(3×推断 warmup, 500, warmup_period, history 深度))."""
    _run(strategy_cls())

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], expected)]


def test_auto_preload_count_takes_max_over_slots(
    broker_config: _BrokerConfig,
) -> None:
    """Slot 的需求也计入: 主策略不需要历史, slot 推断 300 → 900."""
    _run(_HistoryProbe(), strategies_by_slot={"slot_a": _InferredSma300()})

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], 900)]


# ---------------------------------------------------------------------------
# preload_bars 显式控制(H2)
# ---------------------------------------------------------------------------


def test_preload_bars_zero_disables(broker_config: _BrokerConfig) -> None:
    """preload_bars=0 关闭自动预加载."""
    strategy = _HistoryProbe()
    _run(strategy, preload_bars=0)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == []
    assert strategy.on_bar_calls == 1


@pytest.mark.parametrize(("given", "expected"), [(7, 7), (20000, 10000)])
def test_preload_bars_explicit_count(
    broker_config: _BrokerConfig, given: int, expected: int
) -> None:
    """显式根数原样使用(截到 [1, 10000]), 不再套 500 下限."""
    _run(_InferredSma300(), preload_bars=given)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], expected)]


@pytest.mark.parametrize("bad", [-1, True, 1.5, "10"])
def test_preload_bars_rejects_invalid(bad: Any) -> None:
    """负数/bool/浮点/字符串 fail-fast."""
    with pytest.raises(ValueError, match="preload_bars"):
        LiveRunner(
            strategy_cls=_HistoryProbe,
            instruments=[_instrument()],
            broker="replay",
            preload_bars=bad,
        )


def _history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-08-20 15:00", periods=3, freq="D", tz="Asia/Shanghai"
            ),
            "symbol": [_SYMBOL] * 3,
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1000.0] * 3,
        }
    )


def test_explicit_preload_history_wins_over_provider(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """显式 preload_history 完全优先: 不调 provider, 也不告警(未给 preload_bars)."""
    strategy = _HistoryProbe()
    with caplog.at_level(logging.WARNING):
        _run(strategy, preload_history=_history_frame())

    assert broker_config.provider is not None
    assert broker_config.provider.calls == []
    assert strategy.captured == [[2.0, 3.0, 20.0]]
    assert [r for r in caplog.records if "preload_bars" in r.getMessage()] == []


def test_explicit_preload_history_with_preload_bars_warns_once(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """同时给 preload_history 与正的 preload_bars: 以前者为准并告警一次."""
    strategy = _HistoryProbe()
    with caplog.at_level(logging.WARNING):
        _run(strategy, preload_history=_history_frame(), preload_bars=50)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == []
    assert strategy.captured == [[2.0, 3.0, 20.0]]
    warned = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "preload_bars" in r.getMessage()
    ]
    assert len(warned) == 1


def test_no_provider_keeps_existing_behaviour(caplog: Any) -> None:
    """不提供 provider 的 broker 行为不变: 不预热, 不打预热日志."""
    strategy = _HistoryProbe()
    with caplog.at_level(logging.INFO):
        run_live(
            strategy_cls=strategy,
            instruments=[_instrument()],
            broker="replay",
            trading_mode="paper",
            gateway_options={"bars": [_bar(20.0, "2026-08-25 15:00")], "freq": "1d"},
            duration="60s",
        )

    assert strategy.on_bar_calls == 1
    assert int(getattr(strategy, "_history_depth", 0) or 0) == 0
    assert [r for r in caplog.records if "预热" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# 失败告警、照常启动(H4)
# ---------------------------------------------------------------------------


def test_provider_error_warns_and_session_runs(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """Provider 抛异常: WARNING 带异常类型与信息, 会话照常."""
    assert broker_config.provider is not None
    broker_config.provider.error = ConnectionError("history down")
    strategy = _HistoryProbe()
    with caplog.at_level(logging.WARNING):
        _run(strategy)

    assert strategy.on_bar_calls == 1, "provider 失败不应阻止会话"
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    messages = [r.getMessage() for r in warned]
    assert any("ConnectionError" in m and "history down" in m for m in messages), (
        messages
    )


def test_provider_empty_logs_info_and_session_runs(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """Provider 返回空: 一条 INFO, 无告警, 会话照常."""
    assert broker_config.provider is not None
    broker_config.provider.result = []
    strategy = _HistoryProbe()
    with caplog.at_level(logging.INFO):
        _run(strategy)

    assert strategy.on_bar_calls == 1
    infos = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "history_provider" in r.getMessage()
    ]
    assert infos, "空返回应有一条 INFO"
    # 未传 trading_calendar 的一次性缺省日历告警与预加载无关，不计入
    assert not [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "未传 trading_calendar" not in r.getMessage()
    ]


def test_provider_unmatched_symbols_warns_and_session_runs(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """自动模式下 provider 给的数据与挂载标的对不上: 告警继续, 不 fail-fast."""
    assert broker_config.provider is not None
    broker_config.provider.result = _history_bars("000001.SZ")
    strategy = _HistoryProbe()
    with caplog.at_level(logging.WARNING):
        _run(strategy)

    assert strategy.on_bar_calls == 1
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# tick 驱动默认不开(H5)
# ---------------------------------------------------------------------------


def test_tick_drive_skips_auto_preload(
    broker_config: _BrokerConfig, caplog: Any
) -> None:
    """Metadata drive=tick 时自动模式不调 provider, INFO 提示 preload_bars."""
    broker_config.drive = "tick"
    strategy = _HistoryProbe()
    with caplog.at_level(logging.INFO):
        _run(strategy)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == []
    assert any(
        "tick" in r.getMessage() and "preload_bars" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO
    )


def test_tick_freq_skips_auto_preload(broker_config: _BrokerConfig) -> None:
    """声明的 freq 为 tick 也视为 tick 驱动, 不自动预加载."""
    _run(_HistoryProbe(), freq="tick")

    assert broker_config.provider is not None
    assert broker_config.provider.calls == []


def test_tick_drive_with_explicit_preload_bars_calls_provider(
    broker_config: _BrokerConfig,
) -> None:
    """Tick 驱动下显式 preload_bars=N 仍调用 provider."""
    broker_config.drive = "tick"
    _run(_HistoryProbe(), preload_bars=5)

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], 5)]


def test_bar_drive_metadata_keeps_auto_preload(broker_config: _BrokerConfig) -> None:
    """drive=bar 照常自动预加载."""
    broker_config.drive = "bar"
    _run(_HistoryProbe())

    assert broker_config.provider is not None
    assert broker_config.provider.calls == [([_SYMBOL], 500)]


# ---------------------------------------------------------------------------
# 端到端: 与显式 preload_history 同一条路径(H6)
# ---------------------------------------------------------------------------


class _DeclaredSma(Strategy):
    def on_start(self) -> None:
        self.sma = self.I(aq.SMA(2), name="sma2", pane=0, color="#e91e63", plot=True)

    def on_bar(self, bar: Any) -> None:
        pass


def test_auto_preload_streams_preload_then_run_phase(
    broker_config: _BrokerConfig,
) -> None:
    """端到端: 自动预加载的历史指标点带 phase=preload, 之后接着 phase=run."""
    events: List[BacktestStreamEvent] = []
    _run(_DeclaredSma(), on_event=events.append)

    messages = [to_indicator_message(e) for e in events]
    points = [
        m["indicator"]
        for m in messages
        if m is not None and "indicator" in m and m["indicator"]
    ]
    points = [p for p in points if p["indicator_key"] == "sma2"]
    assert [(p["value"], p["phase"]) for p in points] == [
        (10.5, "preload"),
        (11.5, "preload"),
        (16.0, "run"),
    ]
    assert not np.isnan(points[-1]["value"])


# ---------------------------------------------------------------------------
# 行情/交易分开指定时 provider 随行情侧保留(H1)
# ---------------------------------------------------------------------------


def test_split_brokers_keep_market_side_history_provider(
    broker_config: _BrokerConfig,
) -> None:
    """行情/交易分开指定时 history_provider 随行情侧 bundle 保留."""

    def _trader_only(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, symbols, use_aggregator, kwargs)
        return GatewayBundle(metadata={"broker": _TRADER_ONLY})

    register_broker(_TRADER_ONLY, _trader_only)
    try:
        bundle = create_gateway_bundle(
            market_broker=_BROKER,
            trader_broker=_TRADER_ONLY,
            feed=DataFeed.create_live(),  # type: ignore[attr-defined]
            symbols=[_SYMBOL],
            bars=[_bar(20.0, "2026-08-25 15:00")],
        )
    finally:
        unregister_broker(_TRADER_ONLY)

    assert bundle.history_provider is broker_config.provider


def test_gateway_bundle_history_provider_defaults_to_none() -> None:
    """GatewayBundle 新字段默认 None, 既有 broker 不受影响."""
    assert GatewayBundle().history_provider is None


def _split_bundle(
    market_metadata: Dict[str, Any], trader_metadata: Dict[str, Any]
) -> GatewayBundle:
    """用两个桩 broker 按行情/交易分开指定构建合并后的 bundle."""
    provider = _FakeProvider()

    def _market(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, symbols, use_aggregator, kwargs)
        return GatewayBundle(metadata=dict(market_metadata), history_provider=provider)

    def _trader(
        feed: Any, symbols: Sequence[str], use_aggregator: bool, **kwargs: Any
    ) -> GatewayBundle:
        _ = (feed, symbols, use_aggregator, kwargs)
        return GatewayBundle(metadata=dict(trader_metadata))

    register_broker("market_stub_hp_test", _market)
    register_broker(_TRADER_ONLY, _trader)
    try:
        bundle = create_gateway_bundle(
            market_broker="market_stub_hp_test",
            trader_broker=_TRADER_ONLY,
            feed=DataFeed.create_live(),  # type: ignore[attr-defined]
            symbols=[_SYMBOL],
        )
    finally:
        unregister_broker(_TRADER_ONLY)
        unregister_broker("market_stub_hp_test")
    assert bundle.history_provider is provider
    return bundle


def test_split_brokers_market_side_wins_for_market_data_keys() -> None:
    """交易侧写的 drive/freq 不得盖掉行情侧(否则静默关掉预加载、改掉 self.freq)."""
    bundle = _split_bundle(
        {"drive": "bar", "freq": "1min", "bounded_event_total": 3},
        {"drive": "tick", "freq": "1d", "bounded_event_total": 99, "broker": "t"},
    )

    assert bundle.metadata is not None
    assert bundle.metadata["drive"] == "bar"
    assert bundle.metadata["freq"] == "1min"
    assert bundle.metadata["bounded_event_total"] == 3
    assert bundle.metadata["broker"] == "t", "其余键仍按交易侧覆盖"


def test_split_brokers_drop_trader_only_market_data_keys() -> None:
    """行情侧没声明的行情口径键, 交易侧写了也不采纳."""
    bundle = _split_bundle({}, {"drive": "tick", "freq": "1d"})

    assert bundle.metadata is not None
    assert "drive" not in bundle.metadata
    assert "freq" not in bundle.metadata
