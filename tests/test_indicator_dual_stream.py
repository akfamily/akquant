"""tick/bar 双流下声明式指标只由 bar 推进.

双流(同一标的既有基础 bar 又有逐笔 tick)里, 未开 ``intrabar`` 的基础周期
``self.I()`` 指标此前会被 tick 与 bar **各推进一次**: 分发只比较 ``freq``, 而
tick 与基础 bar 的 ``freq`` 都是 ``None``。结果是 bar 收盘时的值混进了逐笔价,
例如 SMA(2) 等于"最后一笔 tick 与收盘价"的均值, 而不是最近两根 bar 收盘价的
均值——指标值是错的, 也没有任何报错。

纯 tick 会话(没有 bar)里 tick 推进基础指标是正确且被依赖的行为, 不变。
是否双流按标的判定, 会话开始前由入口注入(回测按数据, 实盘按网关声明)。
"""

from __future__ import annotations

from typing import Any, List, Tuple

import akquant as aq
import pandas as pd
import pytest
from akquant import Bar, Strategy, run_backtest
from akquant.akquant import Tick
from akquant.strategy_framework_hooks import (
    is_dual_stream_symbol,
    normalize_dual_stream_declaration,
)

_BASE_NS = int(pd.Timestamp("2024-01-02 09:31:00", tz="Asia/Shanghai").value)
_HALF_MIN_NS = 30_000_000_000
_MIN_NS = 60_000_000_000


def _ticks(
    symbol: str, count: int, *, start: float = 10.0, step: float = 0.5
) -> List[Tick]:
    """每 30 秒一笔 tick, 价格每笔 +step."""
    return [
        Tick(
            timestamp=_BASE_NS + i * _HALF_MIN_NS,
            price=start + step * i,
            volume=100.0,
            symbol=symbol,
        )
        for i in range(count)
    ]


def _assert_sma2_follows_bars(
    at_bar: List[Tuple[float, Any]], closes: List[float]
) -> None:
    """每个 (收盘价, SMA(2)) 都等于"上一根与这一根 bar 收盘价"的均值.

    期望值按**完整**收盘价序列算, 不按 on_bar 里见过的: warmup 推断会让首根
    bar 不触发 on_bar, 但它照样推进指标。
    """
    assert at_bar
    for close, got in at_bar:
        i = closes.index(close)
        expected = None if i == 0 else (closes[i - 1] + closes[i]) / 2
        assert got == (pytest.approx(expected) if expected is not None else None), (
            close,
            got,
        )


class _BarProbe(Strategy):
    """记录每根 bar 收盘时 SMA(2) 的值, 以及 tick 回调里看到的值."""

    def __init__(self) -> None:
        super().__init__()
        self.at_bar: List[Tuple[str, float, Any]] = []
        self.at_tick: List[Tuple[str, float, Any]] = []

    def on_start(self) -> None:
        self.sma = self.I(factory=lambda: aq.SMA(2), name="sma")

    def on_tick(self, tick: Tick) -> None:
        self.at_tick.append((str(tick.symbol), float(tick.price), self.sma[0]))

    def on_bar(self, bar: Bar) -> None:
        self.at_bar.append((str(bar.symbol), float(bar.close), self.sma[0]))


def _run(data: List[Any], symbols: List[str], **kwargs: Any) -> _BarProbe:
    probe = run_backtest(
        strategy=_BarProbe,
        data=data,
        symbols=symbols,
        initial_cash=1e5,
        show_progress=False,
        **kwargs,
    ).strategy
    assert isinstance(probe, _BarProbe)
    return probe


def test_dual_stream_backtest_base_indicator_follows_bars_only() -> None:
    """Tick 聚合成 bar 的双流回测: SMA(2) 等于最近两根 bar 收盘价的均值."""
    probe = _run(_ticks("X", 12), ["X"], freq="1min")

    # 每分钟两笔 tick(10.0+0.5i), 第 k 根 bar 收盘价是该分钟第二笔
    closes = [10.0 + 0.5 * (2 * k + 1) for k in range(6)]
    _assert_sma2_follows_bars([(c, v) for _, c, v in probe.at_bar], closes)


def test_dual_stream_ticks_do_not_move_base_indicator_between_bars() -> None:
    """双流下 tick 回调里读到的仍是上一根 bar 的确认值, 不随逐笔变化."""
    probe = _run(_ticks("X", 12), ["X"], freq="1min")

    bar_values = [got for _, _, got in probe.at_bar]
    tick_values = [got for _, _, got in probe.at_tick]
    assert tick_values, "双流回测应触发 on_tick"
    assert set(tick_values) <= set(bar_values) | {None}


def test_pure_tick_backtest_ticks_still_drive_base_indicator() -> None:
    """纯 tick 会话(不传 freq, 没有 bar): tick 照旧推进基础指标."""
    probe = _run(_ticks("X", 4), ["X"])

    assert not probe.at_bar
    prices = [price for _, price, _ in probe.at_tick]
    values = [got for _, _, got in probe.at_tick]
    assert values[1] == pytest.approx((prices[0] + prices[1]) / 2)
    assert values[3] == pytest.approx((prices[2] + prices[3]) / 2)


def test_mixed_input_only_symbols_with_bars_skip_ticks() -> None:
    """混合 [Bar, Tick] 输入: 按标的判定双流.

    有 bar 的标的只由 bar 推进, 只有 tick 的标的照旧由 tick 推进。
    """
    bars = [
        Bar(
            _BASE_NS + (k + 1) * _MIN_NS - 1,
            20.0 + 10 * k,
            20.0 + 10 * k,
            20.0 + 10 * k,
            20.0 + 10 * k,
            100.0,
            "A",
        )
        for k in range(3)
    ]
    data: List[Any] = [
        *bars,
        *_ticks("A", 6, start=1000.0),
        *_ticks("B", 4, start=1.0, step=1.0),
    ]

    probe = _run(data, ["A", "B"])

    _assert_sma2_follows_bars(
        [(c, v) for sym, c, v in probe.at_bar if sym == "A"], [20.0, 30.0, 40.0]
    )

    b_ticks = [(price, got) for sym, price, got in probe.at_tick if sym == "B"]
    assert b_ticks[1][1] == pytest.approx((1.0 + 2.0) / 2)
    assert b_ticks[3][1] == pytest.approx((3.0 + 4.0) / 2)


def test_live_replay_dual_stream_base_indicator_follows_bars_only() -> None:
    """实盘 replay 同时回放 bar 与 tick: 网关声明双流, SMA(2) 只由 bar 推进."""
    from akquant import run_live
    from akquant.akquant import AssetType, Instrument

    bars = [
        Bar(
            _BASE_NS + (k + 1) * _MIN_NS - 1,
            10.0 + k,
            10.5 + k,
            9.5 + k,
            10.0 + k,
            100.0,
            "600000",
        )
        for k in range(4)
    ]
    ticks = _ticks("600000", 8, start=500.0)
    seen: List[Tuple[float, Any]] = []

    class LiveProbe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), name="sma")

        def on_tick(self, tick: Tick) -> None:
            pass

        def on_bar(self, bar: Bar) -> None:
            seen.append((float(bar.close), self.sma[0]))

    run_live(
        strategy_cls=LiveProbe,
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
        gateway_options={"bars": bars, "ticks": ticks, "freq": "1min"},
        cash=1e6,
        show_progress=False,
        duration="30s",
    )

    _assert_sma2_follows_bars(seen, [10.0, 11.0, 12.0, 13.0])


def test_replay_bundle_declares_dual_stream_symbols() -> None:
    """Replay 按标的声明: 只有同时回放了 bar 与 tick 的标的算双流."""
    from akquant import DataFeed
    from akquant.gateway.brokers.replay.gateway import build_replay_bundle

    bar = Bar(_BASE_NS + _MIN_NS - 1, 1.0, 1.0, 1.0, 1.0, 1.0, "A")
    bundle = build_replay_bundle(
        DataFeed(),
        ["A", "B"],
        True,
        bars=[bar],
        ticks=[*_ticks("A", 1), *_ticks("B", 1)],
    )
    assert bundle.metadata is not None
    assert bundle.metadata["dual_stream"] == frozenset({"A"})


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (None, False),
        (False, False),
        (True, True),
        (["A", "B"], frozenset({"A", "B"})),
        (frozenset(), False),
    ],
)
def test_normalize_dual_stream_declaration(declared: Any, expected: Any) -> None:
    """网关元数据 ``dual_stream``: 布尔表示整个会话, 集合表示这些标的; 缺省为否."""
    assert normalize_dual_stream_declaration(declared) == expected


def test_is_dual_stream_symbol_reads_session_flag() -> None:
    """True 覆盖全部标的, 集合只覆盖其中的标的."""

    class Holder:
        _framework_dual_stream: Any = False

    holder = Holder()
    assert is_dual_stream_symbol(holder, "A") is False
    holder._framework_dual_stream = True
    assert is_dual_stream_symbol(holder, "A") is True
    holder._framework_dual_stream = frozenset({"A"})
    assert is_dual_stream_symbol(holder, "A") is True
    assert is_dual_stream_symbol(holder, "B") is False
