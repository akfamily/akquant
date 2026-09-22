"""实盘: 多 slot(主策略 + study)各自订阅不同窗口周期, 订阅合并下发与指标归属.

回测侧 ``test_window_subscriptions`` 已覆盖跨 slot 合并; 实盘走的是同一个
``configure_engine_window_subscriptions(engine, targets, ...)``, 但此前没有测试
钉住 ``targets`` 确实包含全部 slot。这里用 ``broker="replay"`` + 网关声明 ``freq``
(窗口零延迟闭合)复现三种订阅关系:

- 主策略 15min(全标的)
- study A 5min(全标的)
- study B 5min(只限一个标的) —— 与 A 同周期、标的范围重叠但不完全相同,
  正是 ``_merge_scoped_specs`` 必须合并成一条 ``(None, "5min")`` spec 的场景;
  不合并会被 Rust 侧 ``dedup_and_validate`` 拒绝, 会话直接起不来。
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List

import akquant as aq
import pandas as pd
import pytest
from akquant import AssetType, Bar, Instrument, Strategy, Study, run_live

SYMBOLS = ("600000", "600001")


def _instrument(symbol: str) -> Instrument:
    return Instrument(
        symbol=symbol,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars(minutes: int) -> List[Bar]:
    """两只标的、每分钟一根, 09:31 起; 同一时刻两只标的各一根."""
    idx = pd.date_range(
        "2024-01-02 09:31:00", periods=minutes, freq="1min", tz="Asia/Shanghai"
    )
    out: List[Bar] = []
    for i, ts in enumerate(idx):
        for k, symbol in enumerate(SYMBOLS):
            base = 10.0 + 100.0 * k
            out.append(
                Bar(
                    int(ts.value),
                    base + i,
                    base + i + 0.5,
                    base + i - 0.5,
                    base + i + 0.1,
                    100.0,
                    symbol,
                )
            )
    return out


# 各 slot 的窗口回调计数: (owner, freq_label, symbol) -> 次数。模块级是因为
# run_live 自己实例化策略类, 测试拿不到实例。
WINDOW_HITS: "collections.Counter[tuple[str, str, str]]" = collections.Counter()


class MainStrategy(Strategy):
    """主策略: 15min 窗口(全标的) + 基础周期 SMA."""

    def __init__(self) -> None:
        """在 __init__ 订阅 15min 窗口(subscribe_bars 只认 __init__)."""
        super().__init__()
        self.subscribe_bars("15min")

    def on_start(self) -> None:
        """声明基础周期 SMA 与 15min 窗口 SMA."""
        # 两只标的 → 必须 factory=; 传实例会被"共享实例不可跨 symbol"护栏挡住
        self.sma = self.I(factory=lambda: aq.SMA(2), name="sma_base", plot=True)
        self.sma15 = self.I(
            factory=lambda: aq.SMA(1), name="sma15", freq="15min", plot=True
        )

    def on_bar(self, bar: Bar) -> None:
        """不交易, 只让引擎推进."""
        pass

    def on_window_bar(self, bar: Bar) -> None:
        """计数 15min 窗口回调."""
        WINDOW_HITS[("main", str(bar.freq), bar.symbol)] += 1


class FiveMinStudy(Study):
    """study A: 5min 窗口, 全标的."""

    study_id = "five_all"

    def __init__(self) -> None:
        """订阅 5min 窗口, 全标的."""
        super().__init__()
        self.subscribe_bars("5min")

    def on_start(self) -> None:
        """声明 5min 窗口指标."""
        self.rsi5 = self.I(
            factory=lambda: aq.SMA(1), name="rsi5", freq="5min", plot=True
        )

    def on_window_bar(self, bar: Bar) -> None:
        """计数 5min 窗口回调."""
        WINDOW_HITS[("five_all", str(bar.freq), bar.symbol)] += 1


class FiveMinOneSymbolStudy(Study):
    """study B: 同为 5min, 但只订阅第一只标的 —— 与 A 范围重叠不相同."""

    study_id = "five_one"

    def __init__(self) -> None:
        """订阅 5min 窗口, 只限第一只标的."""
        super().__init__()
        self.subscribe_bars("5min", symbols=[SYMBOLS[0]])

    def on_start(self) -> None:
        """声明只限第一只标的的 5min 窗口指标."""
        self.x5 = self.I(
            factory=lambda: aq.SMA(1),
            name="x5",
            freq="5min",
            symbols=[SYMBOLS[0]],
            plot=True,
        )

    def on_window_bar(self, bar: Bar) -> None:
        """计数 5min 窗口回调(应只有第一只标的)."""
        WINDOW_HITS[("five_one", str(bar.freq), bar.symbol)] += 1


@pytest.fixture()
def live_stream() -> Dict[str, Any]:
    """跑一次 32 分钟的 replay 实盘, 返回按 owner 分组的指标点与窗口回调计数."""
    WINDOW_HITS.clear()
    points: List[dict[str, Any]] = []

    def on_event(event: Any) -> None:
        if aq.is_indicator_stream_event(event):
            message = aq.to_indicator_message(event)
            if message is not None and message["type"] == "point":
                points.append(message["indicator"])

    # 32 根 1min bar: 09:31..10:02 → 15min 窗口 09:45 / 10:00 闭合 + 尾部 flush;
    # 5min 窗口 09:35/40/45/50/55/10:00 闭合 + 尾部 flush
    run_live(
        strategy_cls=MainStrategy,
        instruments=[_instrument(s) for s in SYMBOLS],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(32), "freq": "1min"},
        cash=1_000_000,
        show_progress=False,
        on_event=on_event,
        duration="60s",
        strategy_id="main",
        studies=[FiveMinStudy, FiveMinOneSymbolStudy],
    )
    by_owner: Dict[str, List[dict[str, Any]]] = collections.defaultdict(list)
    for p in points:
        by_owner[p["owner_strategy_id"]].append(p)
    return {"points": by_owner, "hits": dict(WINDOW_HITS)}


def test_live_each_slot_gets_only_its_own_window_freq(
    live_stream: Dict[str, Any],
) -> None:
    """主策略只收 15min 窗口, 两个 study 只收 5min 窗口 —— 订阅按 slot 隔离."""
    hits = live_stream["hits"]
    freqs_by_owner: Dict[str, set[str]] = collections.defaultdict(set)
    for owner, freq, _ in hits:
        freqs_by_owner[owner].add(freq)

    assert freqs_by_owner["main"] == {"15min"}
    assert freqs_by_owner["five_all"] == {"5min"}
    assert freqs_by_owner["five_one"] == {"5min"}


def test_live_overlapping_same_freq_subscriptions_merge_and_scope(
    live_stream: Dict[str, Any],
) -> None:
    """A(全标的)与 B(单标的)同为 5min: 会话能起来(合并成功), 且 B 只收到自己的标的.

    合并后 Rust 只聚合一次, 两个 slot 共享同一份 5min 窗口 —— A 在两只标的上的
    回调次数必须相同, 且等于 B 在第一只标的上的次数(同一份窗口, 派发次数一致)。
    """
    hits = live_stream["hits"]
    a_sym0 = hits.get(("five_all", "5min", SYMBOLS[0]), 0)
    a_sym1 = hits.get(("five_all", "5min", SYMBOLS[1]), 0)
    b_sym0 = hits.get(("five_one", "5min", SYMBOLS[0]), 0)
    b_sym1 = hits.get(("five_one", "5min", SYMBOLS[1]), 0)

    # 09:35/40/45/50/55/10:00 六根整窗 + 10:02 尾部 flush = 7
    assert a_sym0 == 7, hits
    assert a_sym1 == 7, hits
    assert b_sym0 == 7, hits
    assert b_sym1 == 0, "study B 只订阅了第一只标的, 不该收到第二只"


def test_live_main_window_count_matches_15min_grid(live_stream: Dict[str, Any]) -> None:
    """15min 窗口: 09:45 / 10:00 整窗 + 10:02 尾部 flush = 3, 两只标的各自独立."""
    hits = live_stream["hits"]
    for symbol in SYMBOLS:
        assert hits.get(("main", "15min", symbol), 0) == 3, hits


def test_live_indicator_points_carry_slot_owner_and_freq_scope(
    live_stream: Dict[str, Any],
) -> None:
    """流事件里各 slot 的指标点带自己的 owner; 窗口指标点数与窗口回调数一致."""
    by_owner = live_stream["points"]
    hits = live_stream["hits"]

    assert set(by_owner) == {"main", "five_all", "five_one"}
    keys = {owner: {p["indicator_key"] for p in pts} for owner, pts in by_owner.items()}
    assert keys["main"] == {"sma_base", "sma15"}
    assert keys["five_all"] == {"rsi5"}
    assert keys["five_one"] == {"x5"}

    # 窗口指标(SMA(1) 即窗口 close)每次窗口闭合上报一点, 与该 slot 的窗口回调次数一致
    def window_points(owner: str, key: str) -> Dict[str, int]:
        counts: Dict[str, int] = collections.Counter()
        for p in by_owner[owner]:
            if p["indicator_key"] == key:
                counts[p["symbol"]] += 1
        return dict(counts)

    assert window_points("five_all", "rsi5") == {
        SYMBOLS[0]: hits[("five_all", "5min", SYMBOLS[0])],
        SYMBOLS[1]: hits[("five_all", "5min", SYMBOLS[1])],
    }
    assert window_points("five_one", "x5") == {
        SYMBOLS[0]: hits[("five_one", "5min", SYMBOLS[0])]
    }
    assert window_points("main", "sma15") == {
        SYMBOLS[0]: hits[("main", "15min", SYMBOLS[0])],
        SYMBOLS[1]: hits[("main", "15min", SYMBOLS[1])],
    }
