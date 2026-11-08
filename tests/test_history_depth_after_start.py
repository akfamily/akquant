"""运行开始之后才设的历史深度也要落到共享历史缓冲区.

历史缓冲区默认只留 1 行(规则版本 2.3.0 起), 容量全靠策略声明的历史深度撑起来。
``engine.run()`` 开头只按主策略的 ``_history_depth`` 配一次容量, 于是下面几种
写法设的深度此前都到不了缓冲区, ``get_history(N)`` 前面永远是 NaN、也不告警:

- 实盘(``run_live``)里在 ``on_start`` 设: 主策略的 ``on_start`` 在 ``engine.run()``
  内部触发, 晚于那次配置;
- 槽位策略自己设: ``engine.run()`` 只读主策略;
- 直接用 ``Engine.run`` 时在 ``on_start`` 设;
- 运行中(``on_bar`` 等回调里)再调 ``set_history_depth``。

现在 ``engine.run()`` 在全部 ``on_start`` 之后按各策略深度的最大值补齐容量,
``set_history_depth`` 在已有上下文时立即扩容, 取历史时按"已确保过的容量"判断
是否要扩容(声明够而容量没跟上时静默补齐)。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import akquant
import numpy as np
import pandas as pd
import pytest
from akquant import AssetType, Bar, Instrument, Strategy, run_live
from akquant.strategy_history import _ensure_history_depth

SYMBOL = "600000"
N_BARS = 12
CLOSES = [float(10 + i) for i in range(N_BARS)]


def _instrument() -> Instrument:
    return Instrument(
        symbol=SYMBOL,
        asset_type=AssetType.Stock,
        multiplier=1.0,
        margin_ratio=1.0,
        tick_size=0.01,
        lot_size=100,
        option_type=None,
        strike_price=None,
        expiry_date=None,
    )


def _bars() -> List[Bar]:
    idx = pd.date_range(
        "2024-01-02 09:31", periods=N_BARS, freq="1min", tz="Asia/Shanghai"
    )
    return [
        Bar(int(ts.value), close, close, close, close, 100.0, SYMBOL)
        for ts, close in zip(idx, CLOSES)
    ]


def _frame() -> pd.DataFrame:
    idx = pd.date_range(
        "2024-01-02 09:31", periods=N_BARS, freq="1min", tz="Asia/Shanghai"
    )
    return pd.DataFrame(
        {
            "date": idx,
            "open": CLOSES,
            "high": CLOSES,
            "low": CLOSES,
            "close": CLOSES,
            "volume": 100.0,
            "symbol": SYMBOL,
        }
    )


def _run_replay(
    strategy: Strategy, strategies_by_slot: Optional[Dict[str, Any]] = None
) -> None:
    run_live(
        strategy_cls=strategy,
        instruments=[_instrument()],
        broker="replay",
        trading_mode="paper",
        gateway_options={"bars": _bars(), "freq": "1min"},
        cash=1_000_000.0,
        show_progress=False,
        duration="60s",
        strategies_by_slot=strategies_by_slot,
    )


def _run_backtest(
    strategy: Any, strategies_by_slot: Optional[Dict[str, Any]] = None
) -> None:
    akquant.run_backtest(
        data=_frame(),
        strategy=strategy,
        symbols=SYMBOL,
        show_progress=False,
        strategies_by_slot=strategies_by_slot,
    )


def _depth_warnings(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "历史" in r.getMessage()
    ]


def _assert_complete_from_fifth(seen: Dict[int, np.ndarray]) -> None:
    """第 5 根 bar 起 get_history(5) 每个值都在, 且等于最近 5 根收盘价."""
    assert sorted(seen) == list(range(5, N_BARS + 1))
    for k, window in seen.items():
        np.testing.assert_array_equal(
            window, np.array(CLOSES[k - 5 : k]), err_msg=str(k)
        )


class _DepthOnStart(Strategy):
    """在 on_start 设深度 5, 第 5 根 bar 起才第一次取历史.

    第一次取历史就在第 5 根: 若容量此前没跟上, 取历史时再补齐也只能从这一刻起
    攒数据, 前 4 根已经丢了 —— 所以这能区分"深度在开跑前就生效"与"取历史时才补"。
    """

    def __init__(self) -> None:
        super().__init__()
        self.k = 0
        self.seen: Dict[int, np.ndarray] = {}

    def on_start(self) -> None:
        self.set_history_depth(5)

    def on_bar(self, bar: Bar) -> None:
        self.k += 1
        if self.k >= 5:
            self.seen[self.k] = self.get_history(5, SYMBOL, "close")


class _Idle(Strategy):
    """主策略: 什么都不设、不取历史."""

    def on_bar(self, bar: Bar) -> None:
        pass


def test_replay_on_start_depth_reaches_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘 on_start 里 set_history_depth(5): 第 5 根 bar 起数据完整, 不告警."""
    strategy = _DepthOnStart()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run_replay(strategy)

    _assert_complete_from_fifth(strategy.seen)
    assert _depth_warnings(caplog) == []


def test_replay_slot_on_start_depth_reaches_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """实盘槽位策略在自己的 on_start 设深度, 主策略不设: 同样生效."""
    slot = _DepthOnStart()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run_replay(_Idle(), strategies_by_slot={"slot_a": slot})

    _assert_complete_from_fifth(slot.seen)
    assert _depth_warnings(caplog) == []


def test_backtest_slot_on_start_depth_reaches_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """回测槽位策略在自己的 on_start 设深度, 主策略不设: 同样生效."""
    slot = _DepthOnStart()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run_backtest(_Idle, strategies_by_slot={"slot_a": slot})

    _assert_complete_from_fifth(slot.seen)
    assert _depth_warnings(caplog) == []


def test_engine_run_on_start_depth_reaches_buffer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """直接用 Engine.run, 在 on_start 设深度: 同样生效."""
    engine = akquant.Engine()
    engine.use_simple_market(0.0)
    engine.configure_sessions(sessions={"default": "ALWAYS_OPEN"})
    engine.set_cash(100000.0)
    feed = akquant.DataFeed()
    for bar in _bars():
        feed.add_bar(bar)
    feed.sort()
    engine.add_data(feed)
    strategy = _DepthOnStart()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        engine.run(strategy, False)

    _assert_complete_from_fifth(strategy.seen)
    assert _depth_warnings(caplog) == []


def test_set_history_depth_mid_run_grows_immediately(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """运行中(第 1 根 bar)才 set_history_depth(5): 当场扩容, 第 6 根起完整."""

    class DepthOnFirstBar(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.k = 0
            self.seen: Dict[int, np.ndarray] = {}

        def on_bar(self, bar: Bar) -> None:
            self.k += 1
            if self.k == 1:
                self.set_history_depth(5)
            if self.k >= 6:
                self.seen[self.k] = self.get_history(5, SYMBOL, "close")

    strategy = DepthOnFirstBar()
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run_replay(strategy)

    assert sorted(strategy.seen) == list(range(6, N_BARS + 1))
    for k, window in strategy.seen.items():
        np.testing.assert_array_equal(
            window, np.array(CLOSES[k - 5 : k]), err_msg=str(k)
        )
    assert _depth_warnings(caplog) == []


class _FakeCtx:
    """只记录 ensure_history_capacity 调用的假上下文."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = capacity
        self.calls: List[int] = []

    def ensure_history_capacity(self, capacity: int) -> bool:
        self.calls.append(capacity)
        if capacity <= self.capacity:
            return False
        self.capacity = capacity
        return True


class _Bare(Strategy):
    def on_bar(self, bar: Bar) -> None:
        pass


def _bare(depth: int, ctx: Any, warmup: int = 0) -> Any:
    strategy = _Bare()
    strategy._history_depth = depth
    strategy.warmup_period = warmup
    strategy.ctx = ctx
    return strategy


def test_declared_depth_without_capacity_grows_silently_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """声明够而容量没跟上: 静默补齐; 补齐过之后不再碰缓冲区(不每次拿写锁)."""
    ctx = _FakeCtx(capacity=1)
    strategy = _bare(depth=5, ctx=ctx)
    with caplog.at_level(logging.WARNING, logger="akquant"):
        for _ in range(3):
            _ensure_history_depth(strategy, 5)
        _ensure_history_depth(strategy, 3)

    assert ctx.calls == [5]
    assert ctx.capacity == 5
    assert _depth_warnings(caplog) == []


def test_new_context_is_ensured_again() -> None:
    """换了上下文(新一次运行)时重新确认容量, 不沿用上一次的记录."""
    first = _FakeCtx(capacity=1)
    strategy = _bare(depth=5, ctx=first)
    _ensure_history_depth(strategy, 5)
    second = _FakeCtx(capacity=1)
    strategy.ctx = second
    _ensure_history_depth(strategy, 5)

    assert first.calls == [5]
    assert second.calls == [5]


def test_declared_depth_too_small_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """要的比声明的多: 扩容并告警一次."""
    ctx = _FakeCtx(capacity=5)
    strategy = _bare(depth=5, ctx=ctx)
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ensure_history_depth(strategy, 8)
        _ensure_history_depth(strategy, 9)

    assert ctx.calls == [8, 9]
    hits = _depth_warnings(caplog)
    assert len(hits) == 1, hits
    assert "count=8" in hits[0]


def test_set_history_depth_with_context_grows_immediately() -> None:
    """已有上下文时 set_history_depth 当场扩容, 之后取同样深度不再碰缓冲区."""
    ctx = _FakeCtx(capacity=1)
    strategy = _bare(depth=0, ctx=ctx)
    strategy.set_history_depth(5)
    assert ctx.calls == [5]
    assert ctx.capacity == 5

    _ensure_history_depth(strategy, 5)
    assert ctx.calls == [5]


def test_set_history_depth_without_context_does_nothing_else() -> None:
    """没有上下文(__init__ / 回测 on_start)时只记下深度."""
    strategy = _Bare()
    strategy.set_history_depth(7)
    assert strategy._history_depth == 7
