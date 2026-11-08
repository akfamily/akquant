"""策略要的历史比框架保留的多时: 自动加长并告警一次.

两种情况: ``get_history(count=N)`` 而历史深度小于 N; ``self.I(..., lookback=L)``
之后取 ``ind[n]`` 且 n >= L。此前都是悄悄返回 NaN / None, 且永远拿不到完整数据。
现在第一次超出时把保留长度加到够用并告警一次, 之后的 bar 都能拿到完整数据。
只是预热期数据还不够(没超出保留长度)时不告警。
"""

import logging
from typing import Any, List

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import Bar, Strategy
from akquant.indicator_declaration import IndicatorDeclaration
from akquant.strategy_history import _ensure_history_depth


def _frame(closes: List[float]) -> pd.DataFrame:
    idx = pd.date_range(
        "2024-01-02 09:31", periods=len(closes), freq="min", tz="Asia/Shanghai"
    )
    return pd.DataFrame(
        {
            "date": idx,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000.0,
            "symbol": "X",
        }
    )


def _run(strategy: type, closes: List[float]) -> Any:
    return aq.run_backtest(
        data=_frame(closes), strategy=strategy, symbols="X", show_progress=False
    )


def _warnings(caplog: pytest.LogCaptureFixture, needle: str) -> List[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


CLOSES = [float(100 + i) for i in range(200)]


def test_get_history_over_depth_grows_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """深度 10 时取 50 根: 告警一次, 加长后满 50 根的 bar 起拿到完整数据."""
    seen: dict[int, np.ndarray] = {}

    class Probe(Strategy):
        def on_start(self) -> None:
            self.set_history_depth(10)
            self.k = 0

        def on_bar(self, bar: Bar) -> None:
            self.k += 1
            if self.k >= 20:
                seen[self.k] = self.get_history(count=50, symbol="X", field="close")

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES)

    # 第一次请求时缓冲只留了 10 根, 之前的数据补不回来
    assert int(np.isnan(seen[20]).sum()) == 40
    # 加长后再过 50 根, 数据完整且正确
    assert not np.isnan(seen[80]).any()
    np.testing.assert_array_equal(seen[80], np.array(CLOSES[30:80]))
    hits = _warnings(caplog, "count=50")
    assert len(hits) == 1, hits
    assert "set_history_depth(50)" in hits[0]


def test_get_history_multi_over_depth_grows_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """get_history_multi 同样加长."""
    seen: dict[int, Any] = {}

    class Probe(Strategy):
        def on_start(self) -> None:
            self.set_history_depth(5)
            self.k = 0

        def on_bar(self, bar: Bar) -> None:
            self.k += 1
            if self.k >= 10:
                seen[self.k] = self.get_history_multi(
                    count=30, symbol="X", fields=("close", "high")
                )

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES)

    np.testing.assert_array_equal(seen[60]["close"], np.array(CLOSES[30:60]))
    assert len(_warnings(caplog, "count=30")) == 1


def test_get_history_within_depth_does_not_warn_during_warmup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """没超出深度、只是预热期数据不够(前面补 NaN): 正常情况, 不告警."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.set_history_depth(50)

        def on_bar(self, bar: Bar) -> None:
            self.get_history(count=50, symbol="X", field="close")

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES[:20])

    assert _warnings(caplog, "count=50") == []


def test_indicator_index_over_lookback_grows_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """lookback=3 时取 [10]: 告警一次, 之后拿到 10 根之前的真实值."""
    seen: dict[int, Any] = {}

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), lookback=3)
            self.k = 0

        def on_bar(self, bar: Bar) -> None:
            self.k += 1
            seen[self.k] = (self.sma[10], self.sma[0])

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES[:60])

    # 第 k 根 bar 的 [10] 等于第 k-10 根 bar 的 [0]
    assert seen[40][0] == seen[30][1]
    assert seen[40][0] is not None
    hits = _warnings(caplog, "lookback")
    assert len(hits) == 1, hits
    assert "lookback=11" in hits[0]


def test_indicator_index_within_lookback_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """没超出 lookback、只是预热期还没有值(返回 None): 正常情况, 不告警."""

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(5), lookback=10)

        def on_bar(self, bar: Bar) -> None:
            _ = self.sma[9]

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES[:8])

    assert _warnings(caplog, "lookback") == []


# --- 自动加长的上限: max(10000, 策略显式声明的深度) ---


class _LimitCtx:
    """只记录 ensure_history_capacity 调用的假上下文."""

    def __init__(self) -> None:
        self.calls: List[int] = []

    def ensure_history_capacity(self, capacity: int) -> bool:
        self.calls.append(capacity)
        return True


class _Bare(Strategy):
    def on_bar(self, bar: Bar) -> None:
        pass


def _bare(depth: int, warmup: int = 0) -> Any:
    strategy: Any = _Bare()
    strategy._history_depth = depth
    strategy.warmup_period = warmup
    strategy.ctx = _LimitCtx()
    return strategy


def test_get_history_over_limit_does_not_grow_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """要的超过自动加长上限: 不加长、单独告警一次, 照旧前面补 NaN."""
    seen: dict[int, np.ndarray] = {}

    class Probe(Strategy):
        def on_start(self) -> None:
            self.set_history_depth(10)
            self.k = 0

        def on_bar(self, bar: Bar) -> None:
            self.k += 1
            if self.k in (20, 30, 40):
                seen[self.k] = self.get_history(count=10001, symbol="X")

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES[:40])

    for k, window in seen.items():
        assert len(window) == 10001
        # 缓冲区仍只留 10 根
        np.testing.assert_array_equal(window[-10:], np.array(CLOSES[k - 10 : k]))
        assert int(np.isnan(window).sum()) == 10001 - 10
    hits = _warnings(caplog, "count=10001")
    assert len(hits) == 1, hits
    assert "自动加长上限 10000" in hits[0]
    assert "set_history_depth(10001)" in hits[0]
    assert _warnings(caplog, "已自动加长") == []


def test_history_limit_boundary_is_default_capacity() -> None:
    """没声明更大深度时, 上限是 10000: 10000 加长, 10001 不加长."""
    strategy = _bare(depth=10)
    _ensure_history_depth(strategy, 10001)
    assert strategy.ctx.calls == []
    _ensure_history_depth(strategy, 10000)
    assert strategy.ctx.calls == [10000]


@pytest.mark.parametrize(
    ("depth", "warmup", "warns"),
    [(12000, 0, False), (10, 12000, True)],
    ids=["history_depth", "warmup_period"],
)
def test_explicit_larger_declaration_lifts_limit(
    caplog: pytest.LogCaptureFixture, depth: int, warmup: int, warns: bool
) -> None:
    """显式声明更大的 history_depth / warmup_period 时, 可以加长到 10000 以上."""
    strategy = _bare(depth=depth, warmup=warmup)
    with caplog.at_level(logging.WARNING, logger="akquant"):
        _ensure_history_depth(strategy, 11000)

    assert strategy.ctx.calls == [11000]
    assert _warnings(caplog, "上限") == []
    assert len(_warnings(caplog, "已自动加长到 11000")) == (1 if warns else 0)


def test_indicator_index_over_limit_does_not_grow_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """指标回看超过自动加长上限: 不加长、单独告警一次, 照旧返回 None."""
    seen: List[Any] = []
    lengths: List[int] = []

    class Probe(Strategy):
        def on_start(self) -> None:
            self.sma = self.I(aq.SMA(2), lookback=3)

        def on_bar(self, bar: Bar) -> None:
            seen.append(self.sma[10001])
            lengths.append(len(self.sma))

    with caplog.at_level(logging.WARNING, logger="akquant"):
        _run(Probe, CLOSES[:30])

    assert seen == [None] * 30
    assert max(lengths) == 3
    hits = _warnings(caplog, "lookback")
    assert len(hits) == 1, hits
    assert "自动加长上限 10000" in hits[0]
    assert "lookback=10002" in hits[0]


def test_indicator_limit_boundary() -> None:
    """指标的上限是 max(10000, 声明的 lookback): 10000 加长, 10001 不加长."""
    declaration = IndicatorDeclaration(lookback=3)
    declaration.push_value("X", 1.0)
    declaration.ensure_lookback("X", 10000, "sma")
    assert declaration.lookback == 3
    declaration.ensure_lookback("X", 9999, "sma")
    assert declaration.lookback == 10000
    # 加长到上限后不再加长
    declaration.ensure_lookback("X", 10000, "sma")
    assert declaration.lookback == 10000

    # 声明的 lookback 本身大于 10000 时照常可用, 但不再往上自动加长
    large = IndicatorDeclaration(lookback=12000)
    large.ensure_lookback("X", 11999, "sma")
    assert large.lookback == 12000
    large.ensure_lookback("X", 12000, "sma")
    assert large.lookback == 12000
