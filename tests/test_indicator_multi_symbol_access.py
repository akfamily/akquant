"""多标的指标的按标的取值(#434).

- ``self.I(factory=...)`` 的产物是 ``PrecomputedIndicator`` 时按预计算处理(原先
  每根 bar 调 ``update()`` 抛 NotImplementedError);
- ``binding.of(symbol)`` 在任何回调里按标的取值, 支持 ``[n]`` 回溯与多值分量;
- 在 on_bar/on_tick/窗口回调之外(on_timer、on_cross_section 等), 多标的指标
  不指定标的取值直接报错——原先按停在上一根 bar 的 ``current_bar`` 解析,
  静默返回别的标的的值(on_cross_section 里恒为 None)。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import akquant as aq
import numpy as np
import pandas as pd
import pytest
from akquant import (
    Bar,
    PrecomputedIndicator,
    Strategy,
    load_checkpoint,
    run_backtest,
    save_checkpoint,
)

_BASES = {"A": 10.0, "B": 1000.0}


def _daily(symbols: Optional[List[str]] = None, n: int = 6) -> Dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-02 15:00", periods=n, freq="B", tz="Asia/Shanghai")
    out = {}
    for sym in symbols or list(_BASES):
        close = _BASES[sym] + np.arange(n, dtype=float)
        out[sym] = pd.DataFrame(
            {
                "date": idx,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1e6,
                "symbol": sym,
            }
        )
    return out


def _run(strategy: Strategy, data: Optional[Dict[str, pd.DataFrame]] = None) -> Any:
    return run_backtest(
        data=data or _daily(), strategy=strategy, initial_cash=1e6, show_progress=False
    )


def _range_plus_close(df: pd.DataFrame) -> Any:
    return (df["high"] - df["low"]).to_numpy() + df["close"].to_numpy()


class _Log(Strategy):
    """各用例共用: 回调里把观察到的值或错误记进 ``log``."""

    def __init__(self) -> None:
        """初始化记录."""
        self.seen: List[Any] = []

    def _try(self, tag: str, fn: Any) -> None:
        try:
            self.seen.append((tag, fn()))
        except ValueError as exc:
            self.seen.append((tag, f"ValueError: {exc}"))


# --------------------------------------------------------------------------- #
# A. factory 产物是预计算指标
# --------------------------------------------------------------------------- #


class _FactoryPrecomputed(_Log):
    def on_start(self) -> None:
        self.iv = self.I(
            factory=lambda: PrecomputedIndicator("x", _range_plus_close), name="x"
        )

    def on_bar(self, bar: Bar) -> None:
        self.seen.append((bar.symbol, self.iv[0]))


def test_factory_returning_precomputed_works_per_symbol() -> None:
    """原先第一根 bar 就抛 NotImplementedError; 现在两个标的各取各的预计算值."""
    strat = _FactoryPrecomputed()
    _run(strat)
    first = {sym: value for sym, value in strat.seen[:2]}
    assert first == {"A": pytest.approx(12.0), "B": pytest.approx(1002.0)}


class _CountingFactory(_Log):
    calls = 0

    def on_start(self) -> None:
        def make() -> Any:
            type(self).calls += 1
            return aq.SMA(2)

        self.ma = self.I(factory=make, name="ma")


def test_incremental_factory_called_once_per_symbol() -> None:
    """声明时探测用的那个实例交给第一个标的, factory 不多调用."""
    _CountingFactory.calls = 0
    _run(_CountingFactory())
    assert _CountingFactory.calls == 2


# --------------------------------------------------------------------------- #
# B. .of(symbol)
# --------------------------------------------------------------------------- #


class _OfInCallbacks(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma")
        self.iv = self.I(PrecomputedIndicator("x", _range_plus_close), name="x")
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        self.seen.append(
            (
                "timer",
                self.ma.of("A")[0],
                self.ma.of("B")[0],
                self.ma.of("A")[1],
                self.iv.of("A")[0],
                self.iv.of("B")[0],
                len(self.ma.of("A")),
            )
        )

    def on_cross_section(self, trading_date: Any, timestamp: Any) -> None:
        self.seen.append(("xsect", self.ma.of("A")[0], self.ma.of("B")[0]))


def test_of_reads_each_symbol_in_timer_and_cross_section() -> None:
    """on_timer / on_cross_section 里 .of() 取各标的的值, 回溯照常."""
    strat = _OfInCallbacks()
    _run(strat)
    timers = [entry for entry in strat.seen if entry[0] == "timer"]
    # 首日 14:50 早于首个行情, 不补登记, 定时器从第二个交易日起; 14:50 又早于
    # 当日 15:00 的 bar, 所以第三个定时器看到的是收盘价 10、11、12 之后的值
    _, a0, b0, a1, iv_a, iv_b, length = timers[2]
    assert a0 == pytest.approx(11.5)  # A 的 SMA(2) of (11, 12)
    assert b0 == pytest.approx(1001.5)
    assert a1 == pytest.approx(10.5)  # 再前一根
    assert iv_a == pytest.approx(14.0)  # 振幅 2 + 收盘 12
    assert iv_b == pytest.approx(1004.0)
    assert length == 3  # 已收到 3 根 bar(首根是预热期的 None)
    sections = [entry for entry in strat.seen if entry[0] == "xsect"]
    assert sections, "on_cross_section 未触发"
    last = sections[-1]
    assert last[1] == pytest.approx(14.5)
    assert last[2] == pytest.approx(1004.5)


class _OfInBar(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma")

    def on_bar(self, bar: Bar) -> None:
        other = "B" if bar.symbol == "A" else "A"
        self.seen.append((bar.symbol, self.ma[0], self.ma.of(bar.symbol)[0]))
        self._try("other", lambda: self.ma.of(other)[0])


def test_bar_access_unchanged_and_of_agrees() -> None:
    """on_bar 里不带标的取值仍是当前标的, 与 .of(当前标的) 一致."""
    strat = _OfInBar()
    _run(strat)
    for entry in strat.seen:
        if entry[0] in _BASES:
            assert entry[1] == entry[2]
    assert all(not str(e[1]).startswith("ValueError") for e in strat.seen)


class _Macd(_Log):
    def on_start(self) -> None:
        self.macd = self.I(
            factory=lambda: aq.MACD(2, 3, 2),
            name="macd",
            outputs=("dif", "dea", "hist"),
        )
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        whole = self.macd.of("B")[0]
        self.seen.append(
            (
                whole,
                self.macd.of("B").dif[0],
                self.macd.dif.of("B")[0],
                self.macd.hist.of("B")[0],
            )
        )


def test_of_works_with_components() -> None:
    """.of() 与多值分量两种顺序都能组合."""
    strat = _Macd()
    _run(strat, _daily(n=12))
    ready = [entry for entry in strat.seen if entry[0] is not None]
    assert ready
    whole, dif_a, dif_b, hist = ready[-1]
    assert dif_a == pytest.approx(whole[0])
    assert dif_b == pytest.approx(whole[0])
    assert hist == pytest.approx(whole[2])


class _OfUnknown(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma", symbols=["A", "B"])
        self._try("unknown", lambda: self.ma.of("C"))
        self._try("blank", lambda: self.ma.of(" "))


def test_of_rejects_symbol_outside_declaration() -> None:
    """标的不在声明的 symbols 之内(或为空)时报错."""
    strat = _OfUnknown()
    _run(strat)
    log = dict(strat.seen)
    assert "ValueError" in str(log["unknown"]) and "'C'" in str(log["unknown"])
    assert "ValueError" in str(log["blank"])


# --------------------------------------------------------------------------- #
# C. bar/tick 回调之外, 多标的指标不带标的取值报错
# --------------------------------------------------------------------------- #


class _AmbiguousRead(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma")
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        self._try("timer[0]", lambda: self.ma[0])
        self._try("timer.value", lambda: self.ma.value)
        self._try("timer.is_ready", lambda: self.ma.is_ready)

    def on_cross_section(self, trading_date: Any, timestamp: Any) -> None:
        self._try("xsect[0]", lambda: self.ma[0])


def test_ambiguous_read_outside_bar_callbacks_raises() -> None:
    """定时器 / 横截面里多标的指标不写标的取值报错, 并提示 .of()."""
    strat = _AmbiguousRead()
    _run(strat)
    # 首个定时器早于任何 bar: 指标还没算过任何标的, 返回未就绪而非报错
    tags = {tag for tag, value in strat.seen if "ValueError" in str(value)}
    assert {"timer[0]", "timer.value", "timer.is_ready", "xsect[0]"} <= tags
    message = next(
        str(v) for t, v in strat.seen if t == "timer[0]" and "ValueError" in str(v)
    )
    assert ".of(" in message and "'ma'" in message


class _SingleSymbol(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma")
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        self._try("timer", lambda: self.ma[0])

    def on_cross_section(self, trading_date: Any, timestamp: Any) -> None:
        self._try("xsect", lambda: self.ma[0])


def test_single_symbol_strategy_unaffected() -> None:
    """只跑一个标的时不存在歧义, 不带标的取值照常可用."""
    strat = _SingleSymbol()
    _run(strat, _daily(["A"]))
    values = [value for _, value in strat.seen]
    assert not any("ValueError" in str(v) for v in values)
    assert values[-1] == pytest.approx(14.5)


class _FilteredToOne(_Log):
    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma", symbols=["B"])
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        self._try("timer", lambda: self.ma[0])


def test_indicator_declared_for_one_symbol_is_unambiguous() -> None:
    """多标的策略里只声明给一个标的的指标(如基准), 定时器里不带标的取值仍可用."""
    strat = _FilteredToOne()
    _run(strat)
    values = [value for _, value in strat.seen]
    assert not any("ValueError" in str(v) for v in values)
    assert values[-1] == pytest.approx(1003.5)


class _Shared(_Log):
    def on_start(self) -> None:
        self.ma = self.I(aq.SMA(2), name="ma", symbols=["A"])
        self.schedule_daily("14:50:00", "t")

    def on_timer(self, payload: str) -> None:
        self._try("timer", lambda: self.ma[0])


def test_shared_instance_indicator_is_unambiguous() -> None:
    """共享实例只能用于一个标的, 定时器里不带标的取值取它."""
    strat = _Shared()
    _run(strat)
    assert strat.seen[-1][1] == pytest.approx(13.5)


# --------------------------------------------------------------------------- #
# 窗口回调: subscribe_bars 的回调与 on_bar 一样确定当前标的
# --------------------------------------------------------------------------- #


def _minutes() -> Dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-02 09:31", periods=30, freq="1min", tz="Asia/Shanghai")
    out = {}
    for sym, base in _BASES.items():
        close = base + np.arange(30, dtype=float)
        out[sym] = pd.DataFrame(
            {
                "date": idx,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1e6,
                "symbol": sym,
            }
        )
    return out


class _Window(_Log):
    def __init__(self) -> None:
        super().__init__()
        self.subscribe_bars("5min", self.on_five)

    def on_start(self) -> None:
        self.ma = self.I(factory=lambda: aq.SMA(2), name="ma")

    def on_five(self, bar: Bar) -> None:
        self._try(bar.symbol, lambda: self.ma[0])


def test_window_callback_resolves_its_symbol() -> None:
    """subscribe_bars 回调里不带标的取值取该窗口 bar 的标的."""
    strat = _Window()
    _run(strat, _minutes())
    assert strat.seen
    for sym, value in strat.seen:
        assert "ValueError" not in str(value)
        if value is not None:
            # 基础周期 SMA(2) 取窗口回调所属标的: A 在 10 附近, B 在 1000 附近
            assert (value < 100) == (sym == "A")


# --------------------------------------------------------------------------- #
# 存档: 固定了标的的绑定连标的一起恢复
# --------------------------------------------------------------------------- #


class _PyMomentum:
    """纯 Python 增量指标(可 pickle)."""

    def __init__(self) -> None:
        self.prev: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        self.value = None if self.prev is None else close - self.prev
        self.prev = close
        return self.value


def _make_momentum() -> _PyMomentum:
    return _PyMomentum()


class _Pinned(Strategy):
    def on_start(self) -> None:
        self.mom = self.I(factory=_make_momentum, name="mom")
        self.mom_a = self.mom.of("A")
        self.dif_b = self.mom.of("B")


def test_pinned_binding_survives_checkpoint(tmp_path: Path) -> None:
    """.of() 固定的标的随存档恢复."""
    result = _run(_Pinned())
    path = tmp_path / "pinned.pkl"
    save_checkpoint(result.engine, result.strategy, str(path))  # type: ignore[arg-type]
    _, restored = load_checkpoint(str(path))
    assert restored.mom_a[0] == pytest.approx(1.0)
    assert restored.dif_b[0] == pytest.approx(1.0)
    assert restored.mom_a._symbol == "A"
    assert restored.dif_b._symbol == "B"


class _HybridMomentum(PrecomputedIndicator):
    """继承预计算类又实现了 update(): 一直按增量指标用(examples/60 的写法)."""

    def __init__(self) -> None:
        super().__init__("hybrid", lambda df: df["close"].diff())
        self.prev: Optional[float] = None
        self._value: Optional[float] = None

    def update(self, value: float) -> Optional[float]:  # type: ignore[override]
        self._value = None if self.prev is None else value - self.prev
        self.prev = value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value


class _UsesHybrid(_Log):
    def on_start(self) -> None:
        self.mom = self.I(factory=_HybridMomentum, name="mom", warmup_bars=1)

    def on_bar(self, bar: Bar) -> None:
        self.seen.append((bar.symbol, self.mom[0]))


def test_factory_hybrid_with_update_stays_incremental() -> None:
    """warmup_bars 只对增量指标合法: 这里不报错, 说明仍按增量处理."""
    strat = _UsesHybrid()
    _run(strat)
    assert strat.seen[-1] == ("B", pytest.approx(1.0))
