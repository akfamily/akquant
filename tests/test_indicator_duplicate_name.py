"""self.I 重名声明直接报错, 不再静默覆盖(2026-10-30).

此前同名第二次声明会覆盖第一次, 两个绑定读到同一组值且不报错不告警。
"""

from collections import deque
from pathlib import Path
from typing import Any, List

import akquant as aq
import pandas as pd
import pytest
from akquant import Strategy, run_backtest
from akquant.akquant import Bar
from akquant.backtest import run_from_checkpoint
from akquant.checkpoint import save_checkpoint


def _bars(n: int, start: str = "2024-01-02 09:31:00") -> List[Bar]:
    idx = pd.date_range(start, periods=n, freq="1min", tz="Asia/Shanghai")
    return [
        Bar(
            int(ts.value),
            10.0 + i * 0.3,
            10.5 + i * 0.3,
            9.5 + i * 0.3,
            10.0 + (i * i % 7) * 0.4 + i * 0.1,
            100.0,
            "X",
        )
        for i, ts in enumerate(idx)
    ]


def _run(strategy: Any, n: int = 40) -> Any:
    return run_backtest(
        data=_bars(n), strategy=strategy, symbols="X", show_progress=False
    )


class ExplicitDup(Strategy):
    """显式传了相同的 name."""

    def on_start(self) -> None:
        """声明两个同名指标."""
        self.a = self.I(aq.EMA(3), name="x")
        self.b = self.I(aq.SMA(30), name="x")


class AutoClash(Strategy):
    """显式名字与自动生成的名字撞车."""

    def on_start(self) -> None:
        """先自动得到 ema, 再显式叫 ema."""
        self.a = self.I(aq.EMA(3))
        self.b = self.I(aq.SMA(30), name="ema")


class InitThenStart(Strategy):
    """__init__ 与 on_start 各声明一次同名指标."""

    def __init__(self) -> None:
        """声明 x."""
        super().__init__()
        self.a = self.I(aq.EMA(3), name="x")

    def on_start(self) -> None:
        """再声明 x."""
        self.b = self.I(aq.EMA(30), name="x")


class AutoNamed(Strategy):
    """两个同类指标都不传 name."""

    def __init__(self) -> None:
        """初始化记录列表."""
        super().__init__()
        self.rows: List[tuple] = []

    def on_start(self) -> None:
        """声明 EMA(3) 与 EMA(30)."""
        self.a = self.I(aq.EMA(3))
        self.b = self.I(aq.EMA(30))

    def on_bar(self, bar: Bar) -> None:
        """记录两个指标的当前值."""
        self.rows.append((self.a[0], self.b[0]))


def test_explicit_duplicate_name_raises() -> None:
    """显式重名: 报错信息含名字与两个类型名."""
    with pytest.raises(ValueError) as ei:
        _run(ExplicitDup)
    msg = str(ei.value)
    assert "'x'" in msg and "EMA" in msg and "SMA" in msg and "name=" in msg


def test_explicit_name_clashing_with_auto_name_raises() -> None:
    """显式名字与自动名 ema 撞名同样报错."""
    with pytest.raises(ValueError, match="ema"):
        _run(AutoClash)


def test_init_then_on_start_same_name_raises() -> None:
    """__init__ 与 on_start 同名属于同一轮声明."""
    with pytest.raises(ValueError, match="'x'"):
        _run(InitThenStart)


def test_auto_names_are_independent() -> None:
    """不传 name 时自动命名 ema/ema_2, 两个指标取值不同."""
    r = _run(AutoNamed)
    strat = r.strategy
    assert strat is not None
    assert set(strat._incremental_indicators) == {"ema", "ema_2"}
    assert len(strat.rows) > 5
    diffs = [abs(a - b) for a, b in strat.rows if a == a and b == b]
    assert diffs and max(diffs) > 1e-6


class _PySma:
    """可 pickle 的纯 Python SMA(Rust 指标不能 pickle, warm start 用不了)."""

    def __init__(self, period: int) -> None:
        self.period = period
        self.buf: deque[float] = deque(maxlen=period)

    def update(self, value: float) -> None:
        self.buf.append(float(value))

    @property
    def value(self) -> Any:
        if len(self.buf) < self.period:
            return None
        return sum(self.buf) / self.period


def _make_fast() -> _PySma:
    return _PySma(3)


def _make_slow() -> _PySma:
    return _PySma(10)


class WarmTwoNamed(Strategy):
    """带两个具名指标, 记录其值."""

    def __init__(self) -> None:
        """初始化记录列表."""
        super().__init__()
        self.rows: List[tuple] = []

    def on_start(self) -> None:
        """声明两个具名指标."""
        self.fast = self.I(factory=_make_fast, name="fast")
        self.slow = self.I(factory=_make_slow, name="slow")

    def on_bar(self, bar: Bar) -> None:
        """记录两个指标的当前值."""
        self.rows.append((self.fast[0], self.slow[0]))


def test_warm_start_redeclare_is_not_a_duplicate(tmp_path: Path) -> None:
    """热启动重跑 on_start 的同名声明合法, 指标状态延续."""
    ckpt = tmp_path / "w.pkl"
    r1 = _run(WarmTwoNamed, 30)
    s1 = r1.strategy
    assert s1 is not None
    n1 = len(s1.rows)
    save_checkpoint(r1.engine, s1, str(ckpt))  # type: ignore[arg-type]
    phase2 = _bars(10, "2024-01-02 10:05:00")
    r2 = run_from_checkpoint(str(ckpt), data=phase2, symbols="X", show_progress=False)
    s2 = r2.strategy
    assert s2 is not None and s2.is_restored
    assert len(s2.rows) > n1
    fast, slow = s2.rows[-1]
    assert fast == fast and slow == slow  # 非 NaN: 状态延续
    assert fast != slow
