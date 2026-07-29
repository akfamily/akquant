# -*- coding: utf-8 -*-
"""每 bar 的 Python 跨界次数上限.

改动前实测 4.08 次/bar(未覆写 on_bar、全程无订单的 cross-section 策略):
  _on_bar_event                     1.00
  _flush_pending_order_events       1.02
  getattr _pending_engine_oco_groups      1.02
  getattr _pending_engine_bracket_plans   1.02

本测试把收敛结果锁成可回归的不变量, 而非一次性测量.
"""

from typing import Any, Dict

import akquant as aq
import numpy as np
import pandas as pd
import pytest

N_SYM = 50
N_DAY = 4
TZ = "Asia/Shanghai"

# 引擎每 bar 可能发起的跨界点
_WATCHED = (
    "_on_bar_event",
    "_on_bar_event_and_flush",
    "_flush_pending_order_events",
    "_pending_engine_oco_groups",
    "_pending_engine_bracket_plans",
)


def _make_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-01-02", periods=N_DAY, freq="B", tz=TZ)
    n = N_SYM * N_DAY
    close = 100.0 + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "date": np.repeat(days.values, N_SYM),
            "symbol": np.tile([f"S{i:03d}" for i in range(N_SYM)], N_DAY),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def _count_crossings() -> Dict[str, int]:
    counts: Dict[str, int] = {}

    class _Counting(aq.Strategy):
        def on_cross_section(self, trading_date: object, timestamp: int) -> None:
            pass

        def __getattribute__(self, name: str) -> Any:
            if name in _WATCHED:
                counts[name] = counts.get(name, 0) + 1
            return object.__getattribute__(self, name)

    data = _make_data()
    _ = aq.run_backtest(
        data=data,
        strategy=_Counting(),
        symbols=sorted(data["symbol"].unique().tolist()),
        initial_cash=1e8,
        show_progress=False,
    )
    return counts


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Task 3 只把跨界从 4.08 降到约 3.06/bar(合并 Bar/Tick 两次调用). "
        "Task 4 消除两处 per-bar getattr 后本测试转绿, 届时移除本标记。"
        "strict=True 保证一旦转绿即报错, 故遗忘收紧不会静默发生。"
    ),
)
def test_per_bar_python_crossings_within_budget() -> None:
    """未覆写 on_bar、无订单的策略, 每 bar 跨界不得超过 2 次."""
    bars = N_SYM * N_DAY
    counts = _count_crossings()
    total = sum(counts.values())
    per_bar = total / bars
    assert per_bar <= 2.0, f"每 bar 跨界 {per_bar:.2f} 次, 超出预算 2.0. 明细: {counts}"


def test_flush_no_longer_called_per_bar_on_bar_path() -> None:
    """Bar 路径的 flush 已并入包装方法, 不应再每 bar 单独调用一次."""
    bars = N_SYM * N_DAY
    counts = _count_crossings()
    assert counts.get("_flush_pending_order_events", 0) < bars
