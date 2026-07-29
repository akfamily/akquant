# -*- coding: utf-8 -*-
"""每 bar 的 Python 跨界(Rust→Python pyo3 边界)次数上限.

改动前实测 4.08 次/bar(未覆写 on_bar、全程无订单的 cross-section 策略),
当时这四个名字全部由 Rust 侧发起, 故计数即等于真实跨界次数:
  _on_bar_event                     1.00
  _flush_pending_order_events       1.02
  getattr _pending_engine_oco_groups      1.02
  getattr _pending_engine_bracket_plans   1.02

Task 3 后(合并 Bar/Tick 两次调用)真实跨界降到约 3.0 次/bar:
  _on_bar_event_and_flush           1.00 (Rust 侧单次 call_method1, 取代原
                                          _on_bar_event + _flush_pending_order_events)
  getattr _pending_engine_oco_groups      1.02
  getattr _pending_engine_bracket_plans   1.02
注意不是 3.06——包装方法内部 self._on_bar_event(...) 是纯 Python 内部调用,
不跨 pyo3 边界, 因此 _on_bar_event/_on_tick_event 不再计入 _WATCHED(见下方说明),
3.0 也不是"3.06 少了 0.06"的巧合, 而是去掉了一次原本就不该计的 Python 内部属性访问。

Task 4 后(OCO/bracket 待注册计划改由 _on_*_event_and_flush 的返回值携带,
不再由 Rust 侧每 bar getattr 轮询)真实跨界收敛到约 1.0 次/bar:
  _on_bar_event_and_flush            1.00 (Rust 侧单次 call_method1;
                                           返回 Optional[Tuple[oco, bracket]])
自 Task 4 起, `_pending_engine_oco_groups` / `_pending_engine_bracket_plans`
只由 Python 内部方法 `_take_pending_engine_plans` 读取(self 属性访问,
不跨 pyo3 边界), 因此这两个名字从 _WATCHED 中移除——若仍保留,
__getattribute__ 会把这类纯 Python 内部读取误记成"跨界", 使本测试对
Task 4 的收益完全测不出来(该陷阱与 Task 3 对 _on_bar_event 的处理同源)。

本测试把收敛结果锁成可回归的不变量, 而非一次性测量.
"""

from typing import Any, Dict

import akquant as aq
import numpy as np
import pandas as pd

N_SYM = 50
N_DAY = 4
TZ = "Asia/Shanghai"

# 引擎每 bar 可能发起的跨界点.
#
# 此集合必须严格镜像 Rust 引擎实际发起的 call_method1/getattr 调用点——
# 只收 Rust→Python 的 pyo3 边界穿越, 故意排除纯 Python 内部调用(例如
# _on_bar_event_and_flush 包装方法内部对 self._on_bar_event 的调用: 那是
# Python 对象上的一次属性查找, 不经过 pyo3, 不是"跨界"). 若把它们也计入,
# Task 4 消除两处 getattr 后真实跨界会降到 1.0/bar, 但本集合会因这类
# 内部调用被误记而报出 2.0/bar, 恰好卡在预算线上"侥幸"通过, 从此不再测量
# 它声称测量的东西——这是本次修复要防止的退化。
#
# 当前 Rust 侧调用点(Task 4 改动后):
#   src/engine/core.rs -> _on_bar_event_and_flush(Bar 分支)
#                         _on_tick_event_and_flush(Tick 分支)
#                         _on_timer_event_and_flush(Timer 分支; 紧邻替换原
#                           _on_timer_event + _flush_pending_order_events 配对)
#                         _flush_pending_order_events(仅 flush_terminal_
#                           pending_order_events 的 finalize 路径调用,
#                           与逐 bar 的 Timer 分支无关, 未合并、保留原样)
# 上述三个 _*_and_flush 包装方法均以返回值(Optional[Tuple[oco, bracket]])
# 携带待注册的 OCO 组/bracket 计划, 因此 Rust 侧不再对
# `_pending_engine_oco_groups` / `_pending_engine_bracket_plans` 发起 getattr——
# 这两个名字现在只由 Python 内部的 `_take_pending_engine_plans` 读取
# (self 属性访问, 不跨 pyo3 边界), 故从本集合移除。
# 若上述文件的调用点改变(新增/改名/删除), 必须同步更新本集合, 否则本测试
# 会静默停止测量它声称测量的跨界次数.
_WATCHED = (
    "_on_bar_event_and_flush",
    "_on_tick_event_and_flush",
    "_on_timer_event_and_flush",
    "_flush_pending_order_events",
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
