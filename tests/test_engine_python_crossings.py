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

# `_take_pending_engine_plans` 内部会各读一次这两个列表
# (`oco = self._pending_engine_oco_groups` /
# `bracket = self._pending_engine_bracket_plans`),
# 且只在这三个 wrapper 方法(_on_bar_event_and_flush / _on_tick_event_and_flush /
# _on_timer_event_and_flush)内部被调用一次。这不是 Rust→Python 跨界, 因此
# **不放进 _WATCHED**(放进去会让预算测试重新把这两个名字算作"跨界", 与
# "它们已改为纯 Python 内部读取"的事实矛盾, 也会让预算数值虚高)。
#
# 但完全不监控这两个名字, 会让 Task 4 的收益失去回归锁: 如果将来有人在
# Rust 侧新增的某个 pipeline stage 里重新对它们发起 getattr(即重蹈 Task 4
# 之前 flush_pending_engine_oco_groups/flush_pending_engine_bracket_plans
# 的覆辙), 上面的跨界预算测试完全看不出来(这两个名字根本不在 _WATCHED
# 里), 全量测试和 oco/bracket 功能测试也大概率仍然全绿(数据没丢, 只是
# 多读了一次)——这个新增的轮询会静默潜入代码库。
#
# 因此单独维护这个集合, 只用于
# `test_engine_no_longer_polls_pending_engine_plans` 的回归锁, 不参与跨界
# 预算的计算。`_take_pending_engine_plans` 每次调用对每个列表只读一次,
# 故这两个名字的读取次数应恰好等于 `_FLUSH_WRAPPER_METHODS` 的调用总次数
# (实测 204 == 204, 见该测试)——若 Rust 侧再对它们发起一次 getattr, 读取
# 次数会变成约两倍, 该测试就会变红。
# 不要把这两个名字简化合并回 _WATCHED —— 那样看似省事, 实际会同时破坏
# "跨界预算只算真跨界"与"这两个名字有独立回归锁"两件事。
_PENDING_PLAN_ATTRS = (
    "_pending_engine_oco_groups",
    "_pending_engine_bracket_plans",
)

# 会触发一次 `_take_pending_engine_plans()` 调用的三个 wrapper 方法——
# `_pending_engine_oco_groups` / `_pending_engine_bracket_plans` 的读取次数
# 应恰好等于这三者的调用次数之和
# (见 `test_engine_no_longer_polls_pending_engine_plans`)。
_FLUSH_WRAPPER_METHODS = (
    "_on_bar_event_and_flush",
    "_on_tick_event_and_flush",
    "_on_timer_event_and_flush",
)

# `_count_crossings()` 实际安装 `__getattribute__` 钩子监控的全部名字:
# 真跨界点(_WATCHED)+ 待注册计划的两个内部读取点(_PENDING_PLAN_ATTRS,
# 仅用于独立的回归锁, 不计入跨界预算)。
_ALL_MONITORED = _WATCHED + _PENDING_PLAN_ATTRS


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
            if name in _ALL_MONITORED:
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
    """未覆写 on_bar、无订单的策略, 每 bar 跨界不得超过 2 次.

    只对 _WATCHED(真正的 Rust→Python 跨界点)求和, 不含
    _PENDING_PLAN_ATTRS——后者是 Task 4 起的纯 Python 内部读取, 计入会让
    预算数值失真(详见文件顶部说明与 _PENDING_PLAN_ATTRS 处的注释)。
    """
    bars = N_SYM * N_DAY
    counts = _count_crossings()
    total = sum(counts.get(name, 0) for name in _WATCHED)
    per_bar = total / bars
    assert per_bar <= 2.0, f"每 bar 跨界 {per_bar:.2f} 次, 超出预算 2.0. 明细: {counts}"


def test_flush_no_longer_called_per_bar_on_bar_path() -> None:
    """Bar 路径的 flush 已并入包装方法, 不应再每 bar 单独调用一次."""
    bars = N_SYM * N_DAY
    counts = _count_crossings()
    assert counts.get("_flush_pending_order_events", 0) < bars


def test_engine_no_longer_polls_pending_engine_plans() -> None:
    """Task 4 收益的回归锁.

    `_pending_engine_oco_groups` / `_pending_engine_bracket_plans` 不应再被
    Rust 侧 getattr 轮询。这两个名字被特意排除在跨界预算(_WATCHED)之外
    (它们现在只是 Python 内部 `_take_pending_engine_plans` 的一次 self 属性读取,
    不算跨界),
    但这也意味着上面的预算测试对它们的意外读取完全免疫。本测试单独盯住
    读取次数: `_take_pending_engine_plans` 只在 `_on_bar_event_and_flush`
    / `_on_tick_event_and_flush` / `_on_timer_event_and_flush` 内部各调用
    一次、对每个列表各读一次, 因此读取次数应恰好等于这三个方法的调用总次数
    (实测 204 == 204)。若 Rust 侧(例如新增的某个 pipeline stage)重新对
    这两个名字发起一次 getattr, 读取次数会跳到约两倍, 本测试就会变红——
    即便跨界预算测试、oco/bracket 功能测试、全量测试都可能仍是绿的(数据
    没丢, 只是多轮询了一次)。
    """
    counts = _count_crossings()
    flush_wrapper_calls = sum(counts.get(name, 0) for name in _FLUSH_WRAPPER_METHODS)
    assert flush_wrapper_calls > 0, f"未采到任何 wrapper 调用, 明细: {counts}"
    for name in _PENDING_PLAN_ATTRS:
        reads = counts.get(name, 0)
        assert reads == flush_wrapper_calls, (
            f"{name} 读取 {reads} 次, 期望恰好等于 wrapper 调用次数 "
            f"{flush_wrapper_calls}(_take_pending_engine_plans 每次调用只读一次)——"
            f"偏离说明 Rust 侧可能重新对该名字发起了 getattr 轮询. 明细: {counts}"
        )
