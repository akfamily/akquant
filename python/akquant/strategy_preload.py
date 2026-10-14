"""实盘历史预热在策略侧的两件事: on_start 之后的回放, 首根实盘 bar 的重叠检查.

灌入历史缓冲在 ``live/_runner.py::_apply_preload_history`` 完成, 它把 Rust
``Engine.preload_history`` 返回的回放序列挂到策略的 ``_preload_events`` 上。
回放必须等 ``on_start`` 跑完: ``self.I()`` 的参数化写法要求在 ``on_start`` 里
声明, 而主策略的 ``on_start`` 由 Rust 在 ``engine.run()`` 内触发, runner 侧在它
之后没有时点。
"""

from typing import Any

from .log import build_log_extra, get_logger

logger = get_logger("live.preload")

#: 指标流事件的 ``phase`` 取值: 预热回放产出的点 vs 正常会话事件产出的点。
PHASE_PRELOAD = "preload"
PHASE_RUN = "run"


def replay_preload_history(strategy: Any) -> None:
    """按回放序列推进 warmup 计数与声明式指标, 不触发任何用户回调.

    - 基础 bar(``freq is None``)计入 per-symbol warmup: 预热了 N 根就等于
      已经攒了 N 根, 否则 ``warmup_period`` 会让策略在实盘里再白等一遍。
    - ``self.I()`` 指标走与实盘完全相同的 ``_update_incremental_indicators``,
      窗口 bar 驱动 ``freq=`` 指标; 绘图点经 ``_indicator_phase`` 带上
      ``phase="preload"``。每根基础 bar 开始前 flush 上一轮, 让基础 bar 与它
      闭合出的窗口 bar 落在同一个快照周期。
    - 热启动恢复的策略跳过: 指标状态与 warmup 计数都已从存档恢复, 再回放会
      重复 update。

    intrabar 指标不会在回放中试算出临时点: 试算依赖 ``self.ctx``, 而它要到
    首个行情事件才被赋值, 回放时恒为 ``None``, ``_peek_intrabar_indicators``
    直接返回。若将来改成 on_start 之前就注入 ctx, 这里须显式跳过试算 ——
    那时 ``current_window()`` 是预热末尾的窗口, 对更早的 bar 会算出错位值。
    """
    events = strategy.__dict__.pop("_preload_events", None)
    if not events:
        return
    if strategy.is_restored:
        logger.info(
            "策略从存档恢复, 指标状态与 warmup 计数已随存档恢复, 跳过预热回放"
            "(历史缓冲仍已灌入)",
            extra=build_log_extra(phase="live"),
        )
        return
    counts = strategy._symbol_bar_counts
    recorder = getattr(strategy, "_indicator_recorder", None)
    flush = getattr(recorder, "flush_stream_snapshot", None)
    if not callable(flush):
        flush = None
    strategy._indicator_phase = PHASE_PRELOAD
    try:
        for event in events:
            if event.freq is None:
                if flush is not None:
                    flush()
                counts[str(event.symbol)] += 1
            strategy._update_incremental_indicators(event)
        if flush is not None:
            flush()
    finally:
        strategy._indicator_phase = PHASE_RUN


def check_preload_overlap(strategy: Any, symbol: str, timestamp: int) -> None:
    """实盘首根 bar 不晚于该标的预热末戳时点名告警(只告警, 不丢).

    ``HistoryBuffer::update`` 无时间戳去重, 重叠会让这根 bar 在历史里出现两次;
    但自动丢弃会在预热数据本身戳错时吃掉真实 bar, 重复一根的后果远小于丢一根
    (设计 §4.4)。守卫 dict 由 runner 在所有策略间共享, 每个标的只检查一次。
    """
    guard = getattr(strategy, "_preload_overlap_guard", None)
    if not guard:
        return
    last_ts = guard.pop(symbol, None)
    if last_ts is None or timestamp > last_ts:
        return
    logger.warning(
        "实盘首根 bar 与预热数据重叠: 标的 %s 的首根实盘 bar 时间戳 %s 不晚于预热末戳 "
        "%s, 该 bar 照常派发, 历史里会出现重复。预热数据应截止到上一根已收盘 bar",
        symbol,
        timestamp,
        last_ts,
        extra=build_log_extra(phase="live", symbol=symbol),
    )
