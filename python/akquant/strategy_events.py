from typing import Any

import numpy as np

from .akquant import Bar, StrategyContext, Tick
from .log import build_log_extra, get_logger
from .strategy_framework_hooks import (
    call_user_callback,
    dispatch_cross_section_timer,
    dispatch_lifecycle_timer,
    dispatch_portfolio_update,
    ensure_framework_state,
    is_lifecycle_payload,
    mark_portfolio_dirty,
    register_cross_section_timers,
)
from .strategy_ml import (
    activate_pending_model,
    begin_training_cycle,
    consume_training_trigger,
    finalize_training_cycle,
    should_trigger_training,
)
from .strategy_order_events import check_expiry_events
from .strategy_preload import check_preload_overlap
from .strategy_scheduler import (
    PERIODIC_TIMER_PREFIXES,
    flush_pending_schedules,
    live_now,
    next_occurrence,
)

logger = get_logger("strategy.events")


def _periodic_timer_kind(payload: str) -> str | None:
    """周期定时器 payload（``__daily__|``/``__weekly__|``/``__monthly__|``）的种类."""
    for kind, prefix in PERIODIC_TIMER_PREFIXES.items():
        if payload.startswith(prefix):
            return kind
    return None


def _is_before_active_start(strategy: Any, timestamp: int) -> bool:
    """Return whether the event timestamp is still in the preload-only window."""
    active_start_time = getattr(strategy, "_active_start_time_ns", None)
    return active_start_time is not None and timestamp < int(active_start_time)


def _flush_indicator_snapshots(strategy: Any) -> None:
    """Flush one callback cycle of pending indicator snapshot events."""
    recorder = getattr(strategy, "_indicator_recorder", None)
    if recorder is not None:
        recorder.flush_stream_snapshot()


def _next_symbol_warmup_count(strategy: Any, symbol: str) -> int:
    """返回该 symbol 处理完当前 bar 后的 warmup 计数.

    正常路径就是"在上次计数基础上 +1"。

    热启动兼容: 若当前快照缺 ``_symbol_bar_counts`` 字段(早于 per-symbol
    warmup 计数落地的旧存档, 见 ``Strategy.__setstate__`` 里
    ``_symbol_bar_counts_needs_history_backfill`` 的置位逻辑), 该 symbol
    在本次会话第一次出现时(用 ``in`` 判断 ``symbol`` 是否已作为 key 出现在
    ``_symbol_bar_counts`` 里, 避免 defaultdict 的 ``__getitem__`` 副作用把
    它误判为"已出现过") 改为查询 Rust 历史缓冲区(checkpoint 的
    history_buffer_snapshot 已经把它完整恢复)里该 symbol **实际**有多少根
    bar, 而不是从 0 重新计数——否则会白白重放一遍其实已经攒够的 warmup。
    若该 symbol 在存档时历史本就不足 warmup_period 根(快照恰好存在预热期
    内), 这里查到的实际深度同样小于 warmup_period, 预热会正确地从此处
    继续、而不是直接放行。

    新快照(带 ``_symbol_bar_counts``)不受影响: 该 symbol 只要在存档前出现
    过, key 就已经存在, 不会走到这条回填分支。
    """
    if symbol not in strategy._symbol_bar_counts and getattr(
        strategy, "_symbol_bar_counts_needs_history_backfill", False
    ):
        warmup_period = int(getattr(strategy, "warmup_period", 0) or 0)
        if warmup_period > 0:
            try:
                history = strategy.get_history(
                    count=warmup_period, symbol=symbol, field="close", freq="bar"
                )
                return int(np.count_nonzero(~np.isnan(history)))
            except Exception as exc:
                logger.warning(
                    "热启动 warmup 回填失败(symbol=%s), 该标的将从 0 重新计数: %s",
                    symbol,
                    exc,
                )
    return int(strategy._symbol_bar_counts[symbol]) + 1


def _drive_local_stops(
    strategy: Any,
    symbol: str,
    last: float,
    high: float | None = None,
    low: float | None = None,
) -> None:
    """broker_live: 每 bar/tick 盯价触发本地止损(回测无该方法则跳过)."""
    execution = getattr(strategy, "execution", None)
    check = getattr(execution, "check_stop_triggers", None)
    if callable(check):
        check(symbol, last, high=high, low=low)


def on_bar_event(strategy: Any, bar: Bar, ctx: StrategyContext) -> None:
    """引擎调用的 Bar 回调 (Internal)."""
    ensure_framework_state(strategy)
    strategy.ctx = ctx
    flush_pending_schedules(strategy)
    register_cross_section_timers(strategy)
    strategy._last_event_type = "bar"

    strategy._check_order_events()
    check_expiry_events(strategy)

    symbol = bar.symbol
    current_pos = ctx.get_position(symbol)

    if current_pos == 0:
        strategy._hold_bars[symbol] = 0
        strategy._last_position_signs[symbol] = 0.0
    else:
        current_sign = np.sign(current_pos)
        prev_sign = strategy._last_position_signs[symbol]

        if current_sign != prev_sign:
            strategy._hold_bars[symbol] = 1
        else:
            strategy._hold_bars[symbol] += 1

        strategy._last_position_signs[symbol] = current_sign

    if not strategy._model_configured:
        strategy._auto_configure_model()

    previous_price = strategy._last_prices.get(bar.symbol)
    strategy.current_bar = None
    strategy.current_tick = None

    check_preload_overlap(strategy, str(symbol), int(bar.timestamp))
    if _is_before_active_start(strategy, int(bar.timestamp)):
        return

    if current_pos != 0 and previous_price is not None and previous_price != bar.close:
        mark_portfolio_dirty(strategy)
    strategy.current_bar = bar
    _drive_local_stops(strategy, bar.symbol, bar.close, high=bar.high, low=bar.low)
    if hasattr(strategy, "_update_incremental_indicators"):
        strategy._update_incremental_indicators(bar)
    strategy._last_prices[bar.symbol] = bar.close
    strategy._bar_count += 1
    strategy._symbol_bar_counts[symbol] = _next_symbol_warmup_count(strategy, symbol)
    dispatch_portfolio_update(strategy)

    if strategy._symbol_bar_counts[symbol] < strategy.warmup_period:
        return

    activate_pending_model(strategy)
    should_train = should_trigger_training(strategy)

    call_user_callback(strategy, "on_bar", bar, payload=bar)
    if should_train:
        consume_training_trigger(strategy)
        training_cycle = begin_training_cycle(strategy)
        try:
            call_user_callback(strategy, "on_train_signal", strategy, payload=strategy)
        finally:
            finalize_training_cycle(strategy, training_cycle)
    analyzer_manager = getattr(strategy, "_analyzer_manager", None)
    if analyzer_manager is not None:
        try:
            analyzer_manager.on_bar(
                {
                    "strategy": strategy,
                    "bar": bar,
                    "engine": getattr(strategy, "_engine", None),
                    "ctx": ctx,
                    "owner_strategy_id": str(
                        getattr(ctx, "strategy_id", None)
                        or getattr(strategy, "_owner_strategy_id", "_default")
                    ),
                }
            )
        except Exception:
            pass
    _flush_indicator_snapshots(strategy)


def on_tick_event(strategy: Any, tick: Tick, ctx: StrategyContext) -> None:
    """引擎调用的 Tick 回调 (Internal)."""
    ensure_framework_state(strategy)
    strategy.ctx = ctx
    flush_pending_schedules(strategy)
    register_cross_section_timers(strategy)
    strategy._last_event_type = "tick"
    strategy._check_order_events()
    check_expiry_events(strategy)
    previous_price = strategy._last_prices.get(tick.symbol)
    strategy.current_bar = None
    strategy.current_tick = None

    if _is_before_active_start(strategy, int(tick.timestamp)):
        return

    current_pos = ctx.get_position(tick.symbol)
    if current_pos != 0 and previous_price is not None and previous_price != tick.price:
        mark_portfolio_dirty(strategy)
    strategy.current_tick = tick
    _drive_local_stops(strategy, tick.symbol, tick.price)
    if hasattr(strategy, "_update_incremental_indicators"):
        strategy._update_incremental_indicators(tick)
    strategy._last_prices[tick.symbol] = tick.price
    dispatch_portfolio_update(strategy)
    call_user_callback(strategy, "on_tick", tick, payload=tick)
    _flush_indicator_snapshots(strategy)


def on_timer_event(strategy: Any, payload: str, ctx: StrategyContext) -> None:
    """引擎调用的 Timer 回调 (Internal)."""
    ensure_framework_state(strategy)
    strategy.ctx = ctx
    # 交易日生命周期事件(盘前/开始交易/收盘后)不刷出 on_start 里登记的定时任务:
    # 首次登记仍在首个行情或用户定时器事件上进行, “跳过早于当前时刻的时点”以它为准
    # (只有日期的日线改写到 15:00 后, 首日 09:35 等定时器不补登记)。
    if not is_lifecycle_payload(payload):
        flush_pending_schedules(strategy)
    register_cross_section_timers(strategy)
    strategy._check_order_events()
    # 到期事件也在定时器上送达（按 key 去重）：日终结算后的 on_after_trading
    # 应已看到当日的到期结果，不必等下一交易日首根 bar。
    check_expiry_events(strategy)

    current_time = int(getattr(ctx, "current_time", 0))
    if _is_before_active_start(strategy, current_time):
        return

    dispatch_portfolio_update(strategy)

    if dispatch_lifecycle_timer(strategy, payload):
        _flush_indicator_snapshots(strategy)
        return

    if dispatch_cross_section_timer(strategy, payload):
        _flush_indicator_snapshots(strategy)
        return

    periodic_kind = _periodic_timer_kind(payload)
    if periodic_kind is not None:
        parts = payload.split("|", 2)
        if len(parts) == 3:
            _, time_str, user_payload = parts

            call_user_callback(strategy, "on_timer", user_payload, payload=user_payload)

            # 回测已按交易日逐日预登记; 实盘(无 _trading_days)按交易日历重排下一次
            if not strategy._trading_days:
                try:
                    target = next_occurrence(
                        periodic_kind,
                        time_str,
                        live_now(strategy),
                        ctx.is_trading_day,
                        ctx.next_trading_day,
                        strategy.timezone,
                    )
                    if target is None:
                        logger.warning(
                            "定时任务 %r（%s %s）在交易日历中找不到下一次触发日，"
                            "已停止重排；请在 trading_calendar 中补充后续交易日",
                            user_payload,
                            periodic_kind,
                            time_str,
                        )
                    else:
                        strategy.schedule(target, payload)
                except Exception as exc:
                    owner_strategy_id = (
                        str(getattr(strategy, "_owner_strategy_id", "_default")).strip()
                        or "_default"
                    )
                    logger.warning(
                        "Failed to reschedule live daily timer",
                        exc_info=exc,
                        extra=build_log_extra(
                            phase="strategy",
                            strategy_id=owner_strategy_id,
                            slot=(
                                owner_strategy_id
                                if owner_strategy_id != "_default"
                                else None
                            ),
                        ),
                    )
            _flush_indicator_snapshots(strategy)
            return

    call_user_callback(strategy, "on_timer", payload, payload=payload)
    _flush_indicator_snapshots(strategy)


def flush_pending_order_events(strategy: Any, ctx: StrategyContext) -> None:
    """Flush pending order/trade callbacks without invoking a user market callback."""
    ensure_framework_state(strategy)
    strategy.ctx = ctx
    strategy._check_order_events()
