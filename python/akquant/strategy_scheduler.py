import datetime as dt
from typing import Any, Union

import pandas as pd

from .log import get_logger

logger = get_logger("scheduler")


def schedule(
    strategy: Any, trigger_time: Union[str, dt.datetime, pd.Timestamp], payload: str
) -> None:
    """注册单次定时任务."""
    if strategy.ctx is None:
        pending = getattr(strategy, "_pending_schedules", None)
        if pending is None:
            pending = []
            strategy._pending_schedules = pending
        pending.append((trigger_time, payload))
        return

    ts_ns = 0
    if isinstance(trigger_time, str):
        try:
            dt_obj = pd.to_datetime(trigger_time)
            if dt_obj.tz is None:
                dt_obj = dt_obj.tz_localize(strategy.timezone)
            ts_ns = int(dt_obj.value)
        except Exception:
            ts_ns = 0
    elif isinstance(trigger_time, (dt.datetime, pd.Timestamp)):
        if trigger_time.tzinfo is None:
            trigger_time = pd.Timestamp(trigger_time).tz_localize(strategy.timezone)
        if hasattr(trigger_time, "value"):
            ts_ns = int(trigger_time.value)
        elif isinstance(trigger_time, dt.datetime):
            ts_ns = int(pd.Timestamp(trigger_time).value)
        else:
            ts_ns = 0

    if ts_ns > 0:
        current_time = int(getattr(strategy.ctx, "current_time", 0))
        if (
            payload.startswith("__framework_pre_open__|")
            and current_time > 0
            and ts_ns <= current_time
        ):
            return
        strategy.ctx.schedule(ts_ns, payload)


def flush_pending_schedules(strategy: Any) -> None:
    """在上下文就绪后刷出缓存的定时任务."""
    if strategy.ctx is None:
        return
    pending_daily = getattr(strategy, "_pending_daily_timers", None)
    if pending_daily:
        queued_daily = list(pending_daily)
        pending_daily.clear()
        for time_str, payload in queued_daily:
            schedule_daily(strategy, time_str, payload)
    pending = getattr(strategy, "_pending_schedules", None)
    if not pending:
        return
    queued = list(pending)
    pending.clear()
    for trigger_time, payload in queued:
        schedule(strategy, trigger_time, payload)


def schedule_daily(strategy: Any, time_str: str, payload: str) -> None:
    """注册每日定时任务."""
    if strategy.ctx is None:
        pending_daily = getattr(strategy, "_pending_daily_timers", None)
        if pending_daily is None:
            pending_daily = []
            strategy._pending_daily_timers = pending_daily
        pending_daily.append((time_str, payload))
        return
    wrapped_payload = f"__daily__|{time_str}|{payload}"

    if not strategy._trading_days:
        try:
            t = pd.to_datetime(time_str).time()
        except Exception:
            logger.warning("Error parsing time: %s", time_str)
            return

        now = pd.Timestamp.now(tz=strategy.timezone)
        target = pd.Timestamp.combine(now.date(), t).tz_localize(strategy.timezone)
        if target <= now:
            target += pd.Timedelta(days=1)

        schedule(strategy, target, wrapped_payload)
        return

    _register_on_days(strategy, strategy._trading_days, time_str, wrapped_payload)


def _register_on_days(
    strategy: Any, days: list, time_str: str, wrapped_payload: str
) -> None:
    """在给定交易日子集的 ``time_str`` 时点逐日注册一枚 timer."""
    try:
        t = pd.to_datetime(time_str).time()
    except Exception:
        logger.warning("Error parsing time: %s", time_str)
        return

    for day in days:
        naive_dt = pd.Timestamp.combine(day.date(), t)
        dt_obj = naive_dt.tz_localize(strategy.timezone)
        schedule(strategy, dt_obj, wrapped_payload)


def _first_trading_day_per_group(days: list, key: Any) -> list:
    """按 ``key(day)`` 分组, 取每组最早的交易日(即该周/月的首个交易日)."""
    seen: dict = {}
    for day in days:
        group = key(day)
        if group not in seen:
            seen[group] = day
    return list(seen.values())


def _nth_per_group(days: list, key: Any, n: int, from_end: bool) -> list:
    """按 ``key(day)`` 分组, 取每组第 n 个(from_end 时为倒数第 n 个)交易日.

    组内不足 n 个交易日则跳过该组。n 从 1 开始。
    """
    grouped: dict = {}
    for day in days:
        grouped.setdefault(key(day), []).append(day)
    out = []
    for group in grouped.values():
        if len(group) < n:
            continue
        out.append(group[-n] if from_end else group[n - 1])
    return out


def schedule_weekly(strategy: Any, time_str: str, payload: str) -> None:
    """每周首个交易日在 ``time_str`` 触发(节假日/停牌自动顺延)."""
    if strategy.ctx is None or not strategy._trading_days:
        logger.warning(
            "schedule_weekly requires trading days (backtest context); ignored"
        )
        return
    wrapped_payload = f"__daily__|{time_str}|{payload}"
    days = _first_trading_day_per_group(
        strategy._trading_days, lambda d: (d.isocalendar()[0], d.isocalendar()[1])
    )
    _register_on_days(strategy, days, time_str, wrapped_payload)


def schedule_monthly(strategy: Any, time_str: str, payload: str) -> None:
    """每月首个交易日在 ``time_str`` 触发(节假日/停牌自动顺延)."""
    if strategy.ctx is None or not strategy._trading_days:
        logger.warning(
            "schedule_monthly requires trading days (backtest context); ignored"
        )
        return
    wrapped_payload = f"__daily__|{time_str}|{payload}"
    days = _first_trading_day_per_group(
        strategy._trading_days, lambda d: (d.year, d.month)
    )
    _register_on_days(strategy, days, time_str, wrapped_payload)
