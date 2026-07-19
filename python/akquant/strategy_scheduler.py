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
            add_daily_timer(strategy, time_str, payload)
    pending = getattr(strategy, "_pending_schedules", None)
    if not pending:
        return
    queued = list(pending)
    pending.clear()
    for trigger_time, payload in queued:
        schedule(strategy, trigger_time, payload)


def add_daily_timer(strategy: Any, time_str: str, payload: str) -> None:
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

    try:
        t = pd.to_datetime(time_str).time()
    except Exception:
        logger.warning("Error parsing time: %s", time_str)
        return

    for day in strategy._trading_days:
        naive_dt = pd.Timestamp.combine(day.date(), t)
        dt_obj = naive_dt.tz_localize(strategy.timezone)
        schedule(strategy, dt_obj, wrapped_payload)
