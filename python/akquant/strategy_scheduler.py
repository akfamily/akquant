import datetime as dt
from typing import Any, Union

import pandas as pd

from .log import get_logger

logger = get_logger("scheduler")


_RESERVED_TIMER_PREFIX = "__akq_"
_DAY_CLOSE_TIMER_PREFIX = "__akq_day_close__|"


def _check_payload(payload: str) -> None:
    """拒绝占用引擎内部保留前缀 ``__akq_`` 的 payload."""
    if str(payload).startswith(_RESERVED_TIMER_PREFIX):
        raise ValueError(
            f"定时器 payload 不能以 {_RESERVED_TIMER_PREFIX!r} 开头（引擎内部保留，"
            f"如日终结算专用前缀 {_DAY_CLOSE_TIMER_PREFIX!r}），请换一个 payload 名称"
        )


def schedule(
    strategy: Any, trigger_time: Union[str, dt.datetime, pd.Timestamp], payload: str
) -> None:
    """注册单次定时任务."""
    _check_payload(payload)
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
    _check_payload(payload)
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

    # 首次登记发生在首根 bar 到达时(flush_pending_schedules)。早于当前引擎时刻的
    # 时点已经过去, 若照登会在当前事件后立刻补触发且时间戳倒退(例如只有日期的
    # 日线改写到 15:00 后, 当日 09:25 定时器会按当日收盘价成交), 故跳过。
    current_time = int(getattr(strategy.ctx, "current_time", 0) or 0)
    for day in days:
        naive_dt = pd.Timestamp.combine(day.date(), t)
        dt_obj = naive_dt.tz_localize(strategy.timezone)
        if current_time > 0 and int(dt_obj.value) < current_time:
            continue
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
    _check_payload(payload)
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
    _check_payload(payload)
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
