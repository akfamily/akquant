import datetime as dt
from typing import Any, Callable, Optional, Union

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
    pending_periodic = getattr(strategy, "_pending_periodic_timers", None)
    if pending_periodic:
        queued_periodic = list(pending_periodic)
        pending_periodic.clear()
        for kind, time_str, payload in queued_periodic:
            _schedule_periodic(strategy, kind, time_str, payload)
    pending = getattr(strategy, "_pending_schedules", None)
    if not pending:
        return
    queued = list(pending)
    pending.clear()
    for trigger_time, payload in queued:
        schedule(strategy, trigger_time, payload)


# 周期定时器 payload 的种类前缀。回测预先逐日登记, 三种都用 ``__daily__|``
# (触发时只解包、不重排); 实盘只登记下一次, 触发后按种类重排, 所以 weekly/monthly
# 用各自的前缀。都不以引擎保留前缀 ``__akq_`` 开头。
PERIODIC_TIMER_PREFIXES = {
    "daily": "__daily__|",
    "weekly": "__weekly__|",
    "monthly": "__monthly__|",
}

_MAX_CALENDAR_STEPS = 4000


def next_occurrence(
    kind: str,
    time_str: str,
    now: pd.Timestamp,
    is_trading_day: Callable[[dt.date], bool],
    next_trading_day: Callable[[dt.date], Optional[dt.date]],
    tz: str,
) -> Optional[pd.Timestamp]:
    """按交易日历求周期定时器晚于 ``now`` 的下一次触发时刻.

    - ``daily``: 每个交易日的 ``time_str``;
    - ``weekly``: 每个 ISO 周首个交易日的 ``time_str``;
    - ``monthly``: 每个自然月首个交易日的 ``time_str``。

    恰好等于 ``now`` 的时点视为已过。日历只经两个回调访问(实盘传
    ``ctx.is_trading_day`` / ``ctx.next_trading_day``); 超出日历
    (``next_trading_day`` 返回 None)时返回 None。

    :param kind: ``"daily"`` / ``"weekly"`` / ``"monthly"``
    :param time_str: 本地时刻, 如 ``"14:55"``
    :param now: 当前时刻(不带时区时按 ``tz`` 解释)
    :param is_trading_day: 判断某日是否为交易日
    :param next_trading_day: 某日之后的下一个交易日(超出日历为 None)
    :param tz: 本地时区
    :return: 带 ``tz`` 时区的触发时刻, 或 None
    :raises ValueError: ``kind`` 未知或 ``time_str`` 无法解析
    """
    if kind not in PERIODIC_TIMER_PREFIXES:
        raise ValueError(f"未知的定时器种类 {kind!r}，应为 daily/weekly/monthly")
    try:
        t = pd.to_datetime(time_str).time()
    except Exception as exc:
        raise ValueError(
            f"无法解析定时器时刻 {time_str!r}，请写成 '14:55' 或 '14:55:00'"
        ) from exc
    now_ts = pd.Timestamp(now)
    now_ts = _localize(now_ts, tz) if now_ts.tzinfo is None else now_ts.tz_convert(tz)

    def on_or_after(day: dt.date) -> Optional[dt.date]:
        return day if is_trading_day(day) else next_trading_day(day)

    def at(day: dt.date) -> pd.Timestamp:
        return _localize(pd.Timestamp.combine(day, t), tz)

    def group_start(day: dt.date) -> dt.date:
        if kind == "weekly":
            return day - dt.timedelta(days=day.weekday())
        return day.replace(day=1)

    def next_group_start(start: dt.date) -> dt.date:
        if kind == "weekly":
            return start + dt.timedelta(days=7)
        if start.month == 12:
            return dt.date(start.year + 1, 1, 1)
        return dt.date(start.year, start.month + 1, 1)

    if kind == "daily":
        day = on_or_after(now_ts.date())
        for _ in range(_MAX_CALENDAR_STEPS):
            if day is None:
                return None
            target = at(day)
            if target > now_ts:
                return target
            day = next_trading_day(day)
        return None

    start = group_start(now_ts.date())
    for _ in range(_MAX_CALENDAR_STEPS):
        first = on_or_after(start)
        if first is None:
            return None
        if group_start(first) != start:
            # 该周/月没有交易日: 改从首个交易日所在的周/月取首日
            start = group_start(first)
            continue
        target = at(first)
        if target > now_ts:
            return target
        start = next_group_start(start)
    return None


def _localize(naive: pd.Timestamp, tz: str) -> pd.Timestamp:
    """本地时刻转带时区时间戳, 夏令时安全.

    落在夏令时跳变缺口里的时刻(如纽约 3 月 02:30)顺延到缺口结束(03:00);
    回拨重叠里的时刻(如纽约 11 月 01:30)取第一次出现(夏令时)。
    """
    return naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)


def _wall_clock(tz: str) -> pd.Timestamp:
    """墙钟当前时刻(单独成函数便于测试替换)."""
    return pd.Timestamp.now(tz=tz)


def live_now(strategy: Any) -> pd.Timestamp:
    """实盘当前时刻: 墙钟与当前事件时刻取较晚者(定时器略早送达时不重排回原时点)."""
    now = _wall_clock(strategy.timezone)
    try:
        current = int(getattr(strategy.ctx, "current_time", 0) or 0)
    except (TypeError, ValueError):
        current = 0
    if current > 0:
        event_ts = pd.Timestamp(current, tz="UTC").tz_convert(strategy.timezone)
        if event_ts > now:
            return event_ts
    return now


def schedule_live_next(
    strategy: Any, kind: str, time_str: str, user_payload: str
) -> bool:
    """实盘: 按会话交易日历登记周期定时器的下一次触发, 返回是否已登记.

    日历来自 ``ctx.is_trading_day`` / ``ctx.next_trading_day``(实盘缺省为工作日
    日历)。时刻无法解析时抛 ``ValueError``; 超出日历时告警并返回 False。
    """
    ctx = strategy.ctx
    target = next_occurrence(
        kind,
        time_str,
        live_now(strategy),
        ctx.is_trading_day,
        ctx.next_trading_day,
        strategy.timezone,
    )
    if target is None:
        logger.warning(
            "定时任务 %r（%s %s）在交易日历中找不到下一次触发日，已停止登记；"
            "请在 trading_calendar 中补充后续交易日",
            user_payload,
            kind,
            time_str,
        )
        return False
    schedule(
        strategy, target, f"{PERIODIC_TIMER_PREFIXES[kind]}{time_str}|{user_payload}"
    )
    return True


def schedule_daily(strategy: Any, time_str: str, payload: str) -> None:
    """注册每日定时任务(交易日历上的每个交易日触发, 回测与实盘一致)."""
    _check_payload(payload)
    if strategy.ctx is None:
        pending_daily = getattr(strategy, "_pending_daily_timers", None)
        if pending_daily is None:
            pending_daily = []
            strategy._pending_daily_timers = pending_daily
        pending_daily.append((time_str, payload))
        return

    if not strategy._trading_days:
        try:
            schedule_live_next(strategy, "daily", time_str, payload)
        except ValueError:
            logger.warning("Error parsing time: %s", time_str)
        return

    wrapped_payload = f"__daily__|{time_str}|{payload}"
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
        dt_obj = _localize(naive_dt, strategy.timezone)
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


def _schedule_periodic(strategy: Any, kind: str, time_str: str, payload: str) -> None:
    _check_payload(payload)
    if strategy.ctx is None:
        # on_start 时上下文尚未注入: 与 schedule_daily 一样延后到首个事件再登记
        pending = getattr(strategy, "_pending_periodic_timers", None)
        if pending is None:
            pending = []
            strategy._pending_periodic_timers = pending
        pending.append((kind, time_str, payload))
        return
    if not strategy._trading_days:
        try:
            schedule_live_next(strategy, kind, time_str, payload)
        except ValueError:
            logger.warning("Error parsing time: %s", time_str)
        return
    if kind == "weekly":
        days = _first_trading_day_per_group(
            strategy._trading_days,
            lambda d: (d.isocalendar()[0], d.isocalendar()[1]),
        )
    else:
        days = _first_trading_day_per_group(
            strategy._trading_days, lambda d: (d.year, d.month)
        )
    # 回测逐日预登记, 触发时只解包不重排, 沿用 __daily__ 前缀
    _register_on_days(strategy, days, time_str, f"__daily__|{time_str}|{payload}")


def schedule_weekly(strategy: Any, time_str: str, payload: str) -> None:
    """每周首个交易日在 ``time_str`` 触发(节假日自动顺延; 回测与实盘一致)."""
    _schedule_periodic(strategy, "weekly", time_str, payload)


def schedule_monthly(strategy: Any, time_str: str, payload: str) -> None:
    """每月首个交易日在 ``time_str`` 触发(节假日自动顺延; 回测与实盘一致)."""
    _schedule_periodic(strategy, "monthly", time_str, payload)
