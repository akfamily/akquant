from collections import deque
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from .akquant import TradingSession
from .log import get_logger

logger = get_logger("strategy")

_RUNTIME_DEFAULTS = {
    "enable_precise_day_boundary_hooks": False,
    "portfolio_update_eps": 0.0,
    "error_mode": "raise",
    "re_raise_on_error": True,
}


def _runtime_option(strategy: Any, name: str) -> Any:
    default = _RUNTIME_DEFAULTS[name]
    cfg = getattr(strategy, "runtime_config", None)
    if isinstance(cfg, dict):
        value = cfg.get(name, default)
    else:
        value = getattr(cfg, name, default) if cfg is not None else default
    if name == "portfolio_update_eps":
        try:
            value = float(cast(Any, value))
        except (TypeError, ValueError):
            raise ValueError("portfolio_update_eps must be >= 0") from None
        if value < 0.0:
            raise ValueError("portfolio_update_eps must be >= 0")
    if name == "error_mode":
        mode = str(value).strip().lower()
        if mode not in {"raise", "continue", "legacy"}:
            raise ValueError("error_mode must be one of: raise, continue, legacy")
        value = mode
    return value


def _use_precise_day_boundary_hooks(strategy: Any) -> bool:
    """Return whether precise boundary timers should own daily hook dispatch."""
    if not bool(_runtime_option(strategy, "enable_precise_day_boundary_hooks")):
        return False
    bounds = getattr(strategy, "_trading_day_bounds", None)
    return bool(bounds)


def _default_after_boundary_timer_active(strategy: Any) -> bool:
    """Whether the day-end boundary timer for on_after_trading runs by default.

    on_after_trading is a day-end hook. In the default (lazy) dispatch mode it is
    flushed only once the next day's bar arrives, so the engine clock has already
    advanced to T+1 and any order submitted inside it is stamped with a T+1
    created_at; the next-open fill guard then pushes the fill to T+2 (issue #324).
    Scheduling the day-end timer (fired at T's session close, created_at still in
    T) makes those orders fill on T+1 as intended. Only needed when the strategy
    actually overrides on_after_trading, and independent of the precise-boundary
    opt-in. The _framework_after_trading_done_date guard dedupes against the lazy
    path.
    """
    if not _strategy_overrides_callback(strategy, "on_after_trading"):
        return False
    return bool(getattr(strategy, "_trading_day_bounds", None))


def _is_normal_session(session: Any) -> bool:
    normal = getattr(TradingSession, "Normal", None)
    continuous = getattr(TradingSession, "Continuous", None)
    text = str(session).lower()
    if (
        text == "normal"
        or text.endswith(".normal")
        or text == "continuous"
        or text.endswith(".continuous")
    ):
        return True
    if normal is not None or continuous is not None:
        return bool(session == normal or session == continuous)
    return False


def _is_pre_open_session(session: Any) -> bool:
    pre_open = getattr(TradingSession, "PreOpen", None)
    if pre_open is not None:
        return bool(session == pre_open)
    text = str(session).lower()
    return text == "preopen" or text.endswith(".preopen")


def _functional_wrapper_method(cls: type, callback_name: str) -> Any:
    """取函数式包装器自身定义的该回调转发方法, 非函数式策略返回 None.

    用 ``__dict__`` 而非 getattr 定位打了标记的那个类本身: 标记会被子类继承,
    getattr 无法区分"包装器"与"包装器的子类"。
    """
    for base in cls.mro():
        if base.__dict__.get("_is_functional_wrapper"):
            return base.__dict__.get(callback_name)
    return None


def _strategy_overrides_callback(strategy: Any, callback_name: str) -> bool:
    """判断策略是否重写了框架钩子(相对框架基类的默认实现).

    比较对象必须是 MRO 中**最基础**的那个定义, 即框架基类 ``Strategy`` 上的
    默认实现。若改为比较"MRO 中第一个定义该方法的基类", 用户把钩子写在一层
    公共基类里、具体策略只继承时会拿用户基类跟自己比, 得到 False, 钩子被
    dispatch_time_hooks / collect_pre_open_timer_entries 整段跳过而静默失效。

    函数式策略是例外: ``FunctionalStrategy`` 在类体里无条件定义了全部回调转发
    方法(未提供时空转), 按类比较恒为 True, 快路径整体失效——引擎会为每根 Bar
    分发无人消费的 pre_open / cross_section / time-hooks / portfolio-update。
    因此当该方法正是包装器自身那份时, 改看用户是否真的提供了对应回调函数;
    子类若二次重写, method 不再是包装器那份, 仍走下面的通用比较。
    """
    method = getattr(type(strategy), callback_name, None)
    if method is None:
        return False
    wrapper_method = _functional_wrapper_method(type(strategy), callback_name)
    if wrapper_method is not None and method is wrapper_method:
        attr = f"_{callback_name}_func"
        if hasattr(strategy, attr):
            return getattr(strategy, attr) is not None
    for base in reversed(type(strategy).mro()[1:]):
        base_method = base.__dict__.get(callback_name)
        if base_method is not None:
            return method is not base_method
    return False


# dispatch_time_hooks 可能触发的全部用户回调. 一个都没重写时整段可跳过.
_TIME_HOOK_CALLBACKS = (
    "on_before_trading",
    "on_after_trading",
)


def _needs_time_hooks(strategy: Any) -> bool:
    """策略是否重写了任一会话/交易日钩子(按类判定, 结果缓存).

    未重写则 dispatch_time_hooks 不会触发任何回调, 其内部维护的 session/date
    状态也无人在本模块外读取, 因此整段可安全跳过——省掉逐 bar 的 pandas 时间
    转换与大量 getattr.
    """
    cached = getattr(strategy, "_framework_needs_time_hooks", None)
    if cached is None:
        cached = any(
            _strategy_overrides_callback(strategy, name)
            for name in _TIME_HOOK_CALLBACKS
        )
        strategy._framework_needs_time_hooks = cached
    return cached


def _needs_portfolio_update(strategy: Any) -> bool:
    """策略是否重写了 on_portfolio_update(按类判定, 结果缓存).

    未重写则 dispatch_portfolio_update 只做无人消费的组合快照/取账户/变更检测,
    其维护的 portfolio_dirty / last_portfolio_state / emit_previous 状态仅由本函数
    自身消费(use_previous 标志则由 on_trade 的 try/finally 独立管理), 故可整段跳过.
    """
    cached = getattr(strategy, "_framework_needs_portfolio_update", None)
    if cached is None:
        cached = _strategy_overrides_callback(strategy, "on_portfolio_update")
        strategy._framework_needs_portfolio_update = cached
    return cached


def _build_pre_open_event(
    strategy: Any,
    trading_date: Any,
    source_timestamp: int,
) -> Dict[str, Any]:
    session = None
    if strategy.ctx is not None:
        session = getattr(strategy.ctx, "session", None)
    return {
        "session": session,
        "session_label": "pre_open",
        "trading_date": trading_date,
        "timestamp": source_timestamp,
        "expected_open_at": source_timestamp,
        "market": "default",
    }


def collect_pre_open_timer_entries(strategy: Any) -> List[Tuple[int, str]]:
    """Collect global framework pre-open timers for all trading days."""
    if not _strategy_overrides_callback(strategy, "on_pre_open"):
        return []

    bounds = getattr(strategy, "_trading_day_bounds", None)
    if not bounds:
        return []

    entries: List[Tuple[int, str]] = []
    for day_key, day_bounds in bounds.items():
        if not isinstance(day_bounds, (list, tuple)) or len(day_bounds) != 2:
            continue
        start_ns = int(day_bounds[0])
        if start_ns <= 0:
            continue
        # Fire the pre-open timer strictly before the day's first bar so an order
        # submitted in on_pre_open has created_at < that bar and fills on this
        # day's open (the "decide before open, fill on this open" contract),
        # rather than one bar late (#324). The payload keeps start_ns as the
        # expected_open_at / history-cutoff source timestamp.
        trigger_ns = start_ns - 1
        entries.append((trigger_ns, f"__framework_pre_open__|{day_key}|{start_ns}"))
    return entries


def collect_boundary_timer_entries(strategy: Any) -> List[Tuple[int, str]]:
    """Collect global framework boundary timers for all trading days.

    "before" timers are scheduled only in precise mode. "after" timers are also
    scheduled in the default mode when the strategy overrides on_after_trading,
    so orders placed at day-end fill on the next trading day instead of one day
    late (issue #324).
    """
    precise = _use_precise_day_boundary_hooks(strategy)
    want_after = precise or _default_after_boundary_timer_active(strategy)
    if not want_after:
        return []

    bounds = getattr(strategy, "_trading_day_bounds", None)
    if not bounds:
        return []

    entries: List[Tuple[int, str]] = []
    for day_key, day_bounds in bounds.items():
        if not isinstance(day_bounds, (list, tuple)) or len(day_bounds) != 2:
            continue
        start_ns = int(day_bounds[0])
        end_ns = int(day_bounds[1])
        if precise and start_ns > 0:
            entries.append((start_ns, f"__framework_boundary__|before|{day_key}"))
        if want_after and end_ns > 0:
            entries.append((end_ns + 1, f"__framework_boundary__|after|{day_key}"))
    return entries


def collect_cross_section_timer_entries(
    strategy: Any,
) -> List[Tuple[int, str]]:
    """Collect after-bar rebalance timers aligned to complete price slices."""
    if not _strategy_overrides_callback(strategy, "on_cross_section"):
        return []

    rebalance_timestamps = getattr(
        strategy, "_trading_day_cross_section_timestamps", None
    )
    if not rebalance_timestamps:
        return []

    entries: List[Tuple[int, str]] = []
    for day_key, source_timestamp in rebalance_timestamps.items():
        try:
            source_ts = int(source_timestamp)
        except (TypeError, ValueError):
            continue
        if source_ts <= 0:
            continue
        entries.append(
            (
                source_ts + 1,
                f"__framework_cross_section__|{day_key}|{source_ts}",
            )
        )
    return entries


def _should_reraise_on_error(strategy: Any) -> bool:
    mode = str(_runtime_option(strategy, "error_mode")).strip().lower()
    if mode == "raise":
        return True
    if mode == "continue":
        return False
    return bool(_runtime_option(strategy, "re_raise_on_error"))


def _snapshot_previous_account_details(strategy: Any) -> Optional[Dict[str, float]]:
    """Capture previous-snapshot derived account fields for framework phases.

    frozen_cash / short_market_value are read from the authoritative Rust
    StrategyContext (the Python re-implementations were removed); this snapshot
    carries the current-period values forward as the "previous" snapshot the
    next framework phase reads.
    """
    if strategy.ctx is None:
        return None

    return {
        "frozen_cash": float(getattr(strategy.ctx, "account_frozen_cash", 0.0)),
        "short_market_value": float(
            getattr(strategy.ctx, "account_short_market_value", 0.0)
        ),
        "margin_accrued_interest": float(
            getattr(strategy.ctx, "margin_accrued_interest", 0.0)
        ),
        "margin_daily_interest": float(
            getattr(strategy.ctx, "margin_daily_interest", 0.0)
        ),
    }


def _run_in_framework_phase(
    strategy: Any,
    phase: str,
    timestamp: int,
    callback_name: str,
    *args: Any,
    payload: Optional[Any] = None,
    use_previous_account_snapshot: bool = True,
    hide_current_event: bool = True,
) -> Any:
    """Run a callback with a framework phase and history visibility cutoff."""
    previous_phase = getattr(strategy, "_framework_phase", None)
    previous_cutoff = getattr(strategy, "_framework_history_cutoff_ns", None)
    previous_pre_open = bool(getattr(strategy, "_framework_in_pre_open_phase", False))
    previous_bar = getattr(strategy, "current_bar", None)
    previous_tick = getattr(strategy, "current_tick", None)
    previous_account_snapshot = bool(
        getattr(strategy, "_framework_use_previous_account_snapshot", False)
    )
    previous_account_details = getattr(
        strategy, "_framework_previous_account_details", None
    )
    strategy._framework_phase = phase
    strategy._framework_history_cutoff_ns = int(timestamp)
    strategy._framework_previous_account_details = _snapshot_previous_account_details(
        strategy
    )
    strategy._framework_use_previous_account_snapshot = bool(
        use_previous_account_snapshot
    )
    if hide_current_event:
        strategy.current_bar = None
        strategy.current_tick = None
    if phase == "pre_open":
        strategy._framework_in_pre_open_phase = True
    try:
        return call_user_callback(strategy, callback_name, *args, payload=payload)
    finally:
        strategy._framework_phase = previous_phase
        strategy._framework_history_cutoff_ns = previous_cutoff
        strategy._framework_in_pre_open_phase = previous_pre_open
        strategy._framework_use_previous_account_snapshot = previous_account_snapshot
        strategy._framework_previous_account_details = previous_account_details
        if hide_current_event:
            strategy.current_bar = previous_bar
            strategy.current_tick = previous_tick


def call_user_callback(
    strategy: Any, callback_name: str, *args: Any, payload: Optional[Any] = None
) -> Any:
    """调用用户回调，并在异常时转发到 on_error."""
    callback = getattr(strategy, callback_name)
    previous_callback = getattr(strategy, "_framework_current_callback", None)
    previous_order = getattr(strategy, "_framework_current_order", None)
    previous_trade = getattr(strategy, "_framework_current_trade", None)
    strategy._framework_current_callback = callback_name
    strategy._framework_current_order = (
        payload if callback_name in {"on_order", "on_reject"} else None
    )
    strategy._framework_current_trade = payload if callback_name == "on_trade" else None
    try:
        return callback(*args)
    except Exception as exc:
        if callback_name != "on_error":
            error_payload = (
                payload if payload is not None else (args[0] if args else None)
            )
            try:
                strategy.on_error(exc, callback_name, error_payload)
            except Exception:
                pass
            if not _should_reraise_on_error(strategy):
                return None
        raise
    finally:
        strategy._framework_current_callback = previous_callback
        strategy._framework_current_order = previous_order
        strategy._framework_current_trade = previous_trade


def dispatch_time_hooks(strategy: Any) -> None:
    """分发会话与交易日相关钩子."""
    if strategy.ctx is None:
        return

    # 性能: 策略未重写任何会话/交易日钩子时, 本函数不会触发回调, 直接跳过.
    if not _needs_time_hooks(strategy):
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    if current_time <= 0:
        return

    # 性能: 逐 bar 的 pd.to_datetime(标量) 极慢(占回测总耗时约三成).
    # 本地交易日在同一天内不变, 按 [当日 00:00, 次日 00:00) 的 ns 窗口缓存,
    # 仅跨日时才走一次 pandas 转换. 语义与逐 bar 转换完全一致.
    cache_lo = getattr(strategy, "_framework_local_date_lo_ns", None)
    cache_hi = getattr(strategy, "_framework_local_date_hi_ns", None)
    if (
        cache_lo is not None
        and cache_hi is not None
        and cache_lo <= current_time < cache_hi
    ):
        current_date = strategy._framework_local_date_cache
    else:
        ts = pd.to_datetime(current_time, unit="ns", utc=True).tz_convert(
            strategy.timezone
        )
        current_date = ts.date()
        day_start = ts.normalize()
        lo_ns = day_start.value
        hi_ns = (day_start + pd.Timedelta(hours=24)).normalize().value
        strategy._framework_local_date_cache = current_date
        strategy._framework_local_date_lo_ns = lo_ns
        # DST "回拨" 的 25h 日 hi<=lo, 令窗口为空以强制逐 bar 重算(仍正确).
        strategy._framework_local_date_hi_ns = hi_ns if hi_ns > lo_ns else lo_ns
    current_session = getattr(strategy.ctx, "session", None)
    use_precise_boundaries = _use_precise_day_boundary_hooks(strategy)

    last_date = getattr(strategy, "_framework_last_local_date", None)
    before_done_date = getattr(strategy, "_framework_before_trading_done_date", None)
    after_done_date = getattr(strategy, "_framework_after_trading_done_date", None)

    if (
        not use_precise_boundaries
        and last_date is not None
        and current_date != last_date
        and before_done_date == last_date
        and after_done_date != last_date
    ):
        call_user_callback(
            strategy,
            "on_after_trading",
            last_date,
            current_time,
            payload={"trading_date": last_date, "timestamp": current_time},
        )
        strategy._framework_after_trading_done_date = last_date

    if (
        not use_precise_boundaries
        and _is_normal_session(current_session)
        and getattr(strategy, "_framework_before_trading_done_date", None)
        != current_date
    ):
        _run_in_framework_phase(
            strategy,
            "before_trading",
            current_time,
            "on_before_trading",
            current_date,
            current_time,
            payload={"trading_date": current_date, "timestamp": current_time},
        )
        strategy._framework_before_trading_done_date = current_date

    if (
        not use_precise_boundaries
        and not _is_normal_session(current_session)
        and getattr(strategy, "_framework_before_trading_done_date", None)
        == current_date
        and getattr(strategy, "_framework_after_trading_done_date", None)
        != current_date
    ):
        call_user_callback(
            strategy,
            "on_after_trading",
            current_date,
            current_time,
            payload={"trading_date": current_date, "timestamp": current_time},
        )
        strategy._framework_after_trading_done_date = current_date

    strategy._framework_last_local_date = current_date


def register_pre_open_timers(strategy: Any) -> None:
    """为实现 on_pre_open 的策略注册交易日开盘前框架定时器."""
    if getattr(strategy, "_framework_pre_open_timers_registered", False):
        return
    if not _strategy_overrides_callback(strategy, "on_pre_open"):
        strategy._framework_pre_open_timers_registered = True
        return
    if strategy.ctx is None:
        return

    entries = collect_pre_open_timer_entries(strategy)
    if not entries:
        # 实盘: pre-open 定时器依赖逐交易日的 _trading_day_bounds, 而该字段只由
        # 回测引擎填充, 实盘恒为空 ⇒ on_pre_open 永不触发。此处必须告警并停止
        # 重试: 否则每根 bar/tick 都会重跑一次注册并再次失败(既无声又白做功)。
        # 回测下 bounds 可能只是"还没填", 保持原样按 bar 重试(不告警, 避免误报)。
        if getattr(strategy, "_live_market_data_owner", False):
            logger.warning(
                "实盘下 on_pre_open 不会触发: 该回调依赖回测交易日历推导的盘前"
                "时点, 实盘无此数据源。请改用 schedule_daily(...) 在盘前时点"
                "触发 on_timer, 或把盘前逻辑放到 on_before_trading 中"
            )
            strategy._framework_pre_open_timers_registered = True
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    for start_ns, payload in entries:
        if current_time > 0 and start_ns <= current_time:
            continue
        strategy.ctx.schedule(start_ns, payload)

    strategy._framework_pre_open_timers_registered = True


def register_boundary_timers(strategy: Any) -> None:
    """注册交易日边界定时器，用于触发 on_before/on_after_trading.

    precise 模式注册 before+after 边界定时器;默认模式在策略重写 on_after_trading
    时也注册 after 边界定时器(修复 #324)。
    """
    if strategy.ctx is None:
        return
    if not (
        _use_precise_day_boundary_hooks(strategy)
        or _default_after_boundary_timer_active(strategy)
    ):
        return
    if getattr(strategy, "_framework_boundary_timers_registered", False):
        return

    for trigger_ts, payload in collect_boundary_timer_entries(strategy):
        strategy.ctx.schedule(trigger_ts, payload)

    strategy._framework_boundary_timers_registered = True


def register_cross_section_timers(strategy: Any) -> None:
    """注册框架级 after-bar 调仓定时器，在完整时间片结束后触发."""
    if strategy.ctx is None:
        return
    if getattr(strategy, "_framework_cross_section_timers_registered", False):
        return

    entries = collect_cross_section_timer_entries(strategy)
    if not entries:
        strategy._framework_cross_section_timers_registered = True
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    for trigger_ts, payload in entries:
        if current_time > 0 and trigger_ts <= current_time:
            continue
        strategy.ctx.schedule(trigger_ts, payload)

    strategy._framework_cross_section_timers_registered = True


def dispatch_boundary_timer(strategy: Any, payload: str) -> bool:
    """处理框架级边界定时器，返回是否已消费该 payload.

    "after" 相位在默认与 precise 模式下都派发(修复 #324);"before" 相位仅在
    precise 模式下派发(默认模式不注册 before 边界定时器)。
    """
    if strategy.ctx is None:
        return False
    if not payload.startswith("__framework_boundary__|"):
        return False

    parts = payload.split("|", 2)
    if len(parts) != 3:
        return True

    _, phase, day_text = parts
    try:
        day = pd.Timestamp(day_text).date()
    except Exception:
        return True

    precise = bool(_runtime_option(strategy, "enable_precise_day_boundary_hooks"))
    current_time = int(getattr(strategy.ctx, "current_time", 0))
    if phase == "before":
        if not precise:
            return True
        if getattr(strategy, "_framework_before_trading_done_date", None) != day:
            _run_in_framework_phase(
                strategy,
                "before_trading",
                current_time,
                "on_before_trading",
                day,
                current_time,
                payload={"trading_date": day, "timestamp": current_time},
            )
            strategy._framework_before_trading_done_date = day
        return True

    if phase == "after":
        if getattr(strategy, "_framework_after_trading_done_date", None) != day:
            call_user_callback(
                strategy,
                "on_after_trading",
                day,
                current_time,
                payload={"trading_date": day, "timestamp": current_time},
            )
            strategy._framework_after_trading_done_date = day
        return True

    return True


def dispatch_pre_open_timer(strategy: Any, payload: str) -> bool:
    """处理框架级 pre-open 定时器，返回是否已消费该 payload."""
    if not payload.startswith("__framework_pre_open__|"):
        return False

    parts = payload.split("|", 2)
    if len(parts) != 3:
        return True

    trading_date_text = parts[1]
    trading_date: Any = trading_date_text
    try:
        trading_date = pd.to_datetime(trading_date_text).date()
    except Exception:
        pass
    try:
        source_timestamp = int(parts[2])
    except Exception:
        source_timestamp = int(getattr(strategy.ctx, "current_time", 0))

    done_date = getattr(strategy, "_framework_pre_open_done_date", None)
    if done_date == trading_date:
        return True

    event = _build_pre_open_event(strategy, trading_date, source_timestamp)
    _run_in_framework_phase(
        strategy,
        "pre_open",
        source_timestamp,
        "on_pre_open",
        event,
        payload=event,
    )
    strategy._framework_pre_open_done_date = trading_date
    return True


def dispatch_cross_section_timer(strategy: Any, payload: str) -> bool:
    """处理框架级 after-bar 调仓定时器，返回是否已消费该 payload."""
    if not payload.startswith("__framework_cross_section__|"):
        return False

    parts = payload.split("|", 2)
    if len(parts) != 3:
        return True

    trading_date_text = parts[1]
    trading_date: Any = trading_date_text
    try:
        trading_date = pd.to_datetime(trading_date_text).date()
    except Exception:
        pass
    try:
        source_timestamp = int(parts[2])
    except Exception:
        source_timestamp = int(getattr(strategy.ctx, "current_time", 0))

    done_date = getattr(strategy, "_framework_cross_section_done_date", None)
    if done_date != trading_date:
        _run_in_framework_phase(
            strategy,
            "cross_section",
            source_timestamp,
            "on_cross_section",
            trading_date,
            source_timestamp,
            payload={"trading_date": trading_date, "timestamp": source_timestamp},
            use_previous_account_snapshot=False,
        )
        strategy._framework_cross_section_done_date = trading_date
    return True


def mark_portfolio_dirty(strategy: Any) -> None:
    """标记账户快照需要重新计算."""
    strategy._framework_portfolio_dirty = True


def dispatch_portfolio_update(strategy: Any) -> None:
    """在账户状态变化时分发 on_portfolio_update."""
    if strategy.ctx is None:
        return
    # 性能: 策略未重写 on_portfolio_update 时, 跳过逐 bar 的组合快照/取账户/变更检测.
    if not _needs_portfolio_update(strategy):
        return
    if (
        not getattr(strategy, "_framework_portfolio_dirty", True)
        and getattr(strategy, "_framework_last_portfolio_state", None) is not None
    ):
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    session = getattr(strategy.ctx, "session", None)
    cash = float(strategy.ctx.cash)
    positions = {k: float(v) for k, v in dict(strategy.ctx.positions).items()}
    available_positions = {
        k: float(v) for k, v in dict(strategy.ctx.available_positions).items()
    }
    use_previous_snapshot = bool(
        getattr(strategy, "_framework_emit_previous_portfolio_snapshot", False)
    )
    previous_override = bool(
        getattr(strategy, "_framework_use_previous_account_snapshot", False)
    )
    strategy._framework_use_previous_account_snapshot = use_previous_snapshot
    try:
        equity = float(strategy.equity)
        market_value = float(equity - cash)
        account_snapshot = strategy.get_account()
    finally:
        strategy._framework_use_previous_account_snapshot = previous_override
    if use_previous_snapshot:
        strategy._framework_emit_previous_portfolio_snapshot = False

    state_key: Tuple[Any, ...] = (
        round(cash, 8),
        round(equity, 8),
        tuple(sorted((k, round(v, 8)) for k, v in positions.items())),
        tuple(sorted((k, round(v, 8)) for k, v in available_positions.items())),
    )

    if state_key == getattr(strategy, "_framework_last_portfolio_state", None):
        strategy._framework_portfolio_dirty = False
        return

    eps = float(_runtime_option(strategy, "portfolio_update_eps"))
    last_state = getattr(strategy, "_framework_last_portfolio_state", None)
    if eps > 0.0 and last_state is not None:
        last_positions = last_state[2]
        last_available_positions = last_state[3]
        if (
            state_key[2] == last_positions
            and state_key[3] == last_available_positions
            and abs(cash - float(last_state[0])) <= eps
            and abs(equity - float(last_state[1])) <= eps
        ):
            strategy._framework_portfolio_dirty = False
        return

    strategy._framework_last_portfolio_state = state_key
    snapshot: Dict[str, Any] = {
        "timestamp": current_time,
        "session": session,
        "cash": cash,
        "equity": equity,
        "market_value": market_value,
        "positions": positions,
        "available_positions": available_positions,
        "margin": float(account_snapshot.get("margin", 0.0)),
        "frozen_cash": float(account_snapshot.get("frozen_cash", 0.0)),
    }
    callback_override = bool(
        getattr(strategy, "_framework_use_previous_account_snapshot", False)
    )
    strategy._framework_use_previous_account_snapshot = use_previous_snapshot
    try:
        call_user_callback(strategy, "on_portfolio_update", snapshot, payload=snapshot)
        strategy._framework_portfolio_dirty = False
    finally:
        strategy._framework_use_previous_account_snapshot = callback_override


def dispatch_shutdown_hooks(strategy: Any) -> None:
    """在停止阶段补发未完成的会话/交易日钩子."""
    if strategy.ctx is None:
        return
    if getattr(strategy, "_framework_stop_flushed", False):
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    before_done_date = getattr(strategy, "_framework_before_trading_done_date", None)
    after_done_date = getattr(strategy, "_framework_after_trading_done_date", None)
    if before_done_date is not None and after_done_date != before_done_date:
        call_user_callback(
            strategy,
            "on_after_trading",
            before_done_date,
            current_time,
            payload={"trading_date": before_done_date, "timestamp": current_time},
        )
        strategy._framework_after_trading_done_date = before_done_date

    strategy._framework_stop_flushed = True


def ensure_framework_state(strategy: Any) -> None:
    """确保框架级钩子状态字段存在."""
    # 性能: 本函数幂等且每 bar 被调用多次, 首次初始化后直接短路,
    # 避免逐 bar 数百次 hasattr 探测. checkpoint 恢复时 __getstate__ 会删除
    # 此标志, 强制重新初始化被重置的字段.
    if getattr(strategy, "_framework_state_ready", False):
        return
    if not hasattr(strategy, "_framework_last_local_date"):
        strategy._framework_last_local_date = None
    if not hasattr(strategy, "_framework_before_trading_done_date"):
        strategy._framework_before_trading_done_date = None
    if not hasattr(strategy, "_framework_cross_section_done_date"):
        strategy._framework_cross_section_done_date = None
    if not hasattr(strategy, "_framework_after_trading_done_date"):
        strategy._framework_after_trading_done_date = None
    if not hasattr(strategy, "_framework_pre_open_done_date"):
        strategy._framework_pre_open_done_date = None
    if not hasattr(strategy, "_framework_pre_open_timers_registered"):
        strategy._framework_pre_open_timers_registered = False
    if not hasattr(strategy, "_framework_in_pre_open_phase"):
        strategy._framework_in_pre_open_phase = False
    if not hasattr(strategy, "_framework_phase"):
        strategy._framework_phase = None
    if not hasattr(strategy, "_framework_history_cutoff_ns"):
        strategy._framework_history_cutoff_ns = None
    if not hasattr(strategy, "_framework_freq"):
        # 早于该字段的检查点恢复后没有它; 周期由本次运行的入口重新注入,
        # 不从存档取(续跑时数据源/周期可能已换)。
        strategy._framework_freq = None
    if not hasattr(strategy, "_framework_use_previous_account_snapshot"):
        strategy._framework_use_previous_account_snapshot = False
    if not hasattr(strategy, "_framework_previous_account_details"):
        strategy._framework_previous_account_details = None
    if not hasattr(strategy, "_framework_emit_previous_portfolio_snapshot"):
        strategy._framework_emit_previous_portfolio_snapshot = False
    if not hasattr(strategy, "_framework_last_portfolio_state"):
        strategy._framework_last_portfolio_state = None
    if not hasattr(strategy, "_framework_portfolio_dirty"):
        strategy._framework_portfolio_dirty = True
    if not hasattr(strategy, "_framework_rejected_order_ids"):
        strategy._framework_rejected_order_ids = set()
    if not hasattr(strategy, "_framework_order_event_keys"):
        strategy._framework_order_event_keys = set()
    if not hasattr(strategy, "_framework_order_event_key_order"):
        strategy._framework_order_event_key_order = deque()
    if not hasattr(strategy, "_framework_expiry_event_keys"):
        strategy._framework_expiry_event_keys = set()
    if not hasattr(strategy, "_framework_stop_flushed"):
        strategy._framework_stop_flushed = False
    if not hasattr(strategy, "_framework_live_stop_dispatched"):
        strategy._framework_live_stop_dispatched = False
    if not hasattr(strategy, "_framework_boundary_timers_registered"):
        strategy._framework_boundary_timers_registered = False
    if not hasattr(strategy, "_trading_day_bounds"):
        strategy._trading_day_bounds = {}
    if not hasattr(strategy, "_trading_day_cross_section_timestamps"):
        strategy._trading_day_cross_section_timestamps = {}
    if not hasattr(strategy, "_framework_cross_section_timers_registered"):
        strategy._framework_cross_section_timers_registered = False
    strategy._framework_state_ready = True
