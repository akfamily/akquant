from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from .akquant import TradingSession

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


def _strategy_overrides_callback(strategy: Any, callback_name: str) -> bool:
    method = getattr(type(strategy), callback_name, None)
    if method is None:
        return False
    for base in type(strategy).mro()[1:]:
        base_method = base.__dict__.get(callback_name)
        if base_method is not None:
            return method is not base_method
    return False


# dispatch_time_hooks 可能触发的全部用户回调. 一个都没重写时整段可跳过.
_TIME_HOOK_CALLBACKS = (
    "on_before_trading",
    "on_after_trading",
    "on_session_start",
    "on_session_end",
    "on_daily_rebalance",
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
        entries.append((start_ns, f"__framework_pre_open__|{day_key}|{start_ns}"))
    return entries


def collect_boundary_timer_entries(strategy: Any) -> List[Tuple[int, str]]:
    """Collect global framework boundary timers for all trading days."""
    if not _use_precise_day_boundary_hooks(strategy):
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
        if start_ns > 0:
            entries.append((start_ns, f"__framework_boundary__|before|{day_key}"))
        if end_ns > 0:
            entries.append((end_ns + 1, f"__framework_boundary__|after|{day_key}"))
    return entries


def collect_daily_rebalance_after_bar_timer_entries(
    strategy: Any,
) -> List[Tuple[int, str]]:
    """Collect after-bar rebalance timers aligned to complete price slices."""
    if not _strategy_overrides_callback(strategy, "on_daily_rebalance_after_bar"):
        return []

    rebalance_timestamps = getattr(
        strategy, "_trading_day_after_bar_rebalance_timestamps", None
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
                f"__framework_after_bar_rebalance__|{day_key}|{source_ts}",
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


def _dispatch_daily_rebalance_if_needed(
    strategy: Any, trading_date: Any, timestamp: int
) -> None:
    if getattr(strategy, "_framework_daily_rebalance_done_date", None) == trading_date:
        return
    _run_in_framework_phase(
        strategy,
        "daily_rebalance",
        timestamp,
        "on_daily_rebalance",
        trading_date,
        timestamp,
        payload={"trading_date": trading_date, "timestamp": timestamp},
    )
    strategy._framework_daily_rebalance_done_date = trading_date


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

    last_session = getattr(strategy, "_framework_last_session", None)
    if current_session != last_session:
        if last_session is not None:
            call_user_callback(
                strategy,
                "on_session_end",
                last_session,
                current_time,
                payload={"session": last_session, "timestamp": current_time},
            )
        if current_session is not None:
            call_user_callback(
                strategy,
                "on_session_start",
                current_session,
                current_time,
                payload={"session": current_session, "timestamp": current_time},
            )

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
        and _is_normal_session(current_session)
        and getattr(strategy, "_framework_daily_rebalance_done_date", None)
        != current_date
    ):
        _dispatch_daily_rebalance_if_needed(strategy, current_date, current_time)

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

    strategy._framework_last_session = current_session
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
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    for start_ns, payload in entries:
        if current_time > 0 and start_ns <= current_time:
            continue
        strategy.ctx.schedule(start_ns, payload)

    strategy._framework_pre_open_timers_registered = True


def register_boundary_timers(strategy: Any) -> None:
    """注册交易日边界定时器，用于精确触发 on_before/on_after_trading."""
    if strategy.ctx is None:
        return
    if not _use_precise_day_boundary_hooks(strategy):
        return
    if getattr(strategy, "_framework_boundary_timers_registered", False):
        return

    for trigger_ts, payload in collect_boundary_timer_entries(strategy):
        strategy.ctx.schedule(trigger_ts, payload)

    strategy._framework_boundary_timers_registered = True


def register_daily_rebalance_after_bar_timers(strategy: Any) -> None:
    """注册框架级 after-bar 调仓定时器，在完整时间片结束后触发."""
    if strategy.ctx is None:
        return
    if getattr(
        strategy, "_framework_daily_rebalance_after_bar_timers_registered", False
    ):
        return

    entries = collect_daily_rebalance_after_bar_timer_entries(strategy)
    if not entries:
        strategy._framework_daily_rebalance_after_bar_timers_registered = True
        return

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    for trigger_ts, payload in entries:
        if current_time > 0 and trigger_ts <= current_time:
            continue
        strategy.ctx.schedule(trigger_ts, payload)

    strategy._framework_daily_rebalance_after_bar_timers_registered = True


def dispatch_boundary_timer(strategy: Any, payload: str) -> bool:
    """处理框架级边界定时器，返回是否已消费该 payload."""
    if strategy.ctx is None:
        return False
    if not bool(_runtime_option(strategy, "enable_precise_day_boundary_hooks")):
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

    current_time = int(getattr(strategy.ctx, "current_time", 0))
    if phase == "before":
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
        if getattr(strategy, "_framework_daily_rebalance_done_date", None) != day:
            _run_in_framework_phase(
                strategy,
                "daily_rebalance",
                current_time,
                "on_daily_rebalance",
                day,
                current_time,
                payload={"trading_date": day, "timestamp": current_time},
            )
            strategy._framework_daily_rebalance_done_date = day
            strategy._framework_daily_rebalance_pending_date = None
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


def dispatch_daily_rebalance_after_bar_timer(strategy: Any, payload: str) -> bool:
    """处理框架级 after-bar 调仓定时器，返回是否已消费该 payload."""
    if not payload.startswith("__framework_after_bar_rebalance__|"):
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

    done_date = getattr(
        strategy, "_framework_daily_rebalance_after_bar_done_date", None
    )
    if done_date != trading_date:
        _run_in_framework_phase(
            strategy,
            "daily_rebalance_after_bar",
            source_timestamp,
            "on_daily_rebalance_after_bar",
            trading_date,
            source_timestamp,
            payload={"trading_date": trading_date, "timestamp": source_timestamp},
            use_previous_account_snapshot=False,
        )
        strategy._framework_daily_rebalance_after_bar_done_date = trading_date
    return True


def mark_portfolio_dirty(strategy: Any) -> None:
    """标记账户快照需要重新计算."""
    strategy._framework_portfolio_dirty = True


def dispatch_portfolio_update(strategy: Any) -> None:
    """在账户状态变化时分发 on_portfolio_update."""
    if strategy.ctx is None:
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
    last_session = getattr(strategy, "_framework_last_session", None)
    if last_session is not None:
        call_user_callback(
            strategy,
            "on_session_end",
            last_session,
            current_time,
            payload={"session": last_session, "timestamp": current_time},
        )
        strategy._framework_last_session = None

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
    if not hasattr(strategy, "_framework_last_session"):
        strategy._framework_last_session = None
    if not hasattr(strategy, "_framework_last_local_date"):
        strategy._framework_last_local_date = None
    if not hasattr(strategy, "_framework_before_trading_done_date"):
        strategy._framework_before_trading_done_date = None
    if not hasattr(strategy, "_framework_daily_rebalance_done_date"):
        strategy._framework_daily_rebalance_done_date = None
    if not hasattr(strategy, "_framework_daily_rebalance_after_bar_done_date"):
        strategy._framework_daily_rebalance_after_bar_done_date = None
    if not hasattr(strategy, "_framework_daily_rebalance_pending_date"):
        strategy._framework_daily_rebalance_pending_date = None
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
    if not hasattr(strategy, "_framework_expiry_event_keys"):
        strategy._framework_expiry_event_keys = set()
    if not hasattr(strategy, "_framework_stop_flushed"):
        strategy._framework_stop_flushed = False
    if not hasattr(strategy, "_framework_boundary_timers_registered"):
        strategy._framework_boundary_timers_registered = False
    if not hasattr(strategy, "_trading_day_bounds"):
        strategy._trading_day_bounds = {}
    if not hasattr(strategy, "_trading_day_after_bar_rebalance_timestamps"):
        strategy._trading_day_after_bar_rebalance_timestamps = {}
    if not hasattr(strategy, "_framework_daily_rebalance_after_bar_timers_registered"):
        strategy._framework_daily_rebalance_after_bar_timers_registered = False
    strategy._framework_state_ready = True
