import datetime as dt
from collections import deque
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

from .log import get_logger
from .strategy_runtime_config import RUNTIME_CONFIG_DEFAULTS

logger = get_logger("strategy")


#: 会话的 tick/bar 双流声明: ``False`` 无, ``True`` 全部标的, 集合为这些标的。
DualStream = Union[bool, frozenset[str]]


def normalize_dual_stream_declaration(value: Any) -> DualStream:
    """把网关元数据 / 回测入口给出的双流声明归一成 :data:`DualStream`.

    ``None`` 与空集合都视为"无双流"(缺省保持旧行为: tick 照常推进基础指标)。
    """
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        symbols = frozenset(str(item) for item in value)
        return symbols if symbols else False
    raise TypeError(
        f"dual_stream 须为 bool 或标的集合, 收到 {type(value).__name__}: {value!r}"
    )


def is_dual_stream_symbol(strategy: Any, symbol: Optional[str]) -> bool:
    """该标的在本会话里是否 tick/bar 双流(见 ``_framework_dual_stream``)."""
    declared = getattr(strategy, "_framework_dual_stream", False)
    if declared is True:
        return True
    if not declared or symbol is None:
        return False
    return str(symbol) in declared


def _runtime_option(strategy: Any, name: str) -> Any:
    default = RUNTIME_CONFIG_DEFAULTS[name]
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
    dispatch_lifecycle_timer 等跳过而静默失效。

    函数式策略是例外: ``FunctionalStrategy`` 在类体里无条件定义了全部回调转发
    方法(未提供时空转), 按类比较恒为 True, 快路径整体失效——引擎会为每根 Bar
    分发无人消费的 pre_open / cross_section / 交易日钩子 / portfolio-update。
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


def call_window_bar_callback(strategy: Any, callback: Any, bar: Any) -> Any:
    """调用窗口 bar 回调; 期间 `_framework_current_callback` 记为 "window:<freq>".

    与 :func:`call_user_callback` 同样把异常转发到 ``on_error``(callback_name 报
    ``"on_window_bar"``)。不复用它是因为它按属性名取回调, 而 subscribe_bars 的
    callback 可以是任意可调用对象。
    """
    previous_callback = getattr(strategy, "_framework_current_callback", None)
    freq = getattr(bar, "freq", None)
    strategy._framework_current_callback = f"window:{freq}"
    try:
        return callback(bar)
    except Exception as exc:
        try:
            strategy.on_error(exc, "on_window_bar", bar)
        except Exception:
            pass
        if not _should_reraise_on_error(strategy):
            return None
        raise
    finally:
        strategy._framework_current_callback = previous_callback


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


#: 引擎生成的交易日生命周期事件 payload 前缀（与 Rust ``LIFECYCLE_TIMER_PREFIX`` 一致）.
_LIFECYCLE_PREFIX = "__akq_lifecycle__|"


def is_lifecycle_payload(payload: str) -> bool:
    """定时器 payload 是否为引擎生成的交易日生命周期事件."""
    return payload.startswith(_LIFECYCLE_PREFIX)


#: 由引擎生命周期事件驱动的交易日钩子.
LIFECYCLE_HOOK_NAMES = ("on_pre_open", "on_before_trading", "on_after_trading")


def configure_engine_lifecycle_dispatch(engine: Any, strategies: Iterable[Any]) -> None:
    """按钩子告知引擎是否有策略（含槽位）重写了交易日生命周期钩子.

    没人重写的钩子，引擎不把对应的盘前/开始交易事件派发给 Python；收盘后事件
    没人重写时只在有过期单或到期事件要送达时派发——只写 ``on_bar`` 的策略不再为
    每个交易日多付三次派发。必须在 ``engine.run()`` 之前调用。
    """
    setter = getattr(engine, "set_lifecycle_hooks_dispatch", None)
    if setter is None:
        return
    strategy_list = list(strategies)
    setter(
        *(
            any(_strategy_overrides_callback(s, name) for s in strategy_list)
            for name in LIFECYCLE_HOOK_NAMES
        )
    )


def dispatch_lifecycle_timer(strategy: Any, payload: str) -> bool:
    """处理引擎生成的交易日生命周期事件，返回是否已消费该 payload.

    payload 形如 ``__akq_lifecycle__|<kind>|<YYYY-MM-DD>``，kind 为
    ``pre_open`` / ``before_trading`` / ``after_trading``。每个钩子每个交易日
    最多触发一次（按交易日去重）；未重写的钩子不调用。
    """
    if not is_lifecycle_payload(payload):
        return False
    parts = payload[len(_LIFECYCLE_PREFIX) :].split("|")
    if len(parts) != 2:
        return True
    kind, day_text = parts
    try:
        day = dt.date.fromisoformat(day_text)
    except ValueError:
        return True
    ts = int(getattr(strategy.ctx, "current_time", 0))
    if kind == "pre_open":
        if getattr(strategy, "_framework_pre_open_done_date", None) != day:
            strategy._framework_pre_open_done_date = day
            if _strategy_overrides_callback(strategy, "on_pre_open"):
                event = _build_pre_open_event(strategy, day, ts)
                _run_in_framework_phase(
                    strategy, "pre_open", ts, "on_pre_open", event, payload=event
                )
    elif kind == "before_trading":
        if getattr(strategy, "_framework_before_trading_done_date", None) != day:
            strategy._framework_before_trading_done_date = day
            if _strategy_overrides_callback(strategy, "on_before_trading"):
                _run_in_framework_phase(
                    strategy,
                    "before_trading",
                    ts,
                    "on_before_trading",
                    day,
                    ts,
                    payload={"trading_date": day, "timestamp": ts},
                )
    elif kind == "after_trading":
        if getattr(strategy, "_framework_after_trading_done_date", None) != day:
            strategy._framework_after_trading_done_date = day
            if _strategy_overrides_callback(strategy, "on_after_trading"):
                call_user_callback(
                    strategy,
                    "on_after_trading",
                    day,
                    ts,
                    payload={"trading_date": day, "timestamp": ts},
                )
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


def ensure_framework_state(strategy: Any) -> None:
    """确保框架级钩子状态字段存在."""
    # 性能: 本函数幂等且每 bar 被调用多次, 首次初始化后直接短路,
    # 避免逐 bar 数百次 hasattr 探测. checkpoint 恢复时 __getstate__ 会删除
    # 此标志, 强制重新初始化被重置的字段.
    if getattr(strategy, "_framework_state_ready", False):
        return
    if not hasattr(strategy, "_framework_before_trading_done_date"):
        strategy._framework_before_trading_done_date = None
    if not hasattr(strategy, "_framework_cross_section_done_date"):
        strategy._framework_cross_section_done_date = None
    if not hasattr(strategy, "_framework_after_trading_done_date"):
        strategy._framework_after_trading_done_date = None
    if not hasattr(strategy, "_framework_pre_open_done_date"):
        strategy._framework_pre_open_done_date = None
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
    if not hasattr(strategy, "_framework_dual_stream"):
        # 同 _framework_freq: 由本次运行的入口注入, 早于该字段的存档按"无双流"。
        strategy._framework_dual_stream = False
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
    if not hasattr(strategy, "_framework_live_stop_dispatched"):
        strategy._framework_live_stop_dispatched = False
    if not hasattr(strategy, "_trading_day_cross_section_timestamps"):
        strategy._trading_day_cross_section_timestamps = {}
    if not hasattr(strategy, "_framework_cross_section_timers_registered"):
        strategy._framework_cross_section_timers_registered = False
    strategy._framework_state_ready = True
