import datetime as dt_module
import logging
import os
import sys
import warnings
from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import pandas as pd
from pydantic import ValidationError

from .. import akquant as _akquant_module
from ..akquant import (
    AssetType,
    Bar,
    DataFeed,
    Engine,
    Instrument,
    OptionMarginModel,
    SettlementType,
    Tick,
    TradingSession,
)
from ..analyzer_plugin import AnalyzerManager, AnalyzerPlugin
from ..config import (
    BacktestConfig,
    ChinaFuturesConfig,
    ChinaFuturesInstrumentTemplateConfig,
    ChinaOptionsConfig,
    ChinaStockConfig,
    RiskConfig,
    StrategyConfig,
)
from ..data import ParquetDataCatalog
from ..feed_adapter import DEFAULT_INPUT_TIMEZONE, DataFeedAdapter, FeedSlice
from ..indicator_recording import IndicatorRecorder, IndicatorSink
from ..log import build_log_extra, get_logger, has_configured_handler, register_logger
from ..normalize import (
    coerce_to_pandas,
    dataframe_to_arrays,
    resolve_columns,
    to_indicator_frame,
)
from ..params import unknown_param_message
from ..risk import apply_risk_config
from ..schema import COLUMN_ALIASES
from ..strategy import (
    InstrumentAssetTypeName,
    InstrumentOptionMarginModelName,
    InstrumentOptionTypeName,
    InstrumentSettlementMode,
    InstrumentSnapshot,
    Strategy,
    StrategyConfigurationError,
    StrategyRuntimeConfig,
)
from ..strategy import (
    _legacy_init_arg_names as _legacy_init_arg_names_impl,
)
from ..strategy_framework_hooks import (
    collect_boundary_timer_entries as _collect_boundary_timer_entries_impl,
)
from ..strategy_framework_hooks import (
    collect_cross_section_timer_entries,
)
from ..strategy_framework_hooks import (
    collect_pre_open_timer_entries as _collect_pre_open_timer_entries_impl,
)
from ..strategy_loader import resolve_strategy_input
from ..utils.inspector import infer_warmup_period
from .fill_mode import FillMode
from .result import BacktestResult
from .tick_input import (
    aggregate_ticks_into_feed,
    normalize_market_input,
    parse_freq_to_interval_min,
)

_RUNTIME_CONFIG_FIELDS = {f.name for f in fields(StrategyRuntimeConfig)}
_collect_cross_section_entries_impl = collect_cross_section_timer_entries
DEFAULT_TIMEZONE = "Asia/Shanghai"
_LEGACY_FILL_POLICY_DICT_MSG = (
    "fill_policy no longer accepts a dict. Use a FillMode object instead:\n"
    '  {"price_basis": "open"}                                   -> NextOpen()\n'
    '  {"price_basis": "close", "bar_offset": 0}                 -> CurrentClose()\n'
    '  {"price_basis": "close", "bar_offset": 0,\n'
    '   "temporal": "next_event"}                                -> '
    'CurrentClose(timer_fill_timing="deferred")\n'
    '  {"price_basis": "close", "bar_offset": 1}                 -> NextClose()\n'
    '  {"price_basis": "ohlc4"}                                  -> NextAverage()\n'
    '  {"price_basis": "hl2"}                                    -> NextHighLowMid()'
)
_RUNTIME_EXECUTION_MODE = getattr(cast(Any, _akquant_module), "ExecutionMode", None)
_RUNTIME_MODE_NEXT_OPEN = getattr(_RUNTIME_EXECUTION_MODE, "NextOpen", "next_open")
_RUNTIME_MODE_CURRENT_CLOSE = getattr(
    _RUNTIME_EXECUTION_MODE, "CurrentClose", "current_close"
)
_RUNTIME_MODE_NEXT_CLOSE = getattr(_RUNTIME_EXECUTION_MODE, "NextClose", "next_close")
_RUNTIME_MODE_NEXT_AVERAGE = getattr(
    _RUNTIME_EXECUTION_MODE, "NextAverage", "next_average"
)
_RUNTIME_MODE_NEXT_HIGH_LOW_MID = getattr(
    _RUNTIME_EXECUTION_MODE, "NextHighLowMid", "next_high_low_mid"
)


class BacktestStreamEvent(TypedDict):
    """Backtest stream event payload."""

    run_id: str
    seq: int
    ts: int
    event_type: str
    symbol: Optional[str]
    level: str
    payload: Dict[str, str]


class FillPolicy(TypedDict, total=False):
    """Unified fill semantics for price basis and temporal policy."""

    price_basis: str
    temporal: str
    bar_offset: int


class SlippagePolicy(TypedDict, total=False):
    """Per-order slippage semantics."""

    type: str
    value: float


SlippageInput = Union[float, int, SlippagePolicy, Dict[str, Any], None]


class CommissionPolicy(TypedDict, total=False):
    """Per-order commission semantics."""

    type: str
    value: float


def _normalize_commission_policy(
    commission_policy: Optional[Dict[str, Any]],
    *,
    scope: str,
) -> Optional[CommissionPolicy]:
    if commission_policy is None:
        return None
    if not isinstance(commission_policy, dict):
        raise TypeError(f"{scope} must be a dict when provided")
    raw_type = str(commission_policy.get("type", "percent")).strip().lower()
    if raw_type not in {"percent", "fixed", "per_unit"}:
        raise ValueError(f"{scope}.type must be one of: percent, fixed, per_unit")
    raw_value = commission_policy.get("value", 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{scope}.value must be a number >= 0") from None
    if value < 0:
        raise ValueError(f"{scope}.value must be >= 0")
    return {"type": raw_type, "value": value}


def make_fill_policy(*args: object, **kwargs: object) -> "FillMode":
    """Raise TypeError; use FillMode objects (NextOpen(), CurrentClose(...)) instead."""
    raise TypeError(
        "make_fill_policy() has been removed. Use a FillMode object directly: "
        "NextOpen(), NextClose(), NextAverage(), NextHighLowMid(), "
        'CurrentClose(timer_fill_timing="immediate"|"deferred")'
    )


def _extract_strategy_log_context(
    strategy: Any,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract stable strategy identity fields for structured logs."""
    strategy_id = str(getattr(strategy, "_owner_strategy_id", "") or "").strip() or None
    symbol = None
    current_bar = getattr(strategy, "current_bar", None)
    current_tick = getattr(strategy, "current_tick", None)
    if current_bar is not None:
        symbol = str(getattr(current_bar, "symbol", "") or "").strip() or None
    elif current_tick is not None:
        symbol = str(getattr(current_tick, "symbol", "") or "").strip() or None
    return strategy_id, strategy_id, symbol


def _build_backtest_log_extra(
    *,
    phase: str,
    strategy: Optional[Any] = None,
    strategy_id: Optional[str] = None,
    slot: Optional[str] = None,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """Build structured logging context for backtest/runtime log records."""
    if strategy is not None:
        (
            strategy_id_from_strategy,
            slot_from_strategy,
            symbol_from_strategy,
        ) = _extract_strategy_log_context(strategy)
        if strategy_id is None:
            strategy_id = strategy_id_from_strategy
        if slot is None:
            slot = slot_from_strategy
        if symbol is None:
            symbol = symbol_from_strategy
    return build_log_extra(
        phase=phase,
        strategy_id=strategy_id,
        slot=slot,
        symbol=symbol,
    )


def _prime_framework_pre_open_timers(
    strategies: Sequence[Strategy], engine: Any
) -> None:
    """Prime global pre-open timers before the event loop starts."""
    add_timer = getattr(engine, "add_timer", None)
    if not callable(add_timer):
        return

    unique_timers: dict[str, int] = {}
    for current_strategy in strategies:
        entries = _collect_pre_open_timer_entries_impl(current_strategy)
        if entries:
            current_strategy._framework_pre_open_timers_registered = True
        for timestamp_ns, payload in entries:
            unique_timers[payload] = int(timestamp_ns)

    for payload, timestamp_ns in sorted(
        unique_timers.items(),
        key=lambda item: (int(item[1]), item[0]),
    ):
        add_timer(timestamp_ns, payload)


def _prime_framework_boundary_timers(
    strategies: Sequence[Strategy], engine: Any
) -> None:
    """Prime global boundary timers before the event loop starts."""
    add_timer = getattr(engine, "add_timer", None)
    if not callable(add_timer):
        return

    unique_timers: dict[str, int] = {}
    for current_strategy in strategies:
        entries = _collect_boundary_timer_entries_impl(current_strategy)
        if entries:
            current_strategy._framework_boundary_timers_registered = True
        for timestamp_ns, payload in entries:
            unique_timers[payload] = int(timestamp_ns)

    for payload, timestamp_ns in sorted(
        unique_timers.items(),
        key=lambda item: (int(item[1]), item[0]),
    ):
        add_timer(timestamp_ns, payload)


def _prime_framework_cross_section_timers(
    strategies: Sequence[Strategy], engine: Any
) -> None:
    """Prime daily after-bar rebalance timers before the event loop starts."""
    add_timer = getattr(engine, "add_timer", None)
    if not callable(add_timer):
        return

    unique_timers: dict[str, int] = {}
    for current_strategy in strategies:
        entries = _collect_cross_section_entries_impl(current_strategy)
        if entries:
            current_strategy._framework_cross_section_timers_registered = True
        for timestamp_ns, payload in entries:
            unique_timers[payload] = int(timestamp_ns)

    for payload, timestamp_ns in sorted(
        unique_timers.items(),
        key=lambda item: (int(item[1]), item[0]),
    ):
        add_timer(timestamp_ns, payload)


@dataclass(frozen=True)
class ResolvedExecutionPolicy:
    """Resolved execution semantics for matching."""

    price_basis: str
    bar_offset: int
    temporal: str
    execution_mode: Any
    source: Literal["fill_policy", "legacy"]


@dataclass
class PreparedStreamRuntime:
    """Prepared stream runtime components shared by backtest/checkpoint resume."""

    stream_on_event: Optional[Callable[[BacktestStreamEvent], None]]
    indicator_stream_emitter: Optional[
        Callable[[str, Optional[str], str, Dict[str, str]], None]
    ]
    indicator_stream_point_interval: int
    indicator_stream_snapshot_interval: int
    event_stats_snapshot: Dict[str, Any]
    stream_progress_interval: int
    stream_equity_interval: int
    stream_batch_size: int
    stream_max_buffer: int
    stream_error_mode: str
    stream_mode: str


_SUPPORTED_FILL_TEMPORAL: set[str] = {"same_cycle", "next_event"}


def _basis_offset_to_mode(price_basis: str, bar_offset: int) -> Any:
    """Map an already-validated (price_basis, bar_offset) pair to a mode enum.

    No validation, no raise: callers must pass a triple that FillMode._to_core()
    already produced (see FillMode/fill_mode_from_core for the legality rules).
    """
    if price_basis == "open":
        return _RUNTIME_MODE_NEXT_OPEN
    if price_basis == "close":
        return (
            _RUNTIME_MODE_CURRENT_CLOSE if bar_offset == 0 else _RUNTIME_MODE_NEXT_CLOSE
        )
    if price_basis == "ohlc4":
        return _RUNTIME_MODE_NEXT_AVERAGE
    return _RUNTIME_MODE_NEXT_HIGH_LOW_MID


def _resolve_execution_policy(
    execution_mode: Union[Any, str],
    timer_execution_policy: str,
    fill_policy: Optional[FillPolicy],
    logger: logging.Logger,
) -> ResolvedExecutionPolicy:
    resolved_execution_mode = execution_mode
    resolved_timer_policy = str(timer_execution_policy).strip().lower()
    resolved_price_basis = "open"
    resolved_bar_offset = 1
    resolved_source: Literal["fill_policy", "legacy"] = "legacy"
    if fill_policy is not None:
        if not isinstance(fill_policy, dict):
            raise TypeError("internal: fill_policy must be pre-translated to dict")
        resolved_price_basis = str(fill_policy["price_basis"])
        resolved_bar_offset = int(fill_policy["bar_offset"])
        resolved_timer_policy = str(fill_policy["temporal"])
        resolved_execution_mode = _basis_offset_to_mode(
            resolved_price_basis, resolved_bar_offset
        )
        resolved_source = "fill_policy"

    if isinstance(resolved_execution_mode, str):
        mode_text = str(resolved_execution_mode).strip()
        mode_raw = mode_text.split(".", 1)[-1] if "." in mode_text else mode_text
        mode_compact = mode_raw.replace(" ", "").replace("-", "_")
        mode_key = mode_compact.lower()
        mode_map = {
            "open": (_RUNTIME_MODE_NEXT_OPEN, "open", 1),
            "close": (_RUNTIME_MODE_CURRENT_CLOSE, "close", 0),
            "next_open": (_RUNTIME_MODE_NEXT_OPEN, "open", 1),
            "nextopen": (_RUNTIME_MODE_NEXT_OPEN, "open", 1),
            "current_close": (_RUNTIME_MODE_CURRENT_CLOSE, "close", 0),
            "currentclose": (_RUNTIME_MODE_CURRENT_CLOSE, "close", 0),
            "next_close": (_RUNTIME_MODE_NEXT_CLOSE, "close", 1),
            "nextclose": (_RUNTIME_MODE_NEXT_CLOSE, "close", 1),
            "next_average": (_RUNTIME_MODE_NEXT_AVERAGE, "ohlc4", 1),
            "nextaverage": (_RUNTIME_MODE_NEXT_AVERAGE, "ohlc4", 1),
            "next_high_low_mid": (_RUNTIME_MODE_NEXT_HIGH_LOW_MID, "hl2", 1),
            "nexthighlowmid": (_RUNTIME_MODE_NEXT_HIGH_LOW_MID, "hl2", 1),
            "ohlc4": (_RUNTIME_MODE_NEXT_AVERAGE, "ohlc4", 1),
            "hl2": (_RUNTIME_MODE_NEXT_HIGH_LOW_MID, "hl2", 1),
        }
        mode_tuple = mode_map.get(mode_key)
        if not mode_tuple:
            logger.warning(
                "Unknown execution mode '%s', defaulting to NextOpen",
                resolved_execution_mode,
            )
            mode_tuple = (_RUNTIME_MODE_NEXT_OPEN, "open", 1)
        resolved_mode_enum, mapped_basis, mapped_offset = mode_tuple
        if fill_policy is None:
            resolved_price_basis = mapped_basis
            resolved_bar_offset = mapped_offset
    else:
        resolved_mode_enum = resolved_execution_mode
        if fill_policy is None:
            reverse_mode_map = {
                _RUNTIME_MODE_NEXT_OPEN: ("open", 1),
                _RUNTIME_MODE_CURRENT_CLOSE: ("close", 0),
                _RUNTIME_MODE_NEXT_CLOSE: ("close", 1),
                _RUNTIME_MODE_NEXT_AVERAGE: ("ohlc4", 1),
                _RUNTIME_MODE_NEXT_HIGH_LOW_MID: ("hl2", 1),
            }
            mapped_basis, mapped_offset = reverse_mode_map.get(
                resolved_mode_enum, ("open", 1)
            )
            resolved_price_basis = mapped_basis
            resolved_bar_offset = mapped_offset

    if resolved_timer_policy not in _SUPPORTED_FILL_TEMPORAL:
        raise ValueError(
            "timer_execution_policy must be one of: same_cycle, next_event"
        )

    return ResolvedExecutionPolicy(
        price_basis=resolved_price_basis,
        bar_offset=resolved_bar_offset,
        temporal=resolved_timer_policy,
        execution_mode=resolved_mode_enum,
        source=resolved_source,
    )


def _raise_if_legacy_execution_policy_used(
    *, legacy_mode_used: bool, legacy_timer_used: bool, api_name: str
) -> None:
    if not (legacy_mode_used or legacy_timer_used):
        return
    raise ValueError(
        f"{api_name} no longer accepts execution_mode/timer_execution_policy; "
        "please use fill_policy"
    )


def _index_to_local_trading_days(
    index: pd.DatetimeIndex, timezone: str
) -> pd.DatetimeIndex:
    local_index = index
    if local_index.tz is None:
        local_index = local_index.tz_localize("UTC")
    return cast(pd.DatetimeIndex, local_index.tz_convert(timezone))


def _parse_runtime_boundary_timestamp(
    value: Union[str, Any, pd.Timestamp], timezone: str
) -> pd.Timestamp:
    """Interpret naive runtime boundaries in the configured strategy timezone."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return cast(pd.Timestamp, timestamp.tz_localize(timezone))
    return timestamp


def _boundary_timestamp_to_utc_ns(
    value: Union[str, Any, pd.Timestamp], timezone: str
) -> int:
    timestamp = _parse_runtime_boundary_timestamp(value, timezone)
    return int(timestamp.tz_convert("UTC").value)


def _filter_datetime_index_frame_by_runtime_window(
    frame: pd.DataFrame,
    start_time: Optional[Union[str, Any]],
    end_time: Optional[Union[str, Any]],
    timezone: str,
) -> pd.DataFrame:
    """Filter a datetime-indexed frame using runtime boundaries.

    Naive frame indices follow the same Shanghai-default market-data semantics as
    the main DataFrame -> Bar conversion path, while naive runtime boundaries are
    interpreted in the configured strategy timezone.
    """
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    compare_index = frame.index
    if compare_index.tz is None:
        compare_index = compare_index.tz_localize(DEFAULT_INPUT_TIMEZONE)
    mask = pd.Series(True, index=frame.index)
    if start_time is not None:
        start_ts = _parse_runtime_boundary_timestamp(start_time, timezone)
        mask &= compare_index >= start_ts
    if end_time is not None:
        end_ts = _parse_runtime_boundary_timestamp(end_time, timezone)
        mask &= compare_index <= end_ts
    return cast(pd.DataFrame, frame.loc[mask.to_numpy()])


def _build_trading_day_metadata(
    data_map_for_indicators: Dict[str, pd.DataFrame], timezone: str
) -> Tuple[List[pd.Timestamp], Dict[str, Tuple[int, int]], Dict[str, int]]:
    """Build sorted trading days, per-day bounds, and rebalance timestamps."""
    frames = {
        str(symbol): df
        for symbol, df in data_map_for_indicators.items()
        if not df.empty and isinstance(df.index, pd.DatetimeIndex)
    }
    if not frames:
        return [], {}, {}

    big_index = pd.concat(frames, names=["symbol"]).index
    dates = cast(pd.DatetimeIndex, big_index.get_level_values(-1))
    if dates.tz is None:
        dates = dates.tz_localize("UTC")
    local_index = dates.tz_convert(timezone)
    normalized_index = cast(pd.DatetimeIndex, local_index.normalize())

    # pandas datetime64 may use a non-nanosecond unit (e.g. us from polars);
    # normalize to ns so integer values match Timestamp.value semantics.
    utc_index = cast(
        pd.DatetimeIndex, local_index.tz_convert("UTC").astype("datetime64[ns, UTC]")
    )
    ns = pd.Series(utc_index, copy=False).astype("int64").to_numpy()
    symbols = big_index.get_level_values(0).astype(str)

    base = pd.DataFrame({"day": normalized_index, "sym": symbols, "ns": ns})
    # Drop NaT-indexed rows before aggregating: `asi8` maps NaT to int64's
    # sentinel min, which the vectorized path would otherwise fold into a
    # day's min/max across *all* symbols (the per-symbol loop only tainted
    # that symbol's own NaT bucket). Dropping keeps a stray NaT bar from
    # silently corrupting every symbol's bounds on that slice.
    if normalized_index.hasnans:
        base = base[base["day"].notna()]
        if base.empty:
            return [], {}, {}

    all_dates = sorted(set(base["day"].unique()))

    bounds = base.groupby("day", sort=False)["ns"].agg(["min", "max"])
    day_bounds: Dict[str, Tuple[int, int]] = {
        pd.Timestamp(day_ts).date().isoformat(): (int(start_ns), int(end_ns))
        for day_ts, start_ns, end_ns in bounds.itertuples()
    }

    rebalance = (
        base.groupby(["day", "sym"], sort=False)["ns"].min().groupby("day").max()
    )
    day_rebalance_timestamps: Dict[str, int] = {
        cast(pd.Timestamp, day_ts).date().isoformat(): int(v)
        for day_ts, v in rebalance.items()
    }

    return all_dates, day_bounds, day_rebalance_timestamps


if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

# polars.DataFrame / pyarrow.Table 为平权一等输入(issue #298),
# 运行时由 coerce_to_pandas 统一转 pandas 后走既有数据路径.
BacktestDataInput = Union[
    pd.DataFrame,
    "pl.DataFrame",
    "pl.LazyFrame",
    "pa.Table",
    Dict[str, pd.DataFrame],
    List[Bar],
    List[Tick],
    List[Union[Bar, Tick]],
    DataFeed,
    DataFeedAdapter,
]

_BROKER_PROFILE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "cn_stock_miniqmt": {
        "commission_rate": 0.0003,
        "stamp_tax_rate": 0.001,
        "transfer_fee_rate": 0.00001,
        "min_commission": 5.0,
        "slippage": {"type": "percent", "value": 0.0002},
        "volume_limit_pct": 0.2,
        "lot_size": 100,
    },
    "cn_stock_t1_low_fee": {
        "commission_rate": 0.0002,
        "stamp_tax_rate": 0.001,
        "transfer_fee_rate": 0.000005,
        "min_commission": 3.0,
        "slippage": {"type": "percent", "value": 0.0001},
        "volume_limit_pct": 0.25,
        "lot_size": 100,
    },
    "cn_stock_sim_high_slippage": {
        "commission_rate": 0.0003,
        "stamp_tax_rate": 0.001,
        "transfer_fee_rate": 0.00001,
        "min_commission": 5.0,
        "slippage": {"type": "percent", "value": 0.001},
        "volume_limit_pct": 0.1,
        "lot_size": 100,
    },
}


def _parse_positive_int_option(name: str, value: Any) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _parse_stream_error_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"continue", "fail_fast"}:
        raise ValueError("stream_error_mode must be 'continue' or 'fail_fast'")
    return mode


def _parse_stream_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"observability", "audit"}:
        raise ValueError("stream_mode must be 'observability' or 'audit'")
    return mode


def _noop_stream_event_handler(_event: BacktestStreamEvent) -> None:
    return None


def _prepare_stream_runtime(
    *,
    on_event: Optional[Callable[[BacktestStreamEvent], None]],
    kwargs: Dict[str, Any],
    owner_strategy_id: Optional[str] = None,
    patch_owner_strategy_id: bool = False,
) -> PreparedStreamRuntime:
    stream_on_event = on_event
    internal_stream_callback = kwargs.pop("_stream_on_event", None)
    if internal_stream_callback is not None and stream_on_event is not None:
        raise TypeError("on_event and _stream_on_event cannot be provided together")
    if internal_stream_callback is not None:
        stream_on_event = internal_stream_callback
    if stream_on_event is not None and not callable(stream_on_event):
        raise TypeError("on_event must be callable when provided")
    if stream_on_event is None:
        stream_on_event = _noop_stream_event_handler
    original_stream_handler = stream_on_event
    # 性能: 无真实消费者(仅 noop)时, 引擎每笔 order/trade 仍跨回 Python 跑
    # wrapped_stream_on_event 再丢弃(实测约占总耗时 20%). 返回 None 让下游守卫
    # 跳过 set_stream_callback, emit_stream_event 无回调时本就 early-return.
    # 代价: 非-streaming 回测 result._event_stats 为空(该字段是 streaming
    # 可观测性特性, 无消费者时无意义).
    has_stream_consumer = original_stream_handler is not _noop_stream_event_handler
    indicator_stream_point_interval = _parse_positive_int_option(
        "indicator_stream_point_interval",
        kwargs.pop("indicator_stream_point_interval", 1),
    )
    indicator_stream_snapshot_interval = _parse_positive_int_option(
        "indicator_stream_snapshot_interval",
        kwargs.pop("indicator_stream_snapshot_interval", 1),
    )
    event_stats_snapshot: Dict[str, Any] = {}
    stream_state: Dict[str, Any] = {
        "run_id": None,
        "last_rust_seq": 0,
        "seq_offset": 0,
        "indicator_point_seen": 0,
        "indicator_snapshot_seen": 0,
    }

    def wrapped_stream_on_event(event: BacktestStreamEvent) -> None:
        forwarded_event: Dict[str, Any] = dict(event)
        event_type = str(forwarded_event.get("event_type", ""))
        if event_type == "finished":
            payload_obj = forwarded_event.get("payload", {})
            if isinstance(payload_obj, dict):
                for key in (
                    "processed_events",
                    "dropped_event_count",
                    "callback_error_count",
                    "backpressure_policy",
                    "stream_mode",
                    "sampling_enabled",
                    "sampling_rate",
                    "reason",
                ):
                    if key in payload_obj:
                        event_stats_snapshot[key] = payload_obj.get(key)
        if patch_owner_strategy_id and owner_strategy_id is not None:
            if event_type in {"order", "trade", "risk"}:
                payload_obj = forwarded_event.get("payload", {})
                if isinstance(payload_obj, dict):
                    current_owner = payload_obj.get("owner_strategy_id")
                    if current_owner is None or str(current_owner) == "":
                        patched_payload = dict(payload_obj)
                        patched_payload["owner_strategy_id"] = owner_strategy_id
                        forwarded_event["payload"] = cast(
                            Dict[str, str], patched_payload
                        )
        run_id = forwarded_event.get("run_id")
        if run_id is not None and str(run_id):
            normalized_run_id = str(run_id)
            stream_state["run_id"] = normalized_run_id
            event_stats_snapshot["run_id"] = normalized_run_id
        raw_seq = int(forwarded_event.get("seq", 0))
        forwarded_event["seq"] = raw_seq + int(stream_state["seq_offset"])
        stream_state["last_rust_seq"] = raw_seq
        original_stream_handler(cast(BacktestStreamEvent, forwarded_event))

    def emit_indicator_stream_event(
        event_type: str,
        symbol: Optional[str],
        level: str,
        payload: Dict[str, str],
    ) -> None:
        run_id = stream_state.get("run_id")
        if not run_id:
            return
        if event_type == "indicator_point":
            stream_state["indicator_point_seen"] = (
                int(stream_state["indicator_point_seen"]) + 1
            )
            if (
                int(stream_state["indicator_point_seen"])
                % indicator_stream_point_interval
                != 0
            ):
                return
        elif event_type == "indicator_snapshot":
            stream_state["indicator_snapshot_seen"] = (
                int(stream_state["indicator_snapshot_seen"]) + 1
            )
            if (
                int(stream_state["indicator_snapshot_seen"])
                % indicator_stream_snapshot_interval
                != 0
            ):
                return
        try:
            event_ts = int(str(payload.get("timestamp", "0")))
        except (TypeError, ValueError):
            event_ts = 0
        next_offset = int(stream_state["seq_offset"]) + 1
        stream_state["seq_offset"] = next_offset
        original_stream_handler(
            cast(
                BacktestStreamEvent,
                {
                    "run_id": str(run_id),
                    "seq": int(stream_state["last_rust_seq"]) + next_offset,
                    "ts": event_ts,
                    "event_type": str(event_type),
                    "symbol": None if symbol is None else str(symbol),
                    "level": str(level),
                    "payload": {str(key): str(value) for key, value in payload.items()},
                },
            )
        )

    stream_progress_interval = _parse_positive_int_option(
        "stream_progress_interval", kwargs.pop("stream_progress_interval", 1)
    )
    stream_equity_interval = _parse_positive_int_option(
        "stream_equity_interval", kwargs.pop("stream_equity_interval", 1)
    )
    stream_batch_size = _parse_positive_int_option(
        "stream_batch_size", kwargs.pop("stream_batch_size", 1)
    )
    stream_max_buffer = _parse_positive_int_option(
        "stream_max_buffer", kwargs.pop("stream_max_buffer", 1024)
    )
    stream_error_mode = _parse_stream_error_mode(
        kwargs.pop("stream_error_mode", "continue")
    )
    stream_mode = _parse_stream_mode(kwargs.pop("stream_mode", "observability"))
    if "legacy_execution_policy_compat" in kwargs:
        raise TypeError(
            "legacy_execution_policy_compat is no longer supported; "
            "please use fill_policy"
        )
    return PreparedStreamRuntime(
        stream_on_event=wrapped_stream_on_event if has_stream_consumer else None,
        indicator_stream_emitter=emit_indicator_stream_event,
        indicator_stream_point_interval=indicator_stream_point_interval,
        indicator_stream_snapshot_interval=indicator_stream_snapshot_interval,
        event_stats_snapshot=event_stats_snapshot,
        stream_progress_interval=stream_progress_interval,
        stream_equity_interval=stream_equity_interval,
        stream_batch_size=stream_batch_size,
        stream_max_buffer=stream_max_buffer,
        stream_error_mode=stream_error_mode,
        stream_mode=stream_mode,
    )


def _attach_result_runtime_metadata(
    *,
    result: BacktestResult,
    engine_summary: Any,
    event_stats_snapshot: Dict[str, Any],
    owner_strategy_id: str,
    resolved_policy: Optional[ResolvedExecutionPolicy],
) -> None:
    setattr(result, "_engine_summary", engine_summary)
    setattr(result, "_event_stats", dict(event_stats_snapshot))
    setattr(result, "_owner_strategy_id", owner_strategy_id)
    run_id = event_stats_snapshot.get("run_id")
    result.stream_run_id = None if run_id in (None, "") else str(run_id)
    if resolved_policy is not None:
        setattr(
            result,
            "_resolved_execution_policy",
            {
                "price_basis": resolved_policy.price_basis,
                "bar_offset": resolved_policy.bar_offset,
                "temporal": resolved_policy.temporal,
                "source": resolved_policy.source,
            },
        )
        result.resolved_execution_policy = cast(
            Dict[str, Any], getattr(result, "_resolved_execution_policy")
        )


def _build_resolved_backtest_config(
    *,
    slippage_policy: SlippagePolicy,
    volume_limit_pct: float,
    commission_policy: CommissionPolicy,
    stamp_tax_rate: float,
    transfer_fee_rate: float,
    min_commission: float,
    t_plus_one: bool,
    timezone: str,
    history_depth: int,
    resolved_policy: Optional[ResolvedExecutionPolicy],
) -> Dict[str, Any]:
    """Snapshot-friendly resolved runtime config (issue #282).

    Single source of truth for what checkpoint resume must re-apply. Kept as a plain
    JSON-able dict so it pickles cleanly and stays forward-compatible.
    """
    resolved: Dict[str, Any] = {
        "slippage": dict(slippage_policy),
        "volume_limit_pct": float(volume_limit_pct),
        "commission_policy": dict(commission_policy),
        "stamp_tax_rate": float(stamp_tax_rate),
        "transfer_fee_rate": float(transfer_fee_rate),
        "min_commission": float(min_commission),
        "t_plus_one": bool(t_plus_one),
        "timezone": str(timezone),
        "history_depth": int(history_depth),
    }
    if resolved_policy is not None:
        resolved["fill_policy"] = {
            "price_basis": resolved_policy.price_basis,
            "bar_offset": resolved_policy.bar_offset,
            "temporal": resolved_policy.temporal,
        }
    return resolved


def _attach_indicator_recorder(
    *,
    stream_emitter: Optional[
        Callable[[str, Optional[str], str, Dict[str, str]], None]
    ] = None,
    strategy_instance: Strategy,
    slot_strategy_instances: Dict[str, Strategy],
    recorder: Optional[IndicatorSink] = None,
) -> IndicatorSink:
    """Attach one shared indicator recorder to all strategy instances.

    When ``recorder`` is provided (a public :class:`IndicatorSink`), it is used
    verbatim and the stream emitter is attached to it; otherwise a built-in
    :class:`IndicatorRecorder` is created. This is the injection point behind
    ``run_backtest(indicator_recorder=...)``.
    """
    if recorder is None:
        recorder = IndicatorRecorder(stream_emitter=stream_emitter)
    else:
        recorder.set_stream_emitter(stream_emitter)
    setattr(strategy_instance, "_indicator_recorder", recorder)
    for slot_strategy in slot_strategy_instances.values():
        setattr(slot_strategy, "_indicator_recorder", recorder)
    return recorder


def _normalize_symbols_argument(
    symbols: Union[str, List[str], Tuple[str, ...], set[str]],
    *,
    api_name: str,
) -> List[str]:
    """Normalize symbols input."""
    if isinstance(symbols, str):
        normalized = [symbols]
    elif isinstance(symbols, (list, tuple, set)):
        normalized = [str(item) for item in symbols]
    else:
        raise TypeError("symbols must be str, list, tuple, or set")

    cleaned: List[str] = []
    seen: set[str] = set()
    for item in normalized:
        value = str(item).strip()
        if not value:
            raise ValueError("symbols cannot contain empty values")
        if value in seen:
            continue
        seen.add(value)
        cleaned.append(value)

    # 传空集合(cleaned 为空)时不在此处报错: 调用方 _resolve_effective_symbols
    # 紧接着会用 api_name 拼出更具体的中文报错(说明后果与两条可选修法), 这里
    # 若先行抛出英文短消息("symbols cannot be empty"), 会让那条更有用的报错
    # 永远拿不到执行机会(死代码)。空字符串等"非空集合但含空值"的情形仍在上面
    # 的循环内单独报错, 不受此影响。
    return cleaned


def _resolve_effective_symbols(
    *,
    symbols: Union[str, List[str], Tuple[str, ...], set[str], None],
    kwargs: Dict[str, Any],
    api_name: str,
) -> Tuple[Union[str, List[str], Tuple[str, ...], set[str]], List[str], bool]:
    """解析 symbols 参数.

    第三个返回值 `symbols_explicit` 表示用户是否**显式**传入了 `symbols`。
    过滤只在显式传入时启用 —— 注意本函数在未传时会填入默认值
    `"BENCHMARK"`, 若照它字面去过滤会把所有真实数据都滤掉。
    """
    if "symbol" in kwargs:
        raise ValueError(
            f"{api_name} no longer accepts `symbol`; please use `symbols` only"
        )

    if symbols is None and "symbols" in kwargs:
        symbols = cast(
            Union[str, List[str], Tuple[str, ...], set[str]],
            kwargs.pop("symbols"),
        )
    elif "symbols" in kwargs:
        kwargs.pop("symbols")
    symbols_explicit = symbols is not None
    if symbols is None:
        symbols = "BENCHMARK"
    effective_symbols = _normalize_symbols_argument(
        symbols=symbols,
        api_name=api_name,
    )
    if symbols_explicit and not effective_symbols:
        raise ValueError(
            f"{api_name} 的 symbols 不能为空: 传空集合会得到一个不放行任何标的的"
            "空回测。要跑全部数据请省略 symbols(沿用「数据即订阅」), "
            "要跑指定标的请至少给一个。"
        )
    return symbols, effective_symbols, symbols_explicit


def _merge_symbol_whitelist_sources(
    effective_symbols: List[str],
    config: Any,
    *strategy_instances: Any,
) -> List[str]:
    """合并 effective_symbols、config.instruments、各策略当前的 _subscriptions.

    刻意从**已归一化**的 `effective_symbols`(`List[str]`)出发, 不接受
    `_resolve_effective_symbols` 第一个返回值(原始 `symbols`, 可能是未归一的
    `str`/`tuple`/`set`)——`set("600519")` 会把字符串拆成单字符集合
    `{'6','0','0','5','1','9'}`, 这类误用曾使 subscribe() 白名单校验把
    合法的自身标的都拦下来(见调用点注释)。

    调用时机决定了合并范围: 在 on_start 之前调用只能看到 __init__ 里已完成的
    subscribe; 在 on_start 之后调用能看到 on_start 里新增的 subscribe。两处
    调用都复用这同一份逻辑, 避免分别手写而漂移出不一致的合并顺序。
    """
    merged: List[str] = list(effective_symbols)
    if config and getattr(config, "instruments", None):
        for s in config.instruments:
            if s not in merged:
                merged.append(s)
    for strategy_instance in strategy_instances:
        if hasattr(strategy_instance, "_subscriptions"):
            for s in strategy_instance._subscriptions:
                if s not in merged:
                    merged.append(s)
    return merged


def _strategy_param_field_names(
    strategy_input: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None],
) -> set[str]:
    """Return the declared __param_model__ field names for a Strategy subclass.

    任一 Strategy 子类都由 __init_subclass__/基类保证 __param_model__ 恒存在, 故
    返回空集只意味着**该策略没有声明任何内联参数字段**(通常是仍在用已废弃的构造
    函数签名写法), 不是「豁免校验」的信号——调用方应据此拒绝其全部 kwargs。
    """
    if isinstance(strategy_input, type) and issubclass(strategy_input, Strategy):
        return set(strategy_input.__param_model__.model_fields)
    return set()


def _strategy_own_init_arg_names(
    strategy_input: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None],
) -> List[str]:
    """Return the named args of a Strategy subclass's own ``__init__``, if any.

    只看该类自身命名空间里的 ``__init__``(不含继承而来的), 与
    ``strategy.py`` 的 ``__init_subclass__`` 告警判据保持一致口径; 用于把
    "该 key 是否出现在 __init__ 签名里但未迁移为内联字段"这一判断结果传给
    ``unknown_param_message``——``params.py`` 不能反向导入 ``strategy.py``
    (会成环), 故由本模块算好后传参。
    """
    if not (isinstance(strategy_input, type) and issubclass(strategy_input, Strategy)):
        return []
    own_init = vars(strategy_input).get("__init__")
    if own_init is None:
        return []
    return _legacy_init_arg_names_impl(own_init)


def _accepts_strategy_kwarg(
    strategy_input: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None],
    kwarg_name: str,
) -> bool:
    """Return whether the strategy declares this kwarg as a __param_model__ field."""
    return kwarg_name in _strategy_param_field_names(strategy_input)


def _split_strategy_kwargs(
    strategy_input: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None],
    strategy_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Split kwargs by __param_model__ field membership."""
    if not isinstance(strategy_input, type) or not issubclass(strategy_input, Strategy):
        return strategy_kwargs, []
    field_names = _strategy_param_field_names(strategy_input)
    accepted = {k: v for k, v in strategy_kwargs.items() if k in field_names}
    unknown = sorted(k for k in strategy_kwargs if k not in field_names)
    return accepted, unknown


def _resolve_broker_profile(profile: Optional[str]) -> Dict[str, Any]:
    if profile is None:
        return {}
    key = str(profile).strip().lower()
    if not key:
        return {}
    if key not in _BROKER_PROFILE_TEMPLATES:
        available = ", ".join(sorted(_BROKER_PROFILE_TEMPLATES.keys()))
        raise ValueError(
            f"Unknown broker_profile '{profile}', available profiles: {available}"
        )
    return dict(_BROKER_PROFILE_TEMPLATES[key])


def _resolve_stock_fee_rules(
    *,
    commission_policy: Optional[CommissionPolicy],
    commission_rate: Optional[float],
    stamp_tax_rate: Optional[float],
    transfer_fee_rate: Optional[float],
    min_commission: Optional[float],
    broker_profile_values: Dict[str, Any],
    strategy_config: Optional[Any],
) -> Tuple[CommissionPolicy, float, float, float]:
    resolved_commission_policy = _normalize_commission_policy(
        cast(Optional[Dict[str, Any]], commission_policy),
        scope="commission_policy",
    )
    if resolved_commission_policy is None and commission_rate is not None:
        resolved_commission_policy = {
            "type": "percent",
            "value": float(commission_rate),
        }
    resolved_stamp_tax_rate = stamp_tax_rate
    resolved_transfer_fee_rate = transfer_fee_rate
    resolved_min_commission = min_commission

    if resolved_commission_policy is None:
        profile_commission_rate = cast(
            Optional[float], broker_profile_values.get("commission_rate")
        )
        if profile_commission_rate is not None:
            resolved_commission_policy = {
                "type": "percent",
                "value": float(profile_commission_rate),
            }
    if resolved_stamp_tax_rate is None:
        resolved_stamp_tax_rate = cast(
            Optional[float], broker_profile_values.get("stamp_tax_rate")
        )
    if resolved_transfer_fee_rate is None:
        resolved_transfer_fee_rate = cast(
            Optional[float], broker_profile_values.get("transfer_fee_rate")
        )
    if resolved_min_commission is None:
        resolved_min_commission = cast(
            Optional[float], broker_profile_values.get("min_commission")
        )

    if strategy_config is not None:
        if resolved_commission_policy is None:
            resolved_commission_policy = _normalize_commission_policy(
                cast(
                    Optional[Dict[str, Any]],
                    getattr(strategy_config, "commission_policy", None),
                ),
                scope="strategy_config.commission_policy",
            )
        if resolved_commission_policy is None:
            config_commission_rate = cast(
                Optional[float], getattr(strategy_config, "commission_rate", None)
            )
            if config_commission_rate is not None:
                resolved_commission_policy = {
                    "type": "percent",
                    "value": float(config_commission_rate),
                }
        if resolved_stamp_tax_rate is None:
            resolved_stamp_tax_rate = cast(
                Optional[float], getattr(strategy_config, "stamp_tax_rate", None)
            )
        if resolved_transfer_fee_rate is None:
            resolved_transfer_fee_rate = cast(
                Optional[float], getattr(strategy_config, "transfer_fee_rate", None)
            )
        if resolved_min_commission is None:
            resolved_min_commission = cast(
                Optional[float], getattr(strategy_config, "min_commission", None)
            )

    if resolved_commission_policy is None:
        resolved_commission_policy = {"type": "percent", "value": 0.0}

    return (
        resolved_commission_policy,
        float(resolved_stamp_tax_rate if resolved_stamp_tax_rate is not None else 0.0),
        float(resolved_transfer_fee_rate or 0.0),
        float(resolved_min_commission or 0.0),
    )


def _apply_strategy_config_overrides(
    *,
    strategy_config: Optional[Any],
    strategy_id: Optional[str],
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ],
    strategy_max_order_value: Optional[Dict[str, float]],
    strategy_max_order_size: Optional[Dict[str, float]],
    strategy_max_position_size: Optional[Dict[str, float]],
    strategy_max_daily_loss: Optional[Dict[str, float]],
    strategy_max_drawdown: Optional[Dict[str, float]],
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]],
    strategy_risk_cooldown_bars: Optional[Dict[str, int]],
    strategy_priority: Optional[Dict[str, int]],
    strategy_risk_budget: Optional[Dict[str, float]],
    strategy_fill_policy: Optional[Dict[str, FillMode]],
    strategy_slippage: Optional[Dict[str, SlippageInput]],
    strategy_commission: Optional[Dict[str, CommissionPolicy]],
    portfolio_risk_budget: Optional[float],
    strategy_runtime_config: Optional[Union[StrategyRuntimeConfig, Dict[str, Any]]],
    strategy_source: Optional[Union[str, bytes, os.PathLike[str]]],
    strategy_loader: Optional[str],
    strategy_loader_options: Optional[Dict[str, Any]],
) -> Tuple[
    Optional[str],
    Optional[Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, float]],
    Optional[Dict[str, bool]],
    Optional[Dict[str, int]],
    Optional[Dict[str, int]],
    Optional[Dict[str, float]],
    Optional[Dict[str, FillMode]],
    Optional[Dict[str, SlippageInput]],
    Optional[Dict[str, CommissionPolicy]],
    Optional[float],
    Optional[Union[StrategyRuntimeConfig, Dict[str, Any]]],
    Optional[Union[str, bytes, os.PathLike[str]]],
    Optional[str],
    Optional[Dict[str, Any]],
]:
    if strategy_config is None:
        return (
            strategy_id,
            strategies_by_slot,
            strategy_max_order_value,
            strategy_max_order_size,
            strategy_max_position_size,
            strategy_max_daily_loss,
            strategy_max_drawdown,
            strategy_reduce_only_after_risk,
            strategy_risk_cooldown_bars,
            strategy_priority,
            strategy_risk_budget,
            strategy_fill_policy,
            strategy_slippage,
            strategy_commission,
            portfolio_risk_budget,
            strategy_runtime_config,
            strategy_source,
            strategy_loader,
            strategy_loader_options,
        )

    if strategy_id is None:
        strategy_id = cast(Optional[str], getattr(strategy_config, "strategy_id", None))
    if strategies_by_slot is None:
        strategies_by_slot = cast(
            Optional[
                Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
            ],
            getattr(strategy_config, "strategies_by_slot", None),
        )
    if strategy_max_order_value is None:
        strategy_max_order_value = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_max_order_value", None),
        )
    if strategy_max_order_size is None:
        strategy_max_order_size = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_max_order_size", None),
        )
    if strategy_max_position_size is None:
        strategy_max_position_size = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_max_position_size", None),
        )
    if strategy_max_daily_loss is None:
        strategy_max_daily_loss = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_max_daily_loss", None),
        )
    if strategy_max_drawdown is None:
        strategy_max_drawdown = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_max_drawdown", None),
        )
    if strategy_reduce_only_after_risk is None:
        strategy_reduce_only_after_risk = cast(
            Optional[Dict[str, bool]],
            getattr(strategy_config, "strategy_reduce_only_after_risk", None),
        )
    if strategy_risk_cooldown_bars is None:
        strategy_risk_cooldown_bars = cast(
            Optional[Dict[str, int]],
            getattr(strategy_config, "strategy_risk_cooldown_bars", None),
        )
    if strategy_priority is None:
        strategy_priority = cast(
            Optional[Dict[str, int]],
            getattr(strategy_config, "strategy_priority", None),
        )
    if strategy_risk_budget is None:
        strategy_risk_budget = cast(
            Optional[Dict[str, float]],
            getattr(strategy_config, "strategy_risk_budget", None),
        )
    if strategy_fill_policy is None:
        strategy_fill_policy = cast(
            Optional[Dict[str, FillMode]],
            getattr(strategy_config, "strategy_fill_policy", None),
        )
    if strategy_slippage is None:
        strategy_slippage = cast(
            Optional[Dict[str, SlippageInput]],
            getattr(strategy_config, "strategy_slippage", None),
        )
    if strategy_commission is None:
        strategy_commission = cast(
            Optional[Dict[str, CommissionPolicy]],
            getattr(strategy_config, "strategy_commission", None),
        )
    if portfolio_risk_budget is None:
        portfolio_risk_budget = cast(
            Optional[float],
            getattr(strategy_config, "portfolio_risk_budget", None),
        )
    if strategy_runtime_config is None:
        config_indicator_mode = getattr(strategy_config, "indicator_mode", None)
        if config_indicator_mode is not None:
            strategy_runtime_config = {"indicator_mode": config_indicator_mode}
    if strategy_source is None:
        strategy_source = cast(
            Optional[Union[str, bytes, os.PathLike[str]]],
            getattr(strategy_config, "strategy_source", None),
        )
    if strategy_loader is None:
        strategy_loader = cast(
            Optional[str],
            getattr(strategy_config, "strategy_loader", None),
        )
    if strategy_loader_options is None:
        strategy_loader_options = cast(
            Optional[Dict[str, Any]],
            getattr(strategy_config, "strategy_loader_options", None),
        )

    return (
        strategy_id,
        strategies_by_slot,
        strategy_max_order_value,
        strategy_max_order_size,
        strategy_max_position_size,
        strategy_max_daily_loss,
        strategy_max_drawdown,
        strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars,
        strategy_priority,
        strategy_risk_budget,
        strategy_fill_policy,
        strategy_slippage,
        strategy_commission,
        portfolio_risk_budget,
        strategy_runtime_config,
        strategy_source,
        strategy_loader,
        strategy_loader_options,
    )


def _validate_strategy_risk_inputs(
    *,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ],
    strategy_max_order_value: Optional[Dict[str, float]],
    strategy_max_order_size: Optional[Dict[str, float]],
    strategy_max_position_size: Optional[Dict[str, float]],
    strategy_max_daily_loss: Optional[Dict[str, float]],
    strategy_max_drawdown: Optional[Dict[str, float]],
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]],
    strategy_risk_cooldown_bars: Optional[Dict[str, int]],
    strategy_priority: Optional[Dict[str, int]],
    strategy_risk_budget: Optional[Dict[str, float]],
    portfolio_risk_budget: Optional[float],
    risk_budget_mode: str,
) -> Tuple[Optional[float], str]:
    if strategies_by_slot is not None and not isinstance(strategies_by_slot, dict):
        raise TypeError("strategies_by_slot must be a dict when provided")
    if strategy_max_order_value is not None and not isinstance(
        strategy_max_order_value, dict
    ):
        raise TypeError("strategy_max_order_value must be a dict when provided")
    if strategy_max_order_size is not None and not isinstance(
        strategy_max_order_size, dict
    ):
        raise TypeError("strategy_max_order_size must be a dict when provided")
    if strategy_max_position_size is not None and not isinstance(
        strategy_max_position_size, dict
    ):
        raise TypeError("strategy_max_position_size must be a dict when provided")
    if strategy_max_daily_loss is not None and not isinstance(
        strategy_max_daily_loss, dict
    ):
        raise TypeError("strategy_max_daily_loss must be a dict when provided")
    if strategy_max_drawdown is not None and not isinstance(
        strategy_max_drawdown, dict
    ):
        raise TypeError("strategy_max_drawdown must be a dict when provided")
    if strategy_reduce_only_after_risk is not None and not isinstance(
        strategy_reduce_only_after_risk, dict
    ):
        raise TypeError("strategy_reduce_only_after_risk must be a dict when provided")
    if strategy_risk_cooldown_bars is not None and not isinstance(
        strategy_risk_cooldown_bars, dict
    ):
        raise TypeError("strategy_risk_cooldown_bars must be a dict when provided")
    if strategy_priority is not None and not isinstance(strategy_priority, dict):
        raise TypeError("strategy_priority must be a dict when provided")
    if strategy_risk_budget is not None and not isinstance(strategy_risk_budget, dict):
        raise TypeError("strategy_risk_budget must be a dict when provided")
    if portfolio_risk_budget is not None:
        portfolio_risk_budget = float(portfolio_risk_budget)
        if not pd.notna(portfolio_risk_budget) or portfolio_risk_budget < 0.0:
            raise ValueError("portfolio_risk_budget must be >= 0")
    normalized_mode = str(risk_budget_mode).strip().lower()
    if normalized_mode not in {"order_notional", "trade_notional"}:
        raise ValueError(
            "risk_budget_mode must be 'order_notional' or 'trade_notional'"
        )
    return portfolio_risk_budget, normalized_mode


def _normalize_strategy_fill_policy_map(
    strategy_fill_policy: Optional[Dict[str, FillMode]],
    configured_slot_ids: Sequence[str],
    logger: logging.Logger,
) -> Optional[Dict[str, FillPolicy]]:
    if not strategy_fill_policy:
        return None
    if not isinstance(strategy_fill_policy, dict):
        raise TypeError("strategy_fill_policy must be a dict when provided")
    normalized: Dict[str, FillPolicy] = {}
    for strategy_key, raw_policy in strategy_fill_policy.items():
        strategy_key_str = str(strategy_key).strip()
        if not strategy_key_str:
            raise ValueError("strategy_fill_policy contains empty strategy id")
        if not isinstance(raw_policy, FillMode):
            raise TypeError(
                f"strategy_fill_policy[{strategy_key_str}] must be a FillMode "
                "(NextOpen(), CurrentClose(...), ...)"
            )
        price_basis, bar_offset, temporal = raw_policy._to_core()
        normalized[strategy_key_str] = {
            "price_basis": price_basis,
            "bar_offset": int(bar_offset),
            "temporal": temporal,
        }
    unknown_keys = sorted(set(normalized.keys()).difference(set(configured_slot_ids)))
    if unknown_keys:
        raise ValueError(
            "strategy_fill_policy contains unknown strategy id(s): "
            + ",".join(unknown_keys)
        )
    return normalized


def _normalize_strategy_slippage_map(
    strategy_slippage: Optional[Dict[str, SlippageInput]],
    configured_slot_ids: Sequence[str],
    logger: logging.Logger,
) -> Optional[Dict[str, SlippagePolicy]]:
    if not strategy_slippage:
        return None
    if not isinstance(strategy_slippage, dict):
        raise TypeError("strategy_slippage must be a dict when provided")
    normalized: Dict[str, SlippagePolicy] = {}
    for strategy_key, raw_slippage in strategy_slippage.items():
        strategy_key_str = str(strategy_key).strip()
        if not strategy_key_str:
            raise ValueError("strategy_slippage contains empty strategy id")
        normalized[strategy_key_str] = _normalize_slippage_policy(
            raw_slippage,
            logger=logger,
            scope=f"strategy_slippage[{strategy_key_str}]",
            resolve_ticks=False,
        )
    unknown_keys = sorted(set(normalized.keys()).difference(set(configured_slot_ids)))
    if unknown_keys:
        raise ValueError(
            "strategy_slippage contains unknown strategy id(s): "
            + ",".join(unknown_keys)
        )
    return normalized


def _normalize_strategy_commission_map(
    strategy_commission: Optional[Dict[str, CommissionPolicy]],
    configured_slot_ids: Sequence[str],
) -> Optional[Dict[str, CommissionPolicy]]:
    if not strategy_commission:
        return None
    if not isinstance(strategy_commission, dict):
        raise TypeError("strategy_commission must be a dict when provided")
    normalized: Dict[str, CommissionPolicy] = {}
    for strategy_key, raw_commission in strategy_commission.items():
        strategy_key_str = str(strategy_key).strip()
        if not strategy_key_str:
            raise ValueError("strategy_commission contains empty strategy id")
        if not isinstance(raw_commission, dict):
            raise TypeError(
                f"strategy_commission[{strategy_key_str}] must be a dict "
                "CommissionPolicy"
            )
        normalized_commission = _normalize_commission_policy(
            cast(Dict[str, Any], raw_commission),
            scope=f"strategy_commission[{strategy_key_str}]",
        )
        if normalized_commission is None:
            raise ValueError(
                f"strategy_commission[{strategy_key_str}] must not be empty"
            )
        normalized[strategy_key_str] = normalized_commission
    unknown_keys = sorted(set(normalized.keys()).difference(set(configured_slot_ids)))
    if unknown_keys:
        raise ValueError(
            "strategy_commission contains unknown strategy id(s): "
            + ",".join(unknown_keys)
        )
    return normalized


def _parse_asset_type_name(value: Any) -> Literal["futures", "stock", "fund", "option"]:
    if isinstance(value, AssetType):
        if value == AssetType.Futures:
            return "futures"
        if value == AssetType.Stock:
            return "stock"
        if value == AssetType.Fund:
            return "fund"
        if value == AssetType.Option:
            return "option"
        raise ValueError(f"Unsupported asset_type: {value}")
    if isinstance(value, str):
        v_lower = value.lower()
        if v_lower in {"future", "futures"}:
            return "futures"
        if v_lower == "stock":
            return "stock"
        if v_lower == "fund":
            return "fund"
        if v_lower == "option":
            return "option"
    raise ValueError(f"Unsupported asset_type: {value}")


def _normalize_expiry_date_yyyymmdd(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expiry_date does not support bool")
    if isinstance(value, int):
        yyyymmdd = value
    elif isinstance(value, pd.Timestamp):
        if pd.isna(value):
            raise ValueError("expiry_date timestamp is NaT")
        yyyymmdd = int(value.strftime("%Y%m%d"))
    elif isinstance(value, dt_module.datetime):
        yyyymmdd = int(value.date().strftime("%Y%m%d"))
    elif isinstance(value, dt_module.date):
        yyyymmdd = int(value.strftime("%Y%m%d"))
    elif isinstance(value, str):
        raise TypeError(
            "expiry_date no longer supports str, please use date/datetime/"
            "Timestamp/int(YYYYMMDD)"
        )
    else:
        raise TypeError(
            "expiry_date must be date/datetime/pandas.Timestamp/int(YYYYMMDD)"
        )
    text = str(yyyymmdd)
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"expiry_date must be YYYYMMDD, got: {value}")
    year = int(text[0:4])
    month = int(text[4:6])
    day = int(text[6:8])
    dt_module.date(year, month, day)
    return yyyymmdd


def _asset_type_to_upper_name(
    value: Union[str, AssetType],
) -> InstrumentAssetTypeName:
    parsed = _parse_asset_type_name(value)
    if parsed == "futures":
        return "FUTURES"
    if parsed == "fund":
        return "FUND"
    if parsed == "option":
        return "OPTION"
    return "STOCK"


def _default_tick_size_for_asset_type(value: Any) -> float:
    """按资产类型给出缺省最小变动价位.

    基金/ETF/可转债(FUND)是 0.001, 其余(含类型未知/无法解析时)是 0.01——
    与 Rust 侧 ``Instrument`` 的缺省规则保持一致(深交所交易规则 3.3.13 条)。
    这里对无法解析的输入做兜底而不抛错, 因为调用方(尤其是 prebuilt 兜底路径)
    可能传入形态不规范的值, 那种情况下退回股票缺省 0.01 比让快照构建整体失败更安全。
    """
    try:
        return 0.001 if _parse_asset_type_name(value) == "fund" else 0.01
    except (ValueError, TypeError):
        return 0.01


def _option_type_to_upper_name(value: Any) -> Optional[InstrumentOptionTypeName]:
    if value is None:
        return None
    text = str(value).upper()
    if "CALL" in text:
        return "CALL"
    if "PUT" in text:
        return "PUT"
    raise ValueError(f"Unsupported option_type: {value}")


def _option_margin_model_to_upper_name(
    value: Any,
) -> Optional[InstrumentOptionMarginModelName]:
    if value is None:
        return None
    text = str(value).upper()
    if "VOL" in text and "ADJUST" in text:
        return "US_BROKER_SINGLE_LEG_VOL_ADJUSTED"
    if "US" in text and "BROKER" in text:
        return "US_BROKER_SINGLE_LEG"
    if "CHINA" in text and "SINGLE" in text:
        return "CHINA_SINGLE_LEG"
    if "RATIO" in text:
        return "RATIO"
    raise ValueError(f"Unsupported option_margin_model: {value}")


def _settlement_type_to_upper_name(value: Any) -> Optional[InstrumentSettlementMode]:
    if value is None:
        return None
    text = str(value).upper()
    if "FORCE" in text and "CLOSE" in text:
        return "FORCE_CLOSE"
    if "SETTLEMENT_PRICE" in text:
        return "SETTLEMENT_PRICE"
    if "CASH" in text:
        return "CASH"
    raise ValueError(f"Unsupported settlement_type: {value}")


def _parse_trading_session(value: Any) -> Any:
    if isinstance(value, TradingSession):
        return value
    call_auction = getattr(
        TradingSession, "CallAuction", getattr(TradingSession, "Normal", None)
    )
    pre_open = getattr(
        TradingSession, "PreOpen", getattr(TradingSession, "PreMarket", None)
    )
    continuous = getattr(
        TradingSession, "Continuous", getattr(TradingSession, "Normal", None)
    )
    break_session = getattr(
        TradingSession, "Break", getattr(TradingSession, "Normal", None)
    )
    post_close = getattr(
        TradingSession, "PostClose", getattr(TradingSession, "PostMarket", None)
    )
    closed = getattr(
        TradingSession, "Closed", getattr(TradingSession, "PostMarket", None)
    )
    v_lower = str(value).strip().lower()
    mapping = {
        "call_auction": call_auction,
        "callauction": call_auction,
        "pre_open": pre_open,
        "preopen": pre_open,
        "continuous": continuous,
        "break": break_session,
        "post_close": post_close,
        "postclose": post_close,
        "closed": closed,
    }
    if v_lower in mapping and mapping[v_lower] is not None:
        return mapping[v_lower]
    raise ValueError(f"Unsupported trading session: {value}")


def _china_futures_session_template(
    profile: str,
) -> List[Tuple[str, str, str]]:
    normalized = str(profile).strip().upper()
    commodity_day_template: List[Tuple[str, str, str]] = [
        ("09:00", "10:15", "continuous"),
        ("10:15", "10:30", "break"),
        ("10:30", "11:30", "continuous"),
        ("11:30", "13:30", "break"),
        ("13:30", "15:00", "continuous"),
    ]
    cffex_stock_index_day_template: List[Tuple[str, str, str]] = [
        ("09:30", "11:30", "continuous"),
        ("11:30", "13:00", "break"),
        ("13:00", "15:00", "continuous"),
    ]
    cffex_bond_day_template: List[Tuple[str, str, str]] = [
        ("09:30", "11:30", "continuous"),
        ("11:30", "13:00", "break"),
        ("13:00", "15:15", "continuous"),
    ]
    if normalized in {"CN_FUTURES_DAY", "CN_FUTURES_COMMODITY_DAY"}:
        return commodity_day_template
    if normalized == "CN_FUTURES_CFFEX_STOCK_INDEX_DAY":
        return cffex_stock_index_day_template
    if normalized == "CN_FUTURES_CFFEX_BOND_DAY":
        return cffex_bond_day_template
    if normalized == "CN_FUTURES_NIGHT_23":
        return [("21:00", "23:00", "continuous")] + commodity_day_template
    if normalized == "CN_FUTURES_NIGHT_01":
        return [
            ("21:00", "23:59", "continuous"),
            ("00:00", "01:00", "continuous"),
        ] + commodity_day_template
    if normalized == "CN_FUTURES_NIGHT_0230":
        return [
            ("21:00", "23:59", "continuous"),
            ("00:00", "02:30", "continuous"),
        ] + commodity_day_template
    raise ValueError(f"Unsupported china futures session profile: {profile}")


def _is_data_feed_adapter(value: Any) -> bool:
    return hasattr(value, "load") and callable(getattr(value, "load"))


def _load_data_map_from_adapter(
    adapter: Any,
    symbols: List[str],
    start_time: Optional[Union[str, Any]],
    end_time: Optional[Union[str, Any]],
    timezone: Optional[str],
) -> Tuple[Dict[str, pd.DataFrame], set[str]]:
    """按 requested_symbols 逐标的调用 adapter.load 并汇总成 {symbol: DataFrame}.

    for 循环本身只会为 requested_symbols(即参数 `symbols`, 未传则回落为
    `["BENCHMARK"]`)里的标的发请求 —— 这一点没问题。但**返回值**不能同等
    信任: 调用方 run_backtest 会把这里 data_map 里出现的每个 key 无条件
    append 进它的 `symbols`, 而那正是随后设进 Rust 引擎 `set_symbol_whitelist`
    的同一个列表。若某次 adapter.load 违反契约、在 frame 里混入了未被请求
    的标的, 不做处理的话它会先污染 data_map、再污染 symbols、最终污染白名
    单本身 —— 白名单对它静默失效, 这正是本函数要防的失败模式。

    因此: 当调用方显式传了 symbols(判据与三段前置过滤同一个 ——
    `bool(symbols and "BENCHMARK" not in symbols)`)时, 响应里超出
    requested_symbols 的标的会被丢弃, 通过第二个返回值 `leaked_symbols`
    上报, 由调用方决定怎么告警/记录, 这里不做 IO。未显式传 symbols 时
    (默认 "BENCHMARK" 哨兵)维持原样"数据即订阅"语义, 不丢弃 —— 这与
    DataFrame/dict/List[Bar] 三段前置过滤在同一哨兵下的行为一致。
    """
    effective_timezone = timezone or DEFAULT_TIMEZONE
    request_start = (
        _parse_runtime_boundary_timestamp(start_time, effective_timezone)
        if start_time is not None
        else None
    )
    request_end = (
        _parse_runtime_boundary_timestamp(end_time, effective_timezone)
        if end_time is not None
        else None
    )
    requested_symbols = symbols or ["BENCHMARK"]
    reject_unrequested = bool(symbols and "BENCHMARK" not in symbols)
    requested_symbols_set = set(requested_symbols)
    data_map: Dict[str, pd.DataFrame] = {}
    leaked_symbols: set[str] = set()

    for sym in requested_symbols:
        frame = adapter.load(
            FeedSlice(
                symbol=str(sym),
                start_time=request_start,
                end_time=request_end,
                timezone=timezone,
            )
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("DataFeedAdapter.load must return pandas.DataFrame")
        if frame.empty:
            continue

        if "symbol" in frame.columns:
            grouped = frame.groupby(frame["symbol"].astype(str), sort=False)
            for grouped_symbol, grouped_frame in grouped:
                key = str(grouped_symbol)
                if reject_unrequested and key not in requested_symbols_set:
                    leaked_symbols.add(key)
                    continue
                data_map[key] = grouped_frame.copy()
        else:
            normalized = frame.copy()
            normalized["symbol"] = str(sym)
            data_map[str(sym)] = normalized

    return data_map, leaked_symbols


def _build_strategy_instance(
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None],
    strategy_kwargs: Dict[str, Any],
    strict_strategy_params: bool,
    logger: Any,
    initialize: Optional[Callable[[Any], None]],
    on_start: Optional[Callable[[Any], None]],
    on_resume: Optional[Callable[[Any], None]],
    on_train_signal: Optional[Callable[[Any], None]],
    on_stop: Optional[Callable[[Any], None]],
    on_tick: Optional[Callable[[Any, Any], None]],
    on_order: Optional[Callable[[Any, Any], None]],
    on_trade: Optional[Callable[[Any, Any], None]],
    on_reject: Optional[Callable[[Any, Any], None]],
    on_before_trading: Optional[Callable[[Any, Any, int], None]],
    on_after_trading: Optional[Callable[[Any, Any, int], None]],
    on_cross_section: Optional[Callable[[Any, Any, int], None]],
    on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]],
    on_error: Optional[Callable[[Any, Exception, str, Any], None]],
    on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]],
    on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]],
    on_timer: Optional[Callable[[Any, str], None]],
    context: Optional[Dict[str, Any]],
) -> Strategy:
    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        accepted_kwargs, unknown_keys = _split_strategy_kwargs(
            strategy, strategy_kwargs
        )
        if unknown_keys:
            message = unknown_param_message(
                unknown_keys=unknown_keys,
                declared_fields=sorted(_strategy_param_field_names(strategy)),
                strategy_label=f"{strategy.__module__}.{strategy.__name__}",
                init_signature_names=_strategy_own_init_arg_names(strategy),
            )
            if strict_strategy_params:
                raise TypeError(message)
            logger.warning("%s (已忽略这些参数, 策略将使用字段默认值)", message)
        try:
            return cast(Strategy, strategy(**accepted_kwargs))
        except (TypeError, ValidationError) as e:
            if strict_strategy_params:
                raise TypeError(
                    "Failed to instantiate strategy with provided parameters: "
                    f"{e}. Strategy={strategy.__module__}.{strategy.__name__}"
                ) from e
            logger.warning(
                f"Failed to instantiate strategy with provided parameters: {e}. "
                "Falling back to default constructor (no arguments)."
            )
            return cast(Strategy, strategy())
    if isinstance(strategy, Strategy):
        return strategy
    if callable(strategy):
        return FunctionalStrategy(
            initialize,
            cast(Callable[[Any, Bar], None], strategy),
            on_start=on_start,
            on_resume=on_resume,
            on_train_signal=on_train_signal,
            on_stop=on_stop,
            on_tick=on_tick,
            on_order=on_order,
            on_trade=on_trade,
            on_reject=on_reject,
            on_before_trading=on_before_trading,
            on_after_trading=on_after_trading,
            on_cross_section=on_cross_section,
            on_portfolio_update=on_portfolio_update,
            on_error=on_error,
            on_expiry=on_expiry,
            on_pre_open=on_pre_open,
            on_timer=on_timer,
            context=context,
        )
    if strategy is None:
        raise ValueError("Strategy must be provided.")
    raise ValueError("Invalid strategy type")


class FunctionalStrategy(Strategy):
    """内部策略包装器，用于支持函数式 API (Zipline 风格)."""

    # 本类在类体里无条件定义了全部回调转发方法，未提供对应函数时方法体是空转。
    # 该标记让 _strategy_overrides_callback 改按"用户是否真的提供了回调"判定，
    # 否则"回调未重写就跳过"的快路径会整体失效（详见 strategy_framework_hooks）。
    _is_functional_wrapper = True

    def __init__(
        self,
        initialize: Optional[Callable[[Any], None]],
        on_bar: Optional[Callable[[Any, Bar], None]],
        on_start: Optional[Callable[[Any], None]] = None,
        on_resume: Optional[Callable[[Any], None]] = None,
        on_train_signal: Optional[Callable[[Any], None]] = None,
        on_stop: Optional[Callable[[Any], None]] = None,
        on_tick: Optional[Callable[[Any, Any], None]] = None,
        on_order: Optional[Callable[[Any, Any], None]] = None,
        on_trade: Optional[Callable[[Any, Any], None]] = None,
        on_reject: Optional[Callable[[Any, Any], None]] = None,
        on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
        on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
        on_cross_section: Optional[Callable[[Any, Any, int], None]] = None,
        on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Any, Exception, str, Any], None]] = None,
        on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
        on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
        on_timer: Optional[Callable[[Any, str], None]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the FunctionalStrategy."""
        super().__init__()
        self._initialize = initialize
        self._on_bar_func = on_bar
        self._on_start_func = on_start
        self._on_resume_func = on_resume
        self._on_train_signal_func = on_train_signal
        self._on_stop_func = on_stop
        self._on_tick_func = on_tick
        self._on_order_func = on_order
        self._on_trade_func = on_trade
        self._on_reject_func = on_reject
        self._on_before_trading_func = on_before_trading
        self._on_after_trading_func = on_after_trading
        self._on_cross_section_func = on_cross_section
        self._on_portfolio_update_func = on_portfolio_update
        self._on_error_func = on_error
        self._on_expiry_func = on_expiry
        self._on_pre_open_func = on_pre_open
        self._on_timer_func = on_timer
        self._context = context or {}

        # 将 context 注入到 self 中，模拟 Zipline 的 context 对象
        # 用户可以通过 self.xxx 访问 context 属性
        for k, v in self._context.items():
            setattr(self, k, v)

        # 调用初始化函数
        if self._initialize is not None:
            self._initialize(self)

    def on_bar(self, bar: Bar) -> None:
        """Delegate on_bar event to the user-provided function."""
        if self._on_bar_func is not None:
            self._on_bar_func(self, bar)

    def on_start(self) -> None:
        """Delegate on_start event to the user-provided function."""
        if self._on_start_func is not None:
            self._on_start_func(self)

    def on_resume(self) -> None:
        """Delegate on_resume event to the user-provided function."""
        if self._on_resume_func is not None:
            self._on_resume_func(self)

    def on_train_signal(self, context: Any) -> None:
        """Delegate on_train_signal event to the user-provided function."""
        if self._on_train_signal_func is not None:
            self._on_train_signal_func(self)

    def on_stop(self) -> None:
        """Delegate on_stop event to the user-provided function."""
        if self._on_stop_func is not None:
            self._on_stop_func(self)

    def on_tick(self, tick: Any) -> None:
        """Delegate on_tick event to the user-provided function."""
        if self._on_tick_func is not None:
            self._on_tick_func(self, tick)

    def on_order(self, order: Any) -> None:
        """Delegate on_order event to the user-provided function."""
        if self._on_order_func is not None:
            self._on_order_func(self, order)

    def on_trade(self, trade: Any) -> None:
        """Delegate on_trade event to the user-provided function."""
        if self._on_trade_func is not None:
            self._on_trade_func(self, trade)

    def on_reject(self, order: Any) -> None:
        """Delegate on_reject event to the user-provided function."""
        if self._on_reject_func is not None:
            self._on_reject_func(self, order)

    def on_before_trading(self, trading_date: Any, timestamp: int) -> None:
        """Delegate on_before_trading event to the user-provided function."""
        if self._on_before_trading_func is not None:
            self._on_before_trading_func(self, trading_date, timestamp)

    def on_after_trading(self, trading_date: Any, timestamp: int) -> None:
        """Delegate on_after_trading event to the user-provided function."""
        if self._on_after_trading_func is not None:
            self._on_after_trading_func(self, trading_date, timestamp)

    def on_cross_section(self, trading_date: Any, timestamp: int) -> None:
        """Delegate on_cross_section to the user-provided function."""
        if self._on_cross_section_func is not None:
            self._on_cross_section_func(self, trading_date, timestamp)

    def on_portfolio_update(self, snapshot: Dict[str, Any]) -> None:
        """Delegate on_portfolio_update event to the user-provided function."""
        if self._on_portfolio_update_func is not None:
            self._on_portfolio_update_func(self, snapshot)

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """Delegate on_error event to the user-provided function."""
        if self._on_error_func is not None:
            self._on_error_func(self, error, source, payload)

    def on_expiry(self, event: Dict[str, Any]) -> None:
        """Delegate on_expiry event to the user-provided function."""
        if self._on_expiry_func is not None:
            self._on_expiry_func(self, event)

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        """Delegate on_pre_open event to the user-provided function."""
        if self._on_pre_open_func is not None:
            self._on_pre_open_func(self, event)

    def on_timer(self, payload: str) -> None:
        """Delegate on_timer event to the user-provided function."""
        if self._on_timer_func is not None:
            self._on_timer_func(self, payload)


def _coerce_strategy_runtime_config(
    value: Union[StrategyRuntimeConfig, Dict[str, Any]],
) -> StrategyRuntimeConfig:
    if isinstance(value, StrategyRuntimeConfig):
        return StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=value.enable_precise_day_boundary_hooks,
            portfolio_update_eps=value.portfolio_update_eps,
            error_mode=value.error_mode,
            re_raise_on_error=value.re_raise_on_error,
            indicator_mode=value.indicator_mode,
        )
    if isinstance(value, dict):
        unknown_fields = sorted(set(value.keys()) - _RUNTIME_CONFIG_FIELDS)
        if unknown_fields:
            allowed = ", ".join(sorted(_RUNTIME_CONFIG_FIELDS))
            unknown = ", ".join(unknown_fields)
            raise ValueError(
                "strategy_runtime_config contains unknown fields: "
                f"{unknown}. Allowed fields: {allowed}"
            )
        try:
            return StrategyRuntimeConfig(**value)
        except ValueError as exc:
            raise ValueError(f"invalid strategy_runtime_config: {exc}") from None
    raise TypeError(
        "strategy_runtime_config must be StrategyRuntimeConfig or Dict[str, Any]"
    )


def _runtime_config_conflicts(
    current: StrategyRuntimeConfig, incoming: StrategyRuntimeConfig
) -> List[str]:
    conflicts: List[str] = []
    for key in sorted(_RUNTIME_CONFIG_FIELDS):
        before = getattr(current, key)
        after = getattr(incoming, key)
        if before != after:
            conflicts.append(f"{key}: {before} -> {after}")
    return conflicts


def _should_prepare_precomputed_indicators(strategy_instance: Strategy) -> bool:
    return str(strategy_instance.indicator_mode).strip().lower() == "precompute"


def _resolve_incremental_indicator_warmup_depth(strategy_instance: Strategy) -> int:
    """提取增量指标注册中声明的最大历史预热深度."""
    if str(strategy_instance.indicator_mode).strip().lower() != "incremental":
        return 0
    registrations = getattr(strategy_instance, "_incremental_indicators", {}) or {}
    max_warmup = 0
    for item in registrations.values():
        warmup_bars = 0
        if hasattr(item, "warmup_bars"):
            warmup_bars = int(getattr(item, "warmup_bars", 0) or 0)
        elif isinstance(item, dict):
            warmup_bars = int(item.get("warmup_bars", 0) or 0)
        if warmup_bars > max_warmup:
            max_warmup = warmup_bars
    return max_warmup


def _resolve_runtime_warmup_depth(
    strategy_instance: Strategy,
    history_depth: int,
    warmup_period: int,
    logger: Any,
) -> tuple[int, int]:
    """Resolve final warmup and effective history depth after strategy setup."""
    strategy_warmup = getattr(strategy_instance, "warmup_period", 0)

    inferred_warmup = 0
    try:
        inferred_warmup = infer_warmup_period(type(strategy_instance))
        if inferred_warmup > 0:
            logger.info(f"Auto-inferred warmup period: {inferred_warmup}")
    except Exception as exc:
        logger.debug(f"Failed to infer warmup period: {exc}")

    final_warmup = max(strategy_warmup, inferred_warmup, warmup_period)
    strategy_instance.warmup_period = final_warmup
    effective_depth = max(final_warmup, history_depth)
    return final_warmup, effective_depth


def _to_active_start_time_ns(
    start_time: Optional[Union[str, Any]],
    timezone: str,
) -> Optional[int]:
    """Normalize an optional active start time to UTC nanoseconds."""
    if start_time is None:
        return None
    return _boundary_timestamp_to_utc_ns(start_time, timezone)


def _apply_strategy_runtime_config(
    strategy_instance: Strategy,
    incoming: Union[StrategyRuntimeConfig, Dict[str, Any]],
    runtime_config_override: bool,
    logger: Any,
) -> None:
    cfg = _coerce_strategy_runtime_config(incoming)
    current = strategy_instance.runtime_config
    conflicts = _runtime_config_conflicts(current, cfg)
    if conflicts:
        conflict_text = "; ".join(conflicts)
        warning_key = f"{runtime_config_override}|{conflict_text}"
        warned_keys = getattr(strategy_instance, "_runtime_config_warning_keys", None)
        if not isinstance(warned_keys, set):
            warned_keys = set()
            setattr(strategy_instance, "_runtime_config_warning_keys", warned_keys)
        should_log = warning_key not in warned_keys
        warned_keys.add(warning_key)
        if runtime_config_override:
            if should_log:
                logger.warning(
                    "strategy_runtime_config overrides strategy runtime_config: "
                    f"{conflict_text}"
                )
        else:
            if should_log:
                logger.warning(
                    "strategy_runtime_config is ignored because "
                    f"runtime_config_override=False: {conflict_text}"
                )
            return
    strategy_instance.runtime_config = cfg


def _coerce_analyzer_plugins(
    analyzer_plugins: Optional[Sequence[AnalyzerPlugin]],
) -> List[AnalyzerPlugin]:
    if analyzer_plugins is None:
        return []
    if not isinstance(analyzer_plugins, (list, tuple)):
        raise TypeError("analyzer_plugins must be a list/tuple of analyzer plugins")
    normalized: List[AnalyzerPlugin] = []
    for plugin in analyzer_plugins:
        if not hasattr(plugin, "name"):
            raise TypeError("analyzer plugin must have 'name' attribute")
        if not hasattr(plugin, "on_start") or not callable(getattr(plugin, "on_start")):
            raise TypeError("analyzer plugin must implement on_start(context)")
        if not hasattr(plugin, "on_bar") or not callable(getattr(plugin, "on_bar")):
            raise TypeError("analyzer plugin must implement on_bar(context)")
        if not hasattr(plugin, "on_trade") or not callable(getattr(plugin, "on_trade")):
            raise TypeError("analyzer plugin must implement on_trade(context)")
        if not hasattr(plugin, "on_finish") or not callable(
            getattr(plugin, "on_finish")
        ):
            raise TypeError("analyzer plugin must implement on_finish(context)")
        normalized.append(plugin)
    return normalized


def _warn_unmatched_instrument_config(
    config: Any, data_map_for_indicators: Dict[str, Any], logger: logging.Logger
) -> None:
    """``instruments_config`` 配了、但数据里没有这个 symbol 时点名告警.

    合约快照与撮合层的合约表都按**数据里实际出现的** symbol 建，配错 symbol 的
    条目会被完全静默丢弃，该标的回退到默认合约参数。最典型的撞法是数据用去后缀
    写法（``600487``）而配置写带后缀（``600487.SH``）：实测 ``lot_size`` 从配置的
    100 变回默认 **1.0**（``tick_size`` 恰好与股票默认值相同，唯一露马脚的就是
    ``lot_size``），A 股下单数量随之不再整百，而用户以为配置已生效。

    与运行前那三个集合（``filtered_out_symbols`` 主动排除 / ``symbol_data_missing``
    白名单里有但数据没有 / adapter 泄漏）都不相交：这里比的是「配置 vs 数据」，
    与 ``symbols`` 白名单无关，故不受 ``symbols_explicit`` 约束。

    :param config: 回测配置（``None`` 或无 ``instruments_config`` 时直接返回）。
    :param data_map_for_indicators: 数据里实际出现的标的映射；为空说明走的是
        ``DataFeed`` 对象输入（只写不读、Python 无从枚举），此时跳过检查。
    :param logger: 调用方的 logger（与本模块其余 ``_warn_*`` helper 一致，
        本模块没有模块级 logger）。
    """
    if not data_map_for_indicators or not config or not config.instruments_config:
        return
    configured: set[str]
    if isinstance(config.instruments_config, list):
        configured = {
            text
            for conf_item in config.instruments_config
            if (text := str(getattr(conf_item, "symbol", "") or "").strip())
        }
    elif isinstance(config.instruments_config, dict):
        configured = {
            text for key in config.instruments_config if (text := str(key).strip())
        }
    else:
        return
    unmatched = sorted(configured - set(data_map_for_indicators.keys()))
    if not unmatched:
        return
    logger.warning(
        "instruments_config 配置了 %d 个数据里不存在的标的: %s —— 这些配置会被静默"
        "丢弃, 对应标的按**默认**合约参数撮合(如 lot_size 回退为 1, A 股下单数量将"
        "不再整百)。请核对配置里的 symbol 与数据里的写法是否一致(常见分歧: 带 "
        ".SH/.SZ 后缀 vs 去后缀的纯数字)。数据里实际出现的标的: %s",
        len(unmatched),
        ", ".join(unmatched),
        ", ".join(sorted(data_map_for_indicators.keys())[:20]),
    )


def _warn_if_suspicious_global_slippage(
    slippage: float, logger: logging.Logger
) -> None:
    """Warn when a global slippage value looks like a fixed price delta."""
    if slippage < 0.05:
        return
    logger.warning(
        "Global slippage=%s uses percent semantics in AKQuant. "
        "For example, 0.2 means 20%% slippage, not a fixed 0.2 price delta. "
        "If you intended a fixed offset such as 0.2 index points, set "
        "order-level slippage={'type': 'fixed', 'value': 0.2} instead.",
        slippage,
    )


def _warn_deprecated_float_slippage(
    slippage: Union[int, float], logger: logging.Logger, scope: str
) -> None:
    _ = slippage
    warnings.warn(
        f"{scope} slippage passed as a bare number is deprecated in AKQuant. "
        "Use an explicit policy such as "
        "slippage={'type': 'percent', 'value': 0.0002} or "
        "slippage={'type': 'fixed', 'value': 0.2}.",
        DeprecationWarning,
        stacklevel=3,
    )
    logger.warning(
        "%s slippage passed as a bare number is deprecated in AKQuant. "
        "Please use an explicit policy such as "
        "slippage={'type': 'percent', 'value': 0.0002} or "
        "slippage={'type': 'fixed', 'value': 0.2}.",
        scope,
    )


def _normalize_slippage_policy(
    slippage: SlippageInput,
    *,
    instrument_snapshots: Optional[Dict[str, InstrumentSnapshot]] = None,
    logger: Optional[logging.Logger] = None,
    scope: str = "Global",
    allow_float: bool = True,
    resolve_ticks: bool = True,
) -> SlippagePolicy:
    if slippage is None:
        return {"type": "zero", "value": 0.0}
    if isinstance(slippage, (int, float)):
        if not allow_float:
            raise TypeError(
                f"{scope} slippage must be a dict policy when provided; "
                "bare numeric slippage is no longer accepted here"
            )
        numeric_value = float(slippage)
        if numeric_value < 0:
            raise ValueError(f"{scope} slippage.value must be >= 0")
        if logger is not None and numeric_value != 0.0:
            _warn_deprecated_float_slippage(numeric_value, logger, scope)
            _warn_if_suspicious_global_slippage(numeric_value, logger)
        return {"type": "percent", "value": numeric_value}
    if not isinstance(slippage, dict):
        raise TypeError(f"{scope} slippage must be a dict when provided")
    raw_type = str(slippage.get("type", "percent")).strip().lower()
    raw_value = slippage.get("value", 0.0)
    if raw_type == "zero":
        return {"type": "zero", "value": 0.0}
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"{scope} slippage.value must be a number >= 0") from None
    if value < 0:
        raise ValueError(f"{scope} slippage.value must be >= 0")
    if raw_type in {"percent", "fixed"}:
        return {"type": raw_type, "value": value}
    if raw_type == "ticks":
        if not resolve_ticks:
            return {"type": "ticks", "value": value}
        if not instrument_snapshots:
            raise ValueError(
                f"{scope} slippage.type='ticks' requires instrument configuration"
            )
        tick_sizes = {
            float(snapshot.tick_size)
            for snapshot in instrument_snapshots.values()
            if snapshot.tick_size is not None
        }
        if not tick_sizes:
            raise ValueError(
                f"{scope} slippage.type='ticks' requires at least one tick_size"
            )
        if len(tick_sizes) != 1:
            raise ValueError(
                f"{scope} slippage.type='ticks' requires a single shared tick_size "
                "across all effective instruments"
            )
        tick_size = next(iter(tick_sizes))
        return {"type": "fixed", "value": value * tick_size}
    raise ValueError(
        f"{scope} slippage.type must be one of: percent, fixed, ticks, zero"
    )


def run_backtest(
    data: Optional[BacktestDataInput] = None,
    freq: Optional[str] = None,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = None,
    strategy_source: Optional[Union[str, bytes, os.PathLike[str]]] = None,
    strategy_loader: Optional[str] = None,
    strategy_loader_options: Optional[Dict[str, Any]] = None,
    symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]] = None,
    initial_cash: Optional[float] = None,
    commission_policy: Optional[CommissionPolicy] = None,
    commission_rate: Optional[float] = None,
    stamp_tax_rate: Optional[float] = None,
    transfer_fee_rate: Optional[float] = None,
    min_commission: Optional[float] = None,
    slippage: SlippageInput = None,
    volume_limit_pct: Optional[float] = None,
    timezone: Optional[str] = None,
    t_plus_one: bool = False,
    initialize: Optional[Callable[[Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
    on_resume: Optional[Callable[[Any], None]] = None,
    on_train_signal: Optional[Callable[[Any], None]] = None,
    on_stop: Optional[Callable[[Any], None]] = None,
    on_tick: Optional[Callable[[Any, Any], None]] = None,
    on_order: Optional[Callable[[Any, Any], None]] = None,
    on_trade: Optional[Callable[[Any, Any], None]] = None,
    on_reject: Optional[Callable[[Any, Any], None]] = None,
    on_before_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_after_trading: Optional[Callable[[Any, Any, int], None]] = None,
    on_cross_section: Optional[Callable[[Any, Any, int], None]] = None,
    on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_error: Optional[Callable[[Any, Exception, str, Any], None]] = None,
    on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    on_timer: Optional[Callable[[Any, str], None]] = None,
    context: Optional[Dict[str, Any]] = None,
    history_depth: Optional[int] = None,
    warmup_period: int = 0,
    lot_size: Union[int, Dict[str, int], None] = None,
    show_progress: Optional[bool] = None,
    start_time: Optional[Union[str, Any]] = None,
    end_time: Optional[Union[str, Any]] = None,
    catalog_path: Optional[str] = None,
    config: Optional[BacktestConfig] = None,
    custom_matchers: Optional[Dict[AssetType, Any]] = None,
    risk_config: Optional[Union[Dict[str, Any], RiskConfig]] = None,
    strategy_runtime_config: Optional[
        Union[StrategyRuntimeConfig, Dict[str, Any]]
    ] = None,
    runtime_config_override: bool = True,
    strategy_id: Optional[str] = None,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ] = None,
    strategy_max_order_value: Optional[Dict[str, float]] = None,
    strategy_max_order_size: Optional[Dict[str, float]] = None,
    strategy_max_position_size: Optional[Dict[str, float]] = None,
    strategy_max_daily_loss: Optional[Dict[str, float]] = None,
    strategy_max_drawdown: Optional[Dict[str, float]] = None,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None,
    strategy_priority: Optional[Dict[str, int]] = None,
    strategy_risk_budget: Optional[Dict[str, float]] = None,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = None,
    strategy_slippage: Optional[Dict[str, SlippageInput]] = None,
    strategy_commission: Optional[Dict[str, CommissionPolicy]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    analyzer_plugins: Optional[Sequence[AnalyzerPlugin]] = None,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    indicator_recorder: Optional[IndicatorSink] = None,
    broker_profile: Optional[str] = None,
    fill_policy: Optional[Union[FillMode, FillPolicy]] = None,
    strict_strategy_params: bool = True,
    **kwargs: Any,
) -> BacktestResult:
    """
    简化版回测入口函数.

    :param data: 回测数据，可以是 Pandas DataFrame、``Bar`` 列表、``Tick`` 列表，
        或 ``Bar``/``Tick`` 混合列表。纯 tick 输入下 ``on_bar`` 不触发，
        ``get_history`` 返回成交价序列（tick 以 ``open=high=low=close=price``
        退化写入）；需要真实 OHLC 语义请配合 ``freq`` 聚合。
    :param custom_matchers: 自定义撮合器字典 {AssetType: MatcherInstance}
                 用于覆盖特定资产类型的默认撮合逻辑。
                 例如：传入一个实现了自定义成交规则的 Rust 撮合器实例，
                 或者用于测试目的的 Mock 撮合器。
                 默认情况下，引擎会根据 AssetType 自动选择内置的撮合器
                 (如 StockMatcher, FuturesMatcher 等)。
    :param risk_config: 风控配置，支持字典 (e.g., {"max_position_pct": 0.1})
                        或 RiskConfig 对象。如果同时提供了 config.strategy_config.risk，
                        此参数将覆盖其中的同名字段。
    :param strategy: 策略类、策略实例或 on_bar 回调函数
    :param strategy_source: 策略源码输入（路径字符串 / bytes / PathLike），
                            当 strategy=None 时用于动态加载策略
    :param strategy_loader: 策略加载器名称，默认 "python_plain"，
                            可选 "encrypted_external" 或用户注册加载器
    :param strategy_loader_options: 传给策略加载器的参数字典（可选），
                                    如 {"strategy_attr": "MyStrategy"} 或
                                    {"decrypt_and_load": callable}
    :param symbols: 标的代码或代码列表
    :param initial_cash: 初始资金 (默认 100,000.0)
    :param commission_policy: 佣金策略 (可选)，格式如
                              {"type": "percent", "value": 0.0003}、
                              {"type": "fixed", "value": 3.0}、
                              {"type": "per_unit", "value": 0.01}。
                              显式传入时优先级高于 commission_rate。
    :param commission_rate: 佣金率 (默认 0.0)
    :param stamp_tax_rate: 印花税率 (仅卖出, 默认 0.0)
    :param transfer_fee_rate: 过户费率 (默认 0.0)
    :param min_commission: 最低佣金 (默认 0.0)
    :param slippage: 滑点策略。推荐显式传 dict，如
                     {"type": "percent", "value": 0.0002}、
                     {"type": "fixed", "value": 0.2}、
                     {"type": "ticks", "value": 1}。
                     裸 float 仍兼容，但已不推荐，且按 percent 语义解析。
    :param volume_limit_pct: 成交量限制比例 (默认 0.25)
    :param fill_policy: 统一成交语义配置（可选），传入 ``FillMode`` 对象:
        ``NextOpen()`` 下一根开盘价、``NextClose()`` 下一根收盘价、
        ``CurrentClose()`` 当根收盘价、``NextAverage()`` 下一根 OHLC4 均价、
        ``NextHighLowMid()`` 下一根 HL2 中位价。
        ``CurrentClose`` 可选 ``timer_fill_timing="immediate"|"deferred"`` 控制
        定时器触发下单是否延迟到下一事件成交。
        旧的 dict 形式与 ``make_fill_policy(...)`` 已移除，
        传入 dict 会抛出 ``TypeError``。
    :param legacy_execution_policy_compat: 已移除，不再支持。
    :param strict_strategy_params: 是否严格校验策略构造参数。True 时若参数不匹配将抛错；
                                   False 时保持兼容行为（忽略未知参数并在失败时
                                   回退无参构造）。
    :param timezone: 时区名称 (默认 "Asia/Shanghai")
    :param t_plus_one: 是否启用 T+1 交易规则 (默认 False)
    :param initialize: 初始化回调函数 (仅当 strategy 为函数时使用)
    :param on_start: 启动回调函数 (仅当 strategy 为函数时使用)
    :param on_stop: 停止回调函数 (仅当 strategy 为函数时使用)
    :param on_tick: Tick 回调函数 (仅当 strategy 为函数时使用)
    :param on_order: 订单回调函数 (仅当 strategy 为函数时使用)
    :param on_trade: 成交回调函数 (仅当 strategy 为函数时使用)
    :param on_timer: 定时器回调函数 (仅当 strategy 为函数时使用)
    :param context: 初始上下文数据 (仅当 strategy 为函数时使用)
    :param history_depth: 自动维护历史数据的长度 (0 表示禁用)
    :param warmup_period: 策略预热期 (等同于 history_depth，取最大值)
    :param lot_size: 最小交易单位。如果是 int，则应用于所有标的；
                     如果是 Dict[str, int]，则按代码匹配；如果不传(None)，默认为 1。
    :param show_progress: 是否显示进度条 (默认 True)
    :param start_time: 回测开始时间 (e.g., "2020-01-01 09:30"). 优先级高于
                       config.start_time。若传入 naive 时间，将按当前 `timezone`
                       解释，再转换为 UTC 参与过滤。
    :param end_time: 回测结束时间 (e.g., "2020-12-31 15:00"). 优先级高于
                     config.end_time。若传入 naive 时间，将按当前 `timezone`
                     解释，再转换为 UTC 参与过滤。
    :param catalog_path: 当 data 未显式传入时，按该目录加载 ParquetDataCatalog 数据。
                         不传则使用 ParquetDataCatalog 默认目录。
    :param config: BacktestConfig 配置对象 (可选)
    :param strategy_runtime_config: 策略运行时配置对象或字典 (可选)
    :param runtime_config_override: 是否覆盖策略实例内已有 runtime_config (默认 True)
    :param strategy_id: 策略归属 ID（预留多策略归因字段，默认 "_default"）
    :param strategies_by_slot: 可选 slot->策略映射，用于启用多策略 slot 迭代执行
    :param strategy_max_order_value: 可选策略级单笔下单金额上限映射
                                    （strategy_id->max_value）
    :param strategy_max_order_size: 可选策略级单笔下单数量上限映射
                                   （strategy_id->max_size）
    :param strategy_max_position_size: 可选策略级净持仓数量上限映射
                                       （strategy_id->max_abs_position）
    :param strategy_max_daily_loss: 可选策略级日内亏损上限映射
                                    （strategy_id->max_daily_loss）
    :param strategy_max_drawdown: 可选策略级回撤上限映射
                                  （strategy_id->max_drawdown）
    :param strategy_reduce_only_after_risk: 可选策略级风控触发后仅平仓开关映射
                                            （strategy_id->bool）
    :param strategy_risk_cooldown_bars: 可选策略级风控触发后冷却 bars 映射
                                        （strategy_id->cooldown_bars）
    :param strategy_priority: 可选策略级执行优先级映射（strategy_id->priority）
    :param strategy_risk_budget: 可选策略级累计风险预算映射（strategy_id->budget）
    :param strategy_fill_policy: 可选策略级默认成交策略映射
                                 （strategy_id->fill_policy）。
                                 下单优先级：订单级 > 策略级 > 运行级。
    :param strategy_slippage: 可选策略级默认滑点映射
                              （strategy_id->slippage）。
                              下单优先级：订单级 > 策略级 > 引擎级。
    :param strategy_commission: 可选策略级默认佣金映射
                                （strategy_id->commission）。
                                下单优先级：订单级 > 策略级 > 引擎级。
    :param portfolio_risk_budget: 可选账户级累计风险预算上限
    :param risk_budget_mode: 风险预算口径，支持 order_notional/trade_notional
    :param risk_budget_reset_daily: 风险预算是否按交易日重置
    :param analyzer_plugins: Analyzer 插件列表，
                             接收 on_start/on_bar/on_trade/on_finish 生命周期事件
    :param on_event: 可选流式事件回调。阶段 5 后 `run_backtest` 始终走统一事件内核；
                     不传时内部使用 no-op 回调并保持返回语义不变。
    :param broker_profile: 可选 broker 参数模板名称，
                           如 "cn_stock_miniqmt" / "cn_stock_t1_low_fee" /
                           "cn_stock_sim_high_slippage"，
                           用于快速注入一组回测参数默认值。
    故障速查可参考 docs/zh/advanced/runtime_config.md，
    英文文档参考 docs/en/advanced/runtime_config.md
    :param instruments_config: 标的配置列表或字典 (可选)
    :return: 回测结果 Result 对象

    配置优先级说明 (Parameter Priority):
    ----------------------------------
    本函数参数采用以下优先级顺序解析（由高到低）：

    1. **Explicit Arguments (显式参数)**:
       直接传递给 `run_backtest` 的参数优先级最高。
       例如: `run_backtest(..., start_time="2022-01-01")` 会覆盖 Config 中的设置。

    2. **Configuration Objects (配置对象)**:
       如果显式参数为 `None`，则尝试从 `config` (`BacktestConfig`) 及其子配置
       (`StrategyConfig`) 中读取。
       例如: `config.start_time` 或 `config.strategy_config.initial_cash`。

    3. **Default Values (默认值)**:
       如果上述两者都未提供，则使用系统默认值。
       例如: `initial_cash` 默认为 1,000,000。
    """
    if "_engine_mode" in kwargs:
        raise TypeError("_engine_mode is no longer supported")
    _raise_if_legacy_execution_policy_used(
        legacy_mode_used="execution_mode" in kwargs,
        legacy_timer_used="timer_execution_policy" in kwargs,
        api_name="run_backtest",
    )
    strategy_config = config.strategy_config if config is not None else None
    (
        strategy_id,
        strategies_by_slot,
        strategy_max_order_value,
        strategy_max_order_size,
        strategy_max_position_size,
        strategy_max_daily_loss,
        strategy_max_drawdown,
        strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars,
        strategy_priority,
        strategy_risk_budget,
        strategy_fill_policy,
        strategy_slippage,
        strategy_commission,
        portfolio_risk_budget,
        strategy_runtime_config,
        strategy_source,
        strategy_loader,
        strategy_loader_options,
    ) = _apply_strategy_config_overrides(
        strategy_config=strategy_config,
        strategy_id=strategy_id,
        strategies_by_slot=strategies_by_slot,
        strategy_max_order_value=strategy_max_order_value,
        strategy_max_order_size=strategy_max_order_size,
        strategy_max_position_size=strategy_max_position_size,
        strategy_max_daily_loss=strategy_max_daily_loss,
        strategy_max_drawdown=strategy_max_drawdown,
        strategy_reduce_only_after_risk=strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars=strategy_risk_cooldown_bars,
        strategy_priority=strategy_priority,
        strategy_risk_budget=strategy_risk_budget,
        strategy_fill_policy=strategy_fill_policy,
        strategy_slippage=strategy_slippage,
        strategy_commission=strategy_commission,
        portfolio_risk_budget=portfolio_risk_budget,
        strategy_runtime_config=strategy_runtime_config,
        strategy_source=strategy_source,
        strategy_loader=strategy_loader,
        strategy_loader_options=strategy_loader_options,
    )
    broker_profile_values = _resolve_broker_profile(broker_profile)
    if broker_profile_values:
        if initial_cash is None:
            initial_cash = cast(
                Optional[float], broker_profile_values.get("initial_cash")
            )
        if commission_rate is None:
            commission_rate = cast(
                Optional[float], broker_profile_values.get("commission_rate")
            )
        if slippage is None:
            slippage = cast(SlippageInput, broker_profile_values.get("slippage"))
        if volume_limit_pct is None:
            volume_limit_pct = cast(
                Optional[float], broker_profile_values.get("volume_limit_pct")
            )
        if lot_size is None:
            lot_size = cast(
                Optional[Union[int, Dict[str, int]]],
                broker_profile_values.get("lot_size"),
            )
        if stamp_tax_rate is None:
            stamp_tax_rate = cast(
                Optional[float], broker_profile_values.get("stamp_tax_rate")
            )
        if transfer_fee_rate is None:
            transfer_fee_rate = cast(
                Optional[float], broker_profile_values.get("transfer_fee_rate")
            )
        if min_commission is None:
            min_commission = cast(
                Optional[float], broker_profile_values.get("min_commission")
            )
    portfolio_risk_budget, risk_budget_mode = _validate_strategy_risk_inputs(
        strategies_by_slot=strategies_by_slot,
        strategy_max_order_value=strategy_max_order_value,
        strategy_max_order_size=strategy_max_order_size,
        strategy_max_position_size=strategy_max_position_size,
        strategy_max_daily_loss=strategy_max_daily_loss,
        strategy_max_drawdown=strategy_max_drawdown,
        strategy_reduce_only_after_risk=strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars=strategy_risk_cooldown_bars,
        strategy_priority=strategy_priority,
        strategy_risk_budget=strategy_risk_budget,
        portfolio_risk_budget=portfolio_risk_budget,
        risk_budget_mode=risk_budget_mode,
    )
    risk_budget_reset_daily = bool(risk_budget_reset_daily)
    effective_strategy_id = strategy_id or "_default"
    indicator_stream_requested = (
        on_event is not None or kwargs.get("_stream_on_event") is not None
    )
    prepared_stream_runtime = _prepare_stream_runtime(
        on_event=on_event,
        kwargs=kwargs,
        owner_strategy_id=effective_strategy_id,
        patch_owner_strategy_id=True,
    )
    stream_on_event = prepared_stream_runtime.stream_on_event
    indicator_stream_emitter = (
        prepared_stream_runtime.indicator_stream_emitter
        if indicator_stream_requested
        else None
    )
    event_stats_snapshot = prepared_stream_runtime.event_stats_snapshot
    stream_progress_interval = prepared_stream_runtime.stream_progress_interval
    stream_equity_interval = prepared_stream_runtime.stream_equity_interval
    stream_batch_size = prepared_stream_runtime.stream_batch_size
    stream_max_buffer = prepared_stream_runtime.stream_max_buffer
    stream_error_mode = prepared_stream_runtime.stream_error_mode
    stream_mode = prepared_stream_runtime.stream_mode

    # 0. 设置默认值 (如果未传入且未在 Config 中设置)
    # 优先级: 参数 > Config > 默认值

    # Defaults
    DEFAULT_INITIAL_CASH = float(getattr(StrategyConfig, "initial_cash", 100000.0))
    DEFAULT_COMMISSION_RATE = 0.0
    DEFAULT_SHOW_PROGRESS = True
    DEFAULT_HISTORY_DEPTH = 0

    # Resolve Initial Cash
    if initial_cash is None:
        if config and config.strategy_config:
            initial_cash = config.strategy_config.initial_cash
        else:
            initial_cash = DEFAULT_INITIAL_CASH

    (
        resolved_commission_policy,
        stamp_tax_rate,
        transfer_fee_rate,
        min_commission,
    ) = _resolve_stock_fee_rules(
        commission_policy=commission_policy,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
        min_commission=min_commission,
        broker_profile_values=broker_profile_values,
        strategy_config=strategy_config,
    )
    commission_policy = resolved_commission_policy
    commission_rate = (
        float(commission_policy["value"])
        if commission_policy["type"] == "percent"
        else DEFAULT_COMMISSION_RATE
    )

    # Resolve Slippage & Volume Limit
    if slippage is None:
        if config and config.strategy_config:
            slippage = config.strategy_config.slippage
        else:
            slippage = 0.0

    if volume_limit_pct is None:
        if config and config.strategy_config:
            volume_limit_pct = config.strategy_config.volume_limit_pct
        else:
            volume_limit_pct = 0.25

    # Resolve Timezone
    if timezone is None:
        if config and config.timezone:
            timezone = config.timezone
        else:
            timezone = DEFAULT_TIMEZONE

    # Resolve Show Progress
    if show_progress is None:
        if config and config.show_progress is not None:
            show_progress = config.show_progress
        else:
            show_progress = DEFAULT_SHOW_PROGRESS

    # Resolve History Depth
    if history_depth is None:
        if config and config.history_depth is not None:
            history_depth = config.history_depth
        else:
            history_depth = DEFAULT_HISTORY_DEPTH

    # 1. 确保日志已初始化
    logger = get_logger("backtest")
    if not has_configured_handler(logger.name, namespace_only=True):
        register_logger(console=True, level="INFO")
        logger = get_logger("backtest")
    normalized_analyzers = _coerce_analyzer_plugins(analyzer_plugins)

    # 1.2 检查 PyCharm 环境下的进度条可见性
    if show_progress and "PYCHARM_HOSTED" in os.environ:
        # PyCharm Console 或 Run 窗口未开启模拟终端时，isatty 通常为 False
        if not sys.stderr.isatty():
            logger.warning(
                "Progress bar might be invisible in PyCharm. "
                "Solution: Enable 'Emulate terminal in output console' "
                "in Run Configuration."
            )

    # 1.5 处理 Config 覆盖 (剩余部分)
    # Resolve effective start/end time for filtering
    # Priority: explicit argument > config

    if start_time is None:
        if config and config.start_time:
            start_time = config.start_time

    if end_time is None:
        if config and config.end_time:
            end_time = config.end_time

    # Handle strategy_params explicitly
    if "strategy_params" in kwargs:
        s_params = kwargs.pop("strategy_params")
        if isinstance(s_params, dict):
            kwargs.update(s_params)
    if "strategy_runtime_config" in kwargs:
        kwargs_runtime_config = kwargs.pop("strategy_runtime_config")
        if strategy_runtime_config is None:
            strategy_runtime_config = kwargs_runtime_config
    symbols, effective_symbols, symbols_explicit = _resolve_effective_symbols(
        symbols=symbols,
        kwargs=kwargs,
        api_name="run_backtest",
    )

    strategy_input = resolve_strategy_input(
        strategy=strategy,
        strategy_source=strategy_source,
        strategy_loader=strategy_loader,
        strategy_loader_options=strategy_loader_options,
    )
    strategy_kwargs = dict(kwargs)
    if start_time and _accepts_strategy_kwarg(strategy_input, "start_time"):
        strategy_kwargs["start_time"] = start_time
    if end_time and _accepts_strategy_kwarg(strategy_input, "end_time"):
        strategy_kwargs["end_time"] = end_time
    if (
        symbols is not None
        and "symbols" not in strategy_kwargs
        and _accepts_strategy_kwarg(strategy_input, "symbols")
    ):
        strategy_kwargs["symbols"] = symbols
    strategy_instance = _build_strategy_instance(
        strategy_input,
        strategy_kwargs,
        strict_strategy_params,
        logger,
        initialize,
        on_start,
        on_resume,
        on_train_signal,
        on_stop,
        on_tick,
        on_order,
        on_trade,
        on_reject,
        on_before_trading,
        on_after_trading,
        on_cross_section,
        on_portfolio_update,
        on_error,
        on_expiry,
        on_pre_open,
        on_timer,
        context,
    )
    slot_strategy_instances: Dict[str, Strategy] = {}
    if strategies_by_slot:
        for slot_key, slot_strategy_input in strategies_by_slot.items():
            slot_key_str = str(slot_key).strip()
            if not slot_key_str:
                raise ValueError("strategy slot id cannot be empty")
            slot_strategy_kwargs = dict(kwargs)
            if symbols is not None and _accepts_strategy_kwarg(
                slot_strategy_input, "symbols"
            ):
                slot_strategy_kwargs["symbols"] = symbols
            if start_time and _accepts_strategy_kwarg(
                slot_strategy_input, "start_time"
            ):
                slot_strategy_kwargs["start_time"] = start_time
            if end_time and _accepts_strategy_kwarg(slot_strategy_input, "end_time"):
                slot_strategy_kwargs["end_time"] = end_time
            slot_strategy_instances[slot_key_str] = _build_strategy_instance(
                slot_strategy_input,
                slot_strategy_kwargs,
                strict_strategy_params,
                logger,
                initialize,
                on_start,
                on_resume,
                on_train_signal,
                on_stop,
                on_tick,
                on_order,
                on_trade,
                on_reject,
                on_before_trading,
                on_after_trading,
                on_cross_section,
                on_portfolio_update,
                on_error,
                on_expiry,
                on_pre_open,
                on_timer,
                context,
            )
    all_strategy_instances = [strategy_instance, *slot_strategy_instances.values()]
    configured_slot_ids = [effective_strategy_id]
    for slot_key in slot_strategy_instances.keys():
        if slot_key not in configured_slot_ids:
            configured_slot_ids.append(slot_key)
    normalized_strategy_fill_policy = _normalize_strategy_fill_policy_map(
        strategy_fill_policy,
        configured_slot_ids,
        logger,
    )
    normalized_strategy_slippage = _normalize_strategy_slippage_map(
        strategy_slippage,
        configured_slot_ids,
        logger,
    )
    normalized_strategy_commission = _normalize_strategy_commission_map(
        strategy_commission,
        configured_slot_ids,
    )
    setattr(strategy_instance, "_owner_strategy_id", effective_strategy_id)
    for slot_key, slot_strategy in slot_strategy_instances.items():
        setattr(slot_strategy, "_owner_strategy_id", slot_key)
    # 数据周期注入 self.freq(只读)。freq 此前只用于 tick→bar 聚合、用完即弃,
    # 策略没有任何途径知道自己跑在什么周期上。纯 bar 数据不传 freq 时保持 None
    # (不从相邻 bar 时间戳差推断: 停牌/跨日/午休会让它给出错误答案)。
    # 多策略槽位共用同一份数据, 故所有实例注入同一个值。
    for current_strategy in all_strategy_instances:
        setattr(current_strategy, "_framework_freq", freq)
    indicator_recorder = _attach_indicator_recorder(
        stream_emitter=indicator_stream_emitter,
        strategy_instance=strategy_instance,
        slot_strategy_instances=slot_strategy_instances,
        recorder=indicator_recorder,
    )
    setattr(strategy_instance, "_slot_strategies", dict(slot_strategy_instances))
    setattr(strategy_instance, "_strategy_slot_ids", list(configured_slot_ids))
    if normalized_strategy_fill_policy is not None:
        for current_strategy in all_strategy_instances:
            setattr(
                current_strategy,
                "_strategy_fill_policy_map",
                dict(normalized_strategy_fill_policy),
            )
    if normalized_strategy_slippage is not None:
        for current_strategy in all_strategy_instances:
            setattr(
                current_strategy,
                "_strategy_slippage_map",
                dict(normalized_strategy_slippage),
            )
    if normalized_strategy_commission is not None:
        for current_strategy in all_strategy_instances:
            setattr(
                current_strategy,
                "_strategy_commission_map",
                dict(normalized_strategy_commission),
            )

    if strategy_runtime_config is not None and isinstance(strategy_instance, Strategy):
        _apply_strategy_runtime_config(
            strategy_instance,
            strategy_runtime_config,
            runtime_config_override,
            logger,
        )
        for slot_strategy in slot_strategy_instances.values():
            _apply_strategy_runtime_config(
                slot_strategy,
                strategy_runtime_config,
                runtime_config_override,
                logger,
            )

    # 注入 context
    if context:
        for current_strategy in all_strategy_instances:
            if hasattr(current_strategy, "_context"):
                continue
            for k, v in context.items():
                setattr(current_strategy, k, v)

    # 注入 Config 中的 Risk Config
    if config and config.strategy_config and config.strategy_config.risk:
        for current_strategy in all_strategy_instances:
            if hasattr(current_strategy, "risk_config"):
                current_strategy.risk_config = config.strategy_config.risk  # type: ignore

    # 注入费率/手数配置到 Strategy 实例(单一真源 _cost_config, 绕过费率只读 setter)。
    # commission_policy 此处已归一(见上 resolved_commission_policy); commission_rate 由
    # policy 派生, 不再单独注入。lot_size=None 时不覆盖(保留策略 __init__ 里的赋值)。
    for current_strategy in all_strategy_instances:
        inject = getattr(current_strategy, "_inject_cost_config", None)
        if callable(inject):
            inject(
                commission_policy=commission_policy,
                min_commission=min_commission,
                stamp_tax_rate=stamp_tax_rate,
                transfer_fee_rate=transfer_fee_rate,
                lot_size=lot_size,
            )

    preliminary_symbols: List[str] = list(effective_symbols)
    if config and config.instruments:
        for s in config.instruments:
            if s not in preliminary_symbols:
                preliminary_symbols.append(s)
    preliminary_inst_conf_map: Dict[str, Any] = {}
    if config and config.instruments_config:
        if isinstance(config.instruments_config, list):
            for conf_item in config.instruments_config:
                if conf_item.symbol not in preliminary_inst_conf_map:
                    preliminary_inst_conf_map[conf_item.symbol] = conf_item
        elif isinstance(config.instruments_config, dict):
            for k, v in config.instruments_config.items():
                if k not in preliminary_inst_conf_map:
                    preliminary_inst_conf_map[k] = v
    preliminary_prebuilt_instruments: Dict[str, Any] = {}
    if "instruments" in kwargs:
        raw_instruments = kwargs["instruments"]
        if isinstance(raw_instruments, list):
            for item in raw_instruments:
                preliminary_prebuilt_instruments[item.symbol] = item
        elif isinstance(raw_instruments, dict):
            preliminary_prebuilt_instruments.update(raw_instruments)
    preliminary_default_expiry = _normalize_expiry_date_yyyymmdd(
        kwargs.get("expiry_date", None)
    )
    preliminary_default_option_type = kwargs.get("option_type", None)
    preliminary_default_option_margin_model = kwargs.get("option_margin_model", None)
    preliminary_default_strike_price = kwargs.get("strike_price", None)
    preliminary_default_asset_type = kwargs.get("asset_type", AssetType.Stock)
    preliminary_default_multiplier = kwargs.get("multiplier", 1.0)
    preliminary_default_margin_ratio = kwargs.get("margin_ratio", 1.0)
    # tick_size 缺省值按 asset_type 分流(基金 0.001, 其余 0.01), 显式传入的
    # tick_size 仍优先——见 _default_tick_size_for_asset_type。
    preliminary_default_tick_size = kwargs.get(
        "tick_size", _default_tick_size_for_asset_type(preliminary_default_asset_type)
    )
    preliminary_lot_size = kwargs.get("lot_size", 1)
    preliminary_default_implied_volatility = kwargs.get("implied_volatility", None)
    preliminary_default_reference_volatility = kwargs.get("reference_volatility", None)
    preliminary_default_settlement_type = _settlement_type_to_upper_name(
        kwargs.get("settlement_type", None)
    )
    preliminary_default_settlement_price = kwargs.get("settlement_price", None)
    preliminary_snapshots: Dict[str, InstrumentSnapshot] = {}
    for sym in preliminary_symbols:
        if sym in preliminary_prebuilt_instruments:
            prebuilt = preliminary_prebuilt_instruments[sym]
            preliminary_snapshots[sym] = InstrumentSnapshot(
                symbol=sym,
                asset_type=_asset_type_to_upper_name(
                    getattr(prebuilt, "asset_type", "")
                ),
                multiplier=float(getattr(prebuilt, "multiplier", 1.0)),
                margin_ratio=float(getattr(prebuilt, "margin_ratio", 1.0)),
                option_margin_model=_option_margin_model_to_upper_name(
                    getattr(prebuilt, "option_margin_model", None)
                ),
                tick_size=float(
                    getattr(prebuilt, "tick_size", None)
                    or _default_tick_size_for_asset_type(
                        getattr(prebuilt, "asset_type", "")
                    )
                ),
                lot_size=float(getattr(prebuilt, "lot_size", 1.0) or 1.0),
                implied_volatility=(
                    float(getattr(prebuilt, "implied_volatility"))
                    if getattr(prebuilt, "implied_volatility", None) is not None
                    else None
                ),
                reference_volatility=(
                    float(getattr(prebuilt, "reference_volatility"))
                    if getattr(prebuilt, "reference_volatility", None) is not None
                    else None
                ),
                settlement_type=_settlement_type_to_upper_name(
                    getattr(prebuilt, "settlement_type", None)
                ),
                settlement_price=(
                    float(getattr(prebuilt, "settlement_price"))
                    if getattr(prebuilt, "settlement_price", None) is not None
                    else None
                ),
            )
            continue
        conf = preliminary_inst_conf_map.get(sym)
        symbol_lot_size: Optional[float] = None
        if isinstance(preliminary_lot_size, int):
            symbol_lot_size = float(preliminary_lot_size)
        elif isinstance(preliminary_lot_size, dict):
            raw_lot = preliminary_lot_size.get(sym)
            if raw_lot is not None:
                symbol_lot_size = float(raw_lot)
        if conf is None:
            preliminary_snapshots[sym] = InstrumentSnapshot(
                symbol=sym,
                asset_type=_asset_type_to_upper_name(preliminary_default_asset_type),
                multiplier=float(preliminary_default_multiplier),
                margin_ratio=float(preliminary_default_margin_ratio),
                option_margin_model=_option_margin_model_to_upper_name(
                    preliminary_default_option_margin_model
                    if _asset_type_to_upper_name(preliminary_default_asset_type)
                    == "OPTION"
                    else None
                ),
                tick_size=float(preliminary_default_tick_size),
                lot_size=float(symbol_lot_size or 1.0),
                option_type=_option_type_to_upper_name(preliminary_default_option_type),
                strike_price=(
                    float(preliminary_default_strike_price)
                    if preliminary_default_strike_price is not None
                    else None
                ),
                expiry_date=preliminary_default_expiry,
                implied_volatility=(
                    float(preliminary_default_implied_volatility)
                    if preliminary_default_implied_volatility is not None
                    else None
                ),
                reference_volatility=(
                    float(preliminary_default_reference_volatility)
                    if preliminary_default_reference_volatility is not None
                    else None
                ),
                settlement_type=preliminary_default_settlement_type,
                settlement_price=(
                    float(preliminary_default_settlement_price)
                    if preliminary_default_settlement_price is not None
                    else None
                ),
            )
            continue
        conf_static_attrs = getattr(conf, "static_attrs", {})
        if conf_static_attrs is None:
            conf_static_attrs = {}
        if not isinstance(conf_static_attrs, dict):
            raise TypeError("InstrumentConfig.static_attrs must be Dict[str, scalar]")
        conf_lot = (
            float(conf.lot_size)
            if conf.lot_size is not None
            else float(symbol_lot_size or 1.0)
        )
        preliminary_snapshots[sym] = InstrumentSnapshot(
            symbol=sym,
            asset_type=_asset_type_to_upper_name(conf.asset_type),
            multiplier=float(conf.multiplier),
            margin_ratio=float(conf.margin_ratio),
            option_margin_model=cast(
                Optional[InstrumentOptionMarginModelName],
                conf.option_margin_model,
            ),
            tick_size=float(conf.tick_size),
            lot_size=conf_lot,
            option_type=_option_type_to_upper_name(conf.option_type),
            strike_price=(
                float(conf.strike_price) if conf.strike_price is not None else None
            ),
            expiry_date=_normalize_expiry_date_yyyymmdd(conf.expiry_date),
            underlying_symbol=(
                str(conf.underlying_symbol)
                if conf.underlying_symbol is not None
                else None
            ),
            implied_volatility=(
                float(conf.implied_volatility)
                if conf.implied_volatility is not None
                else None
            ),
            reference_volatility=(
                float(conf.reference_volatility)
                if conf.reference_volatility is not None
                else None
            ),
            settlement_type=_settlement_type_to_upper_name(conf.settlement_type),
            settlement_price=(
                float(conf.settlement_price)
                if conf.settlement_price is not None
                else None
            ),
            static_attrs=dict(conf_static_attrs),
        )
    for current_strategy in all_strategy_instances:
        current_strategy._set_instrument_snapshots(preliminary_snapshots)
    normalized_global_slippage = _normalize_slippage_policy(
        slippage,
        instrument_snapshots=preliminary_snapshots,
        logger=logger,
        scope="Global",
    )

    # symbols 白名单下发给策略实例(供 subscribe() 校验用): 必须在 on_start 之前,
    # 这样 on_start 里的 subscribe 才会被挡住。只在显式传了 symbols 时校验,
    # 且只在回测路径设——实盘的 subscribe 是正常的动态订阅手段, 不校验。
    # 无条件赋值(而非只在 symbols_explicit 时才赋值): 策略实例可能是从 checkpoint
    # 恢复的, `_symbol_whitelist` 会随对象一起被 pickle 持久化(load_checkpoint 用
    # 默认 __dict__ 整体恢复, 会覆盖 Strategy.__new__ 设的 None)——若本次调用
    # 没显式传 symbols 却不去覆盖它, 就会沿用上一段 checkpoint 里的旧白名单,
    # 与本次「不传 symbols = 不过滤」的意图相矛盾。
    # 用 effective_symbols(已归一 List[str])而非原始 symbols: 此处的 `symbols`
    # 仍是 _resolve_effective_symbols 的第一个返回值, 未经归一——symbols="600519"
    # 这种受支持的字符串写法下 set(symbols) 会把它拆成单字符集合, 使 subscribe()
    # 把合法的自身标的都拦下来。用 _merge_symbol_whitelist_sources 在此时(on_start
    # 之前)就把 config.instruments 与 __init__ 里已有的 _subscriptions 并进来,
    # 使这里下发的白名单与下面(:3101 附近, on_start 之后)重新合并出的 `symbols`
    # 在未发生 on_start 内 subscribe 的前提下一致。
    whitelist_for_strategy = (
        set(
            _merge_symbol_whitelist_sources(
                effective_symbols,
                config,
                strategy_instance,
                *slot_strategy_instances.values(),
            )
        )
        if symbols_explicit
        else None
    )
    strategy_instance._symbol_whitelist = whitelist_for_strategy
    for slot_strategy in slot_strategy_instances.values():
        slot_strategy._symbol_whitelist = whitelist_for_strategy

    # 调用 on_start 获取订阅
    # 注意：现在调用 _on_start_internal 来触发自动发现
    if hasattr(strategy_instance, "_on_start_internal"):
        strategy_instance._on_start_internal()
    elif hasattr(strategy_instance, "on_start"):
        strategy_instance.on_start()
    for slot_strategy in slot_strategy_instances.values():
        if hasattr(slot_strategy, "_on_start_internal"):
            slot_strategy._on_start_internal()
        elif hasattr(slot_strategy, "on_start"):
            slot_strategy.on_start()

    _, effective_depth = _resolve_runtime_warmup_depth(
        strategy_instance=strategy_instance,
        history_depth=history_depth,
        warmup_period=warmup_period,
        logger=logger,
    )
    manual_history_depth = max(
        int(getattr(current_strategy, "_history_depth", 0))
        for current_strategy in all_strategy_instances
    )
    indicator_warmup_depth = max(
        _resolve_incremental_indicator_warmup_depth(current_strategy)
        for current_strategy in all_strategy_instances
    )
    effective_depth = max(effective_depth, manual_history_depth, indicator_warmup_depth)
    preserve_pre_start_history = bool(start_time) and effective_depth > 0
    load_start_time = None if preserve_pre_start_history else start_time
    active_start_time_ns = (
        _to_active_start_time_ns(start_time, timezone)
        if preserve_pre_start_history
        else None
    )
    for current_strategy in all_strategy_instances:
        setattr(current_strategy, "_active_start_time_ns", active_start_time_ns)

    # 3. 准备数据源和 Symbol
    feed = DataFeed()
    symbols = []
    data_map_for_indicators = {}
    # 前置过滤: List[Bar] 形态在构建 feed 前跳过白名单外的标的, 省掉排序/
    # 构建 feed 与按 symbol 分组算指标 df 的开销。与 DataFrame(:3221 附近)、
    # dict(:3250 附近) 两段用同一判据, 三种能枚举内容的形态在 symbols 上的
    # 行为保持一致 —— 语义仍由 Rust 层白名单兜底(Task 1), 这里纯属优化,
    # 覆盖不到 DataFeed 对象这种无法枚举的形态也无妨。
    filtered_out_symbols: set[str] = set()
    # Catalog 路径逐标的读取失败时记入此集合(:3464 附近), 供运行前零数据比对
    # (:4572 附近)去重——那条已有英文 warning, 不需要再叠一条中文的。
    catalog_missing_symbols: set[str] = set()

    # 合并 effective_symbols + config.instruments + 各策略当前的 _subscriptions
    # (含 on_start 里新增的订阅——此时 on_start 已跑完)。与上面(:3048 附近,
    # on_start 之前)下发给策略实例供 subscribe() 校验的白名单复用同一个合并
    # 函数, 避免两处手写而漂移出不一致的顺序。
    symbols = _merge_symbol_whitelist_sources(
        effective_symbols,
        config,
        strategy_instance,
        *slot_strategy_instances.values(),
    )

    analyzer_manager = AnalyzerManager()
    for plugin in normalized_analyzers:
        analyzer_manager.register(plugin)
    for current_strategy in all_strategy_instances:
        setattr(current_strategy, "_analyzer_manager", analyzer_manager)

    # Determine Data Loading Strategy
    if data is not None:
        # polars / pyarrow 输入统一转 pandas, 复用既有数据路径(issue #298)
        data = coerce_to_pandas(data)
        # freq 只对含 Tick 的列表有意义。放在归一之前做校验, 使 DataFrame /
        # DataFeed / 纯 bar 列表配 freq 时也能早失败, 而不是静默忽略这个参数。
        if freq is not None and not (
            isinstance(data, list) and any(isinstance(item, Tick) for item in data)
        ):
            raise ValueError(
                f"freq={freq!r} 只在 data 为含 Tick 的列表时有意义。"
                "若要重采样 bar, 请用 akquant.feed_adapter 的 resample()"
            )

        # 含 Tick(或需要校验)的列表先归一成 DataFeed, 交给下面既有的 DataFeed
        # 分支处理: 该路径已实测能正确投递 tick(feed.add_tick -> 引擎
        # Event::Tick), 无需重复实现。
        # 必须在整条 if/elif 分发链之前完成——一旦进入下面的 elif isinstance(data,
        # list) 分支, 把 data 换成 DataFeed 也不会回退到 DataFeed 分支去。
        #
        # 触发条件不是"含 Tick"而是"非纯 Bar 列表"(空列表, 或含至少一个非 Bar
        # 元素, Tick 也算非 Bar): 这样空列表与含非法元素的列表也会先过
        # normalize_market_input 的校验(早失败, 报 ValueError/TypeError 且指名
        # 位置), 而不是漏到下面 elif 分支里 .sort() 时抛出令人费解的
        # AttributeError, 或静默跳过校验直接进入撮合。纯 Bar 列表(不含 Tick 也
        # 不含非法元素)维持原分支不变, 避免多一趟遍历/排序的开销。
        if isinstance(data, list) and (
            not data or any(not isinstance(item, Bar) for item in data)
        ):
            bars_part, ticks_part = normalize_market_input(data)
            # 预计算指标依赖 data_map_for_indicators, 而归一后走的 DataFeed 分支不
            # 构建它。静默丢失指标比报错危险得多, 故显式拒绝并指向可用的替代方案。
            #
            # 判据是"是否真的注册了预计算指标"而非 indicator_mode: 后者默认就是
            # "precompute"(strategy.py), 用它做判据会误伤所有未显式改模式的 tick 用户。
            if ticks_part and any(
                getattr(one_strategy, "_precomputed_indicators", None)
                for one_strategy in all_strategy_instances
            ):
                raise ValueError(
                    "已注册的预计算指标(precompute 模式)不支持含 Tick 的输入: "
                    "预计算指标需要完整的 OHLC DataFrame, 而 tick 只有成交价。请改用 "
                    "register_incremental_indicator(indicator_mode='incremental', "
                    "tick 路径已支持单值指标), 或给 run_backtest 传 freq"
                    "(如 freq='1min')把 tick 聚合成 bar。"
                )
            tick_feed = DataFeed()
            if bars_part:
                tick_feed.add_bars(bars_part)
            for one_tick in ticks_part:
                tick_feed.add_tick(one_tick)
            if freq is not None:
                # 原始 tick 仍投递, 合成 bar 进同一 feed。
                aggregate_ticks_into_feed(
                    tick_feed, ticks_part, parse_freq_to_interval_min(freq)
                )
            tick_feed.sort()
            data = tick_feed
        if isinstance(data, DataFeed):
            # Use provided DataFeed
            feed = data
            # We don't know symbols in feed easily without iteration,
            # but usually feed contains all needed data.
            # We might need to update 'symbols' if they were not provided explicitly?
            # For now, assume user provided symbols or feed covers them.
        elif _is_data_feed_adapter(data):
            adapter_data_map, leaked_adapter_symbols = _load_data_map_from_adapter(
                adapter=data,
                symbols=symbols,
                start_time=load_start_time,
                end_time=end_time,
                timezone=timezone,
            )
            if leaked_adapter_symbols:
                # 与 filtered_out_symbols(用户主动传 symbols 排除的标的)性质不同:
                # 这里是 adapter 违反契约返回了未请求的标的, 单独发一条 warning
                # 提醒, 不混进下面那条 INFO 汇总日志。
                logger.warning(
                    "DataFeedAdapter 返回了 %d 个未请求的标的, 已丢弃: %s",
                    len(leaked_adapter_symbols),
                    ", ".join(sorted(leaked_adapter_symbols)),
                )
            for sym, df in adapter_data_map.items():
                df_prep = to_indicator_frame(df)
                data_map_for_indicators[sym] = df_prep
                arrays = dataframe_to_arrays(df_prep, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                if sym not in symbols:
                    symbols.append(sym)
            feed.sort()
        elif isinstance(data, pd.DataFrame):
            df_input = data
            # Ensure index is datetime
            if not isinstance(df_input.index, pd.DatetimeIndex):
                # Try to find a date column if index is not date
                # Common candidates: "date", "timestamp", "datetime"
                found_date = False
                for col in ["date", "timestamp", "datetime", "Date", "Timestamp"]:
                    if col in df_input.columns:
                        df_input = df_input.set_index(col)
                        found_date = True
                        break

                if not found_date:
                    # try convert index
                    try:
                        df_input.index = pd.to_datetime(df_input.index)
                    except Exception:
                        pass

            # Ensure index is pd.Timestamp compatible
            # (convert datetime.date to Timestamp)
            # This is handled by pd.to_datetime but let's be safe for object index
            # 非 DatetimeIndex 索引统一转 datetime(兼容 pandas 2.x str dtype)
            if not isinstance(df_input.index, pd.DatetimeIndex):
                try:
                    df_input.index = pd.to_datetime(df_input.index)
                except Exception:
                    pass

            # Filter by date if provided
            if isinstance(df_input.index, pd.DatetimeIndex):
                df_input = _filter_datetime_index_frame_by_runtime_window(
                    df_input,
                    load_start_time,
                    end_time,
                    timezone,
                )
            elif load_start_time or end_time:
                if (
                    len(df_input) > 0
                    and isinstance(df_input.index[0], (dt_module.date))
                    and not isinstance(df_input.index[0], dt_module.datetime)
                ):
                    if load_start_time:
                        ts_start = _parse_runtime_boundary_timestamp(
                            load_start_time, timezone
                        )
                        df_input = df_input[df_input.index >= ts_start.date()]
                    if end_time:
                        ts_end = _parse_runtime_boundary_timestamp(end_time, timezone)
                        df_input = df_input[df_input.index <= ts_end.date()]

            df = to_indicator_frame(df_input)
            # 标的列判据对齐项目统一的别名表(schema.COLUMN_ALIASES)。原先只认英文
            # "symbol", 而 AKShare 的标准输出列名是 "股票代码" —— "从 AKShare 取多
            # 标的数据直接丢进 run_backtest" 这个最自然的用法会静默走进下方的单标的
            # 分支。这里重命名成 "symbol" 而非逐处适配下游: 后面的 isin / groupby /
            # dataframe_to_arrays 沿用同一个列名即可。
            resolved_symbol_col = resolve_columns(df).get("symbol")
            if resolved_symbol_col and resolved_symbol_col != "symbol":
                df = df.rename(columns={resolved_symbol_col: "symbol"})
            if "symbol" in df.columns:
                df = df.copy()
                df["symbol"] = df["symbol"].astype(str)
                filter_symbols = bool(symbols and "BENCHMARK" not in symbols)
                if filter_symbols:
                    # 被 isin 滤掉的标的只能从过滤前的 symbol 集合里减去白名单
                    # 求差集得到 —— isin 本身是整体布尔索引, 不是逐条判断,
                    # 拿不到"被剔除了哪些"这个信息, 只能靠过滤前后各留一份
                    # symbol 集合来对比。只记录, 不改这行既有的过滤判据本身。
                    pre_filter_symbols = set(df["symbol"].unique())
                    df = df[df["symbol"].isin(symbols)]
                    filtered_out_symbols.update(pre_filter_symbols.difference(symbols))
                if not df.empty:
                    arrays = dataframe_to_arrays(df)
                    feed.add_arrays(*arrays)  # type: ignore
                    grouped = df.groupby("symbol", sort=False)
                    for grouped_symbol, grouped_df in grouped:
                        sym = str(grouped_symbol)
                        data_map_for_indicators[sym] = grouped_df.copy()
                    detected_symbols = [str(s) for s in df["symbol"].unique().tolist()]
                    if not symbols or symbols == ["BENCHMARK"]:
                        symbols = detected_symbols
                    else:
                        for sym in detected_symbols:
                            if sym not in symbols:
                                symbols.append(sym)
                feed.sort()
            else:
                target_symbol = symbols[0] if symbols else "BENCHMARK"
                # 同一时间戳出现多行 = 多标的被压成一条序列的可靠信号(真单标的数据
                # 每个时间戳只有一行, 故不会误报)。此前这条退化完全静默: 指标与撮合
                # 结果都不正确, instruments_config 按真实标的配的合约参数整体失效,
                # 下单真实 symbol 只会拿到 "Instrument not found"。
                if "timestamp" in df.columns and df["timestamp"].duplicated().any():
                    logger.warning(
                        "DataFrame 未识别到标的列, 已退化为单标的 %s 处理, 但检测到"
                        "同一时间戳存在多行 —— 多标的数据会被压成一条序列(指标与撮合"
                        "结果均不正确, instruments_config 按真实标的配置的合约参数也"
                        "会失效)。多标的请把标的列命名为 %s 之一, 或改用 "
                        "Dict[str, DataFrame] / list[Bar] 输入。",
                        target_symbol,
                        ", ".join(COLUMN_ALIASES["symbol"]),
                    )
                data_map_for_indicators[target_symbol] = df
                arrays = dataframe_to_arrays(df, symbol=target_symbol)
                feed.add_arrays(*arrays)  # type: ignore
                feed.sort()
                if target_symbol not in symbols:
                    symbols = [target_symbol]
        elif isinstance(data, dict):
            # If explicit symbols are provided (i.e., not just the default "BENCHMARK"),
            # we filter the data dictionary to only include requested symbols.
            filter_symbols = "BENCHMARK" not in symbols

            for sym, df in data.items():
                if filter_symbols and sym not in symbols:
                    filtered_out_symbols.add(sym)
                    continue

                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    # Try to find a date column if index is not date
                    found_date = False
                    for col in ["date", "timestamp", "datetime", "Date", "Timestamp"]:
                        if col in df.columns:
                            df = df.set_index(col)
                            df.index = pd.to_datetime(df.index)
                            found_date = True
                            break

                    if not found_date:
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass

                # Filter by date
                if isinstance(df.index, pd.DatetimeIndex):
                    df = _filter_datetime_index_frame_by_runtime_window(
                        df,
                        load_start_time,
                        end_time,
                        timezone,
                    )

                df_prep = to_indicator_frame(df)
                data_map_for_indicators[sym] = df_prep
                arrays = dataframe_to_arrays(df_prep, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                if sym not in symbols:
                    symbols.append(sym)
            feed.sort()
        elif isinstance(data, list):
            if data:
                # Filter by date
                if load_start_time:
                    # Explicitly convert to int to satisfy mypy
                    ts_start_ns = _boundary_timestamp_to_utc_ns(
                        load_start_time, timezone
                    )
                    data = [b for b in data if b.timestamp >= ts_start_ns]  # type: ignore
                if end_time:
                    ts_end_ns = _boundary_timestamp_to_utc_ns(end_time, timezone)
                    data = [b for b in data if b.timestamp <= ts_end_ns]  # type: ignore

                # 前置过滤: 判据与 DataFrame(:3221 附近)/dict(:3250 附近) 两段
                # 完全一致, 避免三种形态在 symbols 上出现不一致的行为。
                filter_symbols = bool(symbols and "BENCHMARK" not in symbols)
                if filter_symbols:
                    kept_bars = []
                    for b in data:
                        if b.symbol not in symbols:  # type: ignore[union-attr]
                            filtered_out_symbols.add(b.symbol)  # type: ignore[union-attr]
                            continue
                        kept_bars.append(b)
                    data = kept_bars

                data.sort(key=lambda b: b.timestamp)
                feed.add_bars(data)  # type: ignore[arg-type]

                # Construct DataFrame for indicator calculation
                # Group by symbol just in case
                # 到达这里时 data 运行时必为 List[Bar](Tick/空/非法元素的列表已在
                # 分发链之前被 normalize_market_input 拦截并转为 DataFeed 走别的
                # 分支); mypy 静态上仍视 data 为 BacktestDataInput 的宽联合类型
                # (加宽是 Step 3 的要求), 故下面的属性访问需要 ignore。
                bars_by_sym: Dict[str, List[Dict[str, Any]]] = {}
                for bar in data:
                    if bar.symbol not in bars_by_sym:
                        bars_by_sym[bar.symbol] = []
                    bars_by_sym[bar.symbol].append(
                        {
                            "timestamp": pd.Timestamp(
                                bar.timestamp, unit="ns", tz="UTC"
                            ),
                            "open": bar.open,  # type: ignore[union-attr]
                            "high": bar.high,  # type: ignore[union-attr]
                            "low": bar.low,  # type: ignore[union-attr]
                            "close": bar.close,  # type: ignore[union-attr]
                            "volume": bar.volume,
                        }
                    )

                for sym, records in bars_by_sym.items():
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df.set_index("timestamp", inplace=True)
                        df.sort_index(inplace=True)
                        data_map_for_indicators[sym] = df
    else:
        # Load from Catalog / Akshare
        if not symbols:
            logger.warning("No symbols specified and no data provided.")

        catalog = ParquetDataCatalog(root_path=catalog_path)
        logger.info(f"Loading backtest data from catalog root: {catalog.root}")
        # start_time / end_time already resolved above

        loaded_count = 0
        for sym in symbols:
            # Try Catalog
            df = catalog.read(
                sym,
                start_time=load_start_time,
                end_time=end_time,
                timezone=timezone,
            )
            if df.empty:
                logger.warning(f"Data not found in catalog for {sym}")
                catalog_missing_symbols.add(str(sym))
                continue

            if not df.empty:
                df = to_indicator_frame(df)
                data_map_for_indicators[sym] = df
                arrays = dataframe_to_arrays(df, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                loaded_count += 1

        if loaded_count > 0:
            feed.sort()
        else:
            if symbols:
                logger.warning("Failed to load data for all requested symbols.")

    # Inject timezone to strategy
    for current_strategy in all_strategy_instances:
        current_strategy.timezone = timezone

    # Inject trading days to strategy (for schedule_daily)
    all_strategy_instances = [strategy_instance, *slot_strategy_instances.values()]
    if data_map_for_indicators:
        (
            all_dates,
            day_bounds,
            day_cross_section_timestamps,
        ) = _build_trading_day_metadata(data_map_for_indicators, timezone)

        for current_strategy in all_strategy_instances:
            if hasattr(current_strategy, "_trading_days") and all_dates:
                current_strategy._trading_days = all_dates
            if hasattr(current_strategy, "_trading_day_bounds"):
                current_strategy._trading_day_bounds = day_bounds
            if hasattr(current_strategy, "_trading_day_cross_section_timestamps"):
                current_strategy._trading_day_cross_section_timestamps = (
                    day_cross_section_timestamps
                )

    # 4. 配置引擎
    engine = Engine()
    cast(Any, engine).active_start_time_ns = active_start_time_ns
    for current_strategy in all_strategy_instances:
        setattr(current_strategy, "_engine", engine)
    _prime_framework_boundary_timers(all_strategy_instances, engine)
    _prime_framework_cross_section_timers(all_strategy_instances, engine)
    _prime_framework_pre_open_timers(all_strategy_instances, engine)
    if analyzer_manager.plugins:
        try:
            analyzer_manager.on_start(
                {
                    "engine": engine,
                    "strategy": strategy_instance,
                    "strategies": list(all_strategy_instances),
                    "slot_strategy_map": {
                        effective_strategy_id: strategy_instance,
                        **slot_strategy_instances,
                    },
                    "symbols": list(symbols),
                }
            )
        except Exception as e:
            logger.error(f"Analyzer on_start error: {e}")
    if hasattr(engine, "set_timezone_name"):
        cast(Any, engine).set_timezone_name(timezone)
    else:
        offset_delta = pd.Timestamp.now(tz=timezone).utcoffset()
        if offset_delta is None:
            raise ValueError(f"Invalid timezone: {timezone}")
        offset = int(offset_delta.total_seconds())
        engine.set_timezone(offset)
    if hasattr(engine, "set_days_per_year"):
        cast(Any, engine).set_days_per_year(
            getattr(config, "days_per_year", 252.0) if config else 252.0
        )
    if hasattr(engine, "set_risk_free_rate"):
        cast(Any, engine).set_risk_free_rate(
            getattr(config, "risk_free_rate", 0.0) if config else 0.0
        )
    engine.set_cash(initial_cash)
    if hasattr(engine, "set_default_strategy_id"):
        cast(Any, engine).set_default_strategy_id(effective_strategy_id)
    if (
        strategies_by_slot
        and hasattr(engine, "set_strategy_slots")
        and hasattr(engine, "set_strategy_for_slot")
    ):
        cast(Any, engine).set_strategy_slots(configured_slot_ids)
        for slot_index, slot_id in enumerate(configured_slot_ids):
            assigned_strategy: Optional[Strategy] = None
            if slot_id == effective_strategy_id:
                assigned_strategy = strategy_instance
            else:
                assigned_strategy = slot_strategy_instances.get(slot_id)
            if assigned_strategy is not None:
                cast(Any, engine).set_strategy_for_slot(slot_index, assigned_strategy)
    if strategy_priority and hasattr(engine, "set_strategy_priorities"):
        normalized_strategy_priority: Dict[str, int] = {}
        for strategy_key, raw_priority in strategy_priority.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_priority contains empty strategy id")
            priority_value = int(raw_priority)
            normalized_strategy_priority[strategy_key_str] = priority_value
        unknown_keys = sorted(
            set(normalized_strategy_priority.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_priority contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_priorities(normalized_strategy_priority)
    if strategy_risk_budget and hasattr(engine, "set_strategy_risk_budget_limits"):
        normalized_strategy_risk_budget: Dict[str, float] = {}
        for strategy_key, raw_budget in strategy_risk_budget.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_risk_budget contains empty strategy id")
            budget_value = float(raw_budget)
            if not pd.notna(budget_value) or budget_value < 0.0:
                raise ValueError(
                    f"strategy_risk_budget for {strategy_key_str} must be >= 0"
                )
            normalized_strategy_risk_budget[strategy_key_str] = budget_value
        unknown_keys = sorted(
            set(normalized_strategy_risk_budget.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_risk_budget contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_risk_budget_limits(
            normalized_strategy_risk_budget
        )
    if hasattr(engine, "set_portfolio_risk_budget_limit"):
        cast(Any, engine).set_portfolio_risk_budget_limit(portfolio_risk_budget)
    if hasattr(engine, "set_risk_budget_mode"):
        cast(Any, engine).set_risk_budget_mode(risk_budget_mode)
    if hasattr(engine, "set_risk_budget_reset_daily"):
        cast(Any, engine).set_risk_budget_reset_daily(risk_budget_reset_daily)
    if strategy_max_order_value and hasattr(
        engine, "set_strategy_max_order_value_limits"
    ):
        normalized_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_order_value.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_order_value contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_order_value for {strategy_key_str} must be >= 0"
                )
            normalized_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_order_value contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_order_value_limits(normalized_limits)
    if strategy_max_order_size and hasattr(
        engine, "set_strategy_max_order_size_limits"
    ):
        normalized_limits_by_size: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_order_size.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_order_size contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_order_size for {strategy_key_str} must be >= 0"
                )
            normalized_limits_by_size[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_limits_by_size.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_order_size contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_order_size_limits(normalized_limits_by_size)
    if strategy_max_position_size and hasattr(
        engine, "set_strategy_max_position_size_limits"
    ):
        normalized_position_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_position_size.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_max_position_size contains empty strategy id"
                )
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_position_size for {strategy_key_str} must be >= 0"
                )
            normalized_position_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_position_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_position_size contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_position_size_limits(
            normalized_position_limits
        )
    if strategy_max_daily_loss and hasattr(
        engine, "set_strategy_max_daily_loss_limits"
    ):
        normalized_daily_loss_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_daily_loss.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_daily_loss contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_daily_loss for {strategy_key_str} must be >= 0"
                )
            normalized_daily_loss_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_daily_loss_limits.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_daily_loss contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_daily_loss_limits(
            normalized_daily_loss_limits
        )
    if strategy_max_drawdown and hasattr(engine, "set_strategy_max_drawdown_limits"):
        normalized_drawdown_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_drawdown.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_drawdown contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_drawdown for {strategy_key_str} must be >= 0"
                )
            normalized_drawdown_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_drawdown_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_drawdown contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_drawdown_limits(normalized_drawdown_limits)
    if strategy_reduce_only_after_risk and hasattr(
        engine, "set_strategy_reduce_only_after_risk"
    ):
        normalized_reduce_only_flags: Dict[str, bool] = {}
        for strategy_key, raw_flag in strategy_reduce_only_after_risk.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_reduce_only_after_risk contains empty strategy id"
                )
            normalized_reduce_only_flags[strategy_key_str] = bool(raw_flag)
        unknown_keys = sorted(
            set(normalized_reduce_only_flags.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_reduce_only_after_risk contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_reduce_only_after_risk(
            normalized_reduce_only_flags
        )
    if strategy_risk_cooldown_bars and hasattr(
        engine, "set_strategy_risk_cooldown_bars"
    ):
        normalized_cooldown_bars: Dict[str, int] = {}
        for strategy_key, raw_bars in strategy_risk_cooldown_bars.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_risk_cooldown_bars contains empty strategy id"
                )
            cooldown_bars = int(raw_bars)
            if cooldown_bars < 0:
                raise ValueError(
                    f"strategy_risk_cooldown_bars for {strategy_key_str} must be >= 0"
                )
            normalized_cooldown_bars[strategy_key_str] = cooldown_bars
        unknown_keys = sorted(
            set(normalized_cooldown_bars.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_risk_cooldown_bars contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_risk_cooldown_bars(normalized_cooldown_bars)
    if history_depth > 0:
        engine.set_history_depth(history_depth)
    if stream_on_event is not None:
        cast(Any, engine).set_stream_callback(stream_on_event)
        cast(Any, engine).set_stream_options(
            stream_progress_interval,
            stream_equity_interval,
            stream_batch_size,
            stream_max_buffer,
            stream_error_mode,
            stream_mode,
        )

    # Register Custom Matchers
    if custom_matchers:
        for asset_type, matcher in custom_matchers.items():
            try:
                cast(Any, engine).register_custom_matcher(asset_type, matcher)
            except Exception as e:
                logger.warning(
                    "Failed to register custom matcher for %s: %s",
                    asset_type,
                    e,
                )

    if fill_policy is not None:
        if isinstance(fill_policy, FillMode):
            _pb, _bo, _tp = fill_policy._to_core()
            fill_policy = {"price_basis": _pb, "bar_offset": _bo, "temporal": _tp}
        elif isinstance(fill_policy, dict):
            raise TypeError(_LEGACY_FILL_POLICY_DICT_MSG)

    resolved_policy = _resolve_execution_policy(
        execution_mode="next_open",
        timer_execution_policy="same_cycle",
        fill_policy=fill_policy,
        logger=logger,
    )
    if not hasattr(engine, "set_fill_mode"):
        raise RuntimeError(
            "Engine binary does not expose set_fill_mode; please rebuild bindings"
        )
    cast(Any, engine).set_fill_mode(
        resolved_policy.execution_mode,
        resolved_policy.temporal,
    )
    default_fill_policy: FillPolicy = {
        "price_basis": resolved_policy.price_basis,
        "bar_offset": resolved_policy.bar_offset,
        "temporal": resolved_policy.temporal,
    }
    for current_strategy in all_strategy_instances:
        setattr(current_strategy, "_default_fill_policy", dict(default_fill_policy))

    # 4.1 市场规则配置
    china_futures_config: Optional[ChinaFuturesConfig] = None
    china_options_config: Optional[ChinaOptionsConfig] = None
    china_stock_config: Optional[ChinaStockConfig] = None
    has_futures_instruments = False
    has_options_instruments = False
    has_non_futures_instruments = False
    if config is not None:
        china_futures_config = config.china_futures
        china_options_config = config.china_options
        china_stock_config = config.china_stock
        if config.instruments_config:
            if isinstance(config.instruments_config, list):
                for inst in config.instruments_config:
                    asset_name = _parse_asset_type_name(inst.asset_type)
                    if asset_name == "futures":
                        has_futures_instruments = True
                    elif asset_name == "option":
                        has_options_instruments = True
                        has_non_futures_instruments = True
                    else:
                        has_non_futures_instruments = True
            elif isinstance(config.instruments_config, dict):
                for inst in config.instruments_config.values():
                    asset_name = _parse_asset_type_name(inst.asset_type)
                    if asset_name == "futures":
                        has_futures_instruments = True
                    elif asset_name == "option":
                        has_options_instruments = True
                        has_non_futures_instruments = True
                    else:
                        has_non_futures_instruments = True
    if not has_futures_instruments or not has_options_instruments:
        default_asset_name = _parse_asset_type_name(
            kwargs.get("asset_type", AssetType.Stock)
        )
        if not has_futures_instruments:
            has_futures_instruments = default_asset_name == "futures"
        if not has_options_instruments:
            has_options_instruments = default_asset_name == "option"
    if (
        not has_futures_instruments
        and china_futures_config
        and china_futures_config.instrument_templates_by_symbol_prefix
    ):
        has_futures_instruments = True

    # Any explicit per-instrument T+1 (sellable_after_days>=1) needs ChinaMarket so
    # on_day_close releases the lock; otherwise a T+1 instrument under t_plus_one=False
    # (SimpleMarket) would silently never become sellable.
    any_t_plus_one_instrument = any(
        getattr(ic, "sellable_after_days", None) is not None
        and getattr(ic, "sellable_after_days", 0) >= 1
        for ic in (kwargs.get("instruments") or [])
    )

    if china_futures_config and has_futures_instruments:
        if (
            not china_futures_config.use_china_futures_market
            or has_non_futures_instruments
        ):
            engine.use_china_market()
        else:
            engine.use_china_futures_market()
        if t_plus_one:
            engine.set_t_plus_one(True)
    elif china_options_config and has_options_instruments:
        if china_options_config.use_china_market:
            engine.use_china_market()
        else:
            if hasattr(engine, "use_simple_market_policy"):
                cast(Any, engine).use_simple_market_policy(
                    commission_policy["type"], float(commission_policy["value"])
                )
            else:
                engine.use_simple_market(commission_rate)
        if t_plus_one:
            engine.set_t_plus_one(True)
    elif t_plus_one or any_t_plus_one_instrument:
        engine.use_china_market()
        if t_plus_one:
            engine.set_t_plus_one(True)
    else:
        if hasattr(engine, "use_simple_market_policy"):
            cast(Any, engine).use_simple_market_policy(
                commission_policy["type"], float(commission_policy["value"])
            )
        else:
            engine.use_simple_market(commission_rate)

    force_session_continuous = True
    if china_futures_config and has_futures_instruments:
        force_session_continuous = not china_futures_config.enforce_sessions
    engine.set_force_session_continuous(force_session_continuous)
    # 无论使用 SimpleMarket 还是 ChinaMarket，都尝试使用统一佣金策略接口。
    if hasattr(engine, "set_stock_fee_policy"):
        cast(Any, engine).set_stock_fee_policy(
            commission_policy["type"],
            float(commission_policy["value"]),
            stamp_tax_rate,
            transfer_fee_rate,
            min_commission,
        )
    else:
        engine.set_stock_fee_rules(
            commission_rate, stamp_tax_rate, transfer_fee_rate, min_commission
        )

    # Configure Execution parameters
    if (
        normalized_global_slippage["type"] != "zero"
        and normalized_global_slippage["value"] > 0
    ):
        if hasattr(engine, "set_slippage"):
            engine.set_slippage(
                normalized_global_slippage["type"],
                normalized_global_slippage["value"],
            )
        else:
            logger.warning(
                "Slippage policy %s set but not supported by Engine.",
                normalized_global_slippage,
            )

    if volume_limit_pct != 0.25:
        if hasattr(engine, "set_volume_limit"):
            engine.set_volume_limit(volume_limit_pct)
        else:
            logger.warning(
                f"Volume limit {volume_limit_pct} set but not supported by Engine."
            )

    # Configure other asset fees if provided
    if "fund_commission" in kwargs:
        engine.set_fund_fee_rules(
            kwargs["fund_commission"],
            kwargs.get("fund_transfer_fee", 0.0),
            kwargs.get("fund_min_commission", 0.0),
        )

    if china_options_config and has_options_instruments:
        if china_options_config.fee_per_contract is not None:
            engine.set_option_fee_rules(china_options_config.fee_per_contract)
    elif "option_commission" in kwargs:
        engine.set_option_fee_rules(kwargs["option_commission"])

    if china_futures_config and has_futures_instruments:
        template_validation_by_prefix: Dict[
            str, Tuple[Optional[bool], Optional[bool]]
        ] = {}
        template_fee_by_prefix: Dict[str, float] = {}
        if china_futures_config.instrument_templates_by_symbol_prefix:
            for template in china_futures_config.instrument_templates_by_symbol_prefix:
                prefix = template.symbol_prefix.strip().upper()
                if not prefix:
                    continue
                if template.commission_rate is not None:
                    template_fee_by_prefix[prefix] = float(template.commission_rate)
                if (
                    template.enforce_tick_size is not None
                    or template.enforce_lot_size is not None
                ):
                    template_validation_by_prefix[prefix] = (
                        template.enforce_tick_size,
                        template.enforce_lot_size,
                    )

        if hasattr(engine, "set_futures_validation_options"):
            cast(Any, engine).set_futures_validation_options(
                bool(china_futures_config.enforce_tick_size),
                bool(china_futures_config.enforce_lot_size),
            )
        else:
            logger.warning(
                "set_futures_validation_options is not available "
                "in current engine binary"
            )
        merged_validation_by_prefix = dict(template_validation_by_prefix)
        if china_futures_config.validation_by_symbol_prefix:
            for validation_rule in china_futures_config.validation_by_symbol_prefix:
                prefix = validation_rule.symbol_prefix.strip().upper()
                if not prefix:
                    continue
                merged_validation_by_prefix[prefix] = (
                    validation_rule.enforce_tick_size,
                    validation_rule.enforce_lot_size,
                )
        if merged_validation_by_prefix:
            for prefix, (tick_opt, lot_opt) in merged_validation_by_prefix.items():
                if hasattr(engine, "set_futures_validation_options_by_prefix"):
                    cast(Any, engine).set_futures_validation_options_by_prefix(
                        prefix,
                        tick_opt,
                        lot_opt,
                    )
                else:
                    logger.warning(
                        "set_futures_validation_options_by_prefix is not available "
                        "in current engine binary"
                    )
                    break

        merged_fee_by_prefix = dict(template_fee_by_prefix)
        if china_futures_config.fee_by_symbol_prefix:
            for fee_rule in china_futures_config.fee_by_symbol_prefix:
                prefix = fee_rule.symbol_prefix.strip().upper()
                if not prefix:
                    continue
                merged_fee_by_prefix[prefix] = float(fee_rule.commission_rate)
        if merged_fee_by_prefix:
            for prefix, commission_rate_value in merged_fee_by_prefix.items():
                if hasattr(engine, "set_futures_fee_rules_by_prefix"):
                    cast(Any, engine).set_futures_fee_rules_by_prefix(
                        prefix,
                        commission_rate_value,
                    )
                else:
                    logger.warning(
                        "set_futures_fee_rules_by_prefix is not available "
                        "in current engine binary"
                    )
                    break
        if china_futures_config.sessions:
            session_ranges: List[Tuple[str, str, TradingSession]] = []
            for session_rule in china_futures_config.sessions:
                session_ranges.append(
                    (
                        session_rule.start,
                        session_rule.end,
                        _parse_trading_session(session_rule.session),
                    )
                )
            if session_ranges:
                engine.set_market_sessions(session_ranges)
        elif china_futures_config.enforce_sessions:
            session_ranges = []
            for start, end, session_name in _china_futures_session_template(
                china_futures_config.session_profile
            ):
                session_ranges.append(
                    (
                        start,
                        end,
                        _parse_trading_session(session_name),
                    )
                )
            if session_ranges:
                engine.set_market_sessions(session_ranges)

    if china_stock_config is not None:
        if hasattr(engine, "set_stock_validation_options"):
            cast(Any, engine).set_stock_validation_options(
                bool(china_stock_config.enforce_tick_size),
            )
        else:
            logger.warning(
                "set_stock_validation_options is not available in current engine binary"
            )

    if china_options_config and has_options_instruments:
        if china_options_config.fee_by_symbol_prefix:
            for option_fee_rule in china_options_config.fee_by_symbol_prefix:
                prefix = option_fee_rule.symbol_prefix.strip().upper()
                if not prefix:
                    continue
                if hasattr(engine, "set_options_fee_rules_by_prefix"):
                    cast(Any, engine).set_options_fee_rules_by_prefix(
                        prefix,
                        float(option_fee_rule.commission_per_contract),
                    )
                else:
                    logger.warning(
                        "set_options_fee_rules_by_prefix is not available "
                        "in current engine binary",
                        extra=_build_backtest_log_extra(
                            phase="risk",
                            strategy_id=effective_strategy_id,
                            slot=effective_strategy_id,
                        ),
                    )
                    break
        if china_options_config.sessions:
            option_session_ranges: List[Tuple[str, str, TradingSession]] = []
            for option_session_rule in china_options_config.sessions:
                option_session_ranges.append(
                    (
                        option_session_rule.start,
                        option_session_rule.end,
                        _parse_trading_session(option_session_rule.session),
                    )
                )
            if option_session_ranges and not (
                china_futures_config
                and has_futures_instruments
                and china_futures_config.sessions
            ):
                engine.set_market_sessions(option_session_ranges)

    # Apply Risk Config

    # 1. Start with config from BacktestConfig
    current_risk_config: Optional[RiskConfig] = None
    if config and config.strategy_config and config.strategy_config.risk:
        current_risk_config = config.strategy_config.risk

    # 2. If risk_config (dict or object) is provided, merge/override
    if risk_config:
        if current_risk_config is None:
            current_risk_config = RiskConfig()

        if isinstance(risk_config, RiskConfig):
            # If explicit RiskConfig object provided, it takes precedence over
            # partial fields?
            # Or should we merge?
            # Strategy: If it's a full object, use it as base, but this might discard
            # config.risk
            # Better strategy: Copy attributes from override to current
            for field in risk_config.__dataclass_fields__:
                val = getattr(risk_config, field)
                if val is not None:
                    setattr(current_risk_config, field, val)
        elif isinstance(risk_config, dict):
            # Update fields from dict
            for k, v in risk_config.items():
                if hasattr(current_risk_config, k):
                    setattr(current_risk_config, k, v)
                else:
                    logger.warning(
                        "Unknown risk config key: %s",
                        k,
                        extra=_build_backtest_log_extra(
                            phase="risk",
                            strategy_id=effective_strategy_id,
                            slot=effective_strategy_id,
                        ),
                    )

    # 3. Apply if exists
    if current_risk_config:
        apply_risk_config(engine, current_risk_config)

    # Get current manager
    rm = engine.risk_manager
    engine.risk_manager = rm

    # 5. 添加标的
    # 解析 Instrument Config
    inst_conf_map = {}

    # Handle explicit Instrument objects passed via kwargs
    prebuilt_instruments = {}
    if "instruments" in kwargs:
        obs = kwargs["instruments"]
        if isinstance(obs, list):
            for o in obs:
                prebuilt_instruments[o.symbol] = o
        elif isinstance(obs, dict):
            prebuilt_instruments.update(obs)

    # From BacktestConfig
    if config and config.instruments_config:
        if isinstance(config.instruments_config, list):
            for c in config.instruments_config:
                if c.symbol not in inst_conf_map:
                    inst_conf_map[c.symbol] = c
        elif isinstance(config.instruments_config, dict):
            for k, v in config.instruments_config.items():
                if k not in inst_conf_map:
                    inst_conf_map[k] = v

    # Default values from kwargs
    default_multiplier = kwargs.get("multiplier", 1.0)
    default_margin_ratio = kwargs.get("margin_ratio", 1.0)
    default_asset_type = kwargs.get("asset_type", AssetType.Stock)
    # tick_size 缺省值按 asset_type 分流(基金 0.001, 其余 0.01), 显式传入的
    # tick_size 仍优先——见 _default_tick_size_for_asset_type。
    default_tick_size = kwargs.get(
        "tick_size", _default_tick_size_for_asset_type(default_asset_type)
    )

    # Option specific fields
    default_option_type = kwargs.get("option_type", None)
    default_option_margin_model = kwargs.get("option_margin_model", None)
    default_strike_price = kwargs.get("strike_price", None)
    default_expiry_date = _normalize_expiry_date_yyyymmdd(
        kwargs.get("expiry_date", None)
    )
    default_implied_volatility = kwargs.get("implied_volatility", None)
    default_reference_volatility = kwargs.get("reference_volatility", None)
    default_settlement_type = kwargs.get("settlement_type", None)
    default_settlement_price = kwargs.get("settlement_price", None)

    def _parse_asset_type(val: Union[str, AssetType]) -> AssetType:
        if isinstance(val, AssetType):
            return val
        if isinstance(val, str):
            v_lower = val.strip().lower()
            if v_lower == "stock":
                return AssetType.Stock
            if v_lower in {"future", "futures"}:
                return AssetType.Futures
            if v_lower == "fund":
                return AssetType.Fund
            if v_lower == "option":
                return AssetType.Option
        raise ValueError(f"Unsupported asset_type: {val}")

    def _parse_option_type(val: Any) -> Any:
        if val is None:
            return None
        # OptionType might not be available in current binary
        try:
            from ..akquant import OptionType  # type: ignore

            if isinstance(val, str):
                if val.lower() == "call":
                    return OptionType.Call
                if val.lower() == "put":
                    return OptionType.Put
                raise ValueError(f"Unsupported option_type: {val}")
            if str(val).endswith(".Call"):
                return OptionType.Call
            if str(val).endswith(".Put"):
                return OptionType.Put
        except ImportError:
            pass
        if isinstance(val, str):
            v = val.strip().upper()
            if v in {"CALL", "PUT"}:
                return v
            raise ValueError(f"Unsupported option_type: {val}")
        return val

    def _parse_option_margin_model(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, OptionMarginModel):
            return val
        if isinstance(val, str):
            s = val.upper()
            if s == "RATIO":
                return OptionMarginModel.Ratio
            if s == "CHINA_SINGLE_LEG":
                return OptionMarginModel.ChinaSingleLeg
            if s == "US_BROKER_SINGLE_LEG":
                return OptionMarginModel.USBrokerSingleLeg
            if s == "US_BROKER_SINGLE_LEG_VOL_ADJUSTED":
                return OptionMarginModel.USBrokerSingleLegVolAdjusted
        raise ValueError(f"Unsupported option_margin_model: {val}")

    def _parse_settlement_type(
        val: Any,
    ) -> Tuple[Any, Optional[InstrumentSettlementMode]]:
        if val is None:
            return None, None
        if isinstance(val, SettlementType):
            if val == SettlementType.Physical:
                raise ValueError("Unsupported settlement_type: Physical")
            return val, _settlement_type_to_upper_name(val)
        if isinstance(val, str):
            key = val.strip().lower()
            if key in {"cash", "cash_last_price"}:
                return SettlementType.Cash, "CASH"
            if key in {"settlement_price", "cash_settlement_price"}:
                return SettlementType.Cash, "SETTLEMENT_PRICE"
            if key in {"force_close", "forceclose"}:
                return SettlementType.ForceClose, "FORCE_CLOSE"
            raise ValueError(f"Unsupported settlement_type: {val}")
        raise TypeError("settlement_type must be SettlementType or str")

    def _match_futures_template(
        symbol: str,
    ) -> Optional[ChinaFuturesInstrumentTemplateConfig]:
        if (
            china_futures_config is None
            or not china_futures_config.instrument_templates_by_symbol_prefix
        ):
            return None
        symbol_upper = symbol.upper()
        best_template: Optional[ChinaFuturesInstrumentTemplateConfig] = None
        best_len = 0
        for tpl in china_futures_config.instrument_templates_by_symbol_prefix:
            prefix = tpl.symbol_prefix.strip().upper()
            if not prefix:
                continue
            if symbol_upper.startswith(prefix) and len(prefix) > best_len:
                best_template = tpl
                best_len = len(prefix)
        return best_template

    instrument_snapshots: Dict[str, InstrumentSnapshot] = {}

    for sym in symbols:
        # Priority: Pre-built Instrument > Config > Default
        if sym in prebuilt_instruments:
            prebuilt = prebuilt_instruments[sym]
            engine.add_instrument(prebuilt)
            instrument_snapshots[sym] = InstrumentSnapshot(
                symbol=sym,
                asset_type=_asset_type_to_upper_name(
                    getattr(prebuilt, "asset_type", "")
                ),
                multiplier=float(getattr(prebuilt, "multiplier", 1.0)),
                margin_ratio=float(getattr(prebuilt, "margin_ratio", 1.0)),
                option_margin_model=_option_margin_model_to_upper_name(
                    getattr(prebuilt, "option_margin_model", None)
                ),
                tick_size=float(
                    getattr(prebuilt, "tick_size", None)
                    or _default_tick_size_for_asset_type(
                        getattr(prebuilt, "asset_type", "")
                    )
                ),
                lot_size=float(getattr(prebuilt, "lot_size", 1.0) or 1.0),
                implied_volatility=(
                    float(getattr(prebuilt, "implied_volatility"))
                    if getattr(prebuilt, "implied_volatility", None) is not None
                    else None
                ),
                reference_volatility=(
                    float(getattr(prebuilt, "reference_volatility"))
                    if getattr(prebuilt, "reference_volatility", None) is not None
                    else None
                ),
                settlement_type=_settlement_type_to_upper_name(
                    getattr(prebuilt, "settlement_type", None)
                ),
                settlement_price=(
                    float(getattr(prebuilt, "settlement_price"))
                    if getattr(prebuilt, "settlement_price", None) is not None
                    else None
                ),
            )
            continue

        # Determine lot_size for this symbol
        current_lot_size = None
        if isinstance(lot_size, int):
            current_lot_size = lot_size
        elif isinstance(lot_size, dict):
            current_lot_size = lot_size.get(sym)

        # Check specific config
        i_conf = inst_conf_map.get(sym)
        futures_template = _match_futures_template(sym)

        if i_conf:
            p_asset_type = _parse_asset_type(i_conf.asset_type)
            p_multiplier = i_conf.multiplier
            p_margin = i_conf.margin_ratio
            # InstrumentConfig.__post_init__ always fills tick_size in with an
            # asset-type-dependent default, so this is defensive rather than
            # load-bearing at runtime — it exists to keep the static type
            # non-Optional for the Instrument() call below.
            p_tick = (
                i_conf.tick_size if i_conf.tick_size is not None else default_tick_size
            )
            # If config has lot_size, use it, otherwise use global setting
            p_lot = (
                i_conf.lot_size
                if i_conf.lot_size is not None
                else float(current_lot_size or 1.0)
            )
            if futures_template and p_asset_type == AssetType.Futures:
                if i_conf.multiplier == 1 and futures_template.multiplier is not None:
                    p_multiplier = futures_template.multiplier
                if (
                    i_conf.margin_ratio == 1
                    and futures_template.margin_ratio is not None
                ):
                    p_margin = futures_template.margin_ratio
                if i_conf.tick_size == 0.01 and futures_template.tick_size is not None:
                    p_tick = futures_template.tick_size
                if i_conf.lot_size is None and futures_template.lot_size is not None:
                    p_lot = futures_template.lot_size

            p_opt_type = _parse_option_type(i_conf.option_type)
            p_option_margin_model = _parse_option_margin_model(
                i_conf.option_margin_model
                if i_conf.option_margin_model is not None
                else ("CHINA_SINGLE_LEG" if p_asset_type == AssetType.Option else None)
            )
            p_strike = i_conf.strike_price
            p_expiry = _normalize_expiry_date_yyyymmdd(i_conf.expiry_date)
            p_underlying = i_conf.underlying_symbol
            p_implied_volatility = i_conf.implied_volatility
            p_reference_volatility = i_conf.reference_volatility
            p_settlement_type, p_settlement_mode = _parse_settlement_type(
                i_conf.settlement_type
            )
            p_settlement_price = i_conf.settlement_price
            p_sellable_after_days = (
                i_conf.sellable_after_days
                if i_conf.sellable_after_days is not None
                else (1 if t_plus_one else 0)
            )
            static_attrs = getattr(i_conf, "static_attrs", {})
            if static_attrs is None:
                static_attrs = {}
            if not isinstance(static_attrs, dict):
                raise TypeError(
                    "InstrumentConfig.static_attrs must be Dict[str, scalar]"
                )
        else:
            if futures_template:
                p_asset_type = AssetType.Futures
                p_multiplier = (
                    futures_template.multiplier
                    if futures_template.multiplier is not None
                    else default_multiplier
                )
                p_margin = (
                    futures_template.margin_ratio
                    if futures_template.margin_ratio is not None
                    else default_margin_ratio
                )
                p_tick = (
                    futures_template.tick_size
                    if futures_template.tick_size is not None
                    else default_tick_size
                )
                p_lot = (
                    futures_template.lot_size
                    if futures_template.lot_size is not None
                    else float(current_lot_size or 1.0)
                )
            else:
                p_asset_type = default_asset_type
                p_multiplier = default_multiplier
                p_margin = default_margin_ratio
                p_tick = default_tick_size
                p_lot = float(current_lot_size or 1.0)

            p_opt_type = default_option_type
            p_option_margin_model = _parse_option_margin_model(
                default_option_margin_model
                if default_option_margin_model is not None
                else ("CHINA_SINGLE_LEG" if p_asset_type == AssetType.Option else None)
            )
            p_strike = default_strike_price
            p_expiry = default_expiry_date
            p_underlying = None
            p_implied_volatility = default_implied_volatility
            p_reference_volatility = default_reference_volatility
            p_settlement_type, p_settlement_mode = _parse_settlement_type(
                default_settlement_type
            )
            p_settlement_price = default_settlement_price
            p_sellable_after_days = 1 if t_plus_one else 0
            static_attrs = {}

        if p_asset_type != AssetType.Futures:
            p_settlement_type = None
            p_settlement_mode = None
            p_settlement_price = None
        if p_asset_type != AssetType.Option:
            p_option_margin_model = None
            p_implied_volatility = None
            p_reference_volatility = None
        if (
            p_settlement_mode == "SETTLEMENT_PRICE"
            and p_settlement_price is None
            and p_asset_type == AssetType.Futures
        ):
            raise ValueError(
                "settlement_price is required for "
                f"settlement_type=settlement_price ({sym})"
            )

        # Validate types before passing to Rust
        if p_lot is not None and not isinstance(p_lot, (int, float)):
            p_lot = 1.0  # Fallback

        # Ensure lot is float for Rust binding if expected
        p_lot_f: float = float(p_lot)

        instr = Instrument(
            sym,
            p_asset_type,
            p_multiplier,
            p_margin,
            p_tick,
            p_opt_type,
            p_strike,
            p_expiry,
            p_lot_f,
            p_underlying,
            p_settlement_type,
            p_settlement_price,
            p_option_margin_model,
            p_implied_volatility,
            p_reference_volatility,
            p_sellable_after_days,
        )
        engine.add_instrument(instr)
        instrument_snapshots[sym] = InstrumentSnapshot(
            symbol=sym,
            asset_type=_asset_type_to_upper_name(p_asset_type),
            multiplier=float(p_multiplier),
            margin_ratio=float(p_margin),
            option_margin_model=_option_margin_model_to_upper_name(
                p_option_margin_model
            ),
            tick_size=float(p_tick),
            lot_size=float(p_lot_f),
            option_type=_option_type_to_upper_name(p_opt_type),
            strike_price=float(p_strike) if p_strike is not None else None,
            expiry_date=p_expiry,
            underlying_symbol=str(p_underlying) if p_underlying is not None else None,
            implied_volatility=(
                float(p_implied_volatility)
                if p_implied_volatility is not None
                else None
            ),
            reference_volatility=(
                float(p_reference_volatility)
                if p_reference_volatility is not None
                else None
            ),
            settlement_type=p_settlement_mode,
            settlement_price=(
                float(p_settlement_price) if p_settlement_price is not None else None
            ),
            static_attrs=dict(static_attrs),
        )

    for current_strategy in all_strategy_instances:
        current_strategy._set_instrument_snapshots(instrument_snapshots)

    # 被前置过滤掉的标的只发一条汇总日志, 不逐个刷屏 —— 传全市场数据只关心
    # 几个标的是本变更的主要动机场景, 逐标的告警会淹没输出。
    if filtered_out_symbols:
        logger.info(
            "已过滤 %d 个不在 symbols 中的标的: %s",
            len(filtered_out_symbols),
            ", ".join(sorted(filtered_out_symbols)),
        )

    # 运行前比对: symbols 里有没有标的实际数据集合里压根没有(能枚举的形态才行——
    # data_map_for_indicators 为空时说明走的是 DataFeed 对象输入, 它只写不读,
    # Python 无从枚举, 只能留给会话末 Strategy._check_symbol_data_coverage 兜底)。
    # 与 filtered_out_symbols(用户主动排除)、adapter 泄漏 warning(数据里有、但
    # 未被请求)是三个不相交的集合——这里是"白名单里有、但数据里没有"。
    symbol_data_missing: set[str] = set()
    # 用 `data_map_for_indicators or catalog_missing_symbols` 而非只看
    # `data_map_for_indicators`: Catalog 路径下若**全部**请求标的都读取失败,
    # `data_map_for_indicators` 会是空 dict(falsy), 导致这段整体被跳过——
    # 下面"跳过已经报过英文 warning 的标的"与"记入 _symbol_data_warned 防止
    # 会话末重复告警"两步都不会执行, 于是 catalog_missing_symbols 里的标的会在
    # 会话末被 `_check_symbol_data_coverage` 再报一次中文告警, 与运行前的英文
    # warning 重复。全空只是"部分读不到"的极限场景, 去重逻辑理应对称覆盖。
    if symbols_explicit and (data_map_for_indicators or catalog_missing_symbols):
        symbol_data_missing = set(symbols) - set(data_map_for_indicators.keys())
        # Catalog 路径逐标的读取失败已经报过一条英文 warning(:3464 附近), 不再
        # 对这些标的重复报中文告警——但仍要记入 _symbol_data_warned(见下方
        # 整体 update), 否则会话末 _check_symbol_data_coverage 会再报第二次。
        for sym in sorted(symbol_data_missing - catalog_missing_symbols):
            logger.warning(
                "标的 %s 在 symbols 中但数据里没有它: 该标的全程不会有任何行情"
                "事件。常见原因是标的代码写错、数据源未覆盖、或所选时间范围内"
                "无交易。",
                sym,
            )
        if symbol_data_missing:
            strategy_instance._symbol_data_warned = set(symbol_data_missing)
            for slot_strategy in slot_strategy_instances.values():
                slot_strategy._symbol_data_warned = set(symbol_data_missing)

    _warn_unmatched_instrument_config(config, data_map_for_indicators, logger)

    # symbols 白名单: 只在用户显式传了 symbols 时启用。
    # 不得直接用局部变量 `symbols`——它在上面的数据加载分支(DataFrame/dict/
    # adapter, :3253/:3334/:3338/:3389 附近)里会被数据里检测到的标的反向
    # 改写/追加, 一旦发生, 这里下发的就不再是用户请求的白名单而是"用户请求 ∪
    # 数据里出现过的标的"——对 Rust 层形同没有过滤, 复活了这个改动本要消灭的
    # 缺陷(未列入白名单的标的照样撮合、套用默认合约参数)。改用干净的合并结果:
    # 重新调用 _merge_symbol_whitelist_sources(与 run_from_checkpoint 的
    # 对应下发点一致, 见 :6098 附近), 从未被污染的 effective_symbols 出发合并
    # config.instruments 与各策略当前的 _subscriptions(此时 on_start 已跑完,
    # 能看到 on_start 里新增的订阅)。
    if symbols_explicit:
        engine.set_symbol_whitelist(
            _merge_symbol_whitelist_sources(
                effective_symbols,
                config,
                strategy_instance,
                *slot_strategy_instances.values(),
            )
        )

    # 6. 添加数据
    engine.add_data(feed)

    # 7. 运行回测
    logger.info("Running backtest via run_backtest()...")

    if effective_depth > 0:
        for current_strategy in all_strategy_instances:
            current_strategy.set_history_depth(effective_depth)

    # 7.5 Prepare Indicators (Precompute mode only)
    if data_map_for_indicators:
        for current_strategy in all_strategy_instances:
            if _should_prepare_precomputed_indicators(current_strategy) and hasattr(
                current_strategy, "_prepare_indicators"
            ):
                current_strategy._prepare_indicators(data_map_for_indicators)
            if hasattr(current_strategy, "_bootstrap_incremental_indicators"):
                current_strategy._bootstrap_incremental_indicators(
                    data_map_for_indicators
                )

    engine_summary: str = ""
    try:
        engine_summary = str(engine.run(strategy_instance, show_progress))
    except Exception as e:
        logger.error(
            "Backtest failed: %s",
            e,
            extra=_build_backtest_log_extra(
                phase="backtest",
                strategy_id=effective_strategy_id,
                slot=effective_strategy_id,
            ),
        )
        raise e
    finally:
        if stream_on_event is not None and hasattr(engine, "clear_stream_callback"):
            try:
                cast(Any, engine).clear_stream_callback()
            except Exception as e:
                logger.debug(f"Failed to clear stream callback: {e}")
        if hasattr(strategy_instance, "_on_stop_internal"):
            try:
                strategy_instance._on_stop_internal()
            except StrategyConfigurationError:
                # 框架配置校验失败必须让调用方看到: 吞掉它会让用户拿到静默失效的结果
                # (例如 H/L 指标全程未更新、ATR 恒为 0)。用户 on_stop 的 bug 仍按下面
                # 的分支容忍。
                raise
            except Exception as e:
                logger.error(
                    "Error in on_stop: %s",
                    e,
                    extra=_build_backtest_log_extra(
                        phase="strategy",
                        strategy=strategy_instance,
                    ),
                )
        elif hasattr(strategy_instance, "on_stop"):
            try:
                strategy_instance.on_stop()
            except Exception as e:
                logger.error(
                    "Error in on_stop: %s",
                    e,
                    extra=_build_backtest_log_extra(
                        phase="strategy",
                        strategy=strategy_instance,
                    ),
                )
        for slot_strategy in slot_strategy_instances.values():
            if hasattr(slot_strategy, "_on_stop_internal"):
                try:
                    slot_strategy._on_stop_internal()
                except StrategyConfigurationError:
                    # 框架配置校验失败必须让调用方看到, 理由同主策略分支。
                    raise
                except Exception as e:
                    logger.error(
                        "Error in slot on_stop: %s",
                        e,
                        extra=_build_backtest_log_extra(
                            phase="strategy",
                            strategy=slot_strategy,
                        ),
                    )
            elif hasattr(slot_strategy, "on_stop"):
                try:
                    slot_strategy.on_stop()
                except Exception as e:
                    logger.error(
                        "Error in slot on_stop: %s",
                        e,
                        extra=_build_backtest_log_extra(
                            phase="strategy",
                            strategy=slot_strategy,
                        ),
                    )

    result = BacktestResult(
        engine.get_results(),
        timezone=timezone,
        initial_cash=initial_cash,
        strategy=strategy_instance,
        engine=engine,
        indicator_outputs=indicator_recorder.build_payload(),
    )
    _attach_result_runtime_metadata(
        result=result,
        engine_summary=engine_summary,
        event_stats_snapshot=event_stats_snapshot,
        owner_strategy_id=effective_strategy_id,
        resolved_policy=resolved_policy,
    )
    # 挂载已解析运行时配置，作为快照持久化的单一来源 (issue #282)
    result.resolved_config = _build_resolved_backtest_config(
        slippage_policy=normalized_global_slippage,
        volume_limit_pct=volume_limit_pct,
        commission_policy=commission_policy,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
        min_commission=min_commission,
        t_plus_one=t_plus_one,
        timezone=timezone,
        history_depth=history_depth,
        resolved_policy=resolved_policy,
    )
    analyzer_outputs: Dict[str, Dict[str, Any]] = {}
    if analyzer_manager.plugins:
        try:
            analyzer_outputs = analyzer_manager.on_finish(
                {
                    "engine": engine,
                    "strategy": strategy_instance,
                    "strategies": list(all_strategy_instances),
                    "slot_strategy_map": {
                        effective_strategy_id: strategy_instance,
                        **slot_strategy_instances,
                    },
                    "result": result,
                }
            )
        except Exception as e:
            logger.error(f"Analyzer on_finish error: {e}")
    result.analyzer_outputs = analyzer_outputs
    return result


def run_from_checkpoint(
    checkpoint_path: str,
    data: Optional[BacktestDataInput] = None,
    show_progress: bool = True,
    symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]] = None,
    commission_policy: Optional[CommissionPolicy] = None,
    strategy_runtime_config: Optional[
        Union[StrategyRuntimeConfig, Dict[str, Any]]
    ] = None,
    runtime_config_override: bool = True,
    strategy_id: Optional[str] = None,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ] = None,
    strategy_max_order_value: Optional[Dict[str, float]] = None,
    strategy_max_order_size: Optional[Dict[str, float]] = None,
    strategy_max_position_size: Optional[Dict[str, float]] = None,
    strategy_max_daily_loss: Optional[Dict[str, float]] = None,
    strategy_max_drawdown: Optional[Dict[str, float]] = None,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = None,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = None,
    strategy_priority: Optional[Dict[str, int]] = None,
    strategy_risk_budget: Optional[Dict[str, float]] = None,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = None,
    strategy_slippage: Optional[Dict[str, SlippageInput]] = None,
    strategy_commission: Optional[Dict[str, CommissionPolicy]] = None,
    portfolio_risk_budget: Optional[float] = None,
    risk_budget_mode: str = "order_notional",
    risk_budget_reset_daily: bool = False,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = None,
    indicator_recorder: Optional[IndicatorSink] = None,
    config: Optional[BacktestConfig] = None,
    **kwargs: Any,
) -> BacktestResult:
    """
    从检查点续跑回测 (Resume Backtest from Checkpoint).

    注意：当前 run_from_checkpoint 的策略实例来自 checkpoint 恢复，
    不会通过 strategy_source / strategy_loader 重新加载策略类。
    如需替换策略实现，请优先使用 run_backtest 或在恢复后通过
    strategies_by_slot 覆盖 slot 策略。

    故障速查可参考 docs/zh/advanced/runtime_config.md，
    英文文档参考 docs/en/advanced/runtime_config.md

    :param kwargs: 其他引擎配置参数 (如 commission_rate, stamp_tax_rate, t_plus_one)
    """
    import os

    from ..checkpoint import load_checkpoint

    logger = get_logger("backtest")
    if not has_configured_handler(logger.name, namespace_only=True):
        register_logger(console=True, level="INFO")
        logger = get_logger("backtest")
    strategy_config = config.strategy_config if config is not None else None
    (
        strategy_id,
        strategies_by_slot,
        strategy_max_order_value,
        strategy_max_order_size,
        strategy_max_position_size,
        strategy_max_daily_loss,
        strategy_max_drawdown,
        strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars,
        strategy_priority,
        strategy_risk_budget,
        strategy_fill_policy,
        strategy_slippage,
        strategy_commission,
        portfolio_risk_budget,
        strategy_runtime_config,
        _ignored_strategy_source,
        _ignored_strategy_loader,
        _ignored_strategy_loader_options,
    ) = _apply_strategy_config_overrides(
        strategy_config=strategy_config,
        strategy_id=strategy_id,
        strategies_by_slot=strategies_by_slot,
        strategy_max_order_value=strategy_max_order_value,
        strategy_max_order_size=strategy_max_order_size,
        strategy_max_position_size=strategy_max_position_size,
        strategy_max_daily_loss=strategy_max_daily_loss,
        strategy_max_drawdown=strategy_max_drawdown,
        strategy_reduce_only_after_risk=strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars=strategy_risk_cooldown_bars,
        strategy_priority=strategy_priority,
        strategy_risk_budget=strategy_risk_budget,
        strategy_fill_policy=strategy_fill_policy,
        strategy_slippage=strategy_slippage,
        strategy_commission=strategy_commission,
        portfolio_risk_budget=portfolio_risk_budget,
        strategy_runtime_config=strategy_runtime_config,
        strategy_source=None,
        strategy_loader=None,
        strategy_loader_options=None,
    )
    portfolio_risk_budget, risk_budget_mode = _validate_strategy_risk_inputs(
        strategies_by_slot=strategies_by_slot,
        strategy_max_order_value=strategy_max_order_value,
        strategy_max_order_size=strategy_max_order_size,
        strategy_max_position_size=strategy_max_position_size,
        strategy_max_daily_loss=strategy_max_daily_loss,
        strategy_max_drawdown=strategy_max_drawdown,
        strategy_reduce_only_after_risk=strategy_reduce_only_after_risk,
        strategy_risk_cooldown_bars=strategy_risk_cooldown_bars,
        strategy_priority=strategy_priority,
        strategy_risk_budget=strategy_risk_budget,
        portfolio_risk_budget=portfolio_risk_budget,
        risk_budget_mode=risk_budget_mode,
    )
    risk_budget_reset_daily = bool(risk_budget_reset_daily)
    indicator_stream_requested = (
        on_event is not None or kwargs.get("_stream_on_event") is not None
    )
    prepared_stream_runtime = _prepare_stream_runtime(on_event=on_event, kwargs=kwargs)
    stream_on_event = prepared_stream_runtime.stream_on_event
    indicator_stream_emitter = (
        prepared_stream_runtime.indicator_stream_emitter
        if indicator_stream_requested
        else None
    )
    event_stats_snapshot = prepared_stream_runtime.event_stats_snapshot
    stream_progress_interval = prepared_stream_runtime.stream_progress_interval
    stream_equity_interval = prepared_stream_runtime.stream_equity_interval
    stream_batch_size = prepared_stream_runtime.stream_batch_size
    stream_max_buffer = prepared_stream_runtime.stream_max_buffer
    stream_error_mode = prepared_stream_runtime.stream_error_mode
    stream_mode = prepared_stream_runtime.stream_mode
    legacy_mode_override = "execution_mode" in kwargs
    legacy_timer_override = "timer_execution_policy" in kwargs
    has_fill_policy_override = "fill_policy" in kwargs
    _raise_if_legacy_execution_policy_used(
        legacy_mode_used=legacy_mode_override,
        legacy_timer_used=legacy_timer_override,
        api_name="run_from_checkpoint",
    )
    fill_policy_override = cast(Optional[FillPolicy], kwargs.pop("fill_policy", None))
    # 显式 fill_policy 覆盖与 run_backtest 对称：接 FillMode、拒 dict（硬切断）。
    # 注意：这里只处理"用户显式入参"，不影响下方 restored_backtest_config 快照分支
    # ——那是 checkpoint 文件里的内部序列化 dict，不是公共 API 输入，须原样透传。
    if fill_policy_override is not None:
        if isinstance(fill_policy_override, FillMode):
            _pb, _bo, _tp = fill_policy_override._to_core()
            fill_policy_override = {
                "price_basis": _pb,
                "bar_offset": _bo,
                "temporal": _tp,
            }
        elif isinstance(fill_policy_override, dict):
            raise TypeError(_LEGACY_FILL_POLICY_DICT_MSG)
    timezone_name = str(kwargs.get("timezone") or "Asia/Shanghai")
    symbols, effective_symbols, symbols_explicit = _resolve_effective_symbols(
        symbols=symbols,
        kwargs=kwargs,
        api_name="run_from_checkpoint",
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. 准备数据源
    feed = None
    data_map_for_indicators: Dict[str, pd.DataFrame] = {}
    # 与 run_backtest 对称(见该函数 :3150 附近同名注释): 记录被 symbols 白名单
    # 挡在 Python 前置过滤之外的标的, 供下面(:6132 附近)补发与 run_backtest
    # 一致的汇总 INFO 日志——两条入口的可观测性不应该有差异。
    filtered_out_symbols: set[str] = set()

    # polars / pyarrow 输入统一转 pandas, 复用既有数据路径(issue #298)
    data = coerce_to_pandas(data)

    if isinstance(data, DataFeed):
        feed = data
    elif _is_data_feed_adapter(data):
        feed = DataFeed()
        adapter_data_map, leaked_adapter_symbols = _load_data_map_from_adapter(
            adapter=data,
            symbols=list(effective_symbols),
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            timezone=timezone_name,
        )
        if leaked_adapter_symbols:
            logger.warning(
                "DataFeedAdapter 返回了 %d 个未请求的标的, 已丢弃: %s",
                len(leaked_adapter_symbols),
                ", ".join(sorted(leaked_adapter_symbols)),
            )
        loaded_count = 0
        for sym, df in adapter_data_map.items():
            if not df.empty:
                df_prep = to_indicator_frame(df)
                data_map_for_indicators[sym] = df_prep
                arrays = dataframe_to_arrays(df_prep, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                loaded_count += 1
        if loaded_count > 0:
            feed.sort()
    elif data is not None:
        # Convert DataFrame/List to DataFeed
        feed = DataFeed()
        symbols = list(effective_symbols)

        data_map = {}
        # Copied logic from run_backtest for data loading
        if isinstance(data, pd.DataFrame):
            df_input = data
            if not isinstance(df_input.index, pd.DatetimeIndex):
                found_date = False
                for col in ["date", "timestamp", "datetime", "Date", "Timestamp"]:
                    if col in df_input.columns:
                        df_input = df_input.set_index(col)
                        found_date = True
                        break

                if not found_date:
                    try:
                        df_input.index = pd.to_datetime(df_input.index)
                    except Exception:
                        pass

            # 非 DatetimeIndex 索引统一转 datetime(兼容 pandas 2.x str dtype)
            if not isinstance(df_input.index, pd.DatetimeIndex):
                try:
                    df_input.index = pd.to_datetime(df_input.index)
                except Exception:
                    pass

            if isinstance(df_input.index, pd.DatetimeIndex):
                df_input = _filter_datetime_index_frame_by_runtime_window(
                    df_input,
                    kwargs.get("start_time"),
                    kwargs.get("end_time"),
                    timezone_name,
                )

            df_prepared = to_indicator_frame(df_input)
            if "symbol" in df_prepared.columns:
                df_prepared = df_prepared.copy()
                df_prepared["symbol"] = df_prepared["symbol"].astype(str)
                filter_symbols = bool(symbols and "BENCHMARK" not in symbols)
                if filter_symbols:
                    # 与 run_backtest 同一手法(:3320 附近): isin 是整体布尔索引,
                    # 拿不到"被剔除了哪些", 只能靠过滤前后的 symbol 集合求差集。
                    pre_filter_symbols = set(df_prepared["symbol"].unique())
                    df_prepared = df_prepared[df_prepared["symbol"].isin(symbols)]
                    filtered_out_symbols.update(pre_filter_symbols.difference(symbols))
                if not df_prepared.empty:
                    grouped = df_prepared.groupby("symbol", sort=False)
                    data_map = {
                        str(grouped_symbol): grouped_df.copy()
                        for grouped_symbol, grouped_df in grouped
                    }
            elif len(symbols) == 1:
                data_map = {symbols[0]: df_prepared}
            else:
                raise ValueError(
                    "Multi-symbol DataFrame input must contain a 'symbol' column"
                )
        elif isinstance(data, list) and data and isinstance(data[0], Bar):
            # Convert List[Bar] to DataFrame for indicators
            # We assume all bars are for the same symbol (or single symbol context)
            feed.add_bars(data)  # type: ignore

            # Construct DataFrame for indicator calculation
            # Group by symbol just in case
            # (BacktestDataInput 因 Task 4 加宽含 Tick, 但本分支的
            # isinstance(data[0], Bar) 运行时守卫未变——run_from_checkpoint 的
            # tick 支持不在本任务范围内, 这里只是 mypy 静态类型跟着别名变宽而已,
            # 无逻辑改动)
            bars_by_sym: Dict[str, List[Dict[str, Any]]] = {}
            for bar in data:
                if bar.symbol not in bars_by_sym:
                    bars_by_sym[bar.symbol] = []
                bars_by_sym[bar.symbol].append(
                    {
                        "timestamp": pd.Timestamp(bar.timestamp, unit="ns", tz="UTC"),
                        "open": bar.open,  # type: ignore[union-attr]
                        "high": bar.high,  # type: ignore[union-attr]
                        "low": bar.low,  # type: ignore[union-attr]
                        "close": bar.close,  # type: ignore[union-attr]
                        "volume": bar.volume,
                    }
                )

            for sym, records in bars_by_sym.items():
                df = pd.DataFrame(records)
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)
                    data_map_for_indicators[sym] = df

        elif isinstance(data, dict):
            # 与 run_backtest 的 dict 分支对称(:3351 附近): 显式传了 symbols 时
            # 同样只放行白名单内的标的, 被挡下的记入 filtered_out_symbols 供下面
            # 的汇总 INFO 日志复用。
            filter_symbols = bool(symbols and "BENCHMARK" not in symbols)
            if filter_symbols:
                data_map = {}
                for sym, df in data.items():
                    if sym not in symbols:
                        filtered_out_symbols.add(sym)
                        continue
                    data_map[sym] = df
            else:
                data_map = data
        else:
            data_map = {}

        loaded_count = 0
        for sym, df in data_map.items():
            if not df.empty:
                df = to_indicator_frame(df)
                data_map_for_indicators[sym] = df
                arrays = dataframe_to_arrays(df, symbol=sym)
                feed.add_arrays(*arrays)  # type: ignore
                loaded_count += 1

        if loaded_count > 0:
            feed.sort()

    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    engine, strategy_instance = load_checkpoint(checkpoint_path, feed)
    # 快照持久化的运行时配置 (issue #282)；旧快照为 None，走逐项推导降级路径
    restored_backtest_config = getattr(
        strategy_instance, "_warm_start_backtest_config", None
    )
    if not isinstance(restored_backtest_config, dict):
        restored_backtest_config = {}
    restored_strategy_id = str(
        getattr(strategy_instance, "_owner_strategy_id", "") or ""
    ).strip()
    restored_engine_strategy_id = ""
    if hasattr(engine, "get_default_strategy_id"):
        restored_engine_strategy_id = str(
            cast(Any, engine).get_default_strategy_id() or ""
        ).strip()
    effective_strategy_id = (
        str(strategy_id).strip()
        if strategy_id is not None and str(strategy_id).strip()
        else restored_strategy_id or restored_engine_strategy_id or "_default"
    )
    restored_slot_ids: List[str] = []
    slot_fetcher = None
    if hasattr(engine, "get_strategy_slot_ids"):
        slot_fetcher = cast(Any, engine).get_strategy_slot_ids
    elif hasattr(engine, "get_strategy_slots"):
        slot_fetcher = cast(Any, engine).get_strategy_slots
    if slot_fetcher is not None:
        try:
            slot_ids = slot_fetcher()
            if isinstance(slot_ids, list):
                restored_slot_ids = [
                    str(slot_id).strip() for slot_id in slot_ids if str(slot_id).strip()
                ]
        except Exception:
            restored_slot_ids = []

    restored_slot_strategy_instances: Dict[str, Strategy] = {}
    raw_restored_slot_strategies = getattr(strategy_instance, "_slot_strategies", None)
    if isinstance(raw_restored_slot_strategies, dict):
        for slot_key, slot_strategy in raw_restored_slot_strategies.items():
            slot_key_str = str(slot_key).strip()
            if not slot_key_str:
                continue
            if isinstance(slot_strategy, Strategy):
                restored_slot_strategy_instances[slot_key_str] = slot_strategy

    slot_strategy_instances = dict(restored_slot_strategy_instances)
    if strategies_by_slot:
        slot_strategy_instances = {}
        for slot_key, slot_strategy_input in strategies_by_slot.items():
            slot_key_str = str(slot_key).strip()
            if not slot_key_str:
                raise ValueError("strategy slot id cannot be empty")
            slot_strategy_instances[slot_key_str] = _build_strategy_instance(
                slot_strategy_input,
                strategy_kwargs={},
                strict_strategy_params=False,
                logger=logger,
                initialize=None,
                on_start=None,
                on_resume=None,
                on_train_signal=None,
                on_stop=None,
                on_tick=None,
                on_order=None,
                on_trade=None,
                on_reject=None,
                on_before_trading=None,
                on_after_trading=None,
                on_cross_section=None,
                on_portfolio_update=None,
                on_error=None,
                on_expiry=None,
                on_pre_open=None,
                on_timer=None,
                context=None,
            )

    configured_slot_ids = [effective_strategy_id]
    source_slot_ids = (
        list(slot_strategy_instances.keys())
        if slot_strategy_instances
        else restored_slot_ids
    )
    for slot_key in source_slot_ids:
        if slot_key not in configured_slot_ids:
            configured_slot_ids.append(slot_key)
    normalized_strategy_fill_policy = _normalize_strategy_fill_policy_map(
        strategy_fill_policy,
        configured_slot_ids,
        logger,
    )
    normalized_strategy_slippage = _normalize_strategy_slippage_map(
        strategy_slippage,
        configured_slot_ids,
        logger,
    )
    normalized_strategy_commission = _normalize_strategy_commission_map(
        strategy_commission,
        configured_slot_ids,
    )

    setattr(strategy_instance, "_owner_strategy_id", effective_strategy_id)
    for slot_key, slot_strategy in slot_strategy_instances.items():
        setattr(slot_strategy, "_owner_strategy_id", slot_key)
    indicator_recorder = _attach_indicator_recorder(
        stream_emitter=indicator_stream_emitter,
        strategy_instance=strategy_instance,
        slot_strategy_instances=slot_strategy_instances,
        recorder=indicator_recorder,
    )
    setattr(strategy_instance, "_slot_strategies", dict(slot_strategy_instances))
    setattr(strategy_instance, "_strategy_slot_ids", list(configured_slot_ids))
    # 无条件赋值(而非只在 normalized_strategy_* is not None 时才赋值): 恢复出来的
    # 策略实例的 `_strategy_fill_policy_map` / `_strategy_slippage_map` /
    # `_strategy_commission_map` 是上一段 checkpoint 存档时随对象一起被
    # save_checkpoint 整体 pickle 下来的旧值(load_checkpoint 用默认 __dict__
    # 整体恢复, 会原样带出旧 map)——本次若没显式传对应的 strategy_* 参数, 必须
    # 显式置 None 覆盖它, 否则旧 map 会继续生效, 并静默压过本次显式传的运行级
    # fill_policy/slippage/commission_rate(见 strategy_trading_api.py 里
    # _resolve_effective_order_* 的解析顺序: 策略级 map 命中就直接返回, 根本不会
    # 走到运行级默认值)。置 None 而非 {}: 与消费端 `if not policy_map: return
    # None` 的判据等价(两者都是"无覆盖"), 且 None 与该属性未被设置过时的
    # getattr 默认值一致, 不引入第三种状态。
    for current_strategy in [strategy_instance, *slot_strategy_instances.values()]:
        setattr(
            current_strategy,
            "_strategy_fill_policy_map",
            dict(normalized_strategy_fill_policy)
            if normalized_strategy_fill_policy is not None
            else None,
        )
        setattr(
            current_strategy,
            "_strategy_slippage_map",
            dict(normalized_strategy_slippage)
            if normalized_strategy_slippage is not None
            else None,
        )
        setattr(
            current_strategy,
            "_strategy_commission_map",
            dict(normalized_strategy_commission)
            if normalized_strategy_commission is not None
            else None,
        )

    if configured_slot_ids and hasattr(engine, "set_strategy_slots"):
        cast(Any, engine).set_strategy_slots(configured_slot_ids)
    if hasattr(engine, "set_default_strategy_id"):
        cast(Any, engine).set_default_strategy_id(effective_strategy_id)
    if hasattr(engine, "set_strategy_for_slot"):
        for slot_index, slot_id in enumerate(configured_slot_ids):
            assigned_strategy: Strategy
            if slot_id == effective_strategy_id:
                assigned_strategy = strategy_instance
            else:
                assigned_strategy = slot_strategy_instances.get(
                    slot_id, strategy_instance
                )
            cast(Any, engine).set_strategy_for_slot(slot_index, assigned_strategy)

    if "strategy_runtime_config" in kwargs:
        kwargs_runtime_config = kwargs.pop("strategy_runtime_config")
        if strategy_runtime_config is None:
            strategy_runtime_config = kwargs_runtime_config
    if strategy_runtime_config is not None and isinstance(strategy_instance, Strategy):
        _apply_strategy_runtime_config(
            strategy_instance,
            strategy_runtime_config,
            runtime_config_override,
            logger,
        )
        for slot_strategy in slot_strategy_instances.values():
            _apply_strategy_runtime_config(
                slot_strategy,
                strategy_runtime_config,
                runtime_config_override,
                logger,
            )
    if strategy_priority and hasattr(engine, "set_strategy_priorities"):
        normalized_strategy_priority: Dict[str, int] = {}
        for strategy_key, raw_priority in strategy_priority.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_priority contains empty strategy id")
            normalized_strategy_priority[strategy_key_str] = int(raw_priority)
        unknown_keys = sorted(
            set(normalized_strategy_priority.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_priority contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_priorities(normalized_strategy_priority)
    if strategy_risk_budget and hasattr(engine, "set_strategy_risk_budget_limits"):
        normalized_strategy_risk_budget: Dict[str, float] = {}
        for strategy_key, raw_budget in strategy_risk_budget.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_risk_budget contains empty strategy id")
            budget_value = float(raw_budget)
            if not pd.notna(budget_value) or budget_value < 0.0:
                raise ValueError(
                    f"strategy_risk_budget for {strategy_key_str} must be >= 0"
                )
            normalized_strategy_risk_budget[strategy_key_str] = budget_value
        unknown_keys = sorted(
            set(normalized_strategy_risk_budget.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_risk_budget contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_risk_budget_limits(
            normalized_strategy_risk_budget
        )
    if hasattr(engine, "set_portfolio_risk_budget_limit"):
        cast(Any, engine).set_portfolio_risk_budget_limit(portfolio_risk_budget)
    if hasattr(engine, "set_risk_budget_mode"):
        cast(Any, engine).set_risk_budget_mode(risk_budget_mode)
    if hasattr(engine, "set_risk_budget_reset_daily"):
        cast(Any, engine).set_risk_budget_reset_daily(risk_budget_reset_daily)
    if strategy_max_order_value and hasattr(
        engine, "set_strategy_max_order_value_limits"
    ):
        normalized_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_order_value.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_order_value contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_order_value for {strategy_key_str} must be >= 0"
                )
            normalized_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_order_value contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_order_value_limits(normalized_limits)
    if strategy_max_order_size and hasattr(
        engine, "set_strategy_max_order_size_limits"
    ):
        normalized_limits_by_size: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_order_size.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_order_size contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_order_size for {strategy_key_str} must be >= 0"
                )
            normalized_limits_by_size[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_limits_by_size.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_order_size contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_order_size_limits(normalized_limits_by_size)
    if strategy_max_position_size and hasattr(
        engine, "set_strategy_max_position_size_limits"
    ):
        normalized_position_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_position_size.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_max_position_size contains empty strategy id"
                )
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_position_size for {strategy_key_str} must be >= 0"
                )
            normalized_position_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_position_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_position_size contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_position_size_limits(
            normalized_position_limits
        )
    if strategy_max_daily_loss and hasattr(
        engine, "set_strategy_max_daily_loss_limits"
    ):
        normalized_daily_loss_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_daily_loss.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_daily_loss contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_daily_loss for {strategy_key_str} must be >= 0"
                )
            normalized_daily_loss_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_daily_loss_limits.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_daily_loss contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_daily_loss_limits(
            normalized_daily_loss_limits
        )
    if strategy_max_drawdown and hasattr(engine, "set_strategy_max_drawdown_limits"):
        normalized_drawdown_limits: Dict[str, float] = {}
        for strategy_key, raw_limit in strategy_max_drawdown.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError("strategy_max_drawdown contains empty strategy id")
            limit_value = float(raw_limit)
            if not pd.notna(limit_value) or limit_value < 0.0:
                raise ValueError(
                    f"strategy_max_drawdown for {strategy_key_str} must be >= 0"
                )
            normalized_drawdown_limits[strategy_key_str] = limit_value
        unknown_keys = sorted(
            set(normalized_drawdown_limits.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_max_drawdown contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_max_drawdown_limits(normalized_drawdown_limits)
    if strategy_reduce_only_after_risk and hasattr(
        engine, "set_strategy_reduce_only_after_risk"
    ):
        normalized_reduce_only_flags: Dict[str, bool] = {}
        for strategy_key, raw_flag in strategy_reduce_only_after_risk.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_reduce_only_after_risk contains empty strategy id"
                )
            normalized_reduce_only_flags[strategy_key_str] = bool(raw_flag)
        unknown_keys = sorted(
            set(normalized_reduce_only_flags.keys()).difference(
                set(configured_slot_ids)
            )
        )
        if unknown_keys:
            raise ValueError(
                "strategy_reduce_only_after_risk contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_reduce_only_after_risk(
            normalized_reduce_only_flags
        )
    if strategy_risk_cooldown_bars and hasattr(
        engine, "set_strategy_risk_cooldown_bars"
    ):
        normalized_cooldown_bars: Dict[str, int] = {}
        for strategy_key, raw_bars in strategy_risk_cooldown_bars.items():
            strategy_key_str = str(strategy_key).strip()
            if not strategy_key_str:
                raise ValueError(
                    "strategy_risk_cooldown_bars contains empty strategy id"
                )
            cooldown_bars = int(raw_bars)
            if cooldown_bars < 0:
                raise ValueError(
                    f"strategy_risk_cooldown_bars for {strategy_key_str} must be >= 0"
                )
            normalized_cooldown_bars[strategy_key_str] = cooldown_bars
        unknown_keys = sorted(
            set(normalized_cooldown_bars.keys()).difference(set(configured_slot_ids))
        )
        if unknown_keys:
            raise ValueError(
                "strategy_risk_cooldown_bars contains unknown strategy id(s): "
                + ",".join(unknown_keys)
            )
        cast(Any, engine).set_strategy_risk_cooldown_bars(normalized_cooldown_bars)

    all_strategy_instances = [strategy_instance, *slot_strategy_instances.values()]
    snapshot_features = getattr(strategy_instance, "_warm_start_snapshot_features", {})
    if not isinstance(snapshot_features, dict):
        snapshot_features = {}
    history_buffer_snapshot_available = bool(
        snapshot_features.get("history_buffer_snapshot", False)
    )
    history_tracking_enabled = any(
        int(getattr(current_strategy, "_history_depth", 0)) > 0
        for current_strategy in all_strategy_instances
    )
    if history_tracking_enabled and not history_buffer_snapshot_available:
        warning_message = (
            "The checkpoint was created by an older AKQuant version and does not "
            "include the history buffer snapshot. Checkpoint resume can continue, but "
            "`get_history()` / `get_history_map()` may differ from a full backtest "
            "until the checkpoint is regenerated with a newer version."
        )
        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
        logger.warning(warning_message)
    if data_map_for_indicators:
        (
            all_dates,
            day_bounds,
            day_cross_section_timestamps,
        ) = _build_trading_day_metadata(data_map_for_indicators, timezone_name)

        for current_strategy in all_strategy_instances:
            if all_dates and hasattr(current_strategy, "_trading_days"):
                current_strategy._trading_days = all_dates
            if hasattr(current_strategy, "_trading_day_bounds"):
                current_strategy._trading_day_bounds = day_bounds
            if hasattr(current_strategy, "_trading_day_cross_section_timestamps"):
                current_strategy._trading_day_cross_section_timestamps = (
                    day_cross_section_timestamps
                )
            setattr(current_strategy, "_engine", engine)

    # Capture restored cash BEFORE running (for correct initial_market_value in result)
    restored_cash = engine.portfolio.cash
    logger.info(f"Restored engine cash: {restored_cash}")

    # 2.5 重新注册标的 (Instrument)
    # 引擎快照通常不包含静态配置 (Instrument)，因此需要为热启动阶段
    # 重新注入相关 instrument。这里尽量与 run_backtest 保持相同解析规则，
    # 以保证 instruments_config / instruments / 默认参数在冷热启动下语义一致。
    symbols_to_add: set[str] = set()
    if data_map_for_indicators:
        symbols_to_add.update(data_map_for_indicators.keys())

    # 如果 data 是 List[Bar]，也收集其中的 symbol
    if isinstance(data, list) and data and isinstance(data[0], Bar):
        for bar in data:  # 只检查前几个可能不够，但通常数据是单一标的
            symbols_to_add.add(bar.symbol)
            # 优化: 如果列表很大，只检查第一个和最后一个? 或者假设单一标的?
            # 这里简单起见，只取第一个，假设列表是针对单一或少数几个标的
            break

    try:
        if hasattr(strategy_instance, "symbol"):
            s = strategy_instance.symbol
            if s:
                symbols_to_add.add(s)
    except Exception:
        # symbol property might raise error if no current bar/tick
        pass

    if config and config.instruments:
        for config_symbol in config.instruments:
            symbols_to_add.add(config_symbol)

    inst_conf_map: Dict[str, Any] = {}
    if config and config.instruments_config:
        if isinstance(config.instruments_config, list):
            for conf_item in config.instruments_config:
                if conf_item.symbol not in inst_conf_map:
                    inst_conf_map[conf_item.symbol] = conf_item
                    symbols_to_add.add(conf_item.symbol)
        elif isinstance(config.instruments_config, dict):
            for key, value in config.instruments_config.items():
                if key not in inst_conf_map:
                    inst_conf_map[key] = value
                    symbols_to_add.add(key)

    prebuilt_instruments: Dict[str, Any] = {}
    if "instruments" in kwargs:
        raw_instruments = kwargs["instruments"]
        if isinstance(raw_instruments, list):
            for item in raw_instruments:
                prebuilt_instruments[item.symbol] = item
                symbols_to_add.add(item.symbol)
        elif isinstance(raw_instruments, dict):
            for key, value in raw_instruments.items():
                prebuilt_instruments[key] = value
                symbols_to_add.add(key)

    default_multiplier = kwargs.get("multiplier", 1.0)
    default_margin_ratio = kwargs.get("margin_ratio", 1.0)
    default_asset_type = kwargs.get("asset_type", AssetType.Stock)
    # tick_size 缺省值按 asset_type 分流(基金 0.001, 其余 0.01), 显式传入的
    # tick_size 仍优先——见 _default_tick_size_for_asset_type。
    default_tick_size = kwargs.get(
        "tick_size", _default_tick_size_for_asset_type(default_asset_type)
    )
    default_option_type = kwargs.get("option_type", None)
    default_option_margin_model = kwargs.get("option_margin_model", None)
    default_strike_price = kwargs.get("strike_price", None)
    default_expiry_date = _normalize_expiry_date_yyyymmdd(
        kwargs.get("expiry_date", None)
    )
    default_implied_volatility = kwargs.get("implied_volatility", None)
    default_reference_volatility = kwargs.get("reference_volatility", None)
    default_settlement_type = kwargs.get("settlement_type", None)
    default_settlement_price = kwargs.get("settlement_price", None)
    lot_size = kwargs.get("lot_size", 1)

    def _parse_asset_type(val: Union[str, AssetType]) -> AssetType:
        if isinstance(val, AssetType):
            return val
        if isinstance(val, str):
            v_lower = val.strip().lower()
            if v_lower == "stock":
                return AssetType.Stock
            if v_lower in {"future", "futures"}:
                return AssetType.Futures
            if v_lower == "fund":
                return AssetType.Fund
            if v_lower == "option":
                return AssetType.Option
        raise ValueError(f"Unsupported asset_type: {val}")

    def _parse_option_type(val: Any) -> Any:
        if val is None:
            return None
        try:
            from ..akquant import OptionType  # type: ignore

            if isinstance(val, str):
                normalized = val.strip().lower()
                if normalized == "call":
                    return OptionType.Call
                if normalized == "put":
                    return OptionType.Put
                raise ValueError(f"Unsupported option_type: {val}")
            if str(val).endswith(".Call"):
                return OptionType.Call
            if str(val).endswith(".Put"):
                return OptionType.Put
        except ImportError:
            pass
        if isinstance(val, str):
            normalized = val.strip().upper()
            if normalized == "CALL":
                return "CALL"
            if normalized == "PUT":
                return "PUT"
            raise ValueError(f"Unsupported option_type: {val}")
        text = str(val)
        if text.endswith(".Call"):
            return "CALL"
        if text.endswith(".Put"):
            return "PUT"
        return val

    def _parse_option_margin_model(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, OptionMarginModel):
            return val
        if isinstance(val, str):
            normalized = val.strip().upper()
            if normalized == "RATIO":
                return OptionMarginModel.Ratio
            if normalized == "CHINA_SINGLE_LEG":
                return OptionMarginModel.ChinaSingleLeg
            if normalized == "US_BROKER_SINGLE_LEG":
                return OptionMarginModel.USBrokerSingleLeg
            if normalized == "US_BROKER_SINGLE_LEG_VOL_ADJUSTED":
                return OptionMarginModel.USBrokerSingleLegVolAdjusted
        raise ValueError(f"Unsupported option_margin_model: {val}")

    def _parse_settlement_type(
        val: Any,
    ) -> Tuple[Any, Optional[InstrumentSettlementMode]]:
        if val is None:
            return None, None
        if isinstance(val, SettlementType):
            if val == SettlementType.Physical:
                raise ValueError("Unsupported settlement_type: Physical")
            return val, _settlement_type_to_upper_name(val)
        if isinstance(val, str):
            key = val.strip().lower()
            if key in {"cash", "cash_last_price"}:
                return SettlementType.Cash, "CASH"
            if key in {"settlement_price", "cash_settlement_price"}:
                return SettlementType.Cash, "SETTLEMENT_PRICE"
            if key in {"force_close", "forceclose"}:
                return SettlementType.ForceClose, "FORCE_CLOSE"
            raise ValueError(f"Unsupported settlement_type: {val}")
        raise TypeError("settlement_type must be SettlementType or str")

    existing_instrument_snapshots = dict(
        getattr(strategy_instance, "_instrument_snapshots", {})
    )
    warm_start_instrument_snapshots: Dict[str, InstrumentSnapshot] = {}
    for sym in sorted(symbols_to_add):
        if sym in prebuilt_instruments:
            prebuilt = prebuilt_instruments[sym]
            engine.add_instrument(prebuilt)
            warm_start_instrument_snapshots[sym] = InstrumentSnapshot(
                symbol=sym,
                asset_type=_asset_type_to_upper_name(
                    getattr(prebuilt, "asset_type", "")
                ),
                multiplier=float(getattr(prebuilt, "multiplier", 1.0)),
                margin_ratio=float(getattr(prebuilt, "margin_ratio", 1.0)),
                option_margin_model=_option_margin_model_to_upper_name(
                    getattr(prebuilt, "option_margin_model", None)
                ),
                tick_size=float(
                    getattr(prebuilt, "tick_size", None)
                    or _default_tick_size_for_asset_type(
                        getattr(prebuilt, "asset_type", "")
                    )
                ),
                lot_size=float(getattr(prebuilt, "lot_size", 1.0) or 1.0),
                option_type=_option_type_to_upper_name(
                    getattr(prebuilt, "option_type", None)
                ),
                strike_price=(
                    float(getattr(prebuilt, "strike_price"))
                    if getattr(prebuilt, "strike_price", None) is not None
                    else None
                ),
                expiry_date=_normalize_expiry_date_yyyymmdd(
                    getattr(prebuilt, "expiry_date", None)
                ),
                underlying_symbol=(
                    str(getattr(prebuilt, "underlying_symbol"))
                    if getattr(prebuilt, "underlying_symbol", None) is not None
                    else None
                ),
                implied_volatility=(
                    float(getattr(prebuilt, "implied_volatility"))
                    if getattr(prebuilt, "implied_volatility", None) is not None
                    else None
                ),
                reference_volatility=(
                    float(getattr(prebuilt, "reference_volatility"))
                    if getattr(prebuilt, "reference_volatility", None) is not None
                    else None
                ),
                settlement_type=_settlement_type_to_upper_name(
                    getattr(prebuilt, "settlement_type", None)
                ),
                settlement_price=(
                    float(getattr(prebuilt, "settlement_price"))
                    if getattr(prebuilt, "settlement_price", None) is not None
                    else None
                ),
            )
            logger.info(
                f"Re-registered configured instrument for checkpoint resume: {sym}"
            )
            continue

        symbol_lot_size: Optional[float] = None
        if isinstance(lot_size, int):
            symbol_lot_size = float(lot_size)
        elif isinstance(lot_size, dict):
            raw_lot_size = lot_size.get(sym)
            if raw_lot_size is not None:
                symbol_lot_size = float(raw_lot_size)

        conf = inst_conf_map.get(sym)
        if conf is None:
            p_asset_type = default_asset_type
            p_multiplier = default_multiplier
            p_margin_ratio = default_margin_ratio
            p_tick_size = default_tick_size
            p_lot_size = float(symbol_lot_size or 1.0)
            p_option_type = _parse_option_type(default_option_type)
            p_option_margin_model = _parse_option_margin_model(
                default_option_margin_model
                if default_option_margin_model is not None
                else ("CHINA_SINGLE_LEG" if p_asset_type == AssetType.Option else None)
            )
            p_strike_price = default_strike_price
            p_expiry_date = default_expiry_date
            p_underlying_symbol = None
            p_implied_volatility = default_implied_volatility
            p_reference_volatility = default_reference_volatility
            p_settlement_type, p_settlement_mode = _parse_settlement_type(
                default_settlement_type
            )
            p_settlement_price = default_settlement_price
            static_attrs: Dict[str, Any] = {}
        else:
            p_asset_type = _parse_asset_type(conf.asset_type)
            p_multiplier = conf.multiplier
            p_margin_ratio = conf.margin_ratio
            p_tick_size = conf.tick_size
            p_lot_size = (
                float(conf.lot_size)
                if conf.lot_size is not None
                else float(symbol_lot_size or 1.0)
            )
            p_option_type = _parse_option_type(conf.option_type)
            p_option_margin_model = _parse_option_margin_model(
                conf.option_margin_model
                if conf.option_margin_model is not None
                else ("CHINA_SINGLE_LEG" if p_asset_type == AssetType.Option else None)
            )
            p_strike_price = conf.strike_price
            p_expiry_date = _normalize_expiry_date_yyyymmdd(conf.expiry_date)
            p_underlying_symbol = conf.underlying_symbol
            p_implied_volatility = conf.implied_volatility
            p_reference_volatility = conf.reference_volatility
            p_settlement_type, p_settlement_mode = _parse_settlement_type(
                conf.settlement_type
            )
            p_settlement_price = conf.settlement_price
            static_attrs = getattr(conf, "static_attrs", {})
            if static_attrs is None:
                static_attrs = {}
            if not isinstance(static_attrs, dict):
                raise TypeError(
                    "InstrumentConfig.static_attrs must be Dict[str, scalar]"
                )

        if p_asset_type != AssetType.Futures:
            p_settlement_type = None
            p_settlement_mode = None
            p_settlement_price = None
        if p_asset_type != AssetType.Option:
            p_option_margin_model = None
            p_implied_volatility = None
            p_reference_volatility = None

        p_sellable_after_days = (
            conf.sellable_after_days
            if conf is not None and conf.sellable_after_days is not None
            else None
        )
        instr = Instrument(
            symbol=sym,
            asset_type=p_asset_type,
            multiplier=p_multiplier,
            margin_ratio=p_margin_ratio,
            tick_size=p_tick_size,
            option_type=p_option_type,
            strike_price=p_strike_price,
            expiry_date=p_expiry_date,
            lot_size=float(p_lot_size),
            underlying_symbol=p_underlying_symbol,
            settlement_type=p_settlement_type,
            settlement_price=p_settlement_price,
            option_margin_model=p_option_margin_model,
            implied_volatility=p_implied_volatility,
            reference_volatility=p_reference_volatility,
            sellable_after_days=p_sellable_after_days,
        )
        engine.add_instrument(instr)
        warm_start_instrument_snapshots[sym] = InstrumentSnapshot(
            symbol=sym,
            asset_type=_asset_type_to_upper_name(p_asset_type),
            multiplier=float(p_multiplier),
            margin_ratio=float(p_margin_ratio),
            option_margin_model=_option_margin_model_to_upper_name(
                p_option_margin_model
            ),
            tick_size=float(p_tick_size),
            lot_size=float(p_lot_size),
            option_type=_option_type_to_upper_name(p_option_type),
            strike_price=(
                float(p_strike_price) if p_strike_price is not None else None
            ),
            expiry_date=p_expiry_date,
            underlying_symbol=(
                str(p_underlying_symbol) if p_underlying_symbol is not None else None
            ),
            implied_volatility=(
                float(p_implied_volatility)
                if p_implied_volatility is not None
                else None
            ),
            reference_volatility=(
                float(p_reference_volatility)
                if p_reference_volatility is not None
                else None
            ),
            settlement_type=p_settlement_mode,
            settlement_price=(
                float(p_settlement_price) if p_settlement_price is not None else None
            ),
            static_attrs=dict(static_attrs),
        )
        logger.info(f"Re-registered configured instrument for checkpoint resume: {sym}")

    merged_instrument_snapshots = dict(existing_instrument_snapshots)
    merged_instrument_snapshots.update(warm_start_instrument_snapshots)
    for current_strategy in all_strategy_instances:
        current_strategy._set_instrument_snapshots(merged_instrument_snapshots)

    # 2.6 Re-configure Market Model
    # Engine restoration might lose market model config if not in State.
    # Default to SimpleMarket (T+0) or ChinaMarket (T+1) based on kwargs.
    broker_profile_values = _resolve_broker_profile(
        cast(Optional[str], kwargs.get("broker_profile"))
    )
    if commission_policy is None:
        commission_policy = cast(
            Optional[CommissionPolicy], kwargs.get("commission_policy")
        )
    # 费率优先级：显式入参 > 快照配置 > _resolve_stock_fee_rules 内部默认。
    # commission_policy 覆盖快照的 commission_policy；其余标量缺省时回退快照。
    if commission_policy is None:
        restored_commission = restored_backtest_config.get("commission_policy")
        if isinstance(restored_commission, dict):
            commission_policy = cast(CommissionPolicy, dict(restored_commission))
    commission_rate_value = cast(Optional[float], kwargs.get("commission_rate"))
    stamp_tax_rate_value = cast(
        Optional[float], kwargs.get("stamp_tax_rate", kwargs.get("stamp_tax"))
    )
    if stamp_tax_rate_value is None:
        stamp_tax_rate_value = cast(
            Optional[float], restored_backtest_config.get("stamp_tax_rate")
        )
    transfer_fee_rate_value = cast(
        Optional[float], kwargs.get("transfer_fee_rate", kwargs.get("transfer_fee"))
    )
    if transfer_fee_rate_value is None:
        transfer_fee_rate_value = cast(
            Optional[float], restored_backtest_config.get("transfer_fee_rate")
        )
    min_commission_value = cast(Optional[float], kwargs.get("min_commission"))
    if min_commission_value is None:
        min_commission_value = cast(
            Optional[float], restored_backtest_config.get("min_commission")
        )
    (
        resolved_commission_policy,
        stamp_tax,
        transfer_fee,
        min_commission,
    ) = _resolve_stock_fee_rules(
        commission_policy=commission_policy,
        commission_rate=commission_rate_value,
        stamp_tax_rate=stamp_tax_rate_value,
        transfer_fee_rate=transfer_fee_rate_value,
        min_commission=min_commission_value,
        broker_profile_values=broker_profile_values,
        strategy_config=strategy_config,
    )
    t_plus_one = kwargs.get("t_plus_one", False)
    if "t_plus_one" not in kwargs and "t_plus_one" in restored_backtest_config:
        t_plus_one = bool(restored_backtest_config.get("t_plus_one"))

    if t_plus_one:
        # ChinaMarket implies T+1 and specific fee rules
        engine.use_china_market()
    else:
        # SimpleMarket implies T+0
        if hasattr(engine, "use_simple_market_policy"):
            cast(Any, engine).use_simple_market_policy(
                resolved_commission_policy["type"],
                float(resolved_commission_policy["value"]),
            )
        else:
            commission = (
                float(resolved_commission_policy["value"])
                if resolved_commission_policy["type"] == "percent"
                else 0.0
            )
            engine.use_simple_market(commission)

    # Apply fee rules if engine supports it
    # (and if not ChinaMarket which has fixed rules?)
    # ChinaMarket usually has hardcoded rules or defaults,
    # but set_stock_fee_rules overrides them?
    # Let's just set it.
    if hasattr(engine, "set_stock_fee_policy"):
        cast(Any, engine).set_stock_fee_policy(
            resolved_commission_policy["type"],
            float(resolved_commission_policy["value"]),
            stamp_tax,
            transfer_fee,
            min_commission,
        )
        logger.info(
            "Re-configured market fees: commission_policy=%s, stamp=%s",
            resolved_commission_policy,
            stamp_tax,
        )
    elif hasattr(engine, "set_stock_fee_rules"):
        commission = (
            float(resolved_commission_policy["value"])
            if resolved_commission_policy["type"] == "percent"
            else 0.0
        )
        engine.set_stock_fee_rules(commission, stamp_tax, transfer_fee, min_commission)
        logger.info(f"Re-configured market fees: comm={commission}, stamp={stamp_tax}")

    # 2.7 Re-configure Slippage & Volume Limit (issue #282)
    # 引擎状态不保留滑点/量比配置，热启动须与 run_backtest 一样重新应用，
    # 否则二阶段会按原价撮合。显式入参优先，缺省时回退 config.strategy_config。
    # 优先级：显式入参 > config.strategy_config > 快照配置 > 默认
    warm_start_slippage: SlippageInput = cast(SlippageInput, kwargs.get("slippage"))
    if warm_start_slippage is None and strategy_config is not None:
        warm_start_slippage = strategy_config.slippage
    if warm_start_slippage is None:
        warm_start_slippage = cast(
            SlippageInput, restored_backtest_config.get("slippage")
        )
    if warm_start_slippage is None:
        warm_start_slippage = 0.0
    normalized_global_slippage = _normalize_slippage_policy(
        warm_start_slippage,
        instrument_snapshots=merged_instrument_snapshots,
        logger=logger,
        scope="Global",
    )
    if (
        normalized_global_slippage["type"] != "zero"
        and normalized_global_slippage["value"] > 0
    ):
        if hasattr(engine, "set_slippage"):
            engine.set_slippage(
                normalized_global_slippage["type"],
                normalized_global_slippage["value"],
            )
        else:
            logger.warning(
                "Slippage policy %s set but not supported by Engine.",
                normalized_global_slippage,
            )

    warm_start_volume_limit = cast(Optional[float], kwargs.get("volume_limit_pct"))
    if warm_start_volume_limit is None and strategy_config is not None:
        warm_start_volume_limit = strategy_config.volume_limit_pct
    if warm_start_volume_limit is None:
        warm_start_volume_limit = cast(
            Optional[float], restored_backtest_config.get("volume_limit_pct")
        )
    if (
        warm_start_volume_limit is not None
        and warm_start_volume_limit != 0.25
        and hasattr(engine, "set_volume_limit")
    ):
        engine.set_volume_limit(warm_start_volume_limit)

    restored_initial_market_value = float(restored_cash)
    if hasattr(engine, "get_account_metrics"):
        try:
            account_metrics = cast(Any, engine).get_account_metrics()
            if isinstance(account_metrics, tuple) and len(account_metrics) >= 1:
                restored_initial_market_value = float(account_metrics[0])
        except Exception:
            restored_initial_market_value = float(restored_cash)
    if hasattr(engine, "set_initial_cash_reference"):
        cast(Any, engine).set_initial_cash_reference(restored_initial_market_value)
    # fill_policy 优先级：显式入参 > 快照配置。缺省时继承快照 (issue #282)
    effective_fill_policy = fill_policy_override
    apply_fill_policy = has_fill_policy_override
    if not apply_fill_policy:
        restored_fill_policy = restored_backtest_config.get("fill_policy")
        if isinstance(restored_fill_policy, dict):
            effective_fill_policy = cast(FillPolicy, dict(restored_fill_policy))
            apply_fill_policy = True
    resolved_policy_warm_start: Optional[ResolvedExecutionPolicy] = None
    if apply_fill_policy:
        resolved_policy_warm_start = _resolve_execution_policy(
            execution_mode="next_open",
            timer_execution_policy="same_cycle",
            fill_policy=effective_fill_policy,
            logger=logger,
        )
        if not hasattr(engine, "set_fill_mode"):
            raise RuntimeError(
                "Engine binary does not expose set_fill_mode; please rebuild bindings"
            )
        cast(Any, engine).set_fill_mode(
            resolved_policy_warm_start.execution_mode,
            resolved_policy_warm_start.temporal,
        )
    if stream_on_event is not None:
        cast(Any, engine).set_stream_callback(stream_on_event)
        cast(Any, engine).set_stream_options(
            stream_progress_interval,
            stream_equity_interval,
            stream_batch_size,
            stream_max_buffer,
            stream_error_mode,
            stream_mode,
        )
    _prime_framework_boundary_timers(all_strategy_instances, engine)
    _prime_framework_cross_section_timers(all_strategy_instances, engine)
    _prime_framework_pre_open_timers(all_strategy_instances, engine)

    # symbols 白名单下发给策略实例(供 subscribe() 校验用): 必须在 on_start 之前,
    # 这样 on_start 里的 subscribe 才会被挡住。与下面(on_start 之后)的
    # engine.set_symbol_whitelist 是两个不同的下发点, 不要合并。
    # 无条件赋值(而非只在 symbols_explicit 时才赋值): 恢复出来的策略实例的
    # `_symbol_whitelist` 是上一段 checkpoint 存档时随对象一起 pickle 下来的
    # 旧值(load_checkpoint 用默认 __dict__ 整体恢复, 会原样带出旧白名单)——
    # 本次若没显式传 symbols, 必须显式置 None 覆盖它, 否则会沿用上一段的旧白名单,
    # 与「这次不传 symbols = 不过滤」的意图相矛盾, 且引擎层(下面的
    # engine.set_symbol_whitelist, 同一个 guard 不执行 ⇒ 不过滤)与策略层
    # (仍按旧白名单拦截 subscribe)会出现矛盾的两副面孔。
    # 用 effective_symbols 而非原始 symbols: 与 run_backtest 同一个理由(见该函数
    # 同名注释)——`symbols` 未经归一, symbols="600519" 这种字符串写法下
    # set(symbols) 会拆成单字符集合。
    whitelist_for_strategy = (
        set(
            _merge_symbol_whitelist_sources(
                effective_symbols,
                config,
                strategy_instance,
                *slot_strategy_instances.values(),
            )
        )
        if symbols_explicit
        else None
    )
    strategy_instance._symbol_whitelist = whitelist_for_strategy
    for slot_strategy in slot_strategy_instances.values():
        slot_strategy._symbol_whitelist = whitelist_for_strategy

    if hasattr(strategy_instance, "_on_start_internal"):
        strategy_instance._on_start_internal()
    elif hasattr(strategy_instance, "on_start"):
        if hasattr(strategy_instance, "is_restored") and strategy_instance.is_restored:
            if hasattr(strategy_instance, "on_resume"):
                strategy_instance.on_resume()
        strategy_instance.on_start()
    for slot_strategy in slot_strategy_instances.values():
        if hasattr(slot_strategy, "_on_start_internal"):
            slot_strategy._on_start_internal()
        elif hasattr(slot_strategy, "on_start"):
            if hasattr(slot_strategy, "is_restored") and slot_strategy.is_restored:
                if hasattr(slot_strategy, "on_resume"):
                    slot_strategy.on_resume()
            slot_strategy.on_start()

    # symbols 白名单: 只在用户显式传了 symbols 时启用, 与 run_backtest 对称。
    # 白名单取合并后的集合(effective_symbols + config.instruments + 各策略的
    # _subscriptions) —— 放在 on_start 之后是有意的: __init__/on_start 里的
    # subscribe 到这里才确定。add_data 已在上面 load_checkpoint() 内部执行过,
    # 但白名单只需在 engine.run() 之前生效即可, 过滤发生在事件分发时(见
    # set_symbol_whitelist 的实现)。DataFeed 对象输入只能靠这一层过滤 —— 它
    # 只写不读, Python 无从枚举内容。
    whitelist_symbols = _merge_symbol_whitelist_sources(
        effective_symbols,
        config,
        strategy_instance,
        *slot_strategy_instances.values(),
    )
    if symbols_explicit:
        engine.set_symbol_whitelist(whitelist_symbols)

    # 被前置过滤掉的标的只发一条汇总日志, 与 run_backtest 对称(见该函数同名
    # 注释, :4572 附近)——两条入口的可观测性不应该有差异。
    if filtered_out_symbols:
        logger.info(
            "已过滤 %d 个不在 symbols 中的标的: %s",
            len(filtered_out_symbols),
            ", ".join(sorted(filtered_out_symbols)),
        )

    # 运行前比对: 与 run_backtest 对称(见该函数同名注释)。symbols 里有没有标的
    # 实际数据集合里压根没有——data_map_for_indicators 为空说明走的是 DataFeed
    # 对象输入, 只能留给会话末 Strategy._check_symbol_data_coverage 兜底。
    # 用 whitelist_symbols(合并后, 与上面下发给引擎的白名单是同一个集合)而非
    # effective_symbols——否则仅通过 config.instruments 或各策略 _subscriptions
    # 进白名单、且零数据的标的会被漏检(运行前与会话末双重漏报)。
    if symbols_explicit and data_map_for_indicators:
        symbol_data_missing = set(whitelist_symbols) - set(
            data_map_for_indicators.keys()
        )
        for sym in sorted(symbol_data_missing):
            logger.warning(
                "标的 %s 在 symbols 中但数据里没有它: 该标的全程不会有任何行情"
                "事件。常见原因是标的代码写错、数据源未覆盖、或所选时间范围内"
                "无交易。",
                sym,
            )
        if symbol_data_missing:
            strategy_instance._symbol_data_warned = set(symbol_data_missing)
            for slot_strategy in slot_strategy_instances.values():
                slot_strategy._symbol_data_warned = set(symbol_data_missing)

    # 热启动同样会踩「配置 symbol 与数据 symbol 对不上 → 配置被静默丢弃」这一条,
    # 与冷启动共用同一个 helper(只在这里多一个调用点, 语义完全一致)。
    _warn_unmatched_instrument_config(config, data_map_for_indicators, logger)

    if data_map_for_indicators:
        for current_strategy in all_strategy_instances:
            if _should_prepare_precomputed_indicators(current_strategy) and hasattr(
                current_strategy, "_prepare_indicators"
            ):
                try:
                    current_strategy._prepare_indicators(data_map_for_indicators)
                except Exception as e:
                    logger.error(
                        f"Failed to update indicators for checkpoint resume: {e}"
                    )
            if hasattr(current_strategy, "_bootstrap_incremental_indicators"):
                try:
                    current_strategy._bootstrap_incremental_indicators(
                        data_map_for_indicators
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to bootstrap incremental indicators on resume: {e}"
                    )

    # 4. 运行

    engine_summary: str = ""
    try:
        engine_summary = str(engine.run(strategy_instance, show_progress))
    except Exception as e:
        logger.error(
            "Checkpoint resume backtest failed: %s",
            e,
            extra=_build_backtest_log_extra(
                phase="backtest",
                strategy_id=effective_strategy_id,
                slot=effective_strategy_id,
            ),
        )
        raise e
    finally:
        if stream_on_event is not None and hasattr(engine, "clear_stream_callback"):
            try:
                cast(Any, engine).clear_stream_callback()
            except Exception as e:
                logger.debug(f"Failed to clear stream callback: {e}")
        if hasattr(strategy_instance, "_on_stop_internal"):
            try:
                strategy_instance._on_stop_internal()
            except StrategyConfigurationError:
                # 框架配置校验失败必须让调用方看到: 吞掉它会让用户拿到静默失效的结果
                # (例如 H/L 指标全程未更新、ATR 恒为 0)。用户 on_stop 的 bug 仍按下面
                # 的分支容忍。
                raise
            except Exception as e:
                logger.error(
                    "Error in on_stop: %s",
                    e,
                    extra=_build_backtest_log_extra(
                        phase="strategy",
                        strategy=strategy_instance,
                    ),
                )
        elif hasattr(strategy_instance, "on_stop"):
            try:
                strategy_instance.on_stop()
            except Exception as e:
                logger.error(
                    "Error in on_stop: %s",
                    e,
                    extra=_build_backtest_log_extra(
                        phase="strategy",
                        strategy=strategy_instance,
                    ),
                )
        for slot_strategy in slot_strategy_instances.values():
            if hasattr(slot_strategy, "_on_stop_internal"):
                try:
                    slot_strategy._on_stop_internal()
                except StrategyConfigurationError:
                    # 框架配置校验失败必须让调用方看到, 理由同主策略分支。
                    raise
                except Exception as e:
                    logger.error(
                        "Error in slot on_stop: %s",
                        e,
                        extra=_build_backtest_log_extra(
                            phase="strategy",
                            strategy=slot_strategy,
                        ),
                    )
            elif hasattr(slot_strategy, "on_stop"):
                try:
                    slot_strategy.on_stop()
                except Exception as e:
                    logger.error(
                        "Error in slot on_stop: %s",
                        e,
                        extra=_build_backtest_log_extra(
                            phase="strategy",
                            strategy=slot_strategy,
                        ),
                    )

    result = BacktestResult(
        engine.get_results(),
        timezone=timezone_name,
        initial_cash=float(restored_cash),
        strategy=strategy_instance,
        engine=engine,
        indicator_outputs=indicator_recorder.build_payload(),
    )
    try:
        result.initial_cash = float(result.metrics.initial_market_value)
    except Exception:
        pass
    _attach_result_runtime_metadata(
        result=result,
        engine_summary=engine_summary,
        event_stats_snapshot=event_stats_snapshot,
        owner_strategy_id=effective_strategy_id,
        resolved_policy=resolved_policy_warm_start,
    )
    return result
