import datetime as dt
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from .akquant import (
    Bar,
    Order,
    StrategyContext,
    Tick,
    TimeInForce,
)
from .gateway.order_receipt import OrderReceipt
from .indicator_recording import IndicatorRecorder
from .sizer import FixedSize, Sizer
from .strategy_events import (
    flush_pending_order_events as _flush_pending_order_events_impl,
)
from .strategy_events import (
    on_bar_event as _on_bar_event_impl,
)
from .strategy_events import (
    on_tick_event as _on_tick_event_impl,
)
from .strategy_events import (
    on_timer_event as _on_timer_event_impl,
)
from .strategy_framework_hooks import (
    call_user_callback as _call_user_callback_impl,
)
from .strategy_framework_hooks import (
    dispatch_shutdown_hooks as _dispatch_shutdown_hooks_impl,
)
from .strategy_framework_hooks import (
    ensure_framework_state as _ensure_framework_state_impl,
)
from .strategy_framework_hooks import (
    register_pre_open_timers as _register_pre_open_timers_impl,
)
from .strategy_history import get_history as _get_history_impl
from .strategy_history import get_history_df as _get_history_df_impl
from .strategy_history import get_rolling_data as _get_rolling_data_impl
from .strategy_history import set_history_depth as _set_history_depth_impl
from .strategy_history import set_rolling_window as _set_rolling_window_impl
from .strategy_logging import log as _log_impl
from .strategy_ml import auto_configure_model as _auto_configure_model_impl
from .strategy_ml import current_validation_window as _current_validation_window_impl
from .strategy_ml import is_model_ready as _is_model_ready_impl
from .strategy_ml import on_train_signal as _on_train_signal_impl
from .strategy_order_events import (
    check_order_events as _check_order_events_impl,
)
from .strategy_order_events import (
    key_value as _key_value_impl,
)
from .strategy_order_events import (
    remember_trade_key as _remember_trade_key_impl,
)
from .strategy_order_events import (
    trade_dedupe_cache_limit as _trade_dedupe_cache_limit_impl,
)
from .strategy_order_events import (
    trade_event_key as _trade_event_key_impl,
)
from .strategy_position import Position
from .strategy_scheduler import add_daily_timer as _add_daily_timer_impl
from .strategy_scheduler import schedule as _schedule_impl
from .strategy_time import format_time as _format_time_impl
from .strategy_time import now as _now_impl
from .strategy_time import to_local_time as _to_local_time_impl
from .strategy_trading_api import (
    buy as _buy_impl,
)
from .strategy_trading_api import (
    calculate_max_buy_qty as _calculate_max_buy_qty_impl,
)
from .strategy_trading_api import (
    cancel_all_orders as _cancel_all_orders_impl,
)
from .strategy_trading_api import (
    cancel_group as _cancel_group_impl,
)
from .strategy_trading_api import (
    cancel_order as _cancel_order_impl,
)
from .strategy_trading_api import (
    close_position as _close_position_impl,
)
from .strategy_trading_api import (
    cover as _cover_impl,
)
from .strategy_trading_api import (
    get_account as _get_account_impl,
)
from .strategy_trading_api import (
    get_available_position as _get_available_position_impl,
)
from .strategy_trading_api import (
    get_cash as _get_cash_impl,
)
from .strategy_trading_api import (
    get_execution_capabilities as _get_execution_capabilities_impl,
)
from .strategy_trading_api import (
    get_holding_bars as _get_holding_bars_impl,
)
from .strategy_trading_api import (
    get_last_target_positions_plan as _get_last_target_positions_plan_impl,
)
from .strategy_trading_api import (
    get_open_orders as _get_open_orders_impl,
)
from .strategy_trading_api import (
    get_portfolio_value as _get_portfolio_value_impl,
)
from .strategy_trading_api import (
    get_position as _get_position_impl,
)
from .strategy_trading_api import (
    get_positions as _get_positions_impl,
)
from .strategy_trading_api import (
    order_target as _order_target_impl,
)
from .strategy_trading_api import (
    order_target_percent as _order_target_percent_impl,
)
from .strategy_trading_api import (
    order_target_value as _order_target_value_impl,
)
from .strategy_trading_api import (
    rebalance_positions as _rebalance_positions_impl,
)
from .strategy_trading_api import (
    rebalance_weights as _rebalance_weights_impl,
)
from .strategy_trading_api import (
    resolve_symbol as _resolve_symbol_impl,
)
from .strategy_trading_api import (
    sell as _sell_impl,
)
from .strategy_trading_api import (
    short as _short_impl,
)
from .strategy_trading_api import (
    submit_order as _submit_order_impl,
)

if TYPE_CHECKING:
    from .indicator import Indicator
    from .ml.model import QuantModel


@dataclass
class StrategyRuntimeConfig:
    """策略运行时行为配置."""

    enable_precise_day_boundary_hooks: bool = False
    portfolio_update_eps: float = 0.0
    error_mode: Literal["raise", "continue", "legacy"] = "raise"
    re_raise_on_error: bool = True
    indicator_mode: Literal["incremental", "precompute"] = "precompute"

    def __post_init__(self) -> None:
        """校验并标准化配置."""
        self.portfolio_update_eps = float(self.portfolio_update_eps)
        if self.portfolio_update_eps < 0.0:
            raise ValueError("portfolio_update_eps must be >= 0")
        mode = str(self.error_mode).strip().lower()
        if mode not in {"raise", "continue", "legacy"}:
            raise ValueError("error_mode must be one of: raise, continue, legacy")
        self.error_mode = cast(Literal["raise", "continue", "legacy"], mode)
        indicator_mode = str(self.indicator_mode).strip().lower()
        if indicator_mode not in {"incremental", "precompute"}:
            raise ValueError("indicator_mode must be one of: incremental, precompute")
        self.indicator_mode = cast(Literal["incremental", "precompute"], indicator_mode)
        self.enable_precise_day_boundary_hooks = bool(
            self.enable_precise_day_boundary_hooks
        )
        self.re_raise_on_error = bool(self.re_raise_on_error)


InstrumentStaticValue = Union[str, int, float, bool]
InstrumentAssetTypeName = Literal["STOCK", "FUTURES", "FUND", "OPTION"]
InstrumentOptionTypeName = Literal["CALL", "PUT"]
InstrumentOptionMarginModelName = Literal[
    "RATIO",
    "CHINA_SINGLE_LEG",
    "US_BROKER_SINGLE_LEG",
    "US_BROKER_SINGLE_LEG_VOL_ADJUSTED",
]
InstrumentSettlementMode = Literal["CASH", "SETTLEMENT_PRICE", "FORCE_CLOSE"]


@dataclass
class IncrementalIndicatorRegistration:
    """增量指标注册定义."""

    source: str
    symbols: Optional[set[str]]
    input_mode: str = "source"
    warmup_bars: int = 0
    factory: Optional[Callable[[], Any]] = None
    base_indicator: Any = None
    primary_symbol: Optional[str] = None
    instances: Dict[str, Any] = field(default_factory=dict)


class IncrementalIndicatorBinding:
    """按当前 symbol 访问底层增量指标实例的轻量代理."""

    def __init__(self, strategy: "Strategy", name: str) -> None:
        """绑定策略实例与指标名，延迟解析当前 symbol 对应的真实指标."""
        self._strategy = strategy
        self._name = name

    def get_instance(self, symbol: Optional[str] = None) -> Any:
        """返回指定 symbol 对应的底层指标实例."""
        return self._strategy._get_incremental_indicator_instance(self._name, symbol)

    @property
    def value(self) -> Any:
        """获取当前 symbol 的最新指标值."""
        return getattr(self.get_instance(), "value", None)

    @property
    def is_ready(self) -> bool:
        """获取当前 symbol 的 ready 状态."""
        return bool(getattr(self.get_instance(), "is_ready", False))

    def update(self, *args: Any, **kwargs: Any) -> Any:
        """兼容手动调用 update 的现有写法."""
        return self.get_instance().update(*args, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        """将未知属性代理到底层当前 symbol 的增量指标实例."""
        return getattr(self.get_instance(), attr)


@dataclass(frozen=True)
class InstrumentSnapshot:
    """策略侧可访问的标的静态属性快照."""

    symbol: str
    asset_type: InstrumentAssetTypeName
    multiplier: float
    margin_ratio: float
    tick_size: float
    lot_size: float
    option_margin_model: Optional[InstrumentOptionMarginModelName] = None
    option_type: Optional[InstrumentOptionTypeName] = None
    strike_price: Optional[float] = None
    expiry_date: Optional[int] = None
    underlying_symbol: Optional[str] = None
    implied_volatility: Optional[float] = None
    reference_volatility: Optional[float] = None
    settlement_type: Optional[InstrumentSettlementMode] = None
    settlement_price: Optional[float] = None
    static_attrs: Dict[str, InstrumentStaticValue] = field(default_factory=dict)


class Strategy:
    """
    策略基类 (Base Strategy Class).

    采用事件驱动设计
    """

    ctx: Optional[StrategyContext]
    execution_mode: Optional[Any]
    execution: Any
    sizer: Sizer
    current_bar: Optional[Bar]
    current_tick: Optional[Tick]
    _history_depth: int
    # Rust maintains HistoryBuffer for indicator calculation.
    # Python side accesses it via self.ctx.history() (efficient copy).
    # No duplicate storage in Python.
    _precomputed_indicators: List["Indicator"]
    _incremental_indicators: Dict[str, IncrementalIndicatorRegistration]
    _subscriptions: List[str]
    _last_prices: Dict[str, float]
    _rolling_train_window: int
    _rolling_step: int
    _bar_count: int
    _model_configured: bool
    model: Optional["QuantModel"]
    _ml_validation_lifecycle: bool
    _ml_model_template: Optional["QuantModel"]
    _ml_active_model: Optional["QuantModel"]
    _ml_pending_model: Optional["QuantModel"]
    _ml_pending_activation_bar: Optional[int]
    _ml_active_window_index: int
    _ml_active_window_start_bar: Optional[int]
    _ml_active_window_end_bar: Optional[int]
    _ml_pending_window_index: int
    _ml_pending_window_start_bar: Optional[int]
    _ml_pending_window_end_bar: Optional[int]
    _known_orders: Dict[str, Order]
    _seen_trade_keys: set[Tuple[Any, ...]]
    _seen_trade_key_order: Deque[Tuple[Any, ...]]
    timezone: str = "Asia/Shanghai"
    warmup_period: int = 0
    _runtime_config: StrategyRuntimeConfig
    _last_event_type: str = ""  # "bar" or "tick"
    _hold_bars: "defaultdict[str, int]"
    _last_position_signs: "defaultdict[str, float]"
    _framework_last_session: Any
    _framework_last_local_date: Optional[dt.date]
    _framework_before_trading_done_date: Optional[dt.date]
    _framework_after_trading_done_date: Optional[dt.date]
    _framework_pre_open_done_date: Optional[dt.date]
    _framework_last_portfolio_state: Any
    _framework_portfolio_dirty: bool
    _framework_rejected_order_ids: set[str]
    _framework_expiry_event_keys: set[Tuple[Any, ...]]
    _framework_stop_flushed: bool
    _framework_boundary_timers_registered: bool
    _framework_pre_open_timers_registered: bool
    _framework_in_pre_open_phase: bool
    _framework_phase: Optional[str]
    _framework_history_cutoff_ns: Optional[int]
    _framework_use_previous_account_snapshot: bool
    _framework_previous_account_details: Optional[Dict[str, float]]
    _framework_emit_previous_portfolio_snapshot: bool
    _framework_current_callback: Optional[str]
    _framework_current_order: Optional[Any]
    _framework_current_trade: Optional[Any]
    _framework_daily_rebalance_after_bar_done_date: Optional[dt.date]
    _framework_daily_rebalance_after_bar_timers_registered: bool
    _trading_day_bounds: Dict[str, Tuple[int, int]]
    _trading_day_after_bar_rebalance_timestamps: Dict[str, int]
    _oco_groups: Dict[str, set[str]]
    _oco_order_to_group: Dict[str, str]
    _use_engine_oco: bool
    _pending_engine_oco_groups: List[Tuple[str, str, str]]
    _use_engine_bracket: bool
    _pending_engine_bracket_plans: List[
        Tuple[
            str,
            Optional[float],
            Optional[float],
            Optional[TimeInForce],
            Optional[str],
            Optional[str],
        ]
    ]
    _pending_brackets: Dict[str, Dict[str, Any]]
    _order_group_seq: int
    _pending_schedules: List[Tuple[Union[str, dt.datetime, pd.Timestamp], str]]
    _pending_daily_timers: List[Tuple[str, str]]
    _instrument_snapshots: Dict[str, InstrumentSnapshot]
    _indicator_recorder: Optional[IndicatorRecorder]

    _trading_days: List[pd.Timestamp]

    # 成本/手数配置单一真源(经下方 property 暴露: 费率只读, lot_size 可写)
    _cost_config: Dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> "Strategy":
        """Create a new Strategy instance."""
        instance = super().__new__(cls)
        # 初始化默认属性，确保 pickle 恢复前也有这些字段
        instance._is_restored = False
        instance.ctx = None
        instance.execution_mode = None

        from .execution.sim import SimExecution

        instance.execution = SimExecution(instance)
        instance.sizer = FixedSize(100)
        instance.current_bar = None
        instance.current_tick = None
        instance._precomputed_indicators = []
        instance._incremental_indicators = {}
        instance._subscriptions = []
        instance._last_prices = {}
        instance._known_orders = {}
        instance._seen_trade_keys = set()
        instance._seen_trade_key_order = deque()
        instance._hold_bars = defaultdict(int)
        instance._last_position_signs = defaultdict(float)
        instance.timezone = getattr(instance, "timezone", "Asia/Shanghai")
        raw_runtime_config = getattr(instance, "_runtime_config", None)

        def _class_attr(name: str, default: Any) -> Any:
            # 基类 Strategy 自身的 __dict__ 里这些名字是 property（数据描述符）；
            # 只有子类真正用普通类属性覆盖时才应采用该值，否则退回默认值。
            value = cls.__dict__.get(name, default)
            return default if isinstance(value, property) else value

        class_enable_hooks = _class_attr("enable_precise_day_boundary_hooks", False)
        class_portfolio_eps = _class_attr("portfolio_update_eps", 0.0)
        class_error_mode = _class_attr("error_mode", "raise")
        class_re_raise = _class_attr("re_raise_on_error", True)
        class_indicator_mode = _class_attr("indicator_mode", "precompute")
        if isinstance(raw_runtime_config, dict):
            instance.runtime_config = StrategyRuntimeConfig(**raw_runtime_config)
        elif isinstance(raw_runtime_config, StrategyRuntimeConfig):
            instance.runtime_config = StrategyRuntimeConfig(
                enable_precise_day_boundary_hooks=raw_runtime_config.enable_precise_day_boundary_hooks,
                portfolio_update_eps=raw_runtime_config.portfolio_update_eps,
                error_mode=raw_runtime_config.error_mode,
                re_raise_on_error=raw_runtime_config.re_raise_on_error,
                indicator_mode=raw_runtime_config.indicator_mode,
            )
        else:
            instance.runtime_config = StrategyRuntimeConfig(
                enable_precise_day_boundary_hooks=bool(class_enable_hooks),
                portfolio_update_eps=float(class_portfolio_eps),
                error_mode=cast(
                    Literal["raise", "continue", "legacy"], str(class_error_mode)
                ),
                re_raise_on_error=bool(class_re_raise),
                indicator_mode=cast(
                    Literal["incremental", "precompute"], str(class_indicator_mode)
                ),
            )
        instance._last_event_type = ""
        instance._trading_days = []

        # 历史数据配置
        instance._history_depth = 0
        instance.warmup_period = getattr(instance, "warmup_period", 0)

        # 滚动训练配置
        instance._rolling_train_window = 0
        instance._rolling_step = 0
        instance._bar_count = 0
        instance._model_configured = False
        instance._start_initialized = False
        instance._ml_validation_lifecycle = False
        instance._ml_model_template = None
        instance._ml_active_model = None
        instance._ml_pending_model = None
        instance._ml_pending_activation_bar = None
        instance._ml_active_window_index = 0
        instance._ml_active_window_start_bar = None
        instance._ml_active_window_end_bar = None
        instance._ml_pending_window_index = 0
        instance._ml_pending_window_start_bar = None
        instance._ml_pending_window_end_bar = None

        # 初始化通常在 __init__ 中的属性，允许子类省略 super().__init__()
        instance.model = None

        # 成本/手数配置单一真源。费率经只读 property 暴露(写入报错, 指向 run_backtest);
        # lot_size 经可写 property 读写(照旧支持策略 __init__ 里赋值)。
        # lot_size 可为 int(全局) 或 Dict[str, int](按标的);
        # 默认 1(美股/加密), A股请设 100。
        instance._cost_config = {
            "commission_policy": {"type": "percent", "value": 0.0},
            "min_commission": 0.0,
            "stamp_tax_rate": 0.0,
            "transfer_fee_rate": 0.0,
            "lot_size": 1,
        }
        instance._owner_strategy_id = None
        instance._framework_last_session = None
        instance._framework_last_local_date = None
        instance._framework_before_trading_done_date = None
        instance._framework_after_trading_done_date = None
        instance._framework_pre_open_done_date = None
        instance._framework_last_portfolio_state = None
        instance._framework_portfolio_dirty = True
        instance._framework_rejected_order_ids = set()
        instance._framework_expiry_event_keys = set()
        instance._framework_stop_flushed = False
        instance._framework_boundary_timers_registered = False
        instance._framework_pre_open_timers_registered = False
        instance._framework_in_pre_open_phase = False
        instance._framework_phase = None
        instance._framework_history_cutoff_ns = None
        instance._framework_use_previous_account_snapshot = False
        instance._framework_previous_account_details = None
        instance._framework_emit_previous_portfolio_snapshot = False
        instance._framework_current_callback = None
        instance._framework_current_order = None
        instance._framework_current_trade = None
        instance._framework_daily_rebalance_after_bar_done_date = None
        instance._framework_daily_rebalance_after_bar_timers_registered = False
        instance._trading_day_bounds = {}
        instance._trading_day_after_bar_rebalance_timestamps = {}
        instance._oco_groups = {}
        instance._oco_order_to_group = {}
        instance._use_engine_oco = False
        instance._pending_engine_oco_groups = []
        instance._use_engine_bracket = False
        instance._pending_engine_bracket_plans = []
        instance._pending_brackets = {}
        instance._order_group_seq = 0
        instance._order_group_lock = threading.RLock()
        instance._pending_schedules = []
        instance._pending_daily_timers = []
        instance._instrument_snapshots = {}
        instance._indicator_recorder = None

        return instance

    def __init__(self) -> None:
        """初始化."""
        self._owner_strategy_id: Optional[str] = None

    def __getstate__(self) -> Dict[str, Any]:
        """
        Pickle 序列化支持.

        排除运行时上下文 (ctx) 和临时对象 (current_bar, current_tick).
        """
        state = self.__dict__.copy()
        if "ctx" in state:
            del state["ctx"]
        if "current_bar" in state:
            del state["current_bar"]
        if "current_tick" in state:
            del state["current_tick"]
        if "_framework_last_session" in state:
            del state["_framework_last_session"]
        if "_framework_last_local_date" in state:
            del state["_framework_last_local_date"]
        if "_framework_before_trading_done_date" in state:
            del state["_framework_before_trading_done_date"]
        if "_framework_after_trading_done_date" in state:
            del state["_framework_after_trading_done_date"]
        if "_framework_last_portfolio_state" in state:
            del state["_framework_last_portfolio_state"]
        if "_framework_portfolio_dirty" in state:
            del state["_framework_portfolio_dirty"]
        if "_framework_rejected_order_ids" in state:
            del state["_framework_rejected_order_ids"]
        if "_framework_stop_flushed" in state:
            del state["_framework_stop_flushed"]
        if "_framework_boundary_timers_registered" in state:
            del state["_framework_boundary_timers_registered"]
        # 派生态(性能短路标志 + 本地交易日缓存): 恢复时删除, 以强制
        # ensure_framework_state 重建被重置字段, 并让日期缓存自我重建.
        for _derived_key in (
            "_framework_state_ready",
            "_framework_local_date_cache",
            "_framework_local_date_lo_ns",
            "_framework_local_date_hi_ns",
        ):
            if _derived_key in state:
                del state[_derived_key]
        if "_framework_current_callback" in state:
            del state["_framework_current_callback"]
        if "_framework_current_order" in state:
            del state["_framework_current_order"]
        if "_framework_current_trade" in state:
            del state["_framework_current_trade"]
        if "_indicator_recorder" in state:
            del state["_indicator_recorder"]
        if "_order_group_lock" in state:
            del state["_order_group_lock"]  # RLock 不可 pickle, __setstate__ 重建
        incremental_indicators = state.get("_incremental_indicators")
        if isinstance(incremental_indicators, dict):
            for name in incremental_indicators.keys():
                if state.get(name).__class__ is IncrementalIndicatorBinding:
                    del state[name]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Pickle 反序列化支持."""
        self.__dict__.update(state)
        # 注意判 state 而非 self.__dict__: __new__ 已预置 _cost_config 默认,
        # 故必须看"传入 snapshot 是否带 _cost_config"来区分新旧格式。
        if "_cost_config" not in state:
            # 兼容旧 snapshot(费率/lot 曾为裸属性, 无 _cost_config): 迁移进 _cost_config
            self._cost_config = {
                "commission_policy": self.__dict__.pop(
                    "commission_policy", {"type": "percent", "value": 0.0}
                ),
                "min_commission": self.__dict__.pop("min_commission", 0.0),
                "stamp_tax_rate": self.__dict__.pop("stamp_tax_rate", 0.0),
                "transfer_fee_rate": self.__dict__.pop("transfer_fee_rate", 0.0),
                "lot_size": self.__dict__.pop("lot_size", 1),
            }
            # 旧 snapshot 的 commission_rate 是派生量, 清掉避免遗留死数据
            self.__dict__.pop("commission_rate", None)
        self._order_group_lock = threading.RLock()  # RLock 不入 pickle, 恢复时重建
        raw_runtime_config = self.__dict__.pop("runtime_config", None)
        if raw_runtime_config is not None:
            self.runtime_config = raw_runtime_config
        elif "_runtime_config" in self.__dict__:
            self.runtime_config = self.__dict__["_runtime_config"]
        else:
            self.runtime_config = StrategyRuntimeConfig(
                enable_precise_day_boundary_hooks=bool(
                    self.__dict__.pop("enable_precise_day_boundary_hooks", False)
                ),
                portfolio_update_eps=float(
                    self.__dict__.pop("portfolio_update_eps", 0.0)
                ),
                error_mode=cast(
                    Literal["raise", "continue", "legacy"],
                    str(self.__dict__.pop("error_mode", "raise")),
                ),
                re_raise_on_error=bool(self.__dict__.pop("re_raise_on_error", True)),
                indicator_mode=cast(
                    Literal["incremental", "precompute"],
                    str(self.__dict__.pop("indicator_mode", "precompute")),
                ),
            )
        self.ctx = None
        if getattr(self, "execution", None) is None:
            from .execution.sim import SimExecution

            self.execution = SimExecution(self)
        self.current_bar = None
        self.current_tick = None
        self._framework_current_callback = None
        self._framework_current_order = None
        self._framework_current_trade = None
        self._is_restored = True  # 标记为已恢复状态
        self._start_initialized = False
        if not hasattr(self, "_seen_trade_keys"):
            self._seen_trade_keys = set()
        if not hasattr(self, "_incremental_indicators"):
            self._incremental_indicators = {}
        if not hasattr(self, "_seen_trade_key_order"):
            self._seen_trade_key_order = deque()
        if not hasattr(self, "_oco_groups"):
            self._oco_groups = {}
        if not hasattr(self, "_oco_order_to_group"):
            self._oco_order_to_group = {}
        if not hasattr(self, "_use_engine_oco"):
            self._use_engine_oco = False
        if not hasattr(self, "_pending_engine_oco_groups"):
            self._pending_engine_oco_groups = []
        if not hasattr(self, "_use_engine_bracket"):
            self._use_engine_bracket = False
        if not hasattr(self, "_pending_engine_bracket_plans"):
            self._pending_engine_bracket_plans = []
        if not hasattr(self, "_pending_brackets"):
            self._pending_brackets = {}
        if not hasattr(self, "_order_group_seq"):
            self._order_group_seq = 0
        if not hasattr(self, "_pending_schedules"):
            self._pending_schedules = []
        if not hasattr(self, "_pending_daily_timers"):
            self._pending_daily_timers = []
        if not hasattr(self, "_instrument_snapshots"):
            self._instrument_snapshots = {}
        self._indicator_recorder = None
        if not hasattr(self, "_ml_validation_lifecycle"):
            self._ml_validation_lifecycle = False
        if not hasattr(self, "_ml_model_template"):
            self._ml_model_template = None
        if not hasattr(self, "_ml_active_model"):
            self._ml_active_model = None
        if not hasattr(self, "_ml_pending_model"):
            self._ml_pending_model = None
        if not hasattr(self, "_ml_pending_activation_bar"):
            self._ml_pending_activation_bar = None
        if not hasattr(self, "_ml_active_window_index"):
            self._ml_active_window_index = 0
        if not hasattr(self, "_ml_active_window_start_bar"):
            self._ml_active_window_start_bar = None
        if not hasattr(self, "_ml_active_window_end_bar"):
            self._ml_active_window_end_bar = None
        if not hasattr(self, "_ml_pending_window_index"):
            self._ml_pending_window_index = 0
        if not hasattr(self, "_ml_pending_window_start_bar"):
            self._ml_pending_window_start_bar = None
        if not hasattr(self, "_ml_pending_window_end_bar"):
            self._ml_pending_window_end_bar = None
        for name in list(getattr(self, "_incremental_indicators", {}).keys()):
            setattr(self, name, IncrementalIndicatorBinding(self, name))
        _ensure_framework_state_impl(self)

    @property
    def runtime_config(self) -> StrategyRuntimeConfig:
        """返回策略运行时配置."""
        cfg = getattr(self, "_runtime_config", None)
        if isinstance(cfg, StrategyRuntimeConfig):
            return cfg
        if isinstance(cfg, dict):
            self._runtime_config = StrategyRuntimeConfig(**cfg)
            return self._runtime_config
        self._runtime_config = StrategyRuntimeConfig()
        return self._runtime_config

    @runtime_config.setter
    def runtime_config(
        self, value: Union[StrategyRuntimeConfig, Dict[str, Any]]
    ) -> None:
        """设置策略运行时配置."""
        if isinstance(value, StrategyRuntimeConfig):
            self._runtime_config = StrategyRuntimeConfig(
                enable_precise_day_boundary_hooks=value.enable_precise_day_boundary_hooks,
                portfolio_update_eps=value.portfolio_update_eps,
                error_mode=value.error_mode,
                re_raise_on_error=value.re_raise_on_error,
                indicator_mode=value.indicator_mode,
            )
            return
        if isinstance(value, dict):
            self._runtime_config = StrategyRuntimeConfig(**value)
            return
        raise TypeError(
            "runtime_config must be StrategyRuntimeConfig or Dict[str, Any]"
        )

    @property
    def enable_precise_day_boundary_hooks(self) -> bool:
        """是否启用边界定时器精确交易日钩子."""
        return self.runtime_config.enable_precise_day_boundary_hooks

    @enable_precise_day_boundary_hooks.setter
    def enable_precise_day_boundary_hooks(self, value: bool) -> None:
        """设置边界定时器精确交易日钩子开关."""
        cfg = self.runtime_config
        self.runtime_config = StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=bool(value),
            portfolio_update_eps=cfg.portfolio_update_eps,
            error_mode=cfg.error_mode,
            re_raise_on_error=cfg.re_raise_on_error,
            indicator_mode=cfg.indicator_mode,
        )

    @property
    def portfolio_update_eps(self) -> float:
        """返回账户快照更新阈值."""
        return self.runtime_config.portfolio_update_eps

    @portfolio_update_eps.setter
    def portfolio_update_eps(self, value: float) -> None:
        """设置账户快照更新阈值."""
        cfg = self.runtime_config
        self.runtime_config = StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=cfg.enable_precise_day_boundary_hooks,
            portfolio_update_eps=float(value),
            error_mode=cfg.error_mode,
            re_raise_on_error=cfg.re_raise_on_error,
            indicator_mode=cfg.indicator_mode,
        )

    @property
    def error_mode(self) -> str:
        """返回错误处理模式."""
        return self.runtime_config.error_mode

    @error_mode.setter
    def error_mode(self, value: str) -> None:
        """设置错误处理模式."""
        cfg = self.runtime_config
        self.runtime_config = StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=cfg.enable_precise_day_boundary_hooks,
            portfolio_update_eps=cfg.portfolio_update_eps,
            error_mode=cast(Literal["raise", "continue", "legacy"], value),
            re_raise_on_error=cfg.re_raise_on_error,
            indicator_mode=cfg.indicator_mode,
        )

    @property
    def re_raise_on_error(self) -> bool:
        """返回是否在 on_error 后继续抛出异常."""
        return self.runtime_config.re_raise_on_error

    @re_raise_on_error.setter
    def re_raise_on_error(self, value: bool) -> None:
        """设置 on_error 后是否继续抛出异常."""
        cfg = self.runtime_config
        self.runtime_config = StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=cfg.enable_precise_day_boundary_hooks,
            portfolio_update_eps=cfg.portfolio_update_eps,
            error_mode=cfg.error_mode,
            re_raise_on_error=bool(value),
            indicator_mode=cfg.indicator_mode,
        )

    @property
    def indicator_mode(self) -> Literal["incremental", "precompute"]:
        """返回指标执行模式."""
        return self.runtime_config.indicator_mode

    @indicator_mode.setter
    def indicator_mode(self, value: Literal["incremental", "precompute"]) -> None:
        cfg = self.runtime_config
        self.runtime_config = StrategyRuntimeConfig(
            enable_precise_day_boundary_hooks=cfg.enable_precise_day_boundary_hooks,
            portfolio_update_eps=cfg.portfolio_update_eps,
            error_mode=cfg.error_mode,
            re_raise_on_error=cfg.re_raise_on_error,
            indicator_mode=value,
        )

    @property
    def is_restored(self) -> bool:
        """检查策略是否是从快照恢复的."""
        return getattr(self, "_is_restored", False)

    def on_start(self) -> None:
        """
        策略启动时调用.

        Warm Start 注意：如果策略是从快照恢复的，此方法仍会被调用。
        如果在此处初始化指标，请务必检查属性是否存在 (hasattr)，
        或者使用 self.is_restored 属性判断，避免覆盖已恢复的状态。
        """
        pass

    def on_resume(self) -> None:
        """
        策略从快照恢复时调用 (Warm Start).

        仅在 Warm Start 模式下调用，且在 on_start 之前调用。
        用于重新建立非持久化连接、加载外部资源等。
        """
        pass

    def _on_start_internal(self) -> None:
        """内部启动回调."""
        if self._start_initialized:
            return

        # 如果是恢复模式，先调用 on_resume
        if self.is_restored:
            self.on_resume()

        self.on_start()
        _register_pre_open_timers_impl(self)
        self._start_initialized = True

    def on_stop(self) -> None:
        """
        策略停止时调用.

        在此处进行资源清理或结果统计.
        """
        pass

    def _on_stop_internal(self) -> None:
        """内部停止回调，用于补发框架级结束钩子."""
        _ensure_framework_state_impl(self)
        _dispatch_shutdown_hooks_impl(self)
        _call_user_callback_impl(self, "on_stop")

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """
        输出日志 (自动添加当前回测时间).

        :param msg: 日志内容
        :param level: 日志等级 (logging.INFO, logging.WARNING, etc.)
        """
        _log_impl(self, msg, level)

    @property
    def symbol(self) -> str:
        """获取当前正在处理的标的代码 (Proxy to current_bar/tick)."""
        return self._resolve_symbol(None)

    @property
    def close(self) -> float:
        """获取当前最新价 (Close 或 LastPrice)."""
        if self.current_bar:
            return self.current_bar.close
        elif self.current_tick:
            return self.current_tick.price
        return 0.0

    @property
    def open(self) -> float:
        """获取当前开盘价 (仅 Bar 模式有效)."""
        if self.current_bar:
            return self.current_bar.open
        return 0.0

    @property
    def high(self) -> float:
        """获取当前最高价 (仅 Bar 模式有效)."""
        if self.current_bar:
            return self.current_bar.high
        return 0.0

    @property
    def low(self) -> float:
        """获取当前最低价 (仅 Bar 模式有效)."""
        if self.current_bar:
            return self.current_bar.low
        return 0.0

    @property
    def volume(self) -> float:
        """获取当前成交量."""
        if self.current_bar:
            return self.current_bar.volume
        elif self.current_tick:
            return self.current_tick.volume
        return 0.0

    def schedule(
        self, trigger_time: Union[str, dt.datetime, pd.Timestamp], payload: str
    ) -> None:
        """
        注册单次定时任务 (Simplified).

        :param trigger_time: 触发时间 (支持 "2023-01-01 14:55:00", datetime, Timestamp)
        :param payload: 回调标识
        """
        _schedule_impl(self, trigger_time, payload)

    def add_daily_timer(self, time_str: str, payload: str) -> None:
        """
        注册每日定时任务 (Daily Timer).

        :param time_str: 时间字符串 (例如 "14:55:00")
        :param payload: 回调标识
        """
        _add_daily_timer_impl(self, time_str, payload)

    def to_local_time(self, timestamp: int) -> pd.Timestamp:
        """
        将 UTC 纳秒时间戳转换为本地时间 (Timestamp).

        :param timestamp: UTC 纳秒时间戳 (int64)
        :return: 本地时间 (pd.Timestamp)
        """
        return _to_local_time_impl(self, timestamp)

    def format_time(self, timestamp: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        将 UTC 纳秒时间戳格式化为本地时间字符串.

        :param timestamp: UTC 纳秒时间戳 (int64)
        :param fmt: 时间格式字符串
        :return: 格式化后的时间字符串
        """
        return _format_time_impl(self, timestamp, fmt)

    @property
    def now(self) -> Optional[pd.Timestamp]:
        """
        获取当前回测时间的本地时间表示.

        如果当前没有 Bar 或 Tick，则返回 None.
        """
        return _now_impl(self)

    def set_history_depth(self, depth: int) -> None:
        """
        设置历史数据回溯长度.

        :param depth: 保留的 Bar 数量 (0 表示不保留)
        """
        _set_history_depth_impl(self, depth)

    def set_rolling_window(self, train_window: int, step: int) -> None:
        """
        设置滚动训练窗口参数.

        :param train_window: 训练数据长度 (Bars)
        :param step: 滚动步长 (每隔多少个 Bar 触发一次训练)
        """
        _set_rolling_window_impl(self, train_window, step)

    def get_history(
        self, count: int, symbol: Optional[str] = None, field: str = "close"
    ) -> np.ndarray:
        """
        获取历史数据 (类似 Zipline data.history).

        :param count: 获取的数据长度 (必须 <= history_depth)
        :param symbol: 标的代码 (默认当前 Bar 的 symbol)
        :param field: 字段名 (open, high, low, close, volume)
        :return: Numpy 数组
        """
        return _get_history_impl(self, count, symbol, field)

    def get_history_map(
        self,
        count: int,
        symbols: Union[str, List[str], Tuple[str, ...], set[str]],
        field: str = "close",
    ) -> Dict[str, np.ndarray]:
        """
        批量获取多个标的的历史数据.

        :param count: 获取的数据长度
        :param symbols: 标的代码或标的列表
        :param field: 字段名 (open, high, low, close, volume)
        :return: {symbol: np.ndarray}
        """
        normalized = self._normalize_indicator_symbols(symbols)
        if normalized is None:
            raise ValueError("symbols cannot be None")
        normalized_symbols = sorted(normalized)
        history_map: Dict[str, np.ndarray] = {}
        for symbol in normalized_symbols:
            history_map[symbol] = self.get_history(
                count=count,
                symbol=symbol,
                field=field,
            )
        return history_map

    def rebalance_to_topn(
        self,
        scores: Dict[str, float],
        top_n: int,
        *,
        weight_mode: Literal["equal", "score"] = "equal",
        long_only: bool = True,
        min_score: Optional[float] = None,
        liquidate_unmentioned: bool = True,
        allow_leverage: bool = False,
        rebalance_tolerance: float = 0.0,
        **kwargs: Any,
    ) -> List[str]:
        """
        根据打分结果选取 TopN 并执行调仓.

        :param scores: 打分字典 {symbol: score}
        :param top_n: 选取数量
        :param weight_mode: 权重模式，"equal" 为等权，"score" 为按分数归一化
        :param long_only: 是否仅允许做多（默认仅保留正分）
        :param min_score: 最低分数阈值（None 时 long_only=True 默认阈值为 0）
        :param liquidate_unmentioned: 是否清仓未入选标的
        :param allow_leverage: 是否允许总权重超过 1.0
        :param rebalance_tolerance: 调仓容忍阈值（按组合市值比例）
        :param kwargs: 透传到 rebalance_weights 的下单参数
        :return: 入选标的列表（按分数从高到低）
        """
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0")
        if weight_mode not in {"equal", "score"}:
            raise ValueError("weight_mode must be one of: equal, score")

        if not scores:
            if liquidate_unmentioned:
                self.rebalance_weights(
                    target_weights={},
                    liquidate_unmentioned=True,
                    allow_leverage=allow_leverage,
                    rebalance_tolerance=rebalance_tolerance,
                    **kwargs,
                )
            return []

        ranking = sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        threshold = min_score
        if threshold is None and long_only:
            threshold = 0.0

        selected: List[str] = []
        for symbol, score in ranking:
            if threshold is not None and score < threshold:
                continue
            selected.append(symbol)
            if len(selected) >= top_n:
                break

        if not selected:
            if liquidate_unmentioned:
                self.rebalance_weights(
                    target_weights={},
                    liquidate_unmentioned=True,
                    allow_leverage=allow_leverage,
                    rebalance_tolerance=rebalance_tolerance,
                    **kwargs,
                )
            return []

        target_weights: Dict[str, float]
        if weight_mode == "equal":
            target_weight = 1.0 / float(len(selected))
            target_weights = {symbol: target_weight for symbol in selected}
        else:
            clipped_scores = {
                symbol: max(float(scores[symbol]), 0.0) for symbol in selected
            }
            score_sum = float(sum(clipped_scores.values()))
            if score_sum <= 0:
                target_weight = 1.0 / float(len(selected))
                target_weights = {symbol: target_weight for symbol in selected}
            else:
                target_weights = {
                    symbol: clipped_scores[symbol] / score_sum for symbol in selected
                }
        self.rebalance_weights(
            target_weights=target_weights,
            liquidate_unmentioned=liquidate_unmentioned,
            allow_leverage=allow_leverage,
            rebalance_tolerance=rebalance_tolerance,
            **kwargs,
        )
        return selected

    def get_history_df(self, count: int, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        获取历史数据 DataFrame (Open, High, Low, Close, Volume).

        :param count: 数据长度
        :param symbol: 标的代码
        :return: pd.DataFrame
        """
        return _get_history_df_impl(self, count, symbol)

    def get_rolling_data(
        self, length: Optional[int] = None, symbol: Optional[str] = None
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        获取滚动训练数据.

        :param length: 数据长度 (默认使用 set_rolling_window 设置的 train_window)
        :param symbol: 标的代码
        :return: (X, y) 默认为 (DataFrame, None)
        """
        return _get_rolling_data_impl(self, length, symbol)

    def on_train_signal(self, context: Any) -> None:
        """
        滚动训练信号回调.

        默认实现：如果配置了 self.model，则自动执行数据准备和训练.

        :param context: 策略上下文 (通常是 self)
        """
        _on_train_signal_impl(self, context)

    def is_model_ready(self) -> bool:
        """返回当前是否已有可用于推理的模型."""
        return _is_model_ready_impl(self)

    def current_validation_window(self) -> Optional[Dict[str, Any]]:
        """返回当前 walk-forward 验证窗口状态."""
        return _current_validation_window_impl(self)

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> Tuple[Any, Any]:
        """
        Prepare features and labels for ML model.

        Must be implemented by user if using auto-training.

        :param df: Raw dataframe from get_rolling_data
        :param mode: "training" or "inference".
                     If "training", return (X, y).
                     If "inference", return (X_last_row, None) or just X.
        :return: (X, y)
        """
        raise NotImplementedError(
            "You must implement prepare_features(self, df, mode) for auto-training"
        )

    def _auto_configure_model(self) -> None:
        """Apply model validation configuration if present."""
        _auto_configure_model_impl(self)

    def set_sizer(self, sizer: Sizer) -> None:
        """设置仓位管理器."""
        self.sizer = sizer

    def register_precomputed_indicator(self, name: str, indicator: "Indicator") -> None:
        """注册预计算指标."""
        if self.indicator_mode != "precompute":
            raise ValueError(
                "register_precomputed_indicator requires indicator_mode='precompute'"
            )
        if indicator not in self._precomputed_indicators:
            self._precomputed_indicators.append(indicator)
        setattr(self, name, indicator)

    def register_incremental_indicator(
        self,
        name: str,
        indicator: Any = None,
        source: str = "close",
        symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]] = None,
        *,
        warmup_bars: int = 0,
        indicator_factory: Optional[Callable[[], Any]] = None,
        input_mode: str = "source",
    ) -> None:
        """注册增量指标."""
        if self.indicator_mode != "incremental":
            raise ValueError(
                "register_incremental_indicator requires indicator_mode='incremental'"
            )
        if self.is_restored and name in self._incremental_indicators:
            setattr(self, name, IncrementalIndicatorBinding(self, name))
            return
        symbol_filter = self._normalize_indicator_symbols(symbols)
        source_key = str(source).strip().lower()
        input_mode_key = self._normalize_incremental_input_mode(source_key, input_mode)
        if indicator is not None and indicator_factory is not None:
            raise ValueError("indicator and indicator_factory cannot both be provided")
        if indicator is None and indicator_factory is None:
            raise ValueError("indicator or indicator_factory is required")
        warmup_bars_value = int(warmup_bars)
        if warmup_bars_value < 0:
            raise ValueError("warmup_bars must be >= 0")
        self._incremental_indicators[name] = IncrementalIndicatorRegistration(
            source=source_key,
            symbols=symbol_filter,
            input_mode=input_mode_key,
            warmup_bars=warmup_bars_value,
            factory=indicator_factory,
            base_indicator=indicator,
        )
        setattr(self, name, IncrementalIndicatorBinding(self, name))

    def subscribe(self, instrument_id: str) -> None:
        """
        Subscribe to market data for an instrument.

        :param instrument_id: The instrument identifier (e.g., '600000').
        """
        if instrument_id not in self._subscriptions:
            self._subscriptions.append(instrument_id)

    def _prepare_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """Pre-calculate indicators for precompute mode."""
        if self.indicator_mode != "precompute":
            return
        if not self._precomputed_indicators:
            return

        for ind in self._precomputed_indicators:
            for sym, df in data.items():
                # Calculate and cache inside indicator
                ind(df, sym)

    def _update_incremental_indicators(self, bar: Bar) -> None:
        if self.indicator_mode != "incremental":
            return
        if not self._incremental_indicators:
            return
        bar_symbol = getattr(bar, "symbol", None)
        active_start_time_ns = getattr(self, "_active_start_time_ns", None)
        if (
            getattr(self, "_incremental_bootstrap_applied", False)
            and active_start_time_ns is not None
            and int(getattr(bar, "timestamp", active_start_time_ns))
            < int(active_start_time_ns)
        ):
            return
        for name in list(self._incremental_indicators.keys()):
            item = self._get_incremental_registration(name)
            symbol_filter = item.symbols
            if symbol_filter is not None and bar_symbol not in symbol_filter:
                continue
            ind = self._get_incremental_indicator_instance(name, bar_symbol)
            args = self._build_incremental_indicator_args(
                payload=bar,
                source=item.source,
                input_mode=item.input_mode,
            )
            ind.update(*args)

    def _bootstrap_incremental_indicators(self, data: Dict[str, pd.DataFrame]) -> None:
        """在实时增量更新前，用历史数据预热增量指标."""
        if self.indicator_mode != "incremental" or self.is_restored:
            return
        if not self._incremental_indicators:
            return
        available_symbols = set(data.keys())
        active_start_time_ns = getattr(self, "_active_start_time_ns", None)
        if active_start_time_ns is None:
            return
        bootstrap_applied = False
        for name in list(self._incremental_indicators.keys()):
            item = self._get_incremental_registration(name)
            if item.warmup_bars <= 0:
                continue
            target_symbols = (
                sorted(item.symbols)
                if item.symbols is not None
                else sorted(available_symbols)
            )
            for symbol in target_symbols:
                df = data.get(symbol)
                if df is None or df.empty:
                    continue
                if active_start_time_ns is not None and isinstance(
                    df.index, pd.DatetimeIndex
                ):
                    boundary = pd.Timestamp(active_start_time_ns, unit="ns", tz="UTC")
                    if df.index.tz is None:
                        boundary = boundary.tz_localize(None)
                    else:
                        boundary = boundary.tz_convert(df.index.tz)
                    df = df[df.index < boundary]
                    if df.empty:
                        continue
                indicator = self._get_incremental_indicator_instance(name, symbol)
                for _, row in df.tail(item.warmup_bars).iterrows():
                    args = self._build_incremental_indicator_args(
                        payload=row,
                        source=item.source,
                        input_mode=item.input_mode,
                    )
                    indicator.update(*args)
                    bootstrap_applied = True
        self._incremental_bootstrap_applied = bootstrap_applied

    def _normalize_indicator_symbols(
        self,
        symbols: Optional[Union[str, List[str], Tuple[str, ...], set[str]]],
    ) -> Optional[set[str]]:
        if symbols is None:
            return None
        if isinstance(symbols, str):
            symbol_text = symbols.strip()
            if not symbol_text:
                raise ValueError("symbols cannot contain empty symbol")
            return {symbol_text}
        if isinstance(symbols, (list, tuple, set)):
            normalized: set[str] = set()
            for symbol in symbols:
                symbol_text = str(symbol).strip()
                if not symbol_text:
                    raise ValueError("symbols cannot contain empty symbol")
                normalized.add(symbol_text)
            if not normalized:
                raise ValueError("symbols cannot be empty")
            return normalized
        raise TypeError("symbols must be str, list[str], tuple[str, ...], set[str]")

    def _normalize_incremental_input_mode(self, source: str, input_mode: str) -> str:
        input_mode_key = str(input_mode).strip().lower()
        valid_modes = {"source", "hl", "hlc", "ohlc", "close_volume"}
        if input_mode_key not in valid_modes:
            raise ValueError(
                "input_mode must be one of: source, hl, hlc, ohlc, close_volume"
            )
        if input_mode_key == "source" and source not in {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }:
            raise ValueError("source must be one of: open, high, low, close, volume")
        return input_mode_key

    def _get_incremental_registration(
        self, name: str
    ) -> IncrementalIndicatorRegistration:
        item = self._incremental_indicators[name]
        if isinstance(item, IncrementalIndicatorRegistration):
            return item
        legacy_item = cast(Dict[str, Any], item)
        registration = IncrementalIndicatorRegistration(
            source=str(legacy_item.get("source", "close")).strip().lower(),
            symbols=legacy_item.get("symbols"),
            input_mode=str(legacy_item.get("input_mode", "source")).strip().lower(),
            warmup_bars=int(legacy_item.get("warmup_bars", 0) or 0),
            factory=legacy_item.get("indicator_factory"),
            base_indicator=legacy_item.get("indicator"),
        )
        self._incremental_indicators[name] = registration
        return registration

    def _get_incremental_indicator_instance(
        self, name: str, symbol: Optional[str] = None
    ) -> Any:
        registration = self._get_incremental_registration(name)
        resolved_symbol = self._resolve_incremental_indicator_symbol(
            registration, symbol
        )
        if resolved_symbol in registration.instances:
            return registration.instances[resolved_symbol]
        if registration.factory is not None:
            indicator = registration.factory()
        elif registration.primary_symbol is None:
            indicator = registration.base_indicator
            registration.primary_symbol = resolved_symbol
        elif registration.primary_symbol == resolved_symbol:
            indicator = registration.base_indicator
        else:
            raise ValueError(
                f"Incremental indicator '{name}' was registered with a shared instance "
                "and cannot be used across multiple symbols. Use indicator_factory "
                "for multi-symbol incremental indicators."
            )
        registration.instances[resolved_symbol] = indicator
        return indicator

    def _resolve_incremental_indicator_symbol(
        self, registration: IncrementalIndicatorRegistration, symbol: Optional[str]
    ) -> str:
        if symbol is not None:
            return str(symbol)
        if self.current_bar is not None:
            return str(self.current_bar.symbol)
        if self.current_tick is not None:
            return str(self.current_tick.symbol)
        if registration.primary_symbol is not None:
            return registration.primary_symbol
        if len(registration.instances) == 1:
            return next(iter(registration.instances))
        if registration.symbols is not None and len(registration.symbols) == 1:
            return next(iter(registration.symbols))
        return "__default__"

    def _build_incremental_indicator_args(
        self, payload: Any, source: str, input_mode: str
    ) -> Tuple[Any, ...]:
        if input_mode == "source":
            return (getattr(payload, source),)
        if input_mode == "hl":
            return (payload.high, payload.low)
        if input_mode == "hlc":
            return (payload.high, payload.low, payload.close)
        if input_mode == "ohlc":
            return (payload.open, payload.high, payload.low, payload.close)
        if input_mode == "close_volume":
            return (payload.close, payload.volume)
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    def on_order(self, order: Any) -> None:
        """
        订单状态更新回调.

        Args:
            order: 订单对象
        """
        pass

    def on_trade(self, trade: Any) -> None:
        """
        成交回调.

        Args:
            trade: 成交对象
        """
        pass

    def on_execution_report(self, report: Any) -> None:
        """执行回报回调（broker_live 专用；回测不触发）。默认 no-op，可覆盖."""
        pass

    def on_session_start(self, session: Any, timestamp: int) -> None:
        """会话开始回调."""
        pass

    def on_session_end(self, session: Any, timestamp: int) -> None:
        """会话结束回调."""
        pass

    def on_before_trading(self, trading_date: dt.date, timestamp: int) -> None:
        """交易日开始前回调."""
        pass

    def on_daily_rebalance(self, trading_date: dt.date, timestamp: int) -> None:
        """交易日调仓回调（每天最多一次）."""
        pass

    def on_daily_rebalance_after_bar(
        self, trading_date: dt.date, timestamp: int
    ) -> None:
        """完整时间片后的交易日调仓回调（每天最多一次）."""
        pass

    def on_after_trading(self, trading_date: dt.date, timestamp: int) -> None:
        """交易日结束后回调."""
        pass

    def on_pre_open(self, event: Dict[str, Any]) -> None:
        """开盘前回调，仅该阶段默认使用 next-open 成交语义."""
        pass

    def on_reject(self, order: Any) -> None:
        """拒单回调."""
        pass

    def on_portfolio_update(self, snapshot: Dict[str, Any]) -> None:
        """账户变化回调."""
        pass

    def on_error(self, error: Exception, source: str, payload: Any = None) -> None:
        """错误回调."""
        pass

    def on_expiry(self, event: Dict[str, Any]) -> None:
        """到期结算回调."""
        pass

    def _check_order_events(self) -> None:
        _check_order_events_impl(self)

    def _trade_event_key(self, trade: Any) -> Tuple[Any, ...]:
        return _trade_event_key_impl(self, trade)

    def _key_value(self, value: Any) -> Any:
        return _key_value_impl(value)

    def _remember_trade_key(self, key: Tuple[Any, ...]) -> bool:
        return _remember_trade_key_impl(self, key)

    def _trade_dedupe_cache_limit(self) -> int:
        return _trade_dedupe_cache_limit_impl(self)

    def _on_bar_event(self, bar: Bar, ctx: StrategyContext) -> None:
        _on_bar_event_impl(self, bar, ctx)

    def _on_tick_event(self, tick: Tick, ctx: StrategyContext) -> None:
        _on_tick_event_impl(self, tick, ctx)

    def _on_timer_event(self, payload: str, ctx: StrategyContext) -> None:
        _on_timer_event_impl(self, payload, ctx)

    def _flush_pending_order_events(self, ctx: StrategyContext) -> None:
        _flush_pending_order_events_impl(self, ctx)

    def on_bar(self, bar: Bar) -> None:
        """
        策略逻辑入口 (Bar 数据).

        用户应重写此方法.
        """
        pass

    @property
    def position(self) -> Position:
        """
        获取当前处理标的的持仓对象.

        支持常见的策略编写语法:
        if self.position.size == 0:
            ...
        """
        if self.ctx is None:
            raise RuntimeError("Context not ready")

        symbol = self._resolve_symbol(None)
        return Position(self.ctx, symbol)

    def on_tick(self, tick: Tick) -> None:
        """
        策略逻辑入口 (Tick 数据).

        用户应重写此方法.
        """
        pass

    def on_timer(self, payload: str) -> None:
        """
        策略逻辑入口 (Timer 事件).

        用户应重写此方法.
        """
        pass

    def _resolve_symbol(self, symbol: Optional[str] = None) -> str:
        return _resolve_symbol_impl(self, symbol)

    def _resolve_record_timestamp(self) -> int:
        """Resolve the current runtime timestamp for indicator recording."""
        if self.current_bar is not None:
            return int(self.current_bar.timestamp)
        if self.current_tick is not None:
            return int(self.current_tick.timestamp)
        if self.ctx is not None:
            current_time = getattr(self.ctx, "current_time", None)
            if current_time is not None:
                return int(current_time)
        raise RuntimeError(
            "record_indicator requires an active bar/tick or context timestamp"
        )

    def record_indicator(
        self,
        name: str,
        value: Any,
        symbol: Optional[str] = None,
        *,
        timestamp: Optional[Any] = None,
        display_name: Optional[str] = None,
        pane: str = "sub",
        render_type: str = "line",
        unit: Optional[str] = None,
        precision: Optional[int] = None,
        color: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        warmup: bool = False,
    ) -> None:
        """Record one indicator point for downstream visualization and export."""
        recorder = getattr(self, "_indicator_recorder", None)
        if recorder is None:
            return

        resolved_symbol = self._resolve_symbol(symbol)
        resolved_timestamp = (
            self._resolve_record_timestamp()
            if timestamp is None
            else int(pd.Timestamp(timestamp).value)
        )
        owner_strategy_id = str(
            getattr(self, "_owner_strategy_id", "_default") or "_default"
        )
        recorder.record(
            name=name,
            value=value,
            symbol=resolved_symbol,
            timestamp=resolved_timestamp,
            owner_strategy_id=owner_strategy_id,
            display_name=display_name,
            pane=pane,
            render_type=render_type,
            unit=unit,
            precision=precision,
            color=color,
            meta=meta,
            warmup=warmup,
        )

    def get_position(self, symbol: Optional[str] = None) -> float:
        """
        获取指定标的的持仓数量.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)

        Returns:
            float: 持仓数量 (正数为多头, 负数为空头)
        """
        return _get_position_impl(self, symbol)

    def get_available_position(self, symbol: Optional[str] = None) -> float:
        """
        获取指定标的的可用持仓数量 (考虑 T+1 等限制).

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)

        Returns:
            float: 可用持仓数量
        """
        return _get_available_position_impl(self, symbol)

    def get_holding_bars(self, symbol: Optional[str] = None) -> int:
        """
        获取当前持仓持有的 Bar 数量.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)

        Returns:
            int: 持有的 Bar 数量. 如果未持仓，返回 0.
        """
        return _get_holding_bars_impl(self, symbol)

    @property
    def positions(self) -> Dict[str, float]:
        """所有标的持仓 {symbol: qty}（等同旧 get_positions()）."""
        return _get_positions_impl(self)

    def _set_instrument_snapshots(
        self, snapshots: Dict[str, InstrumentSnapshot]
    ) -> None:
        """设置策略可访问的标的属性快照."""
        self._instrument_snapshots = dict(snapshots)

    def get_instrument(self, symbol: str) -> InstrumentSnapshot:
        """返回单个标的属性快照."""
        if symbol not in self._instrument_snapshots:
            raise KeyError(f"Instrument config not found for symbol: {symbol}")
        return self._instrument_snapshots[symbol]

    def get_instruments(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, InstrumentSnapshot]:
        """返回标的属性快照字典."""
        if symbols is None:
            return dict(self._instrument_snapshots)
        return {s: self.get_instrument(s) for s in symbols}

    def get_instrument_field(self, symbol: str, field: str) -> Any:
        """返回标的单个字段值."""
        snapshot = self.get_instrument(symbol)
        if hasattr(snapshot, field):
            return getattr(snapshot, field)
        if field in snapshot.static_attrs:
            return snapshot.static_attrs[field]
        raise KeyError(f"Field '{field}' not found for instrument: {symbol}")

    def get_instrument_config(
        self, symbol: str, fields: Union[str, List[str], None] = None
    ) -> Union[Any, Dict[str, Any], InstrumentSnapshot]:
        """返回标的配置数据."""
        snapshot = self.get_instrument(symbol)
        if fields is None:
            return snapshot
        if isinstance(fields, str):
            return self.get_instrument_field(symbol, fields)
        if isinstance(fields, list):
            return {field: self.get_instrument_field(symbol, field) for field in fields}
        raise TypeError("fields must be str, List[str], or None")

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """
        获取当前未完成的订单.

        Args:
            symbol: 标的代码 (如果为 None，返回所有标的订单)

        Returns:
            List[Order]: 订单列表
        """
        return _get_open_orders_impl(self, symbol)

    def get_order(self, order_id: str) -> Optional[Any]:
        """
        获取指定订单详情.

        Args:
            order_id: 订单 ID

        Returns:
            Order: 订单对象，如果未找到则返回 None
        """
        # 经执行后端(与 get_position/get_account 一致): 回测 SimExecution→ctx/
        # _known_orders, broker_live BrokerExecution→柜台挂单扫描(否则恒 None)。
        return self.execution.get_order(order_id)

    def get_account(self) -> Dict[str, Any]:
        """
        获取账户资金详情快照.

        Returns:
            Dict: 包含以下字段:
                - cash: 现金余额 (期货保证金账户下, 开仓不从此扣减保证金,
                  故并非可用于新开仓的资金; 请改用 free_margin)
                - equity: 总权益 (现金 + 持仓市值 + 浮动盈亏)
                - market_value: 持仓总市值 (equity - cash)
                - frozen_cash: 当前未完成订单预占资金
                - margin: 当前仓位占用保证金
                - used_margin: 同 margin
                - free_margin: 可用保证金 (equity - margin), 即可用于新开仓的资金;
                  与拒单信息中的 Available 口径一致
                - borrowed_cash: 融资负债 (现金为负时)
                - short_market_value: 空头市值
                - maintenance_ratio: 维持担保比例
                - account_mode: 账户模式 ("cash" / "margin")
                - accrued_interest: 累计计提利息
                - daily_interest: 当日计提利息
        """
        return _get_account_impl(self)

    def get_trades(self) -> List[Any]:
        """
        获取所有历史成交记录 (Closed Trades).

        Returns:
            List[ClosedTrade]: 已平仓交易列表
        """
        if self.ctx:
            return self.ctx.closed_trades
        return []

    def cancel_order(self, order_id: Union[str, Any]) -> None:
        """
        取消指定订单.

        Args:
            order_id: 订单 ID，接受 str | OrderReceipt（传 OrderReceipt 时取 .primary）
        """
        _cancel_order_impl(self, order_id)

    def cancel_group(self, group_id: Any) -> None:
        """按 group_id（或 OrderReceipt）撤销一个逻辑委托的全部腿."""
        _cancel_group_impl(self, group_id)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """
        取消当前所有未完成的订单.

        Args:
            symbol: 标的代码 (如果不填, 取消所有标的订单)
        """
        _cancel_all_orders_impl(self, symbol)

    def place_oco(
        self,
        first_order_id: Union[str, OrderReceipt],
        second_order_id: Union[str, OrderReceipt],
        group_id: Optional[str] = None,
    ) -> str:
        """
        创建 OCO 订单组.

        当组内任一订单成交时，自动撤销另一订单。

        Args:
            first_order_id: 订单 ID，接受 str | OrderReceipt
                （传 OrderReceipt 时取 .primary）
            second_order_id: 同上
        """
        # first_order_id/second_order_id 现可能是 buy()/sell() 返回的 OrderReceipt
        # （回测拆腿产生多个 id 时）；取其 .primary（首腿 broker_order_id）落地为
        # 引擎/OCO 簿实际使用的单一订单号字符串。
        first = str(getattr(first_order_id, "primary", first_order_id)).strip()
        second = str(getattr(second_order_id, "primary", second_order_id)).strip()
        if not first or not second:
            raise ValueError("OCO order ids cannot be empty")
        if first == second:
            raise ValueError("OCO order ids must be different")

        with self._order_group_lock:
            if group_id is None or not str(group_id).strip():
                self._order_group_seq += 1
                group_id = f"oco-{self._order_group_seq}"
            group_key = str(group_id).strip()

            engine = getattr(self, "_engine", None)
            register_oco = getattr(engine, "register_oco_group", None)
            if callable(register_oco):
                try:
                    register_oco(group_key, first, second)
                    self._use_engine_oco = True
                    return group_key
                except Exception:
                    self._pending_engine_oco_groups.append((group_key, first, second))
                    self._use_engine_oco = True
                    return group_key

            self._detach_oco_order(first)
            self._detach_oco_order(second)

            self._oco_groups[group_key] = {first, second}
            self._oco_order_to_group[first] = group_key
            self._oco_order_to_group[second] = group_key
            return group_key

    def place_bracket(
        self,
        symbol: str,
        quantity: float,
        entry_price: Optional[float] = None,
        stop_trigger_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        entry_tag: Optional[str] = None,
        stop_tag: Optional[str] = None,
        take_profit_tag: Optional[str] = None,
    ) -> str:
        """
        创建 Bracket 订单.

        先提交进场单，待进场成交后自动挂出止损/止盈，并绑定 OCO。
        """
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if stop_trigger_price is None and take_profit_price is None:
            raise ValueError("stop_trigger_price or take_profit_price must be provided")

        entry_receipt = self.buy(
            symbol=symbol,
            quantity=quantity,
            price=entry_price,
            time_in_force=time_in_force,
            tag=entry_tag,
        )
        if not entry_receipt:
            raise RuntimeError("failed to submit bracket entry order")
        # self.buy() 现返回 OrderReceipt（回测拆腿产生多个 id 时）；取其
        # .primary（首腿 broker_order_id）落地为下方引擎登记/挂起字典使用的
        # 单一订单号字符串，与 trade.order_id（Rust 侧始终为 str）保持可比对。
        entry_order_id: str = str(getattr(entry_receipt, "primary", entry_receipt))

        # self.buy(入场单) 在锁外提交; 仅 bracket 记账/引擎登记在锁内(与协调器互斥).
        with self._order_group_lock:
            engine = getattr(self, "_engine", None)
            register_bracket = getattr(engine, "register_bracket_plan", None)
            if callable(register_bracket):
                try:
                    register_bracket(
                        entry_order_id,
                        stop_trigger_price,
                        take_profit_price,
                        time_in_force,
                        stop_tag,
                        take_profit_tag,
                    )
                    self._use_engine_bracket = True
                    return entry_order_id
                except Exception:
                    self._pending_engine_bracket_plans.append(
                        (
                            entry_order_id,
                            stop_trigger_price,
                            take_profit_price,
                            time_in_force,
                            stop_tag,
                            take_profit_tag,
                        )
                    )
                    self._use_engine_bracket = True
                    return entry_order_id

            self._pending_brackets[entry_order_id] = {
                "symbol": symbol,
                "quantity": float(quantity),
                "stop_trigger_price": stop_trigger_price,
                "take_profit_price": take_profit_price,
                "time_in_force": time_in_force,
                "stop_tag": stop_tag,
                "take_profit_tag": take_profit_tag,
            }
            return entry_order_id

    def _process_order_groups(self, trade: Any) -> None:
        with self._order_group_lock:
            self._process_pending_bracket(trade)
            self._process_oco_trade(trade)

    def _process_pending_bracket(self, trade: Any) -> None:
        if self._use_engine_bracket:
            return
        order_id = str(getattr(trade, "order_id", "") or "")
        if not order_id:
            return

        bracket = self._pending_brackets.pop(order_id, None)
        if bracket is None:
            return

        trade_symbol = getattr(trade, "symbol", None)
        symbol = str(trade_symbol) if trade_symbol else str(bracket["symbol"])

        trade_qty = getattr(trade, "quantity", None)
        if trade_qty is None:
            quantity = float(bracket["quantity"])
        else:
            quantity = float(trade_qty)

        stop_order_id = ""
        take_order_id = ""
        stop_trigger_price = bracket["stop_trigger_price"]
        take_profit_price = bracket["take_profit_price"]
        time_in_force = cast(Optional[TimeInForce], bracket["time_in_force"])

        if stop_trigger_price is not None:
            # self.sell() 现返回 OrderReceipt；取其 .primary 落地为下方 place_oco
            # 使用的单一订单号字符串（与 place_bracket 的规整方式一致）。
            stop_receipt = self.sell(
                symbol=symbol,
                quantity=quantity,
                trigger_price=float(stop_trigger_price),
                time_in_force=time_in_force,
                tag=cast(Optional[str], bracket["stop_tag"]),
            )
            stop_order_id = str(getattr(stop_receipt, "primary", stop_receipt))

        if take_profit_price is not None:
            take_receipt = self.sell(
                symbol=symbol,
                quantity=quantity,
                price=float(take_profit_price),
                time_in_force=time_in_force,
                tag=cast(Optional[str], bracket["take_profit_tag"]),
            )
            take_order_id = str(getattr(take_receipt, "primary", take_receipt))

        if stop_order_id and take_order_id:
            self.place_oco(stop_order_id, take_order_id)

    def _process_oco_trade(self, trade: Any) -> None:
        if self._use_engine_oco:
            return
        order_id = str(getattr(trade, "order_id", "") or "")
        if not order_id:
            return

        group_id = self._oco_order_to_group.get(order_id)
        if not group_id:
            return

        group_orders = self._oco_groups.get(group_id, set())
        peer_orders = [oid for oid in group_orders if oid != order_id]
        for peer_order_id in peer_orders:
            self.cancel_order(peer_order_id)

        self._remove_oco_group(group_id)

    def _detach_oco_order(self, order_id: str) -> None:
        old_group = self._oco_order_to_group.get(order_id)
        if not old_group:
            return
        group_orders = self._oco_groups.get(old_group)
        if group_orders is not None:
            group_orders.discard(order_id)
            if len(group_orders) <= 1:
                self._remove_oco_group(old_group)
        self._oco_order_to_group.pop(order_id, None)

    def _remove_oco_group(self, group_id: str) -> None:
        group_orders = self._oco_groups.pop(group_id, set())
        for oid in group_orders:
            self._oco_order_to_group.pop(oid, None)

    def buy(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
        tag: Optional[str] = None,
        fill_policy: Optional[Dict[str, Any]] = None,
        slippage: Optional[Union[float, Dict[str, Any]]] = None,
        commission: Optional[Dict[str, Any]] = None,
    ) -> Union[str, OrderReceipt]:
        """
        买入下单.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 使用 Sizer 计算)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
            tag: 订单标签

        Returns:
            回测模式下为 OrderReceipt（拆腿时携带全部订单 id；str() 取 group_id）；
            实盘模式下暂仍为 str（订单号）。
        """
        return _buy_impl(
            self,
            symbol,
            quantity,
            price,
            time_in_force,
            trigger_price,
            tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
        )

    def sell(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
        tag: Optional[str] = None,
        fill_policy: Optional[Dict[str, Any]] = None,
        slippage: Optional[Union[float, Dict[str, Any]]] = None,
        commission: Optional[Dict[str, Any]] = None,
    ) -> Union[str, OrderReceipt]:
        """
        卖出下单.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 默认卖出当前标的所有持仓)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
            tag: 订单标签

        Returns:
            回测模式下为 OrderReceipt（拆腿时携带全部订单 id；str() 取 group_id）；
            实盘模式下暂仍为 str（订单号）。
        """
        return _sell_impl(
            self,
            symbol,
            quantity,
            price,
            time_in_force,
            trigger_price,
            tag,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
        )

    def submit_order(
        self,
        symbol: Optional[str] = None,
        side: str = "Buy",
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce | str] = None,
        trigger_price: Optional[float] = None,
        tag: Optional[str] = None,
        client_order_id: Optional[str] = None,
        order_type: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        broker_options: Optional[Dict[str, Any]] = None,
        trail_offset: Optional[float] = None,
        trail_reference_price: Optional[float] = None,
        fill_policy: Optional[Dict[str, Any]] = None,
        slippage: Optional[Union[float, Dict[str, Any]]] = None,
        commission: Optional[Dict[str, Any]] = None,
        position_effect: Optional[str] = None,
        reduce_only: bool = False,
        asset_type: str = "stock",
    ) -> Union[str, OrderReceipt]:
        """
        统一下单接口.

        该接口在回测与实盘模式均可调用，实盘模式下会由 LiveRunner 注入增强能力。

        Returns:
            回测模式下为 OrderReceipt（拆腿时携带全部订单 id；str() 取 group_id）；
            实盘模式下暂仍为 str（订单号）。
        """
        return _submit_order_impl(
            self,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            trigger_price=trigger_price,
            tag=tag,
            client_order_id=client_order_id,
            order_type=order_type,
            extra=extra,
            broker_options=broker_options,
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
            fill_policy=fill_policy,
            slippage=slippage,
            commission=commission,
            position_effect=position_effect,
            reduce_only=reduce_only,
            asset_type=asset_type,
        )

    def can_submit_client_order(self, client_order_id: str) -> bool:
        """
        检查 client_order_id 是否可再次提交.

        该方法在 LiveRunner 的 broker_live 模式下会被注入真实实现。
        """
        _ = client_order_id
        return True

    def get_execution_capabilities(self) -> Dict[str, Any]:
        """获取当前执行能力描述."""
        return _get_execution_capabilities_impl(self)

    def get_last_target_positions_plan(self) -> Dict[str, Any]:
        """获取最近一次 rebalance_positions() 生成的调仓计划."""
        return _get_last_target_positions_plan_impl(self)

    def place_trailing_stop(
        self,
        symbol: str,
        quantity: float,
        trail_offset: float,
        side: str = "Sell",
        trail_reference_price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        tag: Optional[str] = None,
    ) -> Union[str, OrderReceipt]:
        """创建跟踪止损单 (StopTrail)."""
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if trail_offset <= 0:
            raise ValueError("trail_offset must be > 0")
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            time_in_force=time_in_force,
            tag=tag,
            order_type="StopTrail",
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
        )

    def place_trailing_stop_limit(
        self,
        symbol: str,
        quantity: float,
        price: float,
        trail_offset: float,
        side: str = "Sell",
        trail_reference_price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        tag: Optional[str] = None,
    ) -> Union[str, OrderReceipt]:
        """创建跟踪止损限价单 (StopTrailLimit)."""
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if price <= 0:
            raise ValueError("price must be > 0")
        if trail_offset <= 0:
            raise ValueError("trail_offset must be > 0")
        return self.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            tag=tag,
            order_type="StopTrailLimit",
            trail_offset=trail_offset,
            trail_reference_price=trail_reference_price,
        )

    @property
    def cash(self) -> float:
        """当前可用现金（等同旧 get_cash()）."""
        return _get_cash_impl(self)

    @property
    def equity(self) -> float:
        """当前账户总权益（现金 + 持仓市值，权威只读；等同旧 get_portfolio_value()）."""
        return _get_portfolio_value_impl(self)

    def _inject_cost_config(
        self,
        *,
        commission_policy: Optional[Dict[str, Any]] = None,
        min_commission: Optional[float] = None,
        stamp_tax_rate: Optional[float] = None,
        transfer_fee_rate: Optional[float] = None,
        lot_size: Any = None,
    ) -> None:
        """框架内部写入成本/手数配置(绕过费率只读 setter)。None 表示不覆盖既有值."""
        cfg = self._cost_config
        if commission_policy is not None:
            cfg["commission_policy"] = dict(commission_policy)
        if min_commission is not None:
            cfg["min_commission"] = float(min_commission)
        if stamp_tax_rate is not None:
            cfg["stamp_tax_rate"] = float(stamp_tax_rate)
        if transfer_fee_rate is not None:
            cfg["transfer_fee_rate"] = float(transfer_fee_rate)
        if lot_size is not None:
            cfg["lot_size"] = lot_size

    @property
    def commission_policy(self) -> Dict[str, Any]:
        """佣金策略(只读; type: percent/fixed/per_unit).

        经 run_backtest/BacktestConfig 配置。
        """
        return cast(Dict[str, Any], self._cost_config["commission_policy"])

    @commission_policy.setter
    def commission_policy(self, value: Any) -> None:
        raise AttributeError(
            "commission_policy 是回测配置项，请用 "
            'run_backtest(commission_policy={"type":"percent","value":0.0003}) '
            "或 BacktestConfig 设置，不要写 self.commission_policy。"
        )

    @property
    def commission_rate(self) -> float:
        """佣金率只读视图(percent 策略取 value, 否则 0.0).

        写请用 run_backtest(commission_rate=)。
        """
        policy = self._cost_config["commission_policy"]
        if str(policy.get("type", "percent")).strip().lower() == "percent":
            return float(policy.get("value", 0.0) or 0.0)
        return 0.0

    @commission_rate.setter
    def commission_rate(self, value: Any) -> None:
        raise AttributeError(
            "commission_rate 是回测配置项，请用 run_backtest(commission_rate=0.0003) "
            "或 BacktestConfig 设置，不要写 self.commission_rate。"
        )

    @property
    def min_commission(self) -> float:
        """最低佣金(只读)。经 run_backtest(min_commission=)/BacktestConfig 配置."""
        return float(self._cost_config["min_commission"])

    @min_commission.setter
    def min_commission(self, value: Any) -> None:
        raise AttributeError(
            "min_commission 是回测配置项，请用 run_backtest(min_commission=) "
            "或 BacktestConfig 设置，不要写 self.min_commission。"
        )

    @property
    def stamp_tax_rate(self) -> float:
        """印花税率(只读; 卖出侧).

        经 run_backtest(stamp_tax_rate=)/BacktestConfig 配置。
        """
        return float(self._cost_config["stamp_tax_rate"])

    @stamp_tax_rate.setter
    def stamp_tax_rate(self, value: Any) -> None:
        raise AttributeError(
            "stamp_tax_rate 是回测配置项，请用 run_backtest(stamp_tax_rate=) "
            "或 BacktestConfig 设置，不要写 self.stamp_tax_rate。"
        )

    @property
    def transfer_fee_rate(self) -> float:
        """过户费率(只读)。经 run_backtest(transfer_fee_rate=)/BacktestConfig 配置."""
        return float(self._cost_config["transfer_fee_rate"])

    @transfer_fee_rate.setter
    def transfer_fee_rate(self, value: Any) -> None:
        raise AttributeError(
            "transfer_fee_rate 是回测配置项，请用 run_backtest(transfer_fee_rate=) "
            "或 BacktestConfig 设置，不要写 self.transfer_fee_rate。"
        )

    @property
    def lot_size(self) -> Any:
        """最小交易单位(可写; int 或 Dict[str, int])。默认 1(美股/加密), A股设 100."""
        return self._cost_config["lot_size"]

    @lot_size.setter
    def lot_size(self, value: Any) -> None:
        self._cost_config["lot_size"] = value

    def order_target(
        self,
        symbol: Optional[str] = None,
        target: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        调整仓位到目标数量.

        :param symbol: 标的代码 (不填默认当前 Bar/Tick 的 symbol)
        :param target: 目标持仓数量 (例如 100, -100)
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        :return: 本次调仓产生的订单 ID; 若无需交易则返回 None
        """
        return _order_target_impl(self, symbol, target, price, **kwargs)

    def _calculate_max_buy_qty(self, symbol: str, price: float, cash: float) -> float:
        """
        计算考虑费率后的最大可买数量.

        :param symbol: 标的代码
        :param price: 交易价格
        :param cash: 可用资金
        :return: 最大可买数量
        """
        return _calculate_max_buy_qty_impl(self, symbol, price, cash)

    def order_target_value(
        self,
        symbol: Optional[str] = None,
        target_value: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        调整仓位到目标价值.

        :param symbol: 标的代码 (不填默认当前 Bar/Tick 的 symbol)
        :param target_value: 目标持仓价值
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        :return: 本次调仓产生的订单 ID; 若无需交易或无法定价则返回 None
        """
        return _order_target_value_impl(self, symbol, target_value, price, **kwargs)

    def order_target_percent(
        self,
        symbol: Optional[str] = None,
        target_percent: Optional[float] = None,
        price: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        调整仓位到目标百分比.

        :param symbol: 标的代码 (不填默认当前 Bar/Tick 的 symbol)
        :param target_percent: 目标持仓比例 (0.5 = 50%)
        :param price: 限价 (可选)
        :param kwargs: 其他下单参数
        :return: 本次调仓产生的订单 ID; 若无需交易则返回 None
        """
        return _order_target_percent_impl(self, symbol, target_percent, price, **kwargs)

    def rebalance_weights(
        self,
        target_weights: Dict[str, float],
        price_map: Optional[Dict[str, float]] = None,
        liquidate_unmentioned: bool = False,
        allow_leverage: bool = False,
        rebalance_tolerance: float = 0.0,
        **kwargs: Any,
    ) -> List[str]:
        """
        按多标的目标权重调仓.

        :param target_weights: 目标权重字典 {symbol: weight}
        :param price_map: 每个标的的委托价格字典（可选）
        :param liquidate_unmentioned: 是否清仓未出现在目标权重中的现有持仓
        :param allow_leverage: 是否允许目标权重和超过 1.0
        :param rebalance_tolerance: 调仓容忍阈值（按组合市值比例）
        :param kwargs: 其他下单参数
        :return: 本次调仓产生的所有订单 ID 列表 (无交易时为空列表)
        """
        return _rebalance_weights_impl(
            self,
            target_weights,
            price_map,
            liquidate_unmentioned,
            allow_leverage,
            rebalance_tolerance,
            **kwargs,
        )

    def rebalance_positions(
        self,
        target_positions: Dict[str, float],
        price_map: Optional[Dict[str, float]] = None,
        liquidate_unmentioned: bool = False,
        rebalance_tolerance: float = 0.0,
        allow_short: Optional[bool] = None,
        strict_short_capability: bool = True,
        missing_price_mode: str = "ignore",
        **kwargs: Any,
    ) -> List[str]:
        """
        按多标的目标持仓数量调仓，支持正负目标仓位.

        :param target_positions: 目标持仓数量字典 {symbol: quantity}
        :param price_map: 每个标的的委托价格字典（可选）
        :param liquidate_unmentioned: 是否将未出现的现有持仓调到 0
        :param rebalance_tolerance: 调仓容忍阈值（按绝对数量）
        :param allow_short: 是否允许负目标仓位；默认按执行环境能力自动推断
        :param strict_short_capability: 当执行环境未声明可做空时是否严格拒绝
        :param missing_price_mode: price_map 缺项时的处理方式: ignore / skip / fail
        :param kwargs: 其他下单参数
        :return: 本次调仓产生的所有订单 ID 列表 (无交易时为空列表)
        """
        return _rebalance_positions_impl(
            self,
            target_positions,
            price_map,
            liquidate_unmentioned,
            rebalance_tolerance,
            allow_short,
            strict_short_capability,
            missing_price_mode,
            **kwargs,
        )

    def close_position(self, symbol: Optional[str] = None) -> None:
        """
        平仓 (Close Position).

        卖出/买入以抵消当前持仓.

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
        """
        _close_position_impl(self, symbol)

    def short(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
        tag: Optional[str] = None,
        fill_policy: Optional[Dict[str, Any]] = None,
        slippage: Optional[Union[float, Dict[str, Any]]] = None,
        commission: Optional[Dict[str, Any]] = None,
        reduce_only: bool = False,
    ) -> None:
        """
        卖出开空 (Short Sell).

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 使用 Sizer 计算)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
        """
        _short_impl(
            self,
            symbol,
            quantity,
            price,
            time_in_force,
            trigger_price,
            tag,
            fill_policy,
            slippage,
            commission,
            reduce_only,
        )

    def cover(
        self,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: Optional[TimeInForce] = None,
        trigger_price: Optional[float] = None,
        tag: Optional[str] = None,
        fill_policy: Optional[Dict[str, Any]] = None,
        slippage: Optional[Union[float, Dict[str, Any]]] = None,
        commission: Optional[Dict[str, Any]] = None,
        reduce_only: bool = False,
    ) -> None:
        """
        买入平空 (Buy to Cover).

        Args:
            symbol: 标的代码 (如果不填, 默认使用当前 Bar/Tick 的 symbol)
            quantity: 数量 (如果不填, 默认平掉当前标的所有空头持仓)
            price: 限价 (None 为市价)
            time_in_force: 订单有效期
            trigger_price: 触发价 (止损/止盈)
        """
        _cover_impl(
            self,
            symbol,
            quantity,
            price,
            time_in_force,
            trigger_price,
            tag,
            fill_policy,
            slippage,
            commission,
            reduce_only,
        )


class VectorizedStrategy(Strategy):
    """
    向量化策略基类 (Vectorized Strategy Base Class).

    支持预计算指标的高速回测模式.
    用户应在回测前使用 Pandas/Numpy 计算好所有指标,
    然后通过本类提供的高速游标访问机制在 on_bar 中读取.
    """

    def __init__(self, precalculated_data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Initialize VectorizedStrategy.

        :param precalculated_data: 预计算数据字典
                                  Structure: {symbol: {indicator_name: numpy_array}}
        """
        super().__init__()
        self.precalc = precalculated_data
        # 游标管理: {symbol: index}
        self.cursors: defaultdict[str, int] = defaultdict(int)

        # 默认禁用 Python 侧历史数据缓存以提升性能
        self.set_history_depth(0)

    def _on_bar_event(self, bar: Bar, ctx: StrategyContext) -> None:
        """Wrap the user on_bar handler internally."""
        super()._on_bar_event(bar, ctx)
        self.cursors[bar.symbol] += 1

    def get_value(self, name: str, symbol: Optional[str] = None) -> float:
        """
        获取当前 Bar 对应的预计算指标值.

        Args:
            name: 指标名称
            symbol: 标的代码 (如果不填, 默认使用当前 Bar 的 symbol)

        Returns:
            指标值 (float). 如果不存在或越界，返回 nan.
        """
        symbol = self._resolve_symbol(symbol)
        idx = self.cursors[symbol]

        try:
            return float(self.precalc[symbol][name][idx])
        except (KeyError, IndexError):
            return float("nan")
