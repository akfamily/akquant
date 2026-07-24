import os
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import pandas as pd
import polars as pl
import pyarrow as pa

from ..akquant import AssetType, Bar, DataFeed
from ..config import BacktestConfig, RiskConfig
from ..feed_adapter import DataFeedAdapter
from ..indicator_recording import IndicatorSink
from ..strategy import Strategy, StrategyRuntimeConfig
from .fill_mode import CurrentClose as CurrentClose
from .fill_mode import FillMode as FillMode
from .fill_mode import NextAverage as NextAverage
from .fill_mode import NextClose as NextClose
from .fill_mode import NextHighLowMid as NextHighLowMid
from .fill_mode import NextOpen as NextOpen
from .merge import MergedResult as MergedResult
from .merge import merge_results as merge_results
from .result import BacktestResult

BacktestDataInput = Union[
    pd.DataFrame,
    pl.DataFrame,
    pl.LazyFrame,
    pa.Table,
    Dict[str, pd.DataFrame],
    List[Bar],
    DataFeed,
    DataFeedAdapter,
]

class FunctionalStrategy(Strategy):
    def __init__(
        self,
        initialize: Optional[Callable[[Any], None]],
        on_bar: Optional[Callable[[Any, Bar], None]],
        on_start: Optional[Callable[[Any], None]] = ...,
        on_resume: Optional[Callable[[Any], None]] = ...,
        on_train_signal: Optional[Callable[[Any], None]] = ...,
        on_stop: Optional[Callable[[Any], None]] = ...,
        on_tick: Optional[Callable[[Any, Any], None]] = ...,
        on_order: Optional[Callable[[Any, Any], None]] = ...,
        on_trade: Optional[Callable[[Any, Any], None]] = ...,
        on_reject: Optional[Callable[[Any, Any], None]] = ...,
        on_before_trading: Optional[Callable[[Any, Any, int], None]] = ...,
        on_after_trading: Optional[Callable[[Any, Any, int], None]] = ...,
        on_cross_section: Optional[Callable[[Any, Any, int], None]] = ...,
        on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
        on_error: Optional[Callable[[Any, Exception, str, Any], None]] = ...,
        on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
        on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
        on_timer: Optional[Callable[[Any, str], None]] = ...,
        context: Optional[Dict[str, Any]] = ...,
    ) -> None: ...

class BacktestStreamEvent(TypedDict):
    run_id: str
    seq: int
    ts: int
    event_type: str
    symbol: Optional[str]
    level: str
    payload: Dict[str, Any]

class SlippagePolicy(TypedDict, total=False):
    type: Literal["percent", "fixed", "ticks", "zero"]
    value: float

SlippagePolicyInput = Union[SlippagePolicy, Dict[str, Any]]

class CommissionPolicy(TypedDict):
    type: Literal["percent", "fixed", "per_unit"]
    value: float

CommissionPolicyInput = Union[CommissionPolicy, Dict[str, Any]]

def make_fill_policy(*args: Any, **kwargs: Any) -> FillMode: ...
def run_backtest(
    data: Optional[BacktestDataInput] = ...,
    strategy: Union[Type[Strategy], Strategy, Callable[[Any, Bar], None], None] = ...,
    strategy_source: Optional[Union[str, bytes, os.PathLike[str]]] = ...,
    strategy_loader: Optional[str] = ...,
    strategy_loader_options: Optional[Dict[str, Any]] = ...,
    symbols: Union[str, List[str], Tuple[str, ...], set[str]] = ...,
    initial_cash: Optional[float] = ...,
    commission_policy: Optional[CommissionPolicyInput] = ...,
    commission_rate: Optional[float] = ...,
    stamp_tax_rate: Optional[float] = ...,
    transfer_fee_rate: Optional[float] = ...,
    min_commission: Optional[float] = ...,
    slippage: Optional[Union[float, SlippagePolicyInput]] = ...,
    volume_limit_pct: Optional[float] = ...,
    timezone: Optional[str] = ...,
    t_plus_one: bool = ...,
    initialize: Optional[Callable[[Any], None]] = ...,
    on_start: Optional[Callable[[Any], None]] = ...,
    on_resume: Optional[Callable[[Any], None]] = ...,
    on_train_signal: Optional[Callable[[Any], None]] = ...,
    on_stop: Optional[Callable[[Any], None]] = ...,
    on_tick: Optional[Callable[[Any, Any], None]] = ...,
    on_order: Optional[Callable[[Any, Any], None]] = ...,
    on_trade: Optional[Callable[[Any, Any], None]] = ...,
    on_reject: Optional[Callable[[Any, Any], None]] = ...,
    on_before_trading: Optional[Callable[[Any, Any, int], None]] = ...,
    on_after_trading: Optional[Callable[[Any, Any, int], None]] = ...,
    on_cross_section: Optional[Callable[[Any, Any, int], None]] = ...,
    on_portfolio_update: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
    on_error: Optional[Callable[[Any, Exception, str, Any], None]] = ...,
    on_expiry: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
    on_pre_open: Optional[Callable[[Any, Dict[str, Any]], None]] = ...,
    on_timer: Optional[Callable[[Any, str], None]] = ...,
    context: Optional[Dict[str, Any]] = ...,
    history_depth: Optional[int] = ...,
    warmup_period: int = ...,
    lot_size: Union[int, Dict[str, int], None] = ...,
    show_progress: Optional[bool] = ...,
    start_time: Optional[Union[str, Any]] = ...,
    end_time: Optional[Union[str, Any]] = ...,
    config: Optional[BacktestConfig] = ...,
    custom_matchers: Optional[Dict[AssetType, Any]] = ...,
    risk_config: Optional[Union[Dict[str, Any], RiskConfig]] = ...,
    strategy_runtime_config: Optional[
        Union[StrategyRuntimeConfig, Dict[str, Any]]
    ] = ...,
    runtime_config_override: bool = ...,
    strategy_id: Optional[str] = ...,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ] = ...,
    strategy_max_order_value: Optional[Dict[str, float]] = ...,
    strategy_max_order_size: Optional[Dict[str, float]] = ...,
    strategy_max_position_size: Optional[Dict[str, float]] = ...,
    strategy_max_daily_loss: Optional[Dict[str, float]] = ...,
    strategy_max_drawdown: Optional[Dict[str, float]] = ...,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = ...,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = ...,
    strategy_priority: Optional[Dict[str, int]] = ...,
    strategy_risk_budget: Optional[Dict[str, float]] = ...,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = ...,
    strategy_slippage: Optional[Dict[str, Union[float, SlippagePolicyInput]]] = ...,
    strategy_commission: Optional[Dict[str, CommissionPolicyInput]] = ...,
    portfolio_risk_budget: Optional[float] = ...,
    risk_budget_mode: Literal["order_notional", "trade_notional"] = ...,
    risk_budget_reset_daily: bool = ...,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = ...,
    indicator_recorder: Optional[IndicatorSink] = ...,
    broker_profile: Optional[str] = ...,
    fill_policy: Optional[FillMode] = ...,
    stream_mode: Literal["observability", "audit"] = ...,
    strict_strategy_params: bool = True,
    **kwargs: Any,
) -> BacktestResult: ...
def run_from_checkpoint(
    checkpoint_path: str,
    data: Optional[BacktestDataInput] = ...,
    show_progress: bool = ...,
    symbols: Union[str, List[str], Tuple[str, ...], set[str]] = ...,
    commission_policy: Optional[CommissionPolicyInput] = ...,
    strategy_runtime_config: Optional[
        Union[StrategyRuntimeConfig, Dict[str, Any]]
    ] = ...,
    runtime_config_override: bool = ...,
    strategy_id: Optional[str] = ...,
    strategies_by_slot: Optional[
        Dict[str, Union[Type[Strategy], Strategy, Callable[[Any, Bar], None]]]
    ] = ...,
    strategy_max_order_value: Optional[Dict[str, float]] = ...,
    strategy_max_order_size: Optional[Dict[str, float]] = ...,
    strategy_max_position_size: Optional[Dict[str, float]] = ...,
    strategy_max_daily_loss: Optional[Dict[str, float]] = ...,
    strategy_max_drawdown: Optional[Dict[str, float]] = ...,
    strategy_reduce_only_after_risk: Optional[Dict[str, bool]] = ...,
    strategy_risk_cooldown_bars: Optional[Dict[str, int]] = ...,
    strategy_priority: Optional[Dict[str, int]] = ...,
    strategy_risk_budget: Optional[Dict[str, float]] = ...,
    strategy_fill_policy: Optional[Dict[str, FillMode]] = ...,
    strategy_slippage: Optional[Dict[str, SlippagePolicyInput]] = ...,
    strategy_commission: Optional[Dict[str, CommissionPolicyInput]] = ...,
    portfolio_risk_budget: Optional[float] = ...,
    risk_budget_mode: Literal["order_notional", "trade_notional"] = ...,
    risk_budget_reset_daily: bool = ...,
    on_event: Optional[Callable[[BacktestStreamEvent], None]] = ...,
    indicator_recorder: Optional[IndicatorSink] = ...,
    config: Optional[BacktestConfig] = ...,
    **kwargs: Any,
) -> BacktestResult: ...

__all__ = [
    "BacktestResult",
    "BacktestStreamEvent",
    "MergedResult",
    "run_backtest",
    "run_from_checkpoint",
    "merge_results",
    "make_fill_policy",
    "FillMode",
    "NextOpen",
    "NextClose",
    "NextAverage",
    "NextHighLowMid",
    "CurrentClose",
    "FunctionalStrategy",
]
