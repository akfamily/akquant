import typing
from importlib import metadata

import pandas as pd

try:
    __version__ = metadata.version("akquant")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__engine_rule_version__ = "2.0.0"  # Increment on behavior-changing updates

from . import akquant as _akquant
from . import log as _log
from . import talib
from .akquant import *  # noqa: F403
from .akquant import ATR, EMA, MACD, RSI, SMA, BollingerBands, Engine
from .analyzer_plugin import AnalyzerManager, AnalyzerTemplate
from .backtest import (  # type: ignore
    BacktestResult,
    BacktestStreamEvent,
    CurrentClose,
    FillMode,
    MergedResult,
    NextAverage,
    NextClose,
    NextHighLowMid,
    NextOpen,
    make_fill_policy,
    merge_results,
    run_backtest,
    run_from_checkpoint,
)
from .bar_generator import BarGenerator
from .checkpoint import load_checkpoint, save_checkpoint
from .config import (
    BacktestConfig,
    ChinaFuturesConfig,
    ChinaFuturesFeeConfig,
    ChinaFuturesInstrumentTemplateConfig,
    ChinaFuturesValidationConfig,
    ChinaOptionsConfig,
    ChinaOptionsFeeConfig,
    ChinaStockConfig,
    InstrumentAssetTypeEnum,
    InstrumentConfig,
    InstrumentOptionMarginModelEnum,
    InstrumentOptionTypeEnum,
    InstrumentSettlementTypeEnum,
    StrategyConfig,
    strategy_config,
)
from .data import DataLoader
from .feed_adapter import (
    CSVFeedAdapter,
    DataFeedAdapter,
    FeedSlice,
    ParquetFeedAdapter,
    ReplayFeedAdapter,
    ResampledFeedAdapter,
)
from .gateway.order_receipt import OrderLeg, OrderReceipt
from .indicator_declaration import IndicatorBinding, IndicatorDeclaration
from .indicator_recording import IndicatorRecorder, IndicatorSink
from .indicator_script import Indicator, IndicatorCannotTradeError
from .indicator_stream import (
    is_indicator_stream_event,
    to_indicator_message,
    to_indicator_messages,
)
from .live import run_live
from .normalize import (
    normalize,
    write_canonical_parquet,
)
from .optimize import OptimizationResult, run_grid_search, run_walk_forward
from .params import (
    BoolParam,
    ChoiceParam,
    DateRange,
    DateRangeParam,
    FloatParam,
    IntParam,
    ListParam,
    ParamModel,
)
from .params_adapter import (
    build_param_grid_from_search_space,
    extract_runtime_kwargs,
    get_strategy_param_schema,
    resolve_param_model,
    validate_strategy_params,
)
from .plot import plot_indicators, plot_result
from .precomputed import PrecomputedIndicator
from .sizer import AllInSizer, FixedSize, PercentSizer, Sizer
from .strategy import InstrumentSnapshot, Strategy, StrategyRuntimeConfig
from .strategy_inspect import InspectResult, inspect_strategy
from .strategy_loader import (
    get_strategy_loader,
    register_plugin_strategy_loaders,
    register_strategy_loader,
    resolve_strategy_input,
)
from .stream_schema import STREAM_SCHEMA_VERSION
from .trade_stream import (
    is_trade_stream_event,
    to_trade_message,
    to_trade_messages,
)
from .utils import (
    fetch_akshare_symbol,
    format_metric_value,
    load_bar_from_df,
    prepare_dataframe,
)

if typing.TYPE_CHECKING:
    from .log import LogConfig as LogConfig  # type: ignore[assignment]
    from .log import configure_logging as configure_logging  # type: ignore[assignment]
    from .log import get_logger as get_logger
    from .log import register_logger as register_logger
    from .log import set_log_level as set_log_level
else:
    globals()["LogConfig"] = _log.LogConfig
    globals()["configure_logging"] = _log.configure_logging
    globals()["get_logger"] = _log.get_logger
    globals()["register_logger"] = _log.register_logger
    globals()["set_log_level"] = _log.set_log_level

__doc__ = _akquant.__doc__
if hasattr(_akquant, "__all__"):  # noqa: F405
    # `_restore_*` 是 pickle 还原函数，仍可经 `akquant.akquant` 访问，不对外导出
    __all__ = [n for n in _akquant.__all__ if not n.startswith("_restore_")] + [  # noqa: F405
        "load_bar_from_df",
        "prepare_dataframe",
        "normalize",
        "write_canonical_parquet",
        "fetch_akshare_symbol",
        "format_metric_value",
        "Sizer",
        "FixedSize",
        "PercentSizer",
        "AllInSizer",
        "Strategy",
        "Indicator",
        "IndicatorCannotTradeError",
        "InspectResult",
        "inspect_strategy",
        "StrategyRuntimeConfig",
        "InstrumentSnapshot",
        "BarGenerator",
        "register_strategy_loader",
        "resolve_strategy_input",
        "get_strategy_loader",
        "register_plugin_strategy_loaders",
        "DataLoader",
        "DataFeedAdapter",
        "FeedSlice",
        "CSVFeedAdapter",
        "ParquetFeedAdapter",
        "ResampledFeedAdapter",
        "ReplayFeedAdapter",
        "LogConfig",
        "configure_logging",
        "get_logger",
        "register_logger",
        "set_log_level",
        "strategy_config",
        "ChinaFuturesConfig",
        "ChinaFuturesFeeConfig",
        "ChinaFuturesInstrumentTemplateConfig",
        "ChinaFuturesValidationConfig",
        "ChinaOptionsConfig",
        "ChinaOptionsFeeConfig",
        "ChinaStockConfig",
        "PrecomputedIndicator",
        "IndicatorBinding",
        "IndicatorDeclaration",
        "IndicatorRecorder",
        "IndicatorSink",
        "is_indicator_stream_event",
        "to_indicator_message",
        "to_indicator_messages",
        "is_trade_stream_event",
        "to_trade_message",
        "to_trade_messages",
        "STREAM_SCHEMA_VERSION",
        "run_backtest",
        "run_from_checkpoint",
        "run_live",
        "merge_results",
        "make_fill_policy",
        "FillMode",
        "NextOpen",
        "NextClose",
        "NextAverage",
        "NextHighLowMid",
        "CurrentClose",
        "plot_result",
        "plot_indicators",
        "BacktestResult",
        "MergedResult",
        "BacktestStreamEvent",
        "run_grid_search",
        "run_walk_forward",
        "OptimizationResult",
        "BacktestConfig",
        "StrategyConfig",
        "InstrumentConfig",
        "InstrumentAssetTypeEnum",
        "InstrumentOptionMarginModelEnum",
        "InstrumentOptionTypeEnum",
        "InstrumentSettlementTypeEnum",
        "save_checkpoint",
        "load_checkpoint",
        "SMA",
        "EMA",
        "MACD",
        "RSI",
        "BollingerBands",
        "ATR",
        "AnalyzerManager",
        "AnalyzerTemplate",
        "ParamModel",
        "DateRange",
        "IntParam",
        "FloatParam",
        "BoolParam",
        "ChoiceParam",
        "DateRangeParam",
        "ListParam",
        "resolve_param_model",
        "get_strategy_param_schema",
        "validate_strategy_params",
        "extract_runtime_kwargs",
        "build_param_grid_from_search_space",
        "OrderLeg",
        "OrderReceipt",
        "talib",
    ]
else:
    __all__ = [
        "load_bar_from_df",
        "prepare_dataframe",
        "normalize",
        "write_canonical_parquet",
        "fetch_akshare_symbol",
        "format_metric_value",
        "Sizer",
        "FixedSize",
        "PercentSizer",
        "AllInSizer",
        "Strategy",
        "Indicator",
        "IndicatorCannotTradeError",
        "InspectResult",
        "inspect_strategy",
        "StrategyRuntimeConfig",
        "InstrumentSnapshot",
        "BarGenerator",
        "register_strategy_loader",
        "resolve_strategy_input",
        "get_strategy_loader",
        "register_plugin_strategy_loaders",
        "DataLoader",
        "DataFeedAdapter",
        "FeedSlice",
        "CSVFeedAdapter",
        "ParquetFeedAdapter",
        "ResampledFeedAdapter",
        "ReplayFeedAdapter",
        "LogConfig",
        "configure_logging",
        "get_logger",
        "register_logger",
        "set_log_level",
        "strategy_config",
        "ChinaOptionsConfig",
        "ChinaOptionsFeeConfig",
        "ChinaStockConfig",
        "PrecomputedIndicator",
        "IndicatorBinding",
        "IndicatorDeclaration",
        "IndicatorRecorder",
        "IndicatorSink",
        "is_indicator_stream_event",
        "to_indicator_message",
        "to_indicator_messages",
        "is_trade_stream_event",
        "to_trade_message",
        "to_trade_messages",
        "STREAM_SCHEMA_VERSION",
        "run_backtest",
        "run_from_checkpoint",
        "run_live",
        "merge_results",
        "make_fill_policy",
        "FillMode",
        "NextOpen",
        "NextClose",
        "NextAverage",
        "NextHighLowMid",
        "CurrentClose",
        "plot_result",
        "plot_indicators",
        "BacktestResult",
        "MergedResult",
        "BacktestStreamEvent",
        "run_grid_search",
        "run_walk_forward",
        "OptimizationResult",
        "BacktestConfig",
        "StrategyConfig",
        "InstrumentConfig",
        "InstrumentAssetTypeEnum",
        "InstrumentOptionMarginModelEnum",
        "InstrumentOptionTypeEnum",
        "InstrumentSettlementTypeEnum",
        "save_checkpoint",
        "load_checkpoint",
        "SMA",
        "EMA",
        "MACD",
        "RSI",
        "BollingerBands",
        "ATR",
        "AnalyzerManager",
        "AnalyzerTemplate",
        "ParamModel",
        "DateRange",
        "IntParam",
        "FloatParam",
        "BoolParam",
        "ChoiceParam",
        "DateRangeParam",
        "ListParam",
        "resolve_param_model",
        "get_strategy_param_schema",
        "validate_strategy_params",
        "extract_runtime_kwargs",
        "build_param_grid_from_search_space",
        "OrderLeg",
        "OrderReceipt",
        "talib",
    ]


_REMOVED_SESSION_CONFIG_CLASSES = {
    "ChinaFuturesSessionConfig": "ChinaFuturesConfig.sessions",
    "ChinaOptionsSessionConfig": "ChinaOptionsConfig.sessions",
}


def __getattr__(name: str) -> object:
    """已删除的时段配置类给出中文迁移提示，其余名字保持默认的属性错误."""
    if name in _REMOVED_SESSION_CONFIG_CLASSES:
        raise ImportError(
            f"{name} 已删除（原用于 {_REMOVED_SESSION_CONFIG_CLASSES[name]}）："
            "时段改由 run_backtest(sessions=...) 配置，例如 "
            'sessions={"rb": "CN_FUTURES_NIGHT_23"}；'
            '不受时段约束请用 sessions={"default": "ALWAYS_OPEN"}；'
            "期权默认用 CN_OPTION，商品期权未单独配置时跟随其标的期货的时段表"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_bar(
    timestamp: int,
    open_px: float,
    high_px: float,
    low_px: float,
    close_px: float,
    volume: float,
    symbol: str,
) -> Bar:  # noqa: F405
    """创建 Bar 对象的辅助函数."""
    return Bar(timestamp, open_px, high_px, low_px, close_px, volume, symbol)  # noqa: F405


def _engine_get_orders_dataframe(self: Engine) -> "pd.DataFrame":  # noqa: F405
    """
    Get all orders as a pandas DataFrame.

    :return: pd.DataFrame with order details.
    """
    import pandas as pd

    orders = self.orders
    if not orders:
        return pd.DataFrame()

    data = []
    for o in orders:
        # Convert nanosecond timestamp to datetime
        created_at_dt = pd.Timestamp(o.created_at, unit="ns", tz="UTC")
        updated_at_dt = pd.Timestamp(o.updated_at, unit="ns", tz="UTC")

        data.append(
            {
                "order_id": str(o.id),  # UUID usually
                "symbol": o.symbol,
                "side": str(o.side),
                "status": str(o.status),
                "price": o.price,
                "quantity": o.quantity,
                "filled": o.filled_quantity,
                "avg_price": o.average_filled_price,
                "created_at": created_at_dt,
                "updated_at": updated_at_dt,
                "reject_reason": o.reject_reason or "",
            }
        )
    return pd.DataFrame(data)


def _engine_configure_risk(
    self: Engine,  # noqa: F405
    max_position_pct: typing.Optional[float] = None,
    sector_concentration: typing.Optional[
        typing.Tuple[float, typing.Dict[str, str]]
    ] = None,
) -> None:
    """
    Configure risk management rules (Shortcut).

    :param max_position_pct: Max position size as a percentage of total equity
                             (e.g. 0.10 for 10%)
    :param sector_concentration: Tuple of (limit, sector_map) e.g.
                                 (0.15, {"600519": "Consumer"})
    """
    # Get current risk manager (might be a copy)
    rm = self.risk_manager

    if max_position_pct is not None:
        rm.add_max_position_percent_rule(max_position_pct)

    if sector_concentration is not None:
        limit, sector_map = sector_concentration
        rm.add_sector_concentration_rule(limit, sector_map)

    # Set it back
    self.risk_manager = rm


Engine.get_orders_dataframe = _engine_get_orders_dataframe  # type: ignore # noqa: F405
Engine.configure_risk = _engine_configure_risk  # type: ignore # noqa: F405
