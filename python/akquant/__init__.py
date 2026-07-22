import typing
from importlib import metadata

import pandas as pd

try:
    __version__ = metadata.version("akquant")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__engine_rule_version__ = "1.3.4"  # Increment on behavior-changing updates

from . import akquant as _akquant
from . import log as _log
from . import talib
from .akquant import *  # noqa: F403
from .akquant import ATR, EMA, MACD, RSI, SMA, BollingerBands, Engine
from .analyzer_plugin import AnalyzerManager, AnalyzerTemplate
from .backtest import (  # type: ignore
    BacktestResult,
    BacktestStreamEvent,
    MergedResult,
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
    ChinaFuturesSessionConfig,
    ChinaFuturesValidationConfig,
    ChinaOptionsConfig,
    ChinaOptionsFeeConfig,
    ChinaOptionsSessionConfig,
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
from .indicator import Indicator, IndicatorSet
from .indicator_recording import IndicatorRecorder, IndicatorSink
from .indicator_stream import (
    is_indicator_stream_event,
    to_indicator_message,
    to_indicator_messages,
)
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
from .sizer import AllInSizer, FixedSize, PercentSizer, Sizer
from .strategy import InstrumentSnapshot, Strategy, StrategyRuntimeConfig
from .strategy_loader import register_strategy_loader, resolve_strategy_input
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
    __all__ = [name for name in _akquant.__all__ if name != "ExecutionMode"] + [  # noqa: F405
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
        "StrategyRuntimeConfig",
        "InstrumentSnapshot",
        "BarGenerator",
        "register_strategy_loader",
        "resolve_strategy_input",
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
        "ChinaFuturesSessionConfig",
        "ChinaFuturesValidationConfig",
        "ChinaOptionsConfig",
        "ChinaOptionsFeeConfig",
        "ChinaOptionsSessionConfig",
        "Indicator",
        "IndicatorRecorder",
        "IndicatorSet",
        "IndicatorSink",
        "is_indicator_stream_event",
        "to_indicator_message",
        "to_indicator_messages",
        "run_backtest",
        "run_from_checkpoint",
        "merge_results",
        "make_fill_policy",
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
        "StrategyRuntimeConfig",
        "InstrumentSnapshot",
        "BarGenerator",
        "register_strategy_loader",
        "resolve_strategy_input",
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
        "ChinaOptionsSessionConfig",
        "Indicator",
        "IndicatorRecorder",
        "IndicatorSet",
        "IndicatorSink",
        "is_indicator_stream_event",
        "to_indicator_message",
        "to_indicator_messages",
        "run_backtest",
        "run_from_checkpoint",
        "merge_results",
        "make_fill_policy",
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
        "resolve_param_model",
        "get_strategy_param_schema",
        "validate_strategy_params",
        "extract_runtime_kwargs",
        "build_param_grid_from_search_space",
        "OrderLeg",
        "OrderReceipt",
        "talib",
    ]


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


def _engine_set_timezone_name(self: Engine, tz_name: str) -> None:  # noqa: F405
    """
    通过时区名称设置引擎时区.

    :param tz_name: 时区名称，例如 "Asia/Shanghai", "UTC", "US/Eastern"
    """
    import datetime

    try:
        import zoneinfo

        tz: typing.Union[datetime.tzinfo, typing.Any] = zoneinfo.ZoneInfo(tz_name)
    except ImportError:
        import pytz

        tz = pytz.timezone(tz_name)

    # Get offset for current time (approximate is usually fine for constant
    # offset zones, but for DST aware zones, we might want a specific date.
    # For simplicity and standard market hours, we use current date or a fixed date)
    now = datetime.datetime.now(tz)
    utc_offset = now.utcoffset()
    if utc_offset is None:
        offset = 0
    else:
        offset = int(utc_offset.total_seconds())
    self.set_timezone(offset)


# Patch Engine class only when the native extension does not provide this API.
if not hasattr(Engine, "set_timezone_name"):
    Engine.set_timezone_name = _engine_set_timezone_name  # type: ignore


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
