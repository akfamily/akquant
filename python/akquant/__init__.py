import typing
from importlib import metadata

try:
    __version__ = metadata.version("akquant")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from . import akquant as _akquant
from .akquant import *  # noqa: F403
from .backtest import BacktestResult, run_backtest  # type: ignore
from .config import BacktestConfig, InstrumentConfig, StrategyConfig, strategy_config
from .data import DataLoader
from .indicator import Indicator, IndicatorSet
from .log import get_logger, register_logger
from .optimize import OptimizationResult, run_optimization
from .plot import plot_result
from .sizer import AllInSizer, FixedSize, PercentSizer, Sizer
from .strategy import Strategy
from .utils import load_bar_from_df, prepare_dataframe

__doc__ = _akquant.__doc__
if hasattr(_akquant, "__all__"):  # noqa: F405
    __all__ = _akquant.__all__ + [  # noqa: F405
        "load_bar_from_df",
        "prepare_dataframe",
        "Sizer",
        "FixedSize",
        "PercentSizer",
        "AllInSizer",
        "Strategy",
        "DataLoader",
        "get_logger",
        "register_logger",
        "strategy_config",
        "Indicator",
        "IndicatorSet",
        "run_backtest",
        "plot_result",
        "BacktestResult",
        "run_optimization",
        "OptimizationResult",
        "BacktestConfig",
        "StrategyConfig",
        "InstrumentConfig",
    ]
else:
    __all__ = [
        "load_bar_from_df",
        "prepare_dataframe",
        "Sizer",
        "FixedSize",
        "PercentSizer",
        "AllInSizer",
        "Strategy",
        "DataLoader",
        "get_logger",
        "register_logger",
        "strategy_config",
        "Indicator",
        "IndicatorSet",
        "run_backtest",
        "plot_result",
        "BacktestResult",
        "run_optimization",
        "OptimizationResult",
        "BacktestConfig",
        "StrategyConfig",
        "InstrumentConfig",
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


# Patch Engine class
Engine.set_timezone_name = _engine_set_timezone_name  # type: ignore # noqa: F405
