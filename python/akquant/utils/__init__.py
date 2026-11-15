from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..akquant import Bar
from ..normalize import dataframe_to_arrays, dataframe_to_bars, to_indicator_frame


def load_bar_from_df(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    column_map: Optional[Dict[str, str]] = None,
    tz: Optional[str] = None,
) -> List[Bar]:
    r"""
    Convert DataFrame to list of akquant.Bar.

    薄委托: 实现已下沉到 :func:`akquant.normalize.dataframe_to_bars`(单一归一化实现).

    :param df: Historical market data
    :type df: pandas.DataFrame
    :param symbol: Symbol code; if not provided, try to use "symbol" column
    :type symbol: str, optional
    :param column_map: Mapping from DataFrame columns to standard fields.
                       Defaults: date->timestamp, open->open, etc.
    :type column_map: Dict[str, str], optional
    :param tz: Timezone for naive timestamps (default Asia/Shanghai)
    :type tz: str, optional
    :return: List of Bar objects
    :rtype: List[Bar]
    """
    return dataframe_to_bars(df, symbol, column_map, tz)


def parse_duration_to_bars(duration: Union[str, int], frequency: str = "1d") -> int:
    """
    Parse duration string to number of bars.

    Assumes A-share trading hours for intraday frequencies.

    :param duration: Duration string (e.g. "1y", "3m", "20d") or integer bars.
    :param frequency: Data frequency ("1d", "1h", "1m"). Default "1d".
    :return: Estimated number of bars.
    """
    if isinstance(duration, int):
        return duration

    import re

    match = re.match(r"(\d+)([ymwd])", duration.lower())
    if not match:
        # Try to parse as int string
        try:
            return int(duration)
        except ValueError:
            raise ValueError(f"Invalid duration format: {duration}")

    value = int(match.group(1))
    unit = match.group(2)

    # Estimated bars per day (A-share)
    if frequency == "1d":
        bars_per_day = 1
    elif frequency == "1h":
        bars_per_day = 4
    elif frequency == "30m":
        bars_per_day = 8
    elif frequency == "15m":
        bars_per_day = 16
    elif frequency == "5m":
        bars_per_day = 48
    elif frequency == "1m":
        bars_per_day = 240
    else:
        bars_per_day = 1

    if unit == "y":
        return int(value * 252 * bars_per_day)
    elif unit == "m":
        return int(value * 21 * bars_per_day)
    elif unit == "w":
        return int(value * 5 * bars_per_day)
    elif unit == "d":
        return int(value * bars_per_day)

    return value


def df_to_arrays(
    df: pd.DataFrame, symbol: Optional[str] = None
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[str],
    Optional[List[str]],
    Optional[Dict[str, np.ndarray]],
]:
    r"""
    将 DataFrame 转换为用于 DataFeed.add_arrays 的数组元组.

    薄委托: 实现已下沉到 :func:`akquant.normalize.dataframe_to_arrays`(单一归一化实现).

    :param df: 输入的 DataFrame
    :param symbol: 标的代码 (可选)
    :return: (timestamps, opens, highs, lows, closes, volumes, symbol, symbols, extra)
    """
    return dataframe_to_arrays(df, symbol)


def prepare_dataframe(
    df: pd.DataFrame, date_col: Optional[str] = None, tz: str = "Asia/Shanghai"
) -> pd.DataFrame:
    r"""
    自动预处理 DataFrame，处理时区并生成标准时间戳列.

    薄委托: 实现已下沉到 :func:`akquant.normalize.to_indicator_frame`(单一归一化实现).

    :param df: 输入 DataFrame
    :param date_col: 日期列名 (若为 None 则自动探测)
    :param tz: 默认时区 (若数据为 Naive 时间，则假定为此时区)
    :return: 处理后的 DataFrame (包含 'timestamp' 列)
    """
    return to_indicator_frame(df, date_col, tz)


_RATIO_PERCENT_METRICS = frozenset(
    {
        "annualized_return",
        "volatility",
        "kelly_criterion",
    }
)

_PERCENT_VALUE_METRICS = frozenset(
    {
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate",
        "loss_rate",
        "exposure_time_pct",
    }
)


def format_percentage(
    value: float,
    source: Literal["ratio", "pct_value"] = "ratio",
    precision: int = 2,
    width: Optional[int] = None,
) -> str:
    r"""
    Format a value as percentage with explicit source unit.

    :param value: Numeric value to format.
    :param source: "ratio" 表示 0.1 -> 10%，"pct_value" 表示 10 -> 10%。
    :param precision: Decimal places.
    :param width: Optional width for right alignment.
    :return: Formatted percentage text.
    """
    if source == "ratio":
        text = f"{float(value):.{precision}%}"
    else:
        text = f"{float(value):.{precision}f}%"
    if width is None:
        return text
    return f"{text:>{width}}"


def format_metric_value(
    metric_name: str,
    value: float,
    precision: int = 2,
    width: Optional[int] = None,
) -> str:
    r"""
    Format metric display text using AKQuant metric unit mapping.

    :param metric_name: Metric field name.
    :param value: Raw metric value.
    :param precision: Decimal places.
    :param width: Optional width for right alignment.
    :return: Formatted metric text.
    """
    name = str(metric_name)
    numeric = float(value)
    if name in _RATIO_PERCENT_METRICS:
        return format_percentage(
            numeric,
            source="ratio",
            precision=precision,
            width=width,
        )
    if name in _PERCENT_VALUE_METRICS:
        return format_percentage(
            numeric,
            source="pct_value",
            precision=precision,
            width=width,
        )
    text = f"{numeric:.{precision}f}"
    if width is None:
        return text
    return f"{text:>{width}}"
