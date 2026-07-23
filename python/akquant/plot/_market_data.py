"""行情数据规范化工具(报告与 LWC 复盘共享).

将用户传入的任意列名/索引形态的行情 DataFrame 归一到规范 OHLCV 模式,
并支持从单表或 ``{symbol: df}`` 字典中按标的切片。原实现位于
``plot/report.py``,为供 ``akquant.lwc`` 复用而抽取至此,行为保持不变。
"""

from __future__ import annotations

from typing import Optional, Union, cast

import pandas as pd


def resolve_market_data_column(
    data: pd.DataFrame, candidates: list[str]
) -> Optional[str]:
    """按候选名(大小写不敏感)解析行情列名.

    :param data: 待解析的 DataFrame.
    :param candidates: 候选列名列表,按优先级排列.
    :return: 命中的实际列名;无命中返回 ``None``.
    """
    columns = list(data.columns)
    lowered = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        resolved = lowered.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None


def normalize_market_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    """将行情 DataFrame 归一到规范 OHLCV 模式.

    统一列名(open/high/low/close/volume/symbol)、以 DatetimeIndex 为索引、
    数值列强制转数、丢弃 OHLC 缺失行、按时间升序。缺少必需 OHLC 列时返回
    空 DataFrame.

    :param data: 原始行情数据.
    :return: 规范化后的副本(可能为空).
    """
    normalized = data.copy()
    resolved_columns = {
        "timestamp": resolve_market_data_column(
            normalized,
            ["date", "timestamp", "datetime", "time", "trade_date", "日期", "时间"],
        ),
        "open": resolve_market_data_column(normalized, ["open", "open_price", "开盘"]),
        "high": resolve_market_data_column(normalized, ["high", "high_price", "最高"]),
        "low": resolve_market_data_column(normalized, ["low", "low_price", "最低"]),
        "close": resolve_market_data_column(
            normalized, ["close", "close_price", "收盘"]
        ),
        "volume": resolve_market_data_column(normalized, ["volume", "vol", "成交量"]),
        "symbol": resolve_market_data_column(
            normalized, ["symbol", "code", "ticker", "ts_code", "股票代码"]
        ),
    }
    rename_map = {
        source: target
        for target, source in resolved_columns.items()
        if source is not None and source != target
    }
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype(str).str.strip()

    if not isinstance(normalized.index, pd.DatetimeIndex):
        if "timestamp" in normalized.columns:
            normalized = normalized.set_index("timestamp")
        else:
            normalized.index = pd.to_datetime(normalized.index, errors="coerce")

    normalized.index = pd.to_datetime(normalized.index, errors="coerce")
    valid_index_mask = ~normalized.index.to_series().isna()
    normalized = normalized.loc[valid_index_mask].copy()
    if normalized.empty:
        return cast(pd.DataFrame, normalized)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(set(normalized.columns)):
        return pd.DataFrame()

    normalized = normalized.dropna(subset=list(required_cols), how="any")
    if normalized.empty:
        return cast(pd.DataFrame, normalized)

    normalized = normalized.sort_index()
    return cast(pd.DataFrame, normalized)


def extract_symbol_market_data(
    market_data: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]], symbol: str
) -> pd.DataFrame:
    """从单表或 ``{symbol: df}`` 字典中取出指定标的的规范化行情.

    :param market_data: 单个 DataFrame、``{symbol: df}`` 字典或 ``None``.
    :param symbol: 目标标的代码.
    :return: 该标的的规范化行情;无数据返回空 DataFrame.
    """
    if market_data is None:
        return pd.DataFrame()
    if isinstance(market_data, dict):
        matched_key = None
        target_symbol = str(symbol).strip()
        for key in market_data.keys():
            if str(key).strip() == target_symbol:
                matched_key = key
                break
        if matched_key is None:
            return pd.DataFrame()
        data = market_data.get(matched_key, pd.DataFrame()).copy()
    elif isinstance(market_data, pd.DataFrame):
        data = market_data.copy()
    else:
        return pd.DataFrame()

    data = normalize_market_data_frame(data)
    if data.empty:
        return cast(pd.DataFrame, data)
    if "symbol" in data.columns:
        target_symbol = str(symbol).strip()
        symbol_mask = data["symbol"].astype(str).str.strip() == target_symbol
        data = data[symbol_mask].copy()
    return cast(pd.DataFrame, data)
