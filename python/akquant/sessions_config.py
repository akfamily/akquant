"""交易时段配置参数的统一规范化.

``run_backtest`` / ``run_from_checkpoint`` / ``run_live`` / ``BarGenerator`` /
``feed_adapter`` 的 ``sessions``、``trading_calendar``、``asset_type`` 共用这里的
规范化逻辑；取值校验（模板名、自定义段的时刻与重叠等）由 Rust 侧
``SessionResolver`` / ``Engine.configure_sessions`` 完成。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from .akquant import AssetType

#: 自定义时段段：``(start, end, session)`` 或 ``(start, end, session, anchor)``。
CustomSegment = Union[Tuple[str, str, str], Tuple[str, str, str, str]]

#: ``sessions`` 参数：代码前缀 → 时段模板名或自定义段序列。
SessionsArg = Mapping[str, Union[str, Sequence[Sequence[str]]]]

#: 规范化后的 ``sessions``。
SessionsMap = Dict[str, Union[str, List[CustomSegment]]]

_SEGMENT_HINT = (
    '每段写 (start, end, session) 或 (start, end, session, anchor)，如 ("21:00", '
    '"02:30", "continuous")；anchor 可选 "prev"（前一交易日，夜盘）或 "day"'
)

AssetTypeName = Literal["futures", "stock", "fund", "option"]


def normalize_sessions(value: Any) -> Optional[SessionsMap]:
    """把 ``sessions`` 归一为 ``{键: 模板名 | [段元组, ...]}``；None 原样返回.

    值为字符串时是内置模板名（如 ``"CN_FUTURES_NIGHT_23"``），为序列时是自定义
    时段表，每段 ``(start "HH:MM", end "HH:MM", session)``，可选第 4 项锚定
    ``"prev"`` / ``"day"``。这里只做类型检查，取值校验在构造解析器时完成。
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            "sessions 须为字典：代码前缀 → 时段模板名或自定义段列表，如 "
            '{"rb": "CN_FUTURES_NIGHT_23"} 或 '
            '{"xx": [("09:30", "11:30", "continuous")]}；'
            f"收到 {type(value).__name__}"
        )
    out: SessionsMap = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"sessions 的键须为字符串（代码前缀），收到 {key!r}")
        if isinstance(item, str):
            out[key] = item
            continue
        if isinstance(item, bytes) or not isinstance(item, Iterable):
            raise TypeError(
                f"sessions[{key!r}] 须为时段模板名（字符串）或自定义段列表"
                f"（{_SEGMENT_HINT}），收到 {type(item).__name__}"
            )
        segments: List[CustomSegment] = []
        for idx, seg in enumerate(item, start=1):
            if isinstance(seg, (str, bytes)) or not isinstance(seg, Sequence):
                raise TypeError(
                    f"sessions[{key!r}] 第 {idx} 段须为字符串元组：{_SEGMENT_HINT}；"
                    f"收到 {seg!r}"
                )
            if len(seg) not in (3, 4):
                raise ValueError(
                    f"sessions[{key!r}] 第 {idx} 段应有 3 或 4 项：{_SEGMENT_HINT}；"
                    f"收到 {len(seg)} 项"
                )
            if not all(isinstance(part, str) for part in seg):
                raise TypeError(
                    f"sessions[{key!r}] 第 {idx} 段的每一项都须为字符串："
                    f"{_SEGMENT_HINT}；收到 {seg!r}"
                )
            segments.append(tuple(seg))  # type: ignore[arg-type]
        out[key] = segments
    return out


def normalize_trading_calendar(value: Any) -> Optional[List[dt.date]]:
    """把 trading_calendar 归一为 date 列表；None 原样返回，空报错."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "trading_calendar 须为交易日序列（date/字符串/Timestamp 的列表），"
            "不能是单个字符串"
        )
    days = sorted({pd.Timestamp(item).date() for item in value})
    if not days:
        raise ValueError(
            "trading_calendar 不能为空：传交易日列表（如 AKShare 的 "
            "tool_trade_date_hist_sina()['trade_date']），或不传按工作日推算"
        )
    return days


def parse_asset_type_name(value: Any) -> AssetTypeName:
    """把 ``AssetType`` 或字符串解析为资产类型名；不支持的类型报 ValueError."""
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
