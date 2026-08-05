"""回测行情输入适配层: 把 Bar / Tick 混合输入归一成引擎可消费的形态.

回测引擎的 Rust 事件层本就支持 ``Event::Tick``, 撮合也认 tick; 缺口只在
``run_backtest`` 的入口类型闸门与若干消费端。本模块只负责入口归一——不碰撮合、
不碰指标、不碰历史。
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

from ..akquant import Bar, Tick

# 只接受能用整数分钟表达的周期: BarAggregator 的 interval_min 是整数。
_FREQ_PATTERN = re.compile(r"^\s*(\d+)\s*(min|m|h|hour)\s*$", re.IGNORECASE)


def normalize_market_input(
    data: Sequence[Any],
) -> Tuple[List[Bar], List[Tick]]:
    """把 Bar/Tick 混合序列分类, 各组按时间戳升序.

    **逐元素校验类型**: ``engine.py`` 现有分发只检查 ``data[0]``, 混合列表中的
    非法元素会漏到 Rust 层抛出难以定位的错误。此处提前抛 ``TypeError`` 并指名
    位置索引与实际类型。

    :param data: ``Bar`` 与 ``Tick`` 的混合序列
    :return: ``(bars, ticks)``, 两组各自已按时间戳升序
    :raises ValueError: ``data`` 为空
    :raises TypeError: 含非 ``Bar`` / ``Tick`` 元素
    """
    if not data:
        raise ValueError(
            "run_backtest(data=[...]) 收到空序列: 请传入至少一个 Bar 或 Tick"
        )

    bars: List[Bar] = []
    ticks: List[Tick] = []
    for index, item in enumerate(data):
        if isinstance(item, Bar):
            bars.append(item)
        elif isinstance(item, Tick):
            ticks.append(item)
        else:
            raise TypeError(
                f"run_backtest(data=[...]) 的第 {index} 个元素类型为 "
                f"{type(item).__name__}, 只接受 Bar 或 Tick"
            )

    bars.sort(key=lambda b: int(b.timestamp))
    ticks.sort(key=lambda t: int(t.timestamp))
    return bars, ticks


def parse_freq_to_interval_min(freq: str) -> int:
    """把 ``freq`` 字符串解析成整数分钟.

    与 ``akquant.feed_adapter`` 的 ``freq`` 词汇保持一致, 但聚合走
    ``BarAggregator``, 它只接受整数分钟。秒级或非整分周期**明确报错**而非静默
    向上取整——静默取整会让用户以为聚合按其预期的粒度进行。

    :param freq: 形如 ``"1min"`` / ``"5min"`` / ``"1h"``
    :return: 对应的整数分钟
    :raises ValueError: 无法用整数分钟表达
    """
    match = _FREQ_PATTERN.match(str(freq))
    if match is None:
        raise ValueError(
            f"freq={freq!r} 无法解析为整数分钟: tick 聚合经 BarAggregator, "
            "仅支持形如 '1min' / '5min' / '1h' 的整分周期。"
            "需要秒级或非整分聚合请改用 akquant.feed_adapter 的 resample()"
        )

    value = int(match.group(1))
    unit = match.group(2).lower()
    minutes = value * 60 if unit in {"h", "hour"} else value
    if minutes <= 0:
        raise ValueError(f"freq={freq!r} 解析出的分钟数必须为正, 得到 {minutes}")
    return minutes
