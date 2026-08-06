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
#
# 词汇刻意只收 min / h 两种单位, 与 pandas(feed_adapter.resample 底层用的
# to_offset)保持一致——实测 to_offset 明确拒绝 "m"(已废弃)且不认识 "hour"。
# 若这里放宽收下它们, 同一个 freq 字符串会在 run_backtest 被接受、在
# feed_adapter.resample 被拒绝, 造成两个入口行为分叉。
_FREQ_PATTERN = re.compile(r"^\s*(\d+)\s*(min|h)\s*$", re.IGNORECASE)


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
    minutes = value * 60 if unit == "h" else value
    if minutes <= 0:
        raise ValueError(f"freq={freq!r} 解析出的分钟数必须为正, 得到 {minutes}")
    return minutes


def aggregate_ticks_into_feed(
    feed: Any, ticks: Sequence[Tick], interval_min: int
) -> None:
    """经 ``BarAggregator`` 把 tick 聚合成 bar 并推入 ``feed``.

    **volume 口径适配（必须切到单笔模式，不能用 live 默认口径）**:
    ``BarAggregator`` 默认把 ``on_tick`` 的 ``volume`` 当累计量算差分——这与
    CTP 的日累计 ``pDepthMarketData.Volume`` 吻合, 故 live 路径正确。但回测
    调用方构造的 ``Tick(volume=100.0)`` 表达的是**单笔量**: 若仍按累计口径
    处理, 每个 symbol 的**首笔量会被无条件丢弃**（首次差分恒为 0）。因此这里
    显式传 ``volume_is_cumulative=False``, 让 ``BarAggregator`` 把 volume 当
    单笔量直接计入, 不做差分（详见 ``src/data/aggregator.rs``）。

    **时间戳必须打在区间结束（因果顺序，不能用 live 默认口径）**:
    ``BarAggregator`` 默认给封闭 bar 打区间**起点**时间戳——这在 live 下无害,
    bar 是在下一区间首个 tick 到达时才发出的, 墙钟顺序即因果顺序。但本函数把
    合成 bar 与源 tick 写进**同一个** ``feed``, 而 ``run_backtest`` 随后会对
    这个 feed 调用 ``sort()``：按起点时间戳排序会让 bar 落到形成它的 tick
    **之前**, 策略的 ``on_bar`` 就会读到尚未发生的 high/low/close。因此这里
    显式传 ``stamp_bar_at_interval_end=True``, 让封闭 bar 打在区间结束
    （下一区间起点前 1ns), 保证排在其源 tick 之后（详见
    ``src/data/aggregator.rs``）。

    **末尾未满周期不发出**: ``BarAggregator`` 不提供 ``flush``, 最后一个未封闭
    的周期不会产生 bar。这是既有行为, 调用方需知悉。

    :param feed: 目标 ``DataFeed``（合成的 bar 会写入其中）
    :param ticks: 已按时间戳升序的 tick 序列
    :param interval_min: 聚合周期（整数分钟）
    """
    from ..akquant import BarAggregator

    aggregator = BarAggregator(
        feed,
        interval_min,
        volume_is_cumulative=False,
        stamp_bar_at_interval_end=True,
    )
    for tick in ticks:
        aggregator.on_tick(
            str(tick.symbol),
            float(tick.price),
            float(tick.volume),
            int(tick.timestamp),
        )
