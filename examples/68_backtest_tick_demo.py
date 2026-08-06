#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回测 tick 输入示例: 纯 tick 模式与 freq 聚合模式的语义对比.

说明:
- run_backtest(data=...) 除 Bar 列表外, 还接受 Tick 列表与 Bar/Tick 混合列表。
- 纯 tick 模式只触发 on_tick; 传 freq 后额外把 tick 聚合成 bar 触发 on_bar。
- 本示例演示四个容易踩的语义, 每个都有对应输出可核对。
"""

from typing import Any, List

import pandas as pd
from akquant import Strategy, run_backtest
from akquant.akquant import Tick
from akquant.backtest.fill_mode import CurrentClose

SYMBOL = "600000"


def _ns(text: str) -> int:
    """把可读时间转成纳秒时间戳.

    Bar/Tick 的构造器带"秒级->纳秒"自动修正: 时间戳小于 1e10 会被乘 1e9。
    所以务必传真实纳秒, 传 100 之类的小整数会被静默改写成 1e11。
    """
    return int(pd.Timestamp(text, tz="Asia/Shanghai").value)


def _make_ticks() -> List[Tick]:
    """构造第 1 分钟内的 4 笔成交, 外加第 2 分钟 1 笔.

    第 2 分钟那笔用来封闭第 1 分钟的合成 bar: 聚合器不提供 flush,
    末尾未满一个周期的 tick 不会产生 bar。
    """
    plan = [
        ("2024-01-02 09:30:00", 10.00, 100.0),
        ("2024-01-02 09:30:15", 10.60, 200.0),  # 本分钟最高
        ("2024-01-02 09:30:30", 9.80, 150.0),  # 本分钟最低
        ("2024-01-02 09:30:45", 10.20, 50.0),
        ("2024-01-02 09:31:00", 11.00, 300.0),  # 封闭上一分钟
    ]
    return [
        Tick(timestamp=_ns(text), price=price, volume=volume, symbol=SYMBOL)
        for text, price, volume in plan
    ]


class TickOnlyStrategy(Strategy):
    """纯 tick 模式: 只有 on_tick 会被触发."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.tick_count = 0
        self.bar_count = 0
        self.recent_prices: List[float] = []

    def on_start(self) -> None:
        """订阅标的并启用历史回溯."""
        self.subscribe(SYMBOL)
        self.set_history_depth(5)

    def on_tick(self, tick: Any) -> None:
        """累计 tick 并读取历史成交价序列."""
        self.tick_count += 1
        values = self.get_history(3, tick.symbol, "close")
        if values is not None:
            self.recent_prices = [round(float(v), 2) for v in values]

    def on_bar(self, bar: Any) -> None:
        """纯 tick 模式下不会被调用."""
        self.bar_count += 1


class AggregatedStrategy(Strategy):
    """Freq 聚合模式: 原始 tick 与合成 bar 都会到达."""

    def __init__(self) -> None:
        """初始化记录容器."""
        self.arrivals: List[str] = []
        self.bars: List[Any] = []

    def on_start(self) -> None:
        """订阅标的."""
        self.subscribe(SYMBOL)

    def on_tick(self, tick: Any) -> None:
        """记录原始 tick 的到达时刻."""
        self.arrivals.append(
            f"tick {_readable(tick.timestamp)} {float(tick.price):>6.2f}"
        )

    def on_bar(self, bar: Any) -> None:
        """记录合成 bar 的到达时刻与 OHLC."""
        self.bars.append(bar)
        self.arrivals.append(
            f"BAR  {_readable(bar.timestamp)} "
            f"O={float(bar.open):.2f} H={float(bar.high):.2f} "
            f"L={float(bar.low):.2f} C={float(bar.close):.2f} "
            f"V={float(bar.volume):.0f}"
        )


def _readable(timestamp_ns: Any) -> str:
    """把纳秒时间戳转回可读的时分秒."""
    stamp = pd.Timestamp(int(timestamp_ns), unit="ns", tz="UTC")
    return str(stamp.tz_convert("Asia/Shanghai").strftime("%H:%M:%S"))


def demo_tick_only() -> None:
    """纯 tick 模式: on_bar 不触发, get_history 返回成交价序列."""
    print("=" * 62)
    print("模式一: 纯 tick (不传 freq)")
    print("=" * 62)

    strategy = TickOnlyStrategy()
    run_backtest(
        data=_make_ticks(),
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    print(f"  on_tick 触发次数: {strategy.tick_count}")
    print(f"  on_bar  触发次数: {strategy.bar_count}   <- 纯 tick 模式下为 0")
    print(f"  最近 3 笔成交价:  {strategy.recent_prices}")
    print()
    print("  要点: tick 以退化 bar 写入历史(open=high=low=close=price),")
    print("        所以 get_history(n, symbol, 'close') 就是最近若干笔成交价。")
    print()


def demo_freq_aggregation() -> None:
    """Freq 聚合模式: tick 与合成 bar 并存, 且 bar 严格晚于其源 tick."""
    print("=" * 62)
    print('模式二: freq="1min" 聚合')
    print("=" * 62)

    strategy = AggregatedStrategy()
    run_backtest(
        data=_make_ticks(),
        freq="1min",
        strategy=strategy,
        symbols=[SYMBOL],
        initial_cash=100_000.0,
        show_progress=False,
        fill_policy=CurrentClose(),
    )

    print("  事件到达顺序:")
    for line in strategy.arrivals:
        print(f"    {line}")
    print()

    if strategy.bars:
        bar = strategy.bars[0]
        print("  要点一(无前视): 合成 bar 的时间戳打在**区间结束**")
        print(f"        bar 落在 {_readable(bar.timestamp)}, 晚于形成它的全部 tick。")
        print("        若打在区间起点, 排序后 bar 会排到源 tick 之前,")
        print("        策略就会读到尚未发生的 high/low —— 那是前视偏差。")
        print()
        print("  要点二(成交量口径): Tick.volume 是**单笔量**, 聚合器直接求和")
        print(f"        第 1 分钟 4 笔 100+200+150+50 = {float(bar.volume):.0f}")
        print()
        print("  要点三(OHLC 非恒等): 有了真实高低价, ATR 等 H/L 类指标才有意义")
        print(
            f"        H={float(bar.high):.2f} != L={float(bar.low):.2f}"
            "  (纯 tick 模式下二者恒等)"
        )
        print()
    print("  要点四(末尾不封闭): 第 2 分钟只有 1 笔且未跨到第 3 分钟,")
    print("        聚合器不提供 flush, 故它不产生 bar —— 只合成了 1 根。")
    print()


def main() -> None:
    """依次演示两种模式."""
    demo_tick_only()
    demo_freq_aggregation()

    print("=" * 62)
    print("更多能力边界见 docs/zh/guide/data.md 的「Tick 输入」一节:")
    print("  - freq 只支持整数分钟, '30s' 会报错并指向 feed_adapter.resample")
    print("  - 预计算指标不支持含 tick 的输入, 请改用增量指标")
    print("  - 纯 tick + H/L 类增量指标会在会话结束时显式报错")
    print("=" * 62)


if __name__ == "__main__":
    main()
