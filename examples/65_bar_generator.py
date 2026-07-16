"""BarGenerator 运行时聚合 Demo.

演示如何在策略内使用 ``BarGenerator`` 把逐笔 1 分钟 bar 流式聚合为 5 分钟 bar：
- 策略 ``on_bar`` 里调用 ``self.bg.update_bar(bar)``。
- 窗口闭合时 ``BarGenerator`` 回调 ``on_5m``，与离线 ``feed.resample`` 时钟对齐语义
  一致。
- 策略结束前调用 ``self.bg.flush()``，闭合尾部未满窗口。

同一份 ``update_bar`` 调用在回测与实盘中完全一致，无需切换聚合逻辑。
"""

from typing import List

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, BarGenerator, Strategy


def build_1m_bars(symbol: str, start_date: str, days: int) -> List[Bar]:
    """Generate deterministic 1-minute A-share session bars.

    :param symbol: Instrument symbol to attach to each bar.
    :param start_date: First trading day (naive date string).
    :param days: Number of consecutive trading days to generate.
    :return: Chronologically ordered list of 1-minute ``Bar`` objects.
    """
    timestamps: List[pd.Timestamp] = []
    current_date = pd.Timestamp(start_date)
    for _ in range(days):
        # Morning session: 09:31-11:30, Afternoon session: 13:01-15:00.
        rng_am = pd.date_range(
            start=current_date + pd.Timedelta(hours=9, minutes=31),
            end=current_date + pd.Timedelta(hours=11, minutes=30),
            freq="1min",
        )
        rng_pm = pd.date_range(
            start=current_date + pd.Timedelta(hours=13, minutes=1),
            end=current_date + pd.Timedelta(hours=15, minutes=0),
            freq="1min",
        )
        timestamps.extend(rng_am)
        timestamps.extend(rng_pm)
        current_date += pd.Timedelta(days=1)

    np.random.seed(42)
    n = len(timestamps)
    changes = np.random.randn(n) * 0.05
    price = 100.0
    bars: List[Bar] = []
    for ts, change in zip(timestamps, changes):
        price += change
        ts_ns = int(pd.Timestamp(ts, tz="Asia/Shanghai").value)
        bars.append(
            Bar(
                timestamp=ts_ns,
                open=price,
                high=price + 0.05,
                low=price - 0.05,
                close=price,
                volume=1000.0,
                symbol=symbol,
            )
        )
    return bars


class BarGeneratorStrategy(Strategy):
    """Aggregate incoming 1-minute bars into 5-minute bars at runtime."""

    def __init__(self) -> None:
        """Bind a BarGenerator instance to the on_5m callback."""
        super().__init__()
        self.five_min_bars: List[Bar] = []
        self.bg = BarGenerator(self.on_5m, window=5, interval="minute")

    def on_bar(self, bar: Bar) -> None:
        """Feed every incoming 1-minute bar into the runtime aggregator.

        :param bar: The current 1-minute bar.
        """
        self.bg.update_bar(bar)

    def on_5m(self, bar: Bar) -> None:
        """Handle one closed 5-minute window bar.

        :param bar: The aggregated 5-minute bar.
        """
        self.five_min_bars.append(bar)
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="UTC").tz_convert(
            "Asia/Shanghai"
        )
        print(
            f"[5min] {ts} O:{bar.open:.2f} H:{bar.high:.2f} "
            f"L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume:.0f}"
        )

    def on_stop(self) -> None:
        """Flush the trailing partial window before the backtest ends."""
        self.bg.flush()
        print(f"\n共聚合出 {len(self.five_min_bars)} 根 5 分钟 bar")


def main() -> None:
    """Run the BarGenerator runtime aggregation demo."""
    symbol = "000001.SZ"
    print("生成 3 天 1 分钟 A 股行情数据...")
    bars = build_1m_bars(symbol, "2024-01-02", days=3)
    print(f"共 {len(bars)} 根 1 分钟 bar")

    print("开始回测...")
    result = aq.run_backtest(
        strategy=BarGeneratorStrategy,
        data=bars,
        symbols=symbol,
        initial_cash=100_000.0,
        commission_rate=0.0,
        show_progress=False,
    )

    print("\n回测结束")
    print(result.metrics_df)


if __name__ == "__main__":
    main()
