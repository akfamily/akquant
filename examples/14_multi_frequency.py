"""多周期回测示例: 日线定趋势, 分钟线执行.

引擎原生多周期上线前, 这个示例靠「伪标的」拼日线: 把同一份分钟数据 resample
成日线, 起个 ``000001.SZ_1D`` 的假代码喂给引擎, 策略里再按 ``bar.symbol`` 分流
处理两条序列。现在 ``Strategy.subscribe_bars("1d", ...)`` 由 Rust 引擎直接从
基础 bar 聚合出日线窗口, 通过 ``on_daily`` 回调派发, 不再需要伪标的、不再需要
手工 resample/打戳。

三条多周期路径怎么选, 见 docs/zh/advanced/multi_timeframe_feed_api.md 的
「怎么选」表格; 本示例演示推荐路径 ``subscribe_bars``。

**时段与闭合时机**: 窗口默认按品种时段表(A 股: 09:30-11:30、13:00-15:00)的
交易块切分, 不需要手写 ``session_windows``; 日线窗口按交易日聚合, 标签为当日
收盘时刻 15:00。``run_backtest(freq="1min")`` 声明了基础周期, 每根分钟 bar 带上
区间起点(``interval_start``), 日线窗口在当日 15:00 那根分钟 bar 之后即时闭合并
派发; 不传 ``freq`` 时引擎不知道基础周期, 要等次日第一根 bar 到达才闭合(晚一根,
并告警)。实盘/replay 由行情网关声明 ``metadata["freq"]``, 见
examples/71_native_multi_timeframe_live.py。
"""

from datetime import timedelta
from typing import Any, List

import akquant as aq
import numpy as np
import pandas as pd
from akquant import BacktestConfig, InstrumentConfig, StrategyConfig


def create_dummy_data_1m(symbol: str, start_date: str, days: int) -> pd.DataFrame:
    """Generate 1-minute dummy data for a few days (A-share hours)."""
    # 时间戳是 bar 区间终点: 9:31-11:30 (120 根), 13:01-15:00 (120 根), 每日 240 根。
    # 09:31 那根覆盖 09:30-09:31(开盘集合竞价之后的连续竞价), 15:00 那根覆盖
    # 14:59-15:00(含收盘集合竞价)。示例按工作日生成, 未传交易日历时引擎也按工作日推算。
    timestamps: List[pd.Timestamp] = []
    base_price = 100.0
    prices: List[float] = []

    print(f"DEBUG: Generating data starting from {start_date}")
    current_date = pd.Timestamp(start_date)
    if current_date.tz is not None:
        current_date = current_date.tz_localize(None)

    for _ in range(days):
        rng_am = pd.date_range(
            start=current_date + timedelta(hours=9, minutes=31),
            end=current_date + timedelta(hours=11, minutes=30),
            freq="1min",
        )
        rng_pm = pd.date_range(
            start=current_date + timedelta(hours=13, minutes=1),
            end=current_date + timedelta(hours=15, minutes=0),
            freq="1min",
        )
        timestamps.extend(rng_am)
        timestamps.extend(rng_pm)
        current_date += pd.offsets.BDay(1)

    n = len(timestamps)
    np.random.seed(42)
    changes = np.random.randn(n) * 0.1
    p = base_price
    for c in changes:
        p += c
        prices.append(p)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": np.array(prices) + 0.05,
            "low": np.array(prices) - 0.05,
            "close": prices,
            "volume": 1000,
            "symbol": symbol,
        },
        index=timestamps,
    )
    date_index = pd.DatetimeIndex(df.index)
    if date_index.tz is None:
        date_index = date_index.tz_localize("Asia/Shanghai")
    df.index = date_index
    return df


class MultiFreqStrategy(aq.Strategy):
    """日线定趋势(SMA), 分钟线执行. 日线由引擎从 1 分钟 bar 聚合, 不再需要伪标的."""

    daily_sma: Any

    def __init__(self) -> None:
        """订阅日线窗口并注册按日线周期驱动的 SMA 增量指标."""
        super().__init__()
        self.ma_window = 3
        self.daily_trend = 0
        # 窗口按 A 股时段表的交易块切分, 不必手写 session_windows
        self.subscribe_bars("1d", callback=self.on_daily)
        self.daily_sma = self.I(
            aq.SMA(self.ma_window), name="daily_sma", source="close", freq="1d"
        )

    def on_daily(self, bar: aq.Bar) -> None:
        """日线窗口闭合回调: 按收盘价与 SMA 的相对位置更新趋势方向."""
        ma = self.daily_sma.value
        if ma is None:
            print(
                f"[1D] {self.format_time(bar.timestamp)} 收盘 {bar.close:.2f}, 攒历史中"
            )
            return
        self.daily_trend = 1 if bar.close > ma else -1
        print(
            f"[1D] {self.format_time(bar.timestamp)} 收盘 {bar.close:.2f} "
            f"SMA {ma:.2f} 趋势 {self.daily_trend}"
        )

    def on_bar(self, bar: aq.Bar) -> None:
        """分钟线执行: 顺日线趋势开仓/平仓."""
        pos = self.ctx.get_position(bar.symbol) if self.ctx else 0.0
        if self.daily_trend == 1 and pos == 0:
            self.buy(bar.symbol, 100)
        elif self.daily_trend == -1 and pos > 0:
            self.sell(bar.symbol, pos)


if __name__ == "__main__":
    print("Generating dummy data (5 Days) for A-shares...")
    df_1m = create_dummy_data_1m("000001.SZ", "2024-01-02", 5)
    print(f"Generated {len(df_1m)} minute bars.")

    stock_1m_config = InstrumentConfig(
        symbol="000001.SZ",
        asset_type="STOCK",
        multiplier=1.0,
    )

    print("\nStarting Multi-Frequency Backtest (A-share Simulated)...")

    config = BacktestConfig(
        strategy_config=StrategyConfig(
            initial_cash=100_000.0,
        ),
        instruments_config=[stock_1m_config],
        show_progress=True,
    )

    result = aq.run_backtest(
        data=df_1m,
        strategy=MultiFreqStrategy,
        symbols=["000001.SZ"],
        config=config,
        freq="1min",  # 声明基础周期: 日线窗口在当日 15:00 即时闭合
    )

    print("\n" + "=" * 50)
    print("Backtest Results")
    print("-" * 50)
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print("=" * 50)
