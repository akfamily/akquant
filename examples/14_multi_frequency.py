"""多周期回测示例: 日线定趋势, 分钟线执行.

引擎原生多周期上线前, 这个示例靠「伪标的」拼日线: 把同一份分钟数据 resample
成日线, 起个 ``000001.SZ_1D`` 的假代码喂给引擎, 策略里再按 ``bar.symbol`` 分流
处理两条序列。现在 ``Strategy.subscribe_bars("1d", ...)`` 由 Rust 引擎直接从
基础 bar 聚合出日线窗口, 通过 ``on_daily`` 回调派发, 不再需要伪标的、不再需要
手工 resample/打戳。

三条多周期路径怎么选, 见 docs/zh/advanced/multi_timeframe_feed_api.md 的
「怎么选」表格; 本示例演示推荐路径 ``subscribe_bars``。

**关于闭合时机的取舍**: 本示例把 1 分钟 DataFrame 整表传给 ``run_backtest``
(纯 bar 输入), 引擎无法从这种输入直接得知基础周期(``self.freq`` 为 ``None``),
因此日线窗口要等**次日第一根 09:31 分钟 bar** 到达后才闭合并派发(晚一根,
见 ``on_daily`` 里打印的提示)。示例里传了 ``session_windows`` 只是用来正确
切分早盘/午盘的会话边界, 不代表窗口会即时闭合——即时闭合还需要基础周期已知
(比如把数据转成 ``list[Bar]`` 且逐根打上 ``freq="1min"``, 或走实盘/replay 网关
声明 ``metadata["freq"]``, 见 examples/71_native_multi_timeframe_live.py)。
"""

from datetime import timedelta
from typing import Any, List

import akquant as aq
import numpy as np
import pandas as pd
from akquant import BacktestConfig, InstrumentConfig, StrategyConfig


def create_dummy_data_1m(symbol: str, start_date: str, days: int) -> pd.DataFrame:
    """Generate 1-minute dummy data for a few days (A-share hours)."""
    # Trading hours: 9:31-11:30 (120 min), 13:01-15:00 (120 min)
    # Total 240 bars per day
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
        current_date += timedelta(days=1)

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
        self.indicator_mode = "incremental"
        self.ma_window = 3
        self.daily_trend = 0
        self.subscribe_bars(
            "1d",
            callback=self.on_daily,
            session_windows=[("09:30", "11:30"), ("13:00", "15:00")],
        )
        self.register_incremental_indicator(
            "daily_sma", aq.SMA(self.ma_window), source="close", freq="1d"
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
    df_1m = create_dummy_data_1m("000001.SZ", "2024-01-01", 5)
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
            indicator_mode="incremental",
        ),
        instruments_config=[stock_1m_config],
        show_progress=True,
    )

    result = aq.run_backtest(
        data=df_1m,
        strategy=MultiFreqStrategy,
        symbols=["000001.SZ"],
        config=config,
    )

    print("\n" + "=" * 50)
    print("Backtest Results")
    print("-" * 50)
    print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.2f}%")
    print("=" * 50)
