from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


def synthetic_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成固定种子的合成日线（几何随机游走），只用于取数失败时让示例跑通."""
    dates = pd.bdate_range(start_date, end_date)
    rng = np.random.default_rng(sum(map(ord, symbol)))
    close = 10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        }
    )


def fetch_daily(
    symbol: str, start_date: str, end_date: str, adjust: str = ""
) -> pd.DataFrame:
    """用 AKShare 取新浪 A 股日线（代码带市场前缀，如 sh600000）.

    取数失败（未安装 AKShare、断网、接口改名等）时改用合成数据，保证示例能跑通；
    合成数据的回测结果没有参考意义。
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist_sina(
            symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust
        )
        if df.empty:
            raise ValueError("返回空表")
        return pd.DataFrame(df)
    except Exception as exc:  # noqa: BLE001 - 示例容错，任何取数问题都回退
        print(f"警告: AKShare 取数失败（{type(exc).__name__}: {exc}），改用合成数据")
        return synthetic_daily(symbol, start_date, end_date)


df_1 = fetch_daily(symbol="sh600000", start_date="20000101", end_date="20261231")
df_1["symbol"] = "600000"
df_2 = fetch_daily(symbol="sh600004", start_date="20000101", end_date="20261231")
df_2["symbol"] = "600004"
df_3 = fetch_daily(symbol="sh600006", start_date="20000101", end_date="20261231")
df_3["symbol"] = "600006"
df = {"600000": df_1, "600004": df_2, "600006": df_3}


class MyStrategy(Strategy):
    """
    Example strategy for testing broker execution.

    This strategy buys on the first bar and holds for 100 bars or until 10% profit.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.bars_held: dict[str, int] = {}
        self.entry_prices: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar data event.

        :param bar: The current bar data
        """
        symbol = bar.symbol
        pos = self.get_position(symbol)

        # 维护持仓计数
        if pos > 0:
            if symbol not in self.bars_held:
                self.bars_held[symbol] = 0
            self.bars_held[symbol] += 1
        else:
            # 如果没有持仓，清理状态
            if symbol in self.bars_held:
                del self.bars_held[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

        # 交易逻辑
        if pos == 0:
            # 简单示例：每个标的买入 33% 仓位
            self.order_target_percent(target_percent=0.33, symbol=symbol)
            # 初始化计数器 (虽然会在下个 bar 的 pos>0 分支中自增，但这里先占位)
            self.bars_held[symbol] = 0
            self.entry_prices[symbol] = bar.close

        elif pos > 0:
            entry_price = self.entry_prices.get(symbol, bar.close)
            current_bars_held = self.bars_held.get(symbol, 0)

            # 计算收益率
            pnl_pct = (bar.close - entry_price) / entry_price

            # 止盈条件：收益率 >= 10%
            if pnl_pct >= 0.10:
                self.sell(symbol, pos)
                print(
                    f"Take Profit Triggered for {symbol}: Entry={entry_price}, "
                    f"Current={bar.close}, PnL={pnl_pct:.2%}"
                )
            # 持仓时间条件：持有满 100 个 Bar
            elif current_bars_held >= 100:
                self.close_position()


events: list[aq.BacktestStreamEvent] = []

result = aq.run_backtest(
    strategy=MyStrategy,
    data=df,
    initial_cash=5000000,
    commission_rate=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    min_commission=5.0,
    lot_size=1,
    start_time="20250101",
    end_time="20250105",
    symbols=["600000", "600004", "600006"],
    on_event=events.append,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(result)
print(result.orders_df)
liquidation_audit_df = result.liquidation_audit_df
if liquidation_audit_df.empty:
    print("liquidation_audit_df is empty")
else:
    print(liquidation_audit_df)
print(f"stream_events={len(events)}")

# Verify metrics manually in Python
equity_curve = result.equity_curve
if not equity_curve.empty:
    print("\n--- Manual Verification ---")

    # 1. Total Return
    initial_equity = result.metrics.initial_market_value
    final_equity = result.metrics.end_market_value
    total_return_pct = (final_equity - initial_equity) / initial_equity
    print(f"Total Return % (Manual): {total_return_pct:.6%}")
    print(f"Total Return % (Rust):   {result.metrics.total_return_pct / 100:.6%}")

    # 2. Annualized Return
    # Duration in days
    # Fix: result.metrics.duration is a datetime.timedelta in Python wrapper
    # if conversion works, or int (nanoseconds) if raw.
    # Let's inspect type first.
    duration_val = result.metrics.duration

    if isinstance(duration_val, int):
        # Nanoseconds
        duration_days = duration_val / (1e9 * 3600 * 24)
    elif hasattr(duration_val, "total_seconds"):
        # timedelta
        duration_days = duration_val.total_seconds() / (3600 * 24)
    else:
        # Fallback
        duration_days = 0.0

    if duration_days > 0:
        annualized_return = (1 + total_return_pct) ** (365 / duration_days) - 1
        print(f"Annualized Return (Manual): {annualized_return:.6f}")
        print(f"Annualized Return (Rust):   {result.metrics.annualized_return:.6f}")

    # 3. Volatility
    # Resample to daily (end of day), keeping only real trading days so the
    # manual figure matches the Rust metric (ffill would inject 0.0 returns
    # on weekends/holidays and understate volatility by ~15%).
    daily_equity = equity_curve.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()
    volatility = daily_returns.std() * (252**0.5)
    print(f"Volatility (Manual): {volatility:.6f}")
    print(f"Volatility (Rust):   {result.metrics.volatility:.6f}")

    # 4. Max Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve - rolling_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (drawdown / rolling_max).min()

    print(f"Max Drawdown (Manual): {max_drawdown:.6f}")
    print(
        f"Max Drawdown (Rust):   {-result.metrics.max_drawdown_value:.6f}"
    )  # Rust stores positive value for DD

    print(f"Max Drawdown % (Manual): {max_drawdown_pct:.6%}")
    print(
        f"Max Drawdown % (Rust):   {-result.metrics.max_drawdown_pct / 100:.6%}"
    )  # Rust stores positive value for DD %

    # 5. Std Error & R2
    y = equity_curve.to_numpy(dtype=float)
    n = len(y)
    x = np.arange(n)

    # Linear Regression: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # Residuals
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Standard Error of Estimate
    # Rust uses divisor (n - 2.0)
    std_error = np.sqrt(ss_res / (n - 2)) if n > 2 else 0.0

    print(f"R2 (Manual):        {r2:.6f}")
    print(f"R2 (Rust):          {result.metrics.equity_r2:.6f}")
    print(f"Std Error (Manual): {std_error:.6f}")
    print(f"Std Error (Rust):   {result.metrics.std_error:.6f}")

# print(result.equity_curve)
