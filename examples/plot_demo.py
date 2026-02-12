"""
Advanced Plotting Demo.

Demonstrates a complete workflow:
1. Generating realistic synthetic market data.
2. Implementing an advanced strategy with Trend + Momentum + Risk Management.
3. Running a backtest with custom configurations.
4. Visualizing results with a professional-grade interactive dashboard using Plotly.
"""

from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
from akquant import (
    BacktestConfig,
    Bar,
    Indicator,
    PercentSizer,
    Strategy,
    StrategyConfig,
    run_backtest,
)
from akquant.backtest import BacktestResult
from plotly.subplots import make_subplots  # type: ignore


# --------------------------------------------------------------------------------
# 1. Advanced Strategy Implementation
# --------------------------------------------------------------------------------
class TrendMomentumStrategy(Strategy):
    """
    Trend Following + Momentum Filter Strategy.

    Logic:
    - Long Entry: Close > EMA(50) (Trend) AND RSI(14) < 70 (Not Overbought)
    - Long Exit: Close < EMA(50) OR RSI(14) > 80 (Take Profit) OR Stop Loss
    """

    def __init__(
        self, ema_period: int = 50, rsi_period: int = 14, stop_loss_pct: float = 0.05
    ):
        """Initialize the strategy."""
        super().__init__()
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.stop_loss_pct = stop_loss_pct

        # Position Sizer: Use 20% of cash per trade
        self.sizer = PercentSizer(percents=20)

        # Track entry prices manually for Stop Loss calculation
        self.entry_prices: dict[str, float] = {}

        # 1. Define Indicators
        # Using lambda for simplicity, but could be TA-Lib calls
        self.ema = Indicator(
            "ema", lambda df: df["close"].ewm(span=ema_period, adjust=False).mean()
        )
        self.rsi = Indicator("rsi", self._calculate_rsi)

        # 2. Register Indicators (Auto-calculation)
        self._indicators = [self.ema, self.rsi]

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI helper."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return cast(pd.Series, 100 - (100 / (1 + rs)))

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar event."""
        # 1. Get Indicator Values
        ema_val = self.ema.get_value(bar.symbol, bar.timestamp)
        rsi_val = self.rsi.get_value(bar.symbol, bar.timestamp)
        current_price = bar.close

        # 2. Get Current Position
        position = self.get_position(bar.symbol)
        avg_price = self.entry_prices.get(bar.symbol, bar.close)

        # 3. Trading Logic
        if position == 0:
            # Entry Condition: Uptrend & Not Overbought
            if current_price > ema_val and rsi_val < 70:
                self.buy(symbol=bar.symbol)  # Size calculated by Sizer
                self.entry_prices[bar.symbol] = current_price
        else:
            # Exit Condition 1: Trend Reversal
            if current_price < ema_val:
                self.sell(symbol=bar.symbol, quantity=position)
            # Exit Condition 2: Take Profit (Overbought)
            elif rsi_val > 80:
                self.sell(symbol=bar.symbol, quantity=position)
            # Exit Condition 3: Stop Loss
            elif current_price < avg_price * (1 - self.stop_loss_pct):
                self.sell(symbol=bar.symbol, quantity=position)

            # If any sell condition met (simplified logic, assuming one of above
            # executed if true)
            # In reality, we should check if sell was called.
            # Here we just clean up if position becomes 0 (next bar) or assume exit.
            # For this demo, we can just let it update on next entry.


# --------------------------------------------------------------------------------
# 2. Data Generation (Realistic Synthetic Data)
# --------------------------------------------------------------------------------
def generate_data(
    start_date: str = "2022-01-01", periods: int = 500, symbol: str = "BTC-USD"
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with trends and volatility."""
    dates = pd.date_range(start=start_date, periods=periods)
    np.random.seed(42)

    # Generate trend + random walk
    t = np.linspace(0, 4 * np.pi, periods)
    trend = 100 + 10 * np.sin(t) + np.linspace(0, 50, periods)
    noise = np.random.normal(0, 2, periods).cumsum()
    close = trend + noise

    # Ensure positive
    close = np.maximum(close, 1.0)

    # Derive other columns
    high = close * (1 + np.abs(np.random.normal(0, 0.02, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.02, periods)))
    open_p = close * (1 + np.random.normal(0, 0.01, periods))
    volume = np.abs(np.random.normal(1000, 500, periods)) + 100

    df = pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


# --------------------------------------------------------------------------------
# 3. Custom Visualization Function
# --------------------------------------------------------------------------------
def plot_advanced_result(
    result: BacktestResult,
    data_df: pd.DataFrame,
    strategy_instance: TrendMomentumStrategy,
    title: str = "Advanced Backtest",
) -> go.Figure:
    """
    Create a professional dashboard.

    Includes:
    1. Candlestick Chart + EMA + Buy/Sell Markers
    2. Equity Curve
    3. RSI Indicator
    4. Drawdown Area
    """
    # Prepare Data
    equity_curve = pd.DataFrame(result.equity_curve, columns=["time", "equity"])
    equity_curve["time"] = pd.to_datetime(equity_curve["time"], unit="ns", utc=True)
    equity_curve.set_index("time", inplace=True)

    trades = result.trades_df

    # Calculate indicators on full dataframe for plotting
    # (In real app, we might get this from strategy, here we re-calc for plotting)
    ema = data_df["close"].ewm(span=strategy_instance.ema_period, adjust=False).mean()

    delta = data_df["close"].diff()
    gain = (
        (delta.where(delta > 0, 0)).rolling(window=strategy_instance.rsi_period).mean()
    )
    loss = (
        (-delta.where(delta < 0, 0)).rolling(window=strategy_instance.rsi_period).mean()
    )
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Create Subplots
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        subplot_titles=("Price & Signals", "RSI", "Equity Curve", "Drawdown"),
    )

    # --- Row 1: Price & Signals ---
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data_df.index,
            open=data_df["open"],
            high=data_df["high"],
            low=data_df["low"],
            close=data_df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    # EMA
    fig.add_trace(
        go.Scatter(
            x=data_df.index,
            y=ema,
            mode="lines",
            line=dict(color="orange", width=1),
            name=f"EMA({strategy_instance.ema_period})",
        ),
        row=1,
        col=1,
    )

    # Trade Markers
    if not trades.empty:
        # Buys
        buys = trades[trades["direction"] == "LONG"]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["entry_time"],
                    y=buys["entry_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="green"),
                    name="Buy Signal",
                ),
                row=1,
                col=1,
            )

        # Sells (Exits)
        sells = trades[trades["direction"] == "LONG"]  # Long exits
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["exit_time"],
                    y=sells["exit_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color="red"),
                    name="Sell Signal",
                ),
                row=1,
                col=1,
            )

    # --- Row 2: RSI ---
    fig.add_trace(
        go.Scatter(
            x=data_df.index,
            y=rsi,
            mode="lines",
            line=dict(color="purple", width=1),
            name="RSI",
        ),
        row=2,
        col=1,
    )
    # Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)

    # --- Row 3: Equity ---
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve["equity"],
            mode="lines",
            line=dict(color="blue"),
            fill="tozeroy",
            name="Equity",
        ),
        row=3,
        col=1,
    )

    # --- Row 4: Drawdown ---
    max_equity = equity_curve["equity"].cummax()
    drawdown = (equity_curve["equity"] - max_equity) / max_equity
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=drawdown,
            mode="lines",
            line=dict(color="red"),
            fill="tozeroy",
            name="Drawdown",
        ),
        row=4,
        col=1,
    )

    # Layout Updates
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=1000,
        template="plotly_dark",  # Use dark theme for pro look
        hovermode="x unified",
    )

    return fig


# --------------------------------------------------------------------------------
# 4. Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" AKQuant Advanced Demo ".center(60))
    print("=" * 60)

    # 1. Prepare Data
    print("\n[1] Generating Market Data...")
    symbol = "BTC-USD"
    df = generate_data(periods=300, symbol=symbol)
    print(f"    Data Shape: {df.shape}")
    print(f"    Date Range: {df.index[0]} -> {df.index[-1]}")

    # 2. Configure Strategy & Backtest
    print("\n[2] Configuring Backtest...")
    strat_config = StrategyConfig(
        initial_cash=100_000.0,
        fee_amount=0.001,  # 0.1% per order
    )

    backtest_config = BacktestConfig(
        strategy_config=strat_config,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        show_progress=True,
    )

    strategy = TrendMomentumStrategy(ema_period=20, rsi_period=14)

    # 3. Run Backtest
    print("\n[3] Running Backtest...")
    result = run_backtest(
        data=df, strategy=strategy, symbol=symbol, config=backtest_config
    )

    # 4. Print Metrics
    print("\n[4] Performance Metrics:")
    metrics = result.metrics
    print(f"    Total Return:      {metrics.total_return_pct:>6.2f}%")
    print(f"    Annualized Return: {metrics.annualized_return:>6.2f}%")
    print(f"    Sharpe Ratio:      {metrics.sharpe_ratio:>6.2f}")
    print(f"    Max Drawdown:      {metrics.max_drawdown_pct:>6.2f}%")
    print(f"    Win Rate:          {metrics.win_rate * 100:>6.2f}%")
    print(f"    Total Trades:      {len(result.trades_df)}")

    # 5. Visualization
    print("\n[5] Generating Visualization...")
    # Method A: Built-in simple plot
    # fig_simple = result.plot(title="Simple Result", show=False)

    # Method B: Advanced Custom Dashboard
    fig = plot_advanced_result(result, df, strategy)

    output_file = "advanced_backtest_result.html"
    fig.write_html(output_file)
    print(f"    SUCCESS! Dashboard saved to: {output_file}")
