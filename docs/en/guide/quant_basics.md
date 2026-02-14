# Quantitative Trading Guide for Beginners (Zero to Hero)

Welcome to the world of quantitative trading! This guide is designed for beginners to systematically master the process of developing quantitative strategies using the AKQuant framework, from theoretical concepts to practical coding.

---

## 0. Why Choose Quant? (Quant vs. Discretionary)

The investment world is mainly divided into two camps: **Discretionary Trading** and **Quantitative Trading**.

*   **Discretionary Trading** is like an **Art**. Traders rely on personal experience, intuition, and qualitative analysis (e.g., news, macro policies) to make decisions. Buffett and Soros are masters of this.
*   **Quantitative Trading** is a **Science**. It uses mathematical models and computer programs to find probabilistic advantages from massive historical data and executes them strictly. Simons (Medallion Fund) is the representative of quant.

### Core Comparison

| Dimension | Discretionary Trading | Quantitative Trading |
| :--- | :--- | :--- |
| **Basis** | Experience, intuition, news, qualitative analysis | **Data, statistical models, code logic** |
| **Execution** | Vulnerable to emotions (Greed/Fear) | **Automated execution, cold and disciplined (100%)** |
| **Coverage** | Limited human energy (dozens of stocks) | **Monitors thousands of stocks/markets simultaneously** |
| **Verifiability** | Hard to verify ("feelings" can't be backtested) | **Highly verifiable (via historical backtesting)** |
| **Cons** | Hard to replicate, unstable state, prone to errors | Model decay risk, may lag in "Black Swan" events |

**Reasons to Choose Quant**:
If you don't want to stare at the screen with heart palpitations every day, if you believe data beats intuition, and if you want to build a "money-making machine" that works for you long-term via code, then quantitative trading is for you.

---

## 1. Basic Concepts: The Awakening of Trader K

To help you understand quantitative trading intuitively, let's tell the story of a trader named "K". Through his journey, we will unlock the core concepts of quant trading.

### Act 1: Slave to Emotions vs. Machine Discipline
**Story**:
K used to be a typical manual trader. He stared at the screen every day, his mood fluctuating with the red and green candles.
*   10:00 AM, price skyrockets: "Buy now! Or I'll miss out!" -> Result: Bought at the peak.
*   2:00 PM, price dives: "It's over! Sell everything!" -> Result: Sold at the bottom.
At the end of the day, K was exhausted and lost money. He realized his biggest enemy was not the market, but **Greed and Fear**.

**Quant Perspective**:
The first advantage of quantitative trading is **Discipline**.
We write trading logic into code (i.e., **Strategy**) and let the computer execute it automatically. Machines have no emotions; they don't get excited by surges or panic during crashes. They coldly execute: "If A happens, do B".

### Act 2: The Mysterious "Time Machine" (Backtesting)
**Story**:
K summarized a rule: "I noticed that every time the price drops for three consecutive days, it bounces back on the fourth day."
He wanted to bet real money on this rule but was unsure: "Did this rule work in the 2018 bear market? How about the 2020 bull market?"
He wished for a time machine to go back and simulate trading with this rule to see how much he could make.

**Quant Perspective**:
This time machine is **Backtesting**.
Backtesting is the process of verifying a trading strategy using **historical data**.
*   **Input**: Your strategy logic (e.g., "Buy after 3 down days").
*   **Data**: Historical OHLCV data (Open, High, Low, Close, Volume).
*   **Output**: Your final return and max loss if you had traded this way.

### Act 3: A Lesson on "Survival" (Risk Management)
**Story**:
K once got lucky, went all-in on a stock, and doubled his money in a month. Thinking he was a genius, he borrowed money to leverage up and went all-in again.
Then a "Black Swan" event hit, and the price halved. K not only lost his profits but also his principal.
A quant veteran told him: "In this market, surviving is more important than making money fast."

**Quant Perspective**:
This is **Risk Management**. In quant reports, we look not only at how much you made (Return) but also at how much risk you took.
*   **Max Drawdown**: The largest decline from a historical peak. A -50% Max Drawdown means your assets could be cut in half.
*   **Sharpe Ratio**: A metric for cost-effectiveness. How much excess return you get per unit of risk.

### Act 4: The Trap of "Carving a Boat to Find a Sword" (Overfitting)
**Story**:
K learned to code and tried to find a perfect curve. He tweaked parameters endlessly and finally created a strategy with a 500% return last year!
He excitedly started live trading, only to lose 10% in a week.
It turned out his code implicitly said: "If date is Jan 5, 2023, buy". He memorized the answers to last year's exam, but this year's questions changed.

**Quant Perspective**:
This is called **Overfitting**. The strategy only memorized the noise in historical data and didn't grasp the true market laws. Excellent quant strategies should be simple in logic and universally applicable, not hard-coded to history.

---

## 1.5 Core Data Format (OHLCV)
Back to the technical side, the most basic data unit in quantitative trading is the Bar (or Candle), which typically contains:
*   **Open**: Opening price
*   **High**: Highest price
*   **Low**: Lowest price
*   **Close**: Closing price
*   **Volume**: Trading volume

---

## 2. Preparation and Installation

### 2.1 Install AKQuant
Ensure your computer has Python 3.8 or later installed. Open your terminal (or CMD) and run:

```bash
pip install akquant
```

### 2.2 Verify Installation
Create a file named `check_env.py`, enter the following code, and run it:

```python
import akquant
print(f"AKQuant version: {akquant.__version__}")
print("Installation successful!")
```

---

## 3. Hands-on: Developing Your First Strategy (Dual Moving Average)

We will implement a classic **Dual Moving Average Strategy**.
*   **Buy Signal (Golden Cross)**: Short-term MA (e.g., 5-day) crosses *above* Long-term MA (e.g., 20-day).
*   **Sell Signal (Death Cross)**: Short-term MA crosses *below* Long-term MA.

### 3.1 Complete Code Example
Create a file named `first_strategy.py`:

```python
import pandas as pd
import numpy as np
from akquant import Strategy, run_backtest

class DualMovingAverageStrategy(Strategy):
    def __init__(self):
        # Define parameters: Short window 5, Long window 20
        self.short_window = 5
        self.long_window = 20

    def on_bar(self, bar):
        # Get historical closing prices
        # history_data returns a DataFrame
        hist = self.history_data(n=self.long_window + 1)

        # If data is insufficient to calculate MA, return immediately
        if len(hist) < self.long_window:
            return

        # Calculate Short and Long MA
        closes = hist['close'].values
        ma_short = np.mean(closes[-self.short_window:])
        ma_long = np.mean(closes[-self.long_window:])

        # Get previous MA values (to detect crossover)
        prev_ma_short = np.mean(closes[-self.short_window-1 : -1])
        prev_ma_long = np.mean(closes[-self.long_window-1 : -1])

        # Get current position
        position = self.get_position(bar.symbol)

        # Trading Logic
        # 1. Golden Cross: Short MA crosses above Long MA, and no position -> Buy
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if position == 0:
                self.buy(bar.symbol, 100) # Buy 100 shares
                print(f"[{bar.datetime}] Buy {bar.symbol} @ {bar.close:.2f}")

        # 2. Death Cross: Short MA crosses below Long MA, and holding position -> Sell
        elif prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position > 0:
                self.sell(bar.symbol, 100) # Sell 100 shares
                print(f"[{bar.datetime}] Sell {bar.symbol} @ {bar.close:.2f}")

# ------------------------------
# Prepare Mock Data and Run
# ------------------------------
if __name__ == "__main__":
    # Generate mock data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31")
    price = 100 + np.cumsum(np.random.randn(len(dates))) # Random walk

    df = pd.DataFrame({
        "date": dates,
        "open": price, "high": price + 1, "low": price - 1, "close": price,
        "volume": 10000,
        "symbol": "DEMO"
    })

    # Run Backtest
    print("Starting backtest...")
    result = run_backtest(
        strategy_class=DualMovingAverageStrategy,
        data=df,
        initial_capital=10000.0 # Initial capital 10k
    )

    # Print summary
    print("\nBacktest finished!")
    print(f"Final Value: {result.final_value:.2f}")
    print(f"Total Return: {result.total_return * 100:.2f}%")
```

### 3.2 Advanced: How to Get Real Data?
The example above uses mock data. In practice, you need real stock data.
We recommend using open-source libraries like `yfinance` (Global data) or `AKShare` (Chinese market data).

**Example: Get Apple (AAPL) data using yfinance**

```bash
pip install yfinance
```

```python
import yfinance as yf

# Download data
df = yf.download("AAPL", start="2022-01-01", end="2023-12-31")

# Data cleaning: yfinance columns are Capitalized, AKQuant needs lowercase
df.columns = [c.lower() for c in df.columns]
df['symbol'] = "AAPL" # Add symbol column

# Pass directly to run_backtest
result = run_backtest(
    strategy_class=DualMovingAverageStrategy,
    data=df,
    initial_capital=10000.0
)
```

---

## 4. Understanding Backtest Reports: Key Metrics

After running a backtest, AKQuant outputs several metrics. Here is what they mean:

| Metric | Meaning & Interpretation |
| :--- | :--- |
| **Total Return** | How much profit/loss the strategy made relative to the initial capital. |
| **Annualized Return** | The expected return if the strategy runs for a year. Useful for comparing strategies of different durations. |
| **Max Drawdown** | The largest peak-to-trough decline. **A key risk metric**. E.g., -20% means you could lose 20% in the worst-case scenario. |
| **Sharpe Ratio** | Measures risk-adjusted return. How much excess return you get per unit of risk. >1 is good, >2 is excellent. |
| **Win Rate** | Percentage of profitable trades. High win rate doesn't guarantee high profit (you could make small profits but huge losses). |
| **SQN** | System Quality Number. Measures the stability of the trading system. Higher is better. |
| **Kelly Criterion** | Optimal position size based on win rate and profit/loss ratio. |
| **Exposure Time %** | Percentage of time the strategy holds a position. Useful for assessing capital efficiency and risk exposure. |
| **VaR / CVaR** | Value at Risk and Conditional VaR. Measures extreme downside risk. |

---

## 5. Common Issues and Debugging Tips

Here are common pitfalls for beginners:

### 5.1 Insufficient Data (IndexError / NaN)
*   **Symptom**: `IndexError` or MA calculation results in `NaN`.
*   **Cause**: Calculating a 20-day MA requires at least 20 days of history. At the start of the backtest, data is accumulating.
*   **Solution**: Check data length at the beginning of `on_bar`:
    ```python
    if len(self.history_data(n=20)) < 20:
        return
    ```

### 5.2 Look-ahead Bias
*   **Symptom**: Unreasonably high returns (e.g., 1000% annualized).
*   **Cause**: Using **future** data to generate signals. E.g., using tomorrow's close price when processing today's bar.
*   **Solution**: Ensure you only use data up to the current bar. AKQuant's `history_data` is safe by default.

### 5.3 Trades Not Executed
*   **Symptom**: Logs show buy signals, but position doesn't change.
*   **Cause**: Insufficient cash (to buy 1 unit) or minimum trade size not met.
*   **Solution**: Print `self.cash` to check available funds; verify if `self.buy` quantity is reasonable.

---

## 6. Next Steps: Learning Path

Congratulations on completing the beginner guide! To become a more professional quant trader, follow this path:

### 6.1 Skill Tree Order
1.  **Python Mastery**: Master `pandas` and `numpy`. Quant trading is 80% data processing, and these are your best weapons.
2.  **Classic Strategies**:
    *   **Turtle Trading**: Learn how to build a complete trend-following system.
    *   **Grid Trading**: Learn automated arbitrage in oscillating markets.
    *   **Multi-Factor**: Learn the stock selection logic used by institutions.
3.  **Risk Management**: Deep dive into Kelly Criterion, volatility targeting, etc.
4.  **Machine Learning**: Try predicting prices or volatility using ML models (see [ML Guide](../ml_guide.md)).

### 6.2 Recommended Resources
*   **Books**: "Inside the Black Box", "Way of the Turtle", "Python for Finance".
*   **Practice**: Read code in [Examples](../examples.md), try modifying parameters, and observe result changes.

Hope this guide helps you start your quantitative journey! For advanced features, please refer to the [API Reference](../api.md).
