# Python for Finance 101

Many beginners feel overwhelmed by code. Don't worry! In AKQuant, you don't need to be a computer scientist. Mastering just the **core 20% of syntax** covers 80% of strategy development needs.

This guide will teach you Python strictly from a **Financial Trading** perspective.

---

## 1. Variables & Data Types: Defining Assets

In Python, variables are like labeled boxes used to store data.

### Basic Variables
*   **int**: For volume, lot size.
*   **float**: For price, returns.
*   **str**: For ticker symbols.

```python
# Define variables
symbol = "AAPL"        # Ticker (String)
price = 150.5          # Current Price (Float)
volume = 100           # Buy Quantity (Integer)
is_holding = True      # Position Status (Boolean)

# Print them out
print(f"Trading {symbol}, Price: {price}")
```

### Lists: A Sequence of Prices
Imagine a candlestick chart. The closing prices of the last 5 days form a list.

```python
# Closing prices of last 5 days
closes = [150.1, 152.0, 149.5, 153.0, 155.0]

# Access data
print(closes[0])   # 1st day price (Python counts from 0!) -> 150.1
print(closes[-1])  # Last day price (Latest) -> 155.0
print(closes[:3])  # First 3 days -> [150.1, 152.0, 149.5]

# Calculate Average
avg_price = sum(closes) / len(closes)
print(f"5-day Average: {avg_price}")
```

### Dictionaries: Your Portfolio
A dictionary is like a ledger. Left side is the Name (Key), right side is the Value.

```python
# Current Portfolio: Ticker -> Shares
portfolio = {
    "AAPL": 100,
    "TSLA": 50,
    "GOOG": 0
}

# Check holdings
print(portfolio["AAPL"]) # -> 100

# Update holdings
portfolio["TSLA"] += 10  # Add 10 shares
```

---

## 2. Logic Control: The Core of Decision

A trading strategy is essentially a set of `if...else` statements: If X happens, do Y.

### If/Else Statements
This is the "trigger" of your strategy.

```python
ma5 = 155.0  # Short-term MA
ma20 = 150.0 # Long-term MA
cash = 5000  # Available Cash

# Golden Cross Logic
if ma5 > ma20:
    print("Signal: Golden Cross!")
    if cash > 1000:
        print("Cash sufficient, executing BUY!")
    else:
        print("Insufficient funds.")
else:
    print("No signal, hold.")
```

### Loops
Backtesting is basically a big loop, iterating through historical data day by day.

```python
# Mock Backtest: Iterate through daily prices
prices = [100, 110, 90, 120, 130]

for p in prices:
    if p > 120:
        print(f"Price {p} broke 120, take profit!")
    else:
        print(f"Price {p} is normal.")
```

---

## 3. Pandas Crash Course: The Quant Tool

In AKQuant, 99% of data (like `history_data`) is in `DataFrame` format. Think of it as a **Supercharged Excel**.

### DataFrame Structure
A DataFrame has Index (Rows, usually Time) and Columns (e.g., Open/Close).

```python
import pandas as pd

# Mock OHLCV Table
data = {
    "close": [100, 101, 102, 101, 103],
    "volume": [1000, 1500, 2000, 1200, 3000]
}
df = pd.DataFrame(data)

# 1. Get a whole column (Series)
print(df["close"])

# 2. Calculate Indicators (Vectorized, very fast)
# Calculate mean of close prices
ma_price = df["close"].mean()

# 3. Get the latest row
last_bar = df.iloc[-1]
print(f"Latest Close: {last_bar['close']}")
```

---

## 4. Classes & Objects: Understanding Strategy Templates

When writing a strategy, you see `class MyStrategy(Strategy):`. What does it mean?

*   **Class**: The **Blueprint**. Defines what the strategy looks like.
*   **Object**: The **Instance**. Built from the blueprint when backtesting runs.
*   **self**: Represents **The Strategy Instance Itself**.
    *   `self.buy()`: I want to buy.
    *   `self.cash`: My cash.
    *   `self.position`: My position.

```python
class MyStrategy:
    def __init__(self, initial_cash):
        # Init: Runs once when strategy starts
        self.cash = initial_cash
        self.name = "MA Strategy"

    def on_bar(self, price):
        # Runs on every bar close
        if price < 100:
            self.buy()

    def buy(self):
        self.cash -= 100
        print(f"{self.name} Executed BUY. Cash left: {self.cash}")

# Use Strategy
strategy = MyStrategy(1000) # Create Instance
strategy.on_bar(99)         # Trigger BUY
```

---

## 5. Summary: AKQuant Cheat Sheet

Here are the most common Python snippets you'll use in AKQuant:

| Scenario | Python Code | Meaning |
| :--- | :--- | :--- |
| **Get Data** | `hist = self.history_data(n=20)` | Get last 20 bars (DataFrame) |
| **Calc MA** | `ma = hist['close'].mean()` | Calculate average of close prices |
| **Latest Price** | `current_price = bar.close` | Get current bar's close price |
| **Signal** | `if ma_short > ma_long:` | If Short MA > Long MA |
| **Check Pos** | `pos = self.get_position(bar.symbol)` | How many shares do I hold? |
| **Order** | `self.buy(bar.symbol, 100)` | Buy 100 shares |

Master these, and you are ready to write your first strategy! Go to [Quant Guide](quant_guide.md) to practice.
