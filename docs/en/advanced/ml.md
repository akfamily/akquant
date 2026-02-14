# Machine Learning & Walk-forward Training Guide (ML Guide)

AKQuant includes a high-performance machine learning training framework designed specifically for quantitative trading. It solves the common "future function" leakage problem in traditional frameworks and provides out-of-the-box support for Walk-forward Validation.

## Core Design Philosophy

### 1. Separation of Signal and Action

A common mistake for beginners is to let the model directly output "Buy/Sell" instructions. In AKQuant, we decouple this process:

*   **Model Layer**: Only responsible for predicting future probabilities or values (Signal) based on historical data. It does not know how much money is in the account or what the current market risk is.
*   **Strategy Layer**: Receives the Signal from the model, combines it with risk control rules, capital management, and market status, and finally makes trading decisions (Action).

### 2. Adapter Pattern

To unify the distinct programming paradigms of Scikit-learn (traditional ML) and PyTorch (deep learning), we introduced an adapter layer:

*   **SklearnAdapter**: Adapts XGBoost, LightGBM, RandomForest, etc.
*   **PyTorchAdapter**: Adapts deep networks like LSTM, Transformer, automatically handling DataLoader and training loops.

Users only need to interface with the unified `QuantModel` interface.

### 3. Walk-forward Validation

On time-series data, random K-Fold cross-validation is incorrect because it uses future data to predict the past. The correct approach is Walk-forward:

1.  **Window 1**: Train on 2020 data, predict 2021 Q1.
2.  **Window 2**: Train on 2020 Q2 - 2021 Q1 data, predict 2021 Q2.
3.  ... Move forward like a rolling wheel.

---

## Quick Start

### 1. Define Strategy

You need to inherit from `Strategy` and implement the `prepare_features` method.

```python
from akquant.strategy import Strategy
from akquant.ml import SklearnAdapter
from sklearn.linear_model import LogisticRegression
import pandas as pd

class MyMLStrategy(Strategy):
    def __init__(self):

        # 1. Initialize Model
        # You can use any sklearn-compatible model here, such as RandomForest, XGBoost
        self.model = SklearnAdapter(LogisticRegression())

        # 2. Configure Walk-forward Validation
        # The framework automatically handles data splitting, model retraining, and parameter freezing
        self.model.set_validation(
            method='walk_forward',
            train_window='1y',   # Use past 1 year of data for training each time
            rolling_step='3m',   # Retrain every 3 months
            frequency='1d',      # Data frequency (used for parsing time strings)
            verbose=True         # Print training logs
        )

        # Ensure historical data length is sufficient
        self.set_history_depth(300)

    def prepare_features(self, df: pd.DataFrame):
        """
        [Must Implement] Feature Engineering Logic

        :param df: Raw OHLCV data (DataFrame)
        :return: (X, y) tuple
        """
        # Simple Example: Use returns as features
        X = pd.DataFrame()
        X['ret1'] = df['close'].pct_change()
        X['ret2'] = df['close'].pct_change(2)
        X = X.fillna(0)

        # Construct Label y (Predict next period's movement)
        # shift(-1) moves future returns to the current row as label
        future_ret = df['close'].pct_change().shift(-1)
        y = (future_ret > 0).astype(int)

        # Note: The framework only handles slicing raw data df by time window, not feature/label alignment
        # Since shift(-1) causes the last row of y to be NaN, it must be manually removed
        # Otherwise sklearn/pytorch training will fail
        return X.iloc[:-1], y.iloc[:-1]

    def on_bar(self, bar):
        # 3. Real-time Prediction & Trading
        # Check if model is ready (first 1 year of data used for cold start training)
        if self._bar_count < 250:
            return

        # Get recent data for feature extraction
        hist_df = self.get_history_df(5)

        # For demonstration, manually construct current features (consistent with prepare_features logic)
        curr_ret1 = (bar.close - hist_df['close'].iloc[-2]) / hist_df['close'].iloc[-2]
        curr_ret2 = (bar.close - hist_df['close'].iloc[-3]) / hist_df['close'].iloc[-3]

        X_curr = pd.DataFrame([[curr_ret1, curr_ret2]], columns=['ret1', 'ret2'])
        X_curr = X_curr.fillna(0)

        try:
            # Get prediction signal (probability)
            # For binary classification, returns probability of Class 1 (Up)
            signal = self.model.predict(X_curr)[0]

            # Place orders combining risk control rules
            if signal > 0.6:
                self.buy(bar.symbol, 100)
            elif signal < 0.4:
                self.sell(bar.symbol, 100)

        except Exception:
            # Model might not have completed initial training
            pass
```

### 2. Run Backtest

```python
from akquant.backtest import run_backtest

run_backtest(
    strategy=MyMLStrategy,
    symbol="600000",
    start_date="20200101",
    end_date="20231231",
    cash=100000.0
)
```

## API Reference

### `model.set_validation`

Configures the model's validation and training method.

```python
def set_validation(
    self,
    method: str = 'walk_forward',
    train_window: str | int = '1y',
    test_window: str | int = '3m',
    rolling_step: str | int = '3m',
    frequency: str = '1d',
    incremental: bool = False,
    verbose: bool = False
)
```

*   `method`: Currently only supports `'walk_forward'`.
*   `train_window`: Training window length. Supports `'1y'` (1 year), `'6m'` (6 months), `'50d'` (50 days) or integer (number of Bars).
*   `rolling_step`: Rolling step size, i.e., how often to retrain the model.
*   `frequency`: Data frequency, used to correctly convert time strings to Bar counts (e.g., under '1d', 1y=252 bars).
*   `incremental`: Whether to use incremental learning (continue from last training) or retrain from scratch. Default is `False`.
*   `verbose`: Whether to print training logs, default is `False`.

### `strategy.prepare_features`

Callback function that must be implemented by the user for feature engineering.

```python
def prepare_features(self, df: pd.DataFrame, mode: str = "training") -> Tuple[Any, Any]
```

*   **Input**:
    *   `df`: Raw dataframe from `get_rolling_data`.
    *   `mode`: `"training"` or `"inference"`.
*   **Output**:
    *   If `mode="training"`, return `(X, y)`.
    *   If `mode="inference"`, return `X` (usually the last row) or `(X, None)`.
*   **Note**: This is a pure function and should not rely on external state. It will be used in both training and prediction stages.

## Deep Learning Support (PyTorch)

Using `PyTorchAdapter`, you can easily integrate deep learning models:

```python
from akquant.ml import PyTorchAdapter
import torch.nn as nn
import torch.optim as optim

# Define Network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Use in Strategy
self.model = PyTorchAdapter(
    network=SimpleNet(),
    criterion=nn.BCELoss(),
    optimizer_cls=optim.Adam,
    lr=0.001,
    epochs=20,
    batch_size=64,
    device='cuda'  # Supports GPU acceleration
)
```
