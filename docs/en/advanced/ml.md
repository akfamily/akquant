# Machine Learning & Rolling Training Guide (ML Guide)

AKQuant includes a high-performance machine learning training framework designed specifically for quantitative trading. It addresses the common "future function" leakage problem in traditional frameworks and provides out-of-the-box support for Walk-forward Validation.

## Core Design Philosophy

### 1. Signal vs. Action Separation

A common mistake for beginners is to let the model output "buy/sell" instructions directly. In AKQuant, we decouple this process:

*   **Model Layer**: Responsible only for predicting future probabilities or values (Signal) based on historical data. It does not know how much money the account has or what the current market risk is.
*   **Strategy Layer**: Receives the Signal from the model and makes buy/sell decisions (Action) combined with risk control rules, capital management, and market status.

### 2. Adapter Pattern

To unify the disparate programming paradigms of Scikit-learn (traditional machine learning) and PyTorch (deep learning), we introduced an adapter layer:

*   **SklearnAdapter**: Adapts XGBoost, LightGBM, RandomForest, etc.
*   **PyTorchAdapter**: Adapts deep networks like LSTM, Transformer, automatically handling DataLoader and training loops.

Users only need to interface with the unified `QuantModel`.

### 3. Walk-forward Validation

On time-series data, random K-Fold cross-validation is incorrect because it uses future data to predict the past. The correct approach is Walk-forward:

1.  **Window 1**: Train on 2020 data, predict 2021 Q1.
2.  **Window 2**: Train on 2020 Q2 - 2021 Q1 data, predict 2021 Q2.
3.  ... Rolling forward like a wheel.

### 4. Preventing Look-ahead Bias

In quantitative ML, the most dangerous error is using future data. AKQuant recommends following these principles:

*   **Features (X)**: Can only use data from time $t$ and before.
*   **Labels (y)**: Describe the state at time $t+1$ (e.g., future returns), but when training at time $t$, we actually use $X$ at time $t$ to fit $y$ at time $t+1$.
*   **Implementation**: Constructing $y$ usually requires `shift(-1)`, which results in the last row of data having no label (because there is no future), so it must be dropped before training.

### 5. Preventing Data Leakage: Using Pipeline

Feature preprocessing (e.g., standardization, normalization) can also introduce Look-ahead Bias. For example, using `StandardScaler` on the entire dataset implies that the training set contains mean and variance information from the future test set.

**Solution**: Encapsulate preprocessing steps in `sklearn.pipeline.Pipeline`.

*   **Encapsulation**: Pipeline treats the Scaler and Model as a whole.
*   **Isolation**: During Walk-forward training, Pipeline calls `fit` (calculating mean/variance) only on the current training window data, then applies it to the validation set.
*   **Consistency**: In the inference phase, Pipeline automatically applies the trained statistics without manual user maintenance.

### 6. Model Lifecycle in the Current Compatibility Mode

The current Walk-forward implementation uses a compatibility-oriented lifecycle:

*   **Training Window**: After the current bar finishes, the framework trains a new model clone on the latest `train_window` bars.
*   **Delayed Activation**: The newly trained model does not predict on the current bar. It becomes active on the next bar.
*   **Effective Range**: `test_window` defines the intended out-of-sample range for the active model.
*   **Rolling Updates**: `rolling_step` controls when the next retraining is triggered. If it is `0`, the framework falls back to `test_window`.
*   **Explicit State Checks**: In `on_bar`, prefer `self.is_model_ready()` and `self.current_validation_window()` before calling `self.model.predict(...)`.
*   **Model Cloning**: The framework calls `QuantModel.clone()` to create a pending model for each training window. Override it if your custom model cannot be deep-copied safely.

---

## Complete Runnable Example

The following code demonstrates how to build a robust strategy combining **Pipeline** and **Walk-forward Validation**.

```python
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from akquant import Strategy, run_backtest
from akquant.ml import SklearnAdapter

class WalkForwardStrategy(Strategy):
    """
    Demo Strategy: Predicting returns using Logistic Regression (with Pipeline preprocessing)
    """

    def __init__(self):
        # 1. Initialize Model (Encapsulate preprocessing and model using Pipeline)
        # StandardScaler: Ensures standardization using training set statistics to prevent leakage
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression())
        ])

        self.model = SklearnAdapter(pipeline)

        # 2. Configure Walk-forward Validation
        # The framework automatically handles data slicing and model retraining
        self.model.set_validation(
            method='walk_forward',
            train_window=50,   # Train on past 50 bars
            test_window=20,    # Keep each fitted model active for 20 OOS bars
            rolling_step=10,   # Retrain every 10 bars
            frequency='1m',    # Data frequency
            incremental=False, # Whether to use incremental learning (Sklearn supports partial_fit)
            verbose=True       # Print training logs
        )

        # Ensure history depth covers training window + feature calculation window
        # Alternatively use self.warmup_period = 60
        self.set_history_depth(60)
        self._last_logged_window_index = 0
        self._last_logged_pending_activation = 0

    def prepare_features(self, df: pd.DataFrame, mode: str = "training") -> Tuple[Any, Any]:
        """
        [Must Implement] Feature Engineering Logic
        Used for both training (generating X, y) and inference (generating X)
        """
        X = pd.DataFrame()
        # Feature 1: 1-period return
        X['ret1'] = df['close'].pct_change()
        # Feature 2: 2-period return
        X['ret2'] = df['close'].pct_change(2)

        if mode == 'inference':
            # Inference Mode: Return only the last row of features, no y needed
            # Note: df passed during inference is the recent history_depth data
            # The last row is the latest bar, we need its features
            return X.iloc[-1:]

        # Training Mode: Construct label y (predict next period's return)
        # shift(-1) moves future return to current row as label
        future_ret = df['close'].pct_change().shift(-1)

        # Combine into one DataFrame to align drops
        data = pd.concat([X, future_ret.rename("future_ret")], axis=1)

        # Drop rows with NaN features (e.g. from history padding or initial pct_change)
        data = data.dropna(subset=["ret1", "ret2"])

        # For training, we must have a valid future return
        data = data.dropna(subset=["future_ret"])

        # Calculate y on valid data
        y = (data["future_ret"] > 0).astype(int)
        X_clean = data[["ret1", "ret2"]]

        return X_clean, y

    def on_bar(self, bar):
        # 3. Real-time Prediction & Trading
        validation_window = self.current_validation_window()
        if validation_window is None:
            return

        pending_activation = validation_window["pending_activation_bar"]
        if (
            not self.is_model_ready()
            and pending_activation is not None
            and pending_activation != self._last_logged_pending_activation
        ):
            print(
                f"Bar {bar.timestamp}: "
                f"Pending Window={validation_window['pending_window_index']} "
                f"Activation Bar={pending_activation}"
            )
            self._last_logged_pending_activation = int(pending_activation)
            return

        if not self.is_model_ready():
            return

        # Get recent history for feature extraction
        # Note: Need enough history to calculate features (e.g. pct_change(2) needs at least 3 bars)
        hist_df = self.get_history_df(10)

        # If data is insufficient, return
        if len(hist_df) < 5:
            return

        # Reuse feature calculation logic!
        # Directly call prepare_features to get current features
        X_curr = self.prepare_features(hist_df, mode='inference')

        try:
            # Get prediction signal (probability)
            # SklearnAdapter returns probability of Class 1 for binary classification
            signal = self.model.predict(X_curr)[0]
            window_index = int(validation_window["window_index"])
            active_start_bar = validation_window["active_start_bar"]
            active_end_bar = validation_window["active_end_bar"]

            if window_index != self._last_logged_window_index:
                print(
                    f"Bar {bar.timestamp}: "
                    f"Activated Window={window_index} "
                    f"ActiveRange=[{active_start_bar}, {active_end_bar}]"
                )
                self._last_logged_window_index = window_index

            print(
                f"Bar {bar.timestamp}: "
                f"Window={window_index} "
                f"ActiveRange=[{active_start_bar}, {active_end_bar}] "
                f"Signal={signal:.4f}"
            )

            # Combine with risk rules for ordering
            # Use self.get_position(symbol) to check position
            pos = self.get_position(bar.symbol)

            if signal > 0.55 and pos == 0:
                self.buy(bar.symbol, 100)
            elif signal < 0.45 and pos > 0:
                self.sell(bar.symbol, pos)

        except Exception:
            # Keep the example resilient to inference-time failures
            pass

if __name__ == "__main__":
    # 1. Generate Synthetic Data
    print("Generating test data...")
    dates = pd.date_range(start="2023-01-01", periods=500, freq="1min")
    # Random walk price
    price = 100 + np.cumsum(np.random.randn(500))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": 1000,
        "symbol": "TEST"
    })

    # 2. Run Backtest
    print("Starting ML Backtest...")
    result = run_backtest(
        data=df,
        strategy=WalkForwardStrategy,
        symbols="TEST",
        lot_size=1,
        fill_policy={"price_basis": "close", "bar_offset": 0, "temporal": "same_cycle"}, # Match at close of current bar
        history_depth=60,
        warmup_period=50,
    )
    print("Backtest Finished.")

    # 3. Print Results
    print(result)
```

### Example Output

After running the code above, you will see output similar to this (including detailed performance metrics):

```text
Generating test data...
Starting ML Backtest...
2026-02-09 15:58:29 | INFO | Running backtest via run_backtest()...
[########################################] 500/500 (0s)
Backtest Finished.
BacktestResult:
                                            Value
name
start_time              2023-01-01 00:00:00+08:00
end_time                2023-01-01 08:19:00+08:00
duration                          0 days, 8:19:00
total_bars                                    500
closed_trade_count                                  12.0
execution_count                                     24.0
open_position_count                                  0.0
initial_market_value                     100000.0
end_market_value                        100120.50
total_pnl                                  120.50
total_return_pct                         0.120500
annualized_return                        0.127450
max_drawdown                                50.00
max_drawdown_pct                         0.049900
win_rate                                58.333333
loss_rate                               41.666667
```

## Advanced Guide

### 1. Feature Engineering Tips

Excellent features are key to ML success. Besides simple returns, consider:

*   **Technical Indicators**: RSI, MACD, Bollinger Bands (recommend using `talib` or `pandas_ta`).
*   **Volatility Features**: Historical volatility, ATR.
*   **Market Microstructure**: Buying/selling pressure, volume-price relationship.
*   **Time Features**: Hour, Day of Week (note these are categorical, may need One-hot encoding).

### 2. Model Persistence (Save/Load)

Trained models can be saved for live trading or subsequent analysis.

```python
# Save
strategy.model.save("my_model.pkl")

# Load (in __init__)
self.model.load("my_model.pkl")
```

### 3. Deep Learning Support (PyTorch)

Use `PyTorchAdapter` to easily integrate deep learning models. You need to define a standard `nn.Module`.

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
    device='cuda'  # Support GPU acceleration
)
```

## API Reference

### `model.set_validation`

Configure model validation and training methods.

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
*   `train_window`: Length of training window. Supports `'1y'` (1 year), `'6m'` (6 months), `'50d'` (50 days), or integer (number of bars).
*   `test_window`: Intended out-of-sample window length for the active model. In compatibility mode, the newly trained model activates on the next bar and covers this range by default.
*   `rolling_step`: Rolling step size, i.e., how often to retrain the model. If it is `0`, the framework falls back to `test_window`.
*   `frequency`: Data frequency, used to correctly convert time strings to bar counts (e.g., 1y = 252 bars under '1d').
*   `incremental`: Whether to use incremental learning (continue training from the last active model) or retrain from scratch. Default is `False`.
*   `verbose`: Whether to print training logs. Default is `False`.

### `model.clone`

Create a model copy for a new training window.

```python
def clone(self) -> QuantModel
```

*   The default implementation uses `copy.deepcopy`.
*   Override this method if your model owns GPU handles, locks, file descriptors, or any state that should not be copied blindly.
*   The framework trains a pending model on the current bar and activates it on the next one, so `clone()` is central to window isolation.

### `strategy.prepare_features`

Callback function that must be implemented by the user for feature engineering.

```python
def prepare_features(self, df: pd.DataFrame, mode: str = "training") -> Tuple[Any, Any]
```

*   **Input**:
    *   `df`: Historical data DataFrame.
    *   `mode`: `"training"` (Training mode) or `"inference"` (Inference mode).
*   **Output**:
    *   `mode="training"`: Return `(X, y)`.
    *   `mode="inference"`: Return `X` (usually the last row).
*   **Note**: This is a pure function and should not rely on external state.

### `strategy.is_model_ready`

Check whether an active model is currently available for inference.

```python
def is_model_ready(self) -> bool
```

*   `True` means it is safe to call `self.model.predict(...)` on the current bar.
*   Before the first training window completes, this typically returns `False`.

### `strategy.current_validation_window`

Return the current Walk-forward lifecycle state.

```python
def current_validation_window(self) -> dict[str, Any] | None
```

The returned dictionary may include:

*   `window_index`: Current active window index
*   `active_start_bar` / `active_end_bar`: Planned active range of the current model
*   `pending_activation_bar`: Bar index where the pending model will become active
*   `pending_window_index`: Pending window index
*   `next_train_bar`: Next scheduled retraining bar index
