from typing import Any, cast

import akquant as aq
import numpy as np
import pandas as pd
from akquant.akquant import Bar
from akquant.backtest import run_backtest
from akquant.ml import SklearnAdapter
from akquant.strategy import Strategy
from sklearn.linear_model import LogisticRegression  # type: ignore


class WalkForwardStrategy(Strategy):
    """
    Demo strategy for Walk-forward Validation.

    Uses a logistic regression model to predict price direction.
    """

    def __init__(self) -> None:
        """Initialize the strategy."""
        super().__init__()

        # 1. Initialize model
        self.model = SklearnAdapter(LogisticRegression())

        # 2. Configure Walk-forward Validation
        # Framework automatically handles data slicing and retraining
        self.model.set_validation(
            method="walk_forward",
            train_window=50,  # Use last 50 bars for training
            test_window=20,  # Configure the out-of-sample horizon
            rolling_step=10,  # Retrain every 10 bars
            frequency="1m",
            verbose=True,  # Print training logs
        )

        # Ensure we have enough history for features + training
        self.set_history_depth(60)
        self._last_logged_window_index = 0
        self._last_logged_pending_activation = 0

        print("WalkForwardStrategy initialized")

    def on_train_signal(self, context: Any) -> None:
        """Override on_train_signal to skip training during warmup."""
        if self._bar_count < 50:
            return
        super().on_train_signal(context)

    def prepare_features(
        self, df: pd.DataFrame, mode: str = "training"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Feature Engineering (Shared by Training and Prediction).

        Returns X, y (y can be None/Invalid for prediction).
        """
        X = pd.DataFrame()
        X["ret1"] = df["close"].pct_change()
        X["ret2"] = df["close"].pct_change(2)
        # Remove fillna(0) to avoid treating padded NaNs as valid zero returns
        # X = X.fillna(0)

        # Construct Label y (Next period return)
        # shift(-1) moves future return to current row
        future_ret = df["close"].pct_change().shift(-1)

        # Combine into one DataFrame to align drops
        data = pd.concat([X, future_ret.rename("future_ret")], axis=1)

        # Drop rows with NaN features (e.g. from history padding or initial pct_change)
        data = data.dropna(subset=["ret1", "ret2"])

        if mode == "training":
            # For training, we must have a valid future return
            data = data.dropna(subset=["future_ret"])

        # Calculate y on valid data
        y = (data["future_ret"] > 0).astype(int)
        X_clean = data[["ret1", "ret2"]]

        return cast(pd.DataFrame, X_clean), cast(pd.Series, y)

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar event.

        :param bar: Current bar data
        """
        if self.model is None:
            return

        validation_window = self.current_validation_window()
        if validation_window is None:
            return

        pending_activation = validation_window["pending_activation_bar"]
        if (
            not self.is_model_ready()
            and pending_activation is not None
            and pending_activation != self._last_logged_pending_activation
        ):
            pending_window_index = int(validation_window["pending_window_index"])
            print(
                f"Bar {bar.timestamp}: "
                f"Pending Window={pending_window_index} "
                f"Activation Bar={pending_activation}"
            )
            self._last_logged_pending_activation = int(pending_activation)
            return

        if not self.is_model_ready():
            return

        hist_df = self.get_history_df(5)

        current_ret1 = (bar.close - hist_df["close"].iloc[-2]) / hist_df["close"].iloc[
            -2
        ]
        current_ret2 = (bar.close - hist_df["close"].iloc[-3]) / hist_df["close"].iloc[
            -3
        ]

        X_curr = pd.DataFrame([[current_ret1, current_ret2]], columns=["ret1", "ret2"])
        X_curr = X_curr.fillna(0)

        try:
            pred_prob = self.model.predict(X_curr)
            signal = (
                pred_prob[0] if isinstance(pred_prob, (list, np.ndarray)) else pred_prob
            )
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
                f"Pred Signal = {signal:.4f}"
            )

            if signal > 0.55:
                self.buy(bar.symbol, 100)
            elif signal < 0.45:
                self.sell(bar.symbol, 100)
        except Exception:
            # Model might not be fitted yet
            pass


if __name__ == "__main__":
    # 生成合成数据
    dates = pd.date_range(start="2023-01-01", periods=20000, freq="1min")
    # 随机漫步价格
    price = 100 + np.cumsum(np.random.randn(20000))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1000,
            "symbol": "TEST",
        }
    )

    # 运行回测
    print("Running Walk-Forward Validation Backtest...")
    run_backtest(
        data=df,
        strategy=WalkForwardStrategy,
        symbols="TEST",
        lot_size=1,
        fill_policy=aq.CurrentClose(),
        history_depth=60,  # Explicitly pass history depth to ensure engine enables it
    )
    print("Backtest finished.")
