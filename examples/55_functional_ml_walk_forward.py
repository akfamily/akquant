#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Function-style walk-forward ML demo."""

from typing import Any, cast

import numpy as np
import pandas as pd
from akquant import CurrentClose
from akquant.backtest import run_backtest
from akquant.ml import SklearnAdapter
from sklearn.linear_model import LogisticRegression  # type: ignore


def initialize(ctx: Any) -> None:
    """Initialize function-style strategy state."""
    ctx.model = SklearnAdapter(LogisticRegression())
    ctx.model.set_validation(
        method="walk_forward",
        train_window=50,
        test_window=20,
        rolling_step=10,
        frequency="1m",
        verbose=True,
    )
    ctx.set_history_depth(60)
    ctx._last_logged_window_index = 0
    ctx._last_logged_pending_activation = 0
    print("Functional walk-forward strategy initialized")


def prepare_features(
    df: pd.DataFrame, mode: str = "training"
) -> tuple[pd.DataFrame, pd.Series]:
    """Build shared features for training and inference."""
    features = pd.DataFrame()
    features["ret1"] = df["close"].pct_change()
    features["ret2"] = df["close"].pct_change(2)
    future_ret = df["close"].pct_change().shift(-1)

    dataset = pd.concat([features, future_ret.rename("future_ret")], axis=1)
    dataset = dataset.dropna(subset=["ret1", "ret2"])
    if mode == "training":
        dataset = dataset.dropna(subset=["future_ret"])

    labels = (dataset["future_ret"] > 0).astype(int)
    x_clean = dataset[["ret1", "ret2"]]
    return cast(pd.DataFrame, x_clean), cast(pd.Series, labels)


def on_train_signal(ctx: Any) -> None:
    """Train a fresh model when the walk-forward window advances."""
    if ctx._bar_count < 50:
        return
    training_df, _ = ctx.get_rolling_data()
    x_train, y_train = prepare_features(training_df, mode="training")
    ctx.model.fit(x_train, y_train)


def on_bar(ctx: Any, bar: Any) -> None:
    """Infer on each bar once the pending model becomes active."""
    if ctx.model is None:
        return

    validation_window = ctx.current_validation_window()
    if validation_window is None:
        return

    pending_activation = validation_window["pending_activation_bar"]
    if (
        not ctx.is_model_ready()
        and pending_activation is not None
        and pending_activation != ctx._last_logged_pending_activation
    ):
        pending_window_index = int(validation_window["pending_window_index"])
        print(
            f"Bar {bar.timestamp}: "
            f"Pending Window={pending_window_index} "
            f"Activation Bar={pending_activation}"
        )
        ctx._last_logged_pending_activation = int(pending_activation)
        return

    if not ctx.is_model_ready():
        return

    hist_df = ctx.get_history_df(5)
    current_ret1 = (bar.close - hist_df["close"].iloc[-2]) / hist_df["close"].iloc[-2]
    current_ret2 = (bar.close - hist_df["close"].iloc[-3]) / hist_df["close"].iloc[-3]
    x_curr = pd.DataFrame([[current_ret1, current_ret2]], columns=["ret1", "ret2"])
    x_curr = x_curr.fillna(0)

    try:
        pred_prob = ctx.model.predict(x_curr)
        signal = (
            pred_prob[0] if isinstance(pred_prob, (list, np.ndarray)) else pred_prob
        )
        window_index = int(validation_window["window_index"])
        active_start_bar = validation_window["active_start_bar"]
        active_end_bar = validation_window["active_end_bar"]
        if window_index != ctx._last_logged_window_index:
            print(
                f"Bar {bar.timestamp}: "
                f"Activated Window={window_index} "
                f"ActiveRange=[{active_start_bar}, {active_end_bar}]"
            )
            ctx._last_logged_window_index = window_index

        print(
            f"Bar {bar.timestamp}: "
            f"Window={window_index} "
            f"ActiveRange=[{active_start_bar}, {active_end_bar}] "
            f"Pred Signal = {signal:.4f}"
        )

        if signal > 0.55:
            ctx.buy(bar.symbol, 100)
        elif signal < 0.45:
            ctx.sell(bar.symbol, 100)
    except Exception:
        pass


def main() -> None:
    """Run the function-style walk-forward demo."""
    dates = pd.date_range(start="2023-01-01", periods=20000, freq="1min")
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

    print("Running functional walk-forward validation backtest...")
    run_backtest(
        data=df,
        strategy=on_bar,
        initialize=initialize,
        on_train_signal=on_train_signal,
        symbols="TEST",
        lot_size=1,
        fill_policy=CurrentClose(),
        history_depth=60,
    )
    print("done_functional_ml_walk_forward")


if __name__ == "__main__":
    main()
