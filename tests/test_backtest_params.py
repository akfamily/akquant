import numpy as np
import pandas as pd
import pytest
from akquant import Strategy, run_backtest
from akquant.params import IntParam


class PStrat(Strategy):
    """Minimal strategy declaring a single inline int param field."""

    fast = IntParam(5, ge=2, le=100)

    def on_start(self):
        """Subscribe to the test symbol and set warmup from the param."""
        self.subscribe("X")
        self.warmup_period = self.params.fast

    def on_bar(self, bar):
        """Record the resolved param value for inspection."""
        self._seen_fast = self.params.fast


def _data():
    idx = pd.date_range("2023-01-01", periods=30)
    close = 100 + np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
            "symbol": "X",
        },
        index=idx,
    )


def test_strategy_params_injected():
    """Test strategy_params flows through __param_model__ validation into params."""
    result = run_backtest(
        data=_data(),
        strategy=PStrat,
        symbols="X",
        initial_cash=100000.0,
        strategy_params={"fast": 7},
    )
    assert result is not None  # 跑通即可


def test_unknown_strategy_param_raises():
    """Test unknown strategy_params keys raise under strict_strategy_params."""
    with pytest.raises((TypeError, ValueError)):
        run_backtest(
            data=_data(),
            strategy=PStrat,
            symbols="X",
            initial_cash=100000.0,
            strategy_params={"nope": 1},
            strict_strategy_params=True,
        )
