"""Grid search parameter injection for function-style strategies."""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


def _make_df(length: int = 200) -> pd.DataFrame:
    """Build deterministic OHLCV data for the sweep."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=length, freq="D")
    prices = 100 + np.cumsum(np.random.randn(length))
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 100000,
            "symbol": "MOCK",
        }
    )


def _initialize(ctx: Any) -> None:
    """Read swept params off ctx and register the warmup period."""
    ctx.long_window = getattr(ctx, "long_window", 20)
    ctx.warmup_period = ctx.long_window + 1


def _on_bar(ctx: Any, bar: Bar) -> None:
    """Trade a dual-MA crossover using the swept parameters."""
    short_window = getattr(ctx, "short_window", 5)
    closes = ctx.get_history(
        count=ctx.long_window + 1, symbol=bar.symbol, field="close"
    )
    if len(closes) < ctx.long_window + 1:
        return
    history = closes[:-1]
    ma_short = history[-short_window:].mean()
    ma_long = history[-ctx.long_window :].mean()
    if ma_short > ma_long and ctx.get_position(bar.symbol) == 0:
        ctx.order_target_percent(symbol=bar.symbol, target_percent=0.95)
    elif ma_short < ma_long and ctx.get_position(bar.symbol) > 0:
        ctx.close_position(bar.symbol)


def test_grid_search_injects_params_into_functional_strategy() -> None:
    """Different param combos must produce different results, not identical rows."""
    results = aq.run_grid_search(
        strategy=_on_bar,
        data=_make_df(),
        param_grid={"short_window": [3, 10], "long_window": [15, 60]},
        initialize=_initialize,
        initial_cash=100_000,
        max_workers=1,
    )
    returns = [round(float(x), 6) for x in results["total_return_pct"].tolist()]
    assert len(returns) == 4
    assert len(set(returns)) > 1, (
        "functional strategy grid search returned identical results for all "
        f"param combos: {returns}"
    )
