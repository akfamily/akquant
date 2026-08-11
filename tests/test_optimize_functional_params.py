"""Grid search parameter injection for function-style strategies."""

from typing import Any, List, Tuple, cast

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


_DEFAULT_SHORT_WINDOW = 5
_DEFAULT_LONG_WINDOW = 20

# 记录策略实际观测到的 (short_window, long_window)。用默认值做哨兵：
# 网格中不含默认值，因此一旦记录里出现默认值，就说明参数被静默丢弃了。
_OBSERVED_PARAMS: List[Tuple[int, int]] = []


def _initialize(ctx: Any) -> None:
    """Read swept params off ctx and register the warmup period."""
    ctx.long_window = getattr(ctx, "long_window", _DEFAULT_LONG_WINDOW)
    ctx.warmup_period = ctx.long_window + 1


def _on_bar(ctx: Any, bar: Bar) -> None:
    """Trade a dual-MA crossover using the swept parameters."""
    short_window = getattr(ctx, "short_window", _DEFAULT_SHORT_WINDOW)
    observed = (int(short_window), int(ctx.long_window))
    if observed not in _OBSERVED_PARAMS:
        _OBSERVED_PARAMS.append(observed)
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
    _OBSERVED_PARAMS.clear()
    results = aq.run_grid_search(
        strategy=cast(Any, _on_bar),
        data=_make_df(),
        param_grid={"short_window": [3, 10], "long_window": [15, 60]},
        initialize=_initialize,
        initial_cash=100_000,
        max_workers=1,
    )
    assert isinstance(results, pd.DataFrame)
    returns = [round(float(x), 6) for x in results["total_return_pct"].tolist()]
    assert len(returns) == 4
    assert len(set(returns)) > 1, (
        "functional strategy grid search returned identical results for all "
        f"param combos: {returns}"
    )
    # 两个键都必须真正到达策略：只注入 long_window 也能让上面的断言通过，
    # 所以这里直接核对策略观测到的参数对集合。
    assert set(_OBSERVED_PARAMS) == {(3, 15), (3, 60), (10, 15), (10, 60)}, (
        f"strategy observed unexpected param pairs: {sorted(_OBSERVED_PARAMS)}"
    )


def test_walk_forward_injects_best_params_into_functional_strategy() -> None:
    """The out-of-sample leg must run with the selected params, not the defaults."""
    _OBSERVED_PARAMS.clear()
    aq.run_walk_forward(
        strategy=cast(Any, _on_bar),
        data=_make_df(),
        param_grid={"short_window": [3, 10], "long_window": [15, 60]},
        train_period=100,
        test_period=50,
        initialize=_initialize,
        initial_cash=100_000,
        max_workers=1,
    )
    assert _OBSERVED_PARAMS, "strategy never ran during walk forward"
    # 样本外回测复用同一个 _on_bar，因此若 best_params 未注入，
    # 记录中会出现默认参数对 (5, 20)。
    defaults = (_DEFAULT_SHORT_WINDOW, _DEFAULT_LONG_WINDOW)
    assert defaults not in _OBSERVED_PARAMS, (
        "walk forward ran a leg with the strategy defaults instead of the "
        f"selected params: {sorted(_OBSERVED_PARAMS)}"
    )
