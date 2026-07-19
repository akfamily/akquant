"""回归测试：run_walk_forward 不应误转发仅 grid_search 专用的 kwargs.

修复前：run_walk_forward 里 `backtest_kwargs = kwargs.copy()` 会把 max_workers
等键一并带入 run_backtest 调用；run_backtest 无对应形参，落入
strategy_kwargs 后被 strict_strategy_params 校验拒绝，抛出
`TypeError: Unknown strategy constructor parameter(s): max_workers`.
"""

import akquant
import numpy as np
import pandas as pd
from akquant import IntParam


class InlineParamStrategy(akquant.Strategy):
    """带内联参数声明的简单策略, 模拟 param_grid 优化对象."""

    dummy = IntParam(0)

    def on_bar(self, bar: akquant.Bar) -> None:
        """No-op: 仅用于验证 kwargs 转发路径, 不关心交易逻辑."""
        return


def _build_data(n: int, symbol: str) -> pd.DataFrame:
    """构造一段小型分钟级合成数据, 供 walk-forward 切片使用."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    returns = rng.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": np.full(n, 1000.0),
            "symbol": symbol,
        }
    )


def test_run_walk_forward_does_not_leak_grid_search_only_kwargs_to_backtest() -> None:
    """max_workers=1 应仅被 run_grid_search 消费, 不应泄漏到样本外 run_backtest."""
    symbol = "WFO_KWARGS_LEAK"
    data = _build_data(n=24, symbol=symbol)

    results = akquant.run_walk_forward(
        strategy=InlineParamStrategy,
        param_grid={"dummy": [1, 2]},
        data=data,
        train_period=10,
        test_period=5,
        initial_cash=100_000.0,
        max_workers=1,
        show_progress=False,
    )

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "equity" in results.columns
    assert "train_start" in results.columns
    assert "train_end" in results.columns
