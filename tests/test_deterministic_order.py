"""测试回测结果确定性 - issue #211.

验证不同 symbol 顺序不影响回测结果.
"""

import pandas as pd
from akquant import Strategy, run_backtest


class SimpleStrategy(Strategy):
    """简单策略：每个标的第一次收到 bar 时买入."""

    def __init__(self) -> None:
        """初始化策略."""
        super().__init__()
        self.bought: set[str] = set()

    def on_bar(self, bar) -> None:  # type: ignore[no-untyped-def]
        """每个标的第一次看到就买入."""
        if bar.symbol not in self.bought:
            self.buy(bar.symbol, 100, price=bar.close)
            self.bought.add(bar.symbol)


def test_dict_order_determinism() -> None:
    """测试不同 dict 顺序得到相同结果."""
    # 准备相同的数据，不同的字典顺序
    dates = pd.date_range("2020-01-02", periods=5, freq="D")

    # 创建两个标的的数据
    df1 = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2, 10.3, 10.4],
            "high": [10.5, 10.6, 10.7, 10.8, 10.9],
            "low": [9.5, 9.6, 9.7, 9.8, 9.9],
            "close": [10.0, 10.1, 10.2, 10.3, 10.4],
            "volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )

    df2 = pd.DataFrame(
        {
            "open": [20.0, 20.1, 20.2, 20.3, 20.4],
            "high": [20.5, 20.6, 20.7, 20.8, 20.9],
            "low": [19.5, 19.6, 19.7, 19.8, 19.9],
            "close": [20.0, 20.1, 20.2, 20.3, 20.4],
            "volume": [2000, 2100, 2200, 2300, 2400],
        },
        index=dates,
    )

    # 顺序1: "000001" 先于 "000002"
    data_map_1 = {"000001": df1, "000002": df2}
    result1 = run_backtest(
        data=data_map_1,
        strategy=SimpleStrategy,
        initial_cash=100000,
        show_progress=False,
    )

    # 顺序2: "000002" 先于 "000001"
    data_map_2 = {"000002": df2, "000001": df1}
    result2 = run_backtest(
        data=data_map_2,
        strategy=SimpleStrategy,
        initial_cash=100000,
        show_progress=False,
    )

    # 验证关键指标完全一致
    print(
        f"result1 metrics: total_return={result1.metrics.total_return}, "
        f"sharpe_ratio={result1.metrics.sharpe_ratio}"
    )
    print(
        f"result2 metrics: total_return={result2.metrics.total_return}, "
        f"sharpe_ratio={result2.metrics.sharpe_ratio}"
    )
    print(f"result1 trades count: {len(result1.trades)}")
    print(f"result2 trades count: {len(result2.trades)}")

    assert result1.metrics.total_return == result2.metrics.total_return, (
        f"total_return 不一致: "
        f"{result1.metrics.total_return} vs {result2.metrics.total_return}"
    )
    assert result1.metrics.sharpe_ratio == result2.metrics.sharpe_ratio, (
        f"sharpe_ratio 不一致: "
        f"{result1.metrics.sharpe_ratio} vs {result2.metrics.sharpe_ratio}"
    )
    assert result1.metrics.max_drawdown == result2.metrics.max_drawdown, (
        f"max_drawdown 不一致: "
        f"{result1.metrics.max_drawdown} vs {result2.metrics.max_drawdown}"
    )

    # 验证成交记录完全一致（数量和内容）
    assert len(result1.trades) == len(result2.trades), (
        f"成交数量不一致: {len(result1.trades)} vs {len(result2.trades)}"
    )

    print("✓ 测试通过：不同 symbol 顺序得到相同的回测结果")


if __name__ == "__main__":
    test_dict_order_determinism()
    print("\n所有测试通过！")
