"""测试排序确定性的单元测试."""

import pandas as pd
from akquant import Strategy, run_backtest


class RecordOrderStrategy(Strategy):
    """记录 bar 接收顺序的策略."""

    def __init__(self) -> None:
        """初始化策略."""
        super().__init__()
        self.bar_order: list[str] = []

    def on_bar(self, bar) -> None:  # type: ignore[no-untyped-def]
        """记录收到的 bar symbol."""
        self.bar_order.append(bar.symbol)


def test_same_timestamp_symbol_order() -> None:
    """测试相同时间戳的不同 symbol 排序确定性."""
    # 创建两个标的在相同时间点的数据
    dates = pd.date_range("2020-01-02 10:00", periods=1, freq="h")

    df_aaa = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        },
        index=dates,
    )

    df_bbb = pd.DataFrame(
        {
            "open": [20.0],
            "high": [21.0],
            "low": [19.0],
            "close": [20.5],
            "volume": [2000],
        },
        index=dates,
    )

    # 测试1: AAA 先于 BBB
    strategy1 = RecordOrderStrategy()
    data_map_1 = {"AAA": df_aaa, "BBB": df_bbb}
    run_backtest(
        data=data_map_1,
        strategy=strategy1,
        initial_cash=100000,
        show_progress=False,
    )

    # 测试2: BBB 先于 AAA
    strategy2 = RecordOrderStrategy()
    data_map_2 = {"BBB": df_bbb, "AAA": df_aaa}
    run_backtest(
        data=data_map_2,
        strategy=strategy2,
        initial_cash=100000,
        show_progress=False,
    )

    # 验证：无论插入顺序如何，回调顺序都应该相同（按 symbol 字典序）
    print(f"测试1收到 bar 顺序: {strategy1.bar_order}")
    print(f"测试2收到 bar 顺序: {strategy2.bar_order}")

    assert strategy1.bar_order == strategy2.bar_order, (
        f"不同插入顺序导致不同的回调顺序: "
        f"{strategy1.bar_order} vs {strategy2.bar_order}"
    )

    assert strategy1.bar_order == ["AAA", "BBB"], (
        f"预期按字典序 ['AAA', 'BBB']，实际是 {strategy1.bar_order}"
    )

    print("✓ 测试通过：相同时间戳的不同 symbol 按字典序排序")


if __name__ == "__main__":
    test_same_timestamp_symbol_order()
    print("\n排序确定性测试通过！")
