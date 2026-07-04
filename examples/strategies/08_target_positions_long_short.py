"""
高级目标仓位示例 (Advanced Target Positions Example).

=====================================================

本示例展示如何使用 `rebalance_positions()` 统一表达多标的正负目标仓位，
并通过 `get_last_target_positions_plan()` 查看最近一次调仓计划。

教学目标 (Learning Objectives):
1. 使用 `rebalance_positions()` 同时表达多头与空头目标。
2. 观察 `reduce-first` 语义下的同周期调仓行为。
3. 使用 `get_last_target_positions_plan()` 解释最近一次调仓决策。

本示例使用合成数据，不依赖外部数据源。
"""

from collections import defaultdict

import akquant as aq
import pandas as pd
from akquant import Bar, Strategy
from akquant.config import RiskConfig


def _build_symbol_df(
    symbol: str, timestamps: list[pd.Timestamp], closes: list[float]
) -> pd.DataFrame:
    rows = []
    for ts, close in zip(timestamps, closes):
        rows.append(
            {
                "date": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 10000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


class TargetPositionsDemoStrategy(Strategy):
    """演示正负目标仓位与调仓计划输出."""

    def __init__(self) -> None:
        """初始化分组同步状态和演示步骤."""
        super().__init__()
        self.pending: dict[int, set[str]] = defaultdict(set)
        self.step = 0
        self.symbols = ["AAA", "BBB"]

    def on_bar(self, bar: Bar) -> None:
        """在同一时间戳收齐两只标的后提交目标仓位调仓."""
        bucket = self.pending[bar.timestamp]
        bucket.add(bar.symbol)
        if len(bucket) < len(self.symbols):
            return
        self.pending.pop(bar.timestamp, None)

        if self.step == 0:
            self.rebalance_positions(
                {"AAA": 100.0},
                liquidate_unmentioned=True,
            )
        elif self.step == 1:
            self.rebalance_positions(
                {"AAA": -100.0, "BBB": 50.0},
                liquidate_unmentioned=True,
                allow_short=True,
                missing_price_mode="fail",
            )

        plan = self.get_last_target_positions_plan()
        if plan:
            print(
                f"[{bar.timestamp_iso}] status={plan.get('status')} "
                f"reduce={plan.get('reduce_legs')} "
                f"increase={plan.get('increase_legs')} "
                f"submitted={plan.get('submitted_legs')} "
                f"skipped={plan.get('skipped_legs')}"
            )
        self.step += 1


if __name__ == "__main__":
    timestamps = [
        pd.Timestamp("2023-01-02 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-03 10:00:00", tz="Asia/Shanghai"),
        pd.Timestamp("2023-01-04 10:00:00", tz="Asia/Shanghai"),
    ]
    data_map = {
        "AAA": _build_symbol_df("AAA", timestamps, [10.0, 10.0, 10.0]),
        "BBB": _build_symbol_df("BBB", timestamps, [10.0, 10.0, 10.0]),
    }

    result = aq.run_backtest(
        data=data_map,
        strategy=TargetPositionsDemoStrategy,
        symbols=["AAA", "BBB"],
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=0.0,
        lot_size=1,
        fill_policy={"price_basis": "close", "temporal": "same_cycle"},
        risk_config=RiskConfig(account_mode="margin", enable_short_sell=True),
        show_progress=False,
    )

    print("\n=== Backtest Result ===")
    print(result)
    print("\n=== Final Positions ===")
    print(result.positions.tail(1))
