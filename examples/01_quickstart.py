from typing import Any

import akquant as aq
import akshare as ak
import pandas as pd
from akquant import Bar, Strategy
from akquant.config import BacktestConfig, RiskConfig, StrategyConfig

df_1 = ak.stock_zh_a_daily(
    symbol="sh600000", start_date="20000101", end_date="20261231"
)
df_1["symbol"] = "600000"
df_2 = ak.stock_zh_a_daily(
    symbol="sh600004", start_date="20000101", end_date="20261231"
)
df_2["symbol"] = "600004"
df_3 = ak.stock_zh_a_daily(
    symbol="sh600006", start_date="20000101", end_date="20261231"
)
df_3["symbol"] = "600006"
df = {"600000": df_1, "600004": df_2, "600006": df_3}


class MyStrategy(Strategy):
    """
    Example strategy for testing broker execution.

    This strategy buys on the first bar and holds for 100 bars or until 10% profit.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize strategy state."""
        super().__init__()
        self.bars_held: dict[str, int] = {}
        self.entry_prices: dict[str, float] = {}

    def on_bar(self, bar: Bar) -> None:
        """
        Handle bar data event.

        :param bar: The current bar data
        """
        symbol = bar.symbol
        pos = self.get_position(symbol)

        # 维护持仓计数
        if pos > 0:
            if symbol not in self.bars_held:
                self.bars_held[symbol] = 0
            self.bars_held[symbol] += 1
        else:
            # 如果没有持仓，清理状态
            if symbol in self.bars_held:
                del self.bars_held[symbol]
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]

        # 交易逻辑
        if pos == 0:
            # 简单示例：每个标的买入 33% 仓位
            self.order_target_percent(target_percent=0.33, symbol=symbol)
            # 初始化计数器 (虽然会在下个 bar 的 pos>0 分支中自增，但这里先占位)
            self.bars_held[symbol] = 0
            self.entry_prices[symbol] = bar.close

        elif pos > 0:
            entry_price = self.entry_prices.get(symbol, bar.close)
            current_bars_held = self.bars_held.get(symbol, 0)

            # 计算收益率
            pnl_pct = (bar.close - entry_price) / entry_price

            # 止盈条件：收益率 >= 10%
            if pnl_pct >= 0.10:
                self.sell(symbol, pos)
                print(
                    f"Take Profit Triggered for {symbol}: Entry={entry_price}, "
                    f"Current={bar.close}, PnL={pnl_pct:.2%}"
                )
            # 持仓时间条件：持有满 100 个 Bar
            elif current_bars_held >= 100:
                self.close_position()


# 配置风险参数：safety_margin
risk_config = RiskConfig(safety_margin=0.0001)
strategy_config = StrategyConfig(risk=risk_config)
backtest_config = BacktestConfig(
    strategy_config=strategy_config,
)

result = aq.run_backtest(
    strategy=MyStrategy,
    data=df,
    initial_cash=5000000,
    commission_rate=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    min_commission=5.0,
    lot_size=1,
    execution_mode=aq.ExecutionMode.NextAverage,
    config=backtest_config,
    start_time="20250101",
    end_time="20250105",
    symbol=["600000", "600004", "600006"],
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print(result)
print(result.orders_df)
# print(result.equity_curve)
