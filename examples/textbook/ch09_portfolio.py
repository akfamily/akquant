"""
第 9 章：基金与资产配置 (Funds & Asset Allocation).

本示例展示了经典的 **股债平衡策略 (60/40 Portfolio)**。
这是资产配置中最基础也最有效的策略之一。

核心逻辑：
1.  **资产选择**：
    *   权益类 (Stock): 沪深300 ETF (高波动，高收益)
    *   固收类 (Bond): 国债 ETF (低波动，稳健收益)
2.  **目标配比**：60% 股票 + 40% 债券。
3.  **再平衡 (Rebalancing)**：
    *   定期（如每季度）检查持仓比例。
    *   如果股票涨多了（占比 > 60%），卖出股票，买入债券。
    *   如果股票跌多了（占比 < 60%），卖出债券，买入股票。

金融学原理：
通过持有相关性低（甚至负相关）的资产，可以在不显著降低预期收益的情况下，大幅降低组合波动率（马科维茨有效前沿）。
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


# 模拟数据生成 (股债双牛/股债跷跷板)
def generate_portfolio_data(length: int = 500) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=length, freq="D")

    # 股票：波动大，长期向上但有回撤
    stock_returns = np.random.normal(0.0005, 0.02, length)  # 日均 0.05%, 波动 2%
    stock_prices = 4.0 * np.cumprod(1 + stock_returns)

    # 债券：波动小，稳健向上
    bond_returns = np.random.normal(0.0001, 0.002, length)  # 日均 0.01%, 波动 0.2%
    bond_prices = 100.0 * np.cumprod(1 + bond_returns)

    # 构造 DataFrame
    df_stock = pd.DataFrame(
        {
            "date": dates,
            "open": stock_prices,
            "high": stock_prices * 1.01,
            "low": stock_prices * 0.99,
            "close": stock_prices,
            "volume": 1000000,
            "symbol": "510300",  # Stock ETF
        }
    )

    df_bond = pd.DataFrame(
        {
            "date": dates,
            "open": bond_prices,
            "high": bond_prices * 1.001,
            "low": bond_prices * 0.999,
            "close": bond_prices,
            "volume": 100000,
            "symbol": "511010",  # Bond ETF
        }
    )

    return pd.concat([df_stock, df_bond])


class RebalanceStrategy(Strategy):
    """股债平衡策略."""

    def on_start(self) -> None:
        """策略启动回调."""
        self.stock_symbol = "510300"
        self.bond_symbol = "511010"
        self.target_stock_weight = 0.6
        self.rebalance_interval = 20  # 每 20 个交易日 (约1个月) 再平衡一次
        self.days_counter = 0

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        # 仅在股票的 Bar 触发逻辑 (避免同一天触发两次)
        if bar.symbol != self.stock_symbol:
            return

        self.days_counter += 1

        # 初始建仓 或 触发再平衡
        if self.days_counter == 1 or self.days_counter % self.rebalance_interval == 0:
            self.rebalance()

    def rebalance(self) -> None:
        """执行再平衡."""
        # 获取当前总资产 (现金 + 持仓市值)
        # 注意：这里简化处理，假设当前时刻已获取到所有资产的最新价格
        # 在实盘中可能需要先查询所有持仓市值
        total_value = self.equity

        if total_value <= 0:
            return

        self.log(f"执行再平衡... 总资产: {total_value:.2f}")

        # 计算目标市值
        target_stock_val = total_value * self.target_stock_weight
        target_bond_val = total_value * (1 - self.target_stock_weight)

        # 调整仓位
        # order_target_value 会自动计算买卖数量
        self.order_target_value(self.stock_symbol, target_stock_val)
        self.order_target_value(self.bond_symbol, target_bond_val)


if __name__ == "__main__":
    df = generate_portfolio_data()

    print("开始运行第 9 章 股债平衡策略示例...")

    result = aq.run_backtest(
        strategy=RebalanceStrategy,
        data=df,
        initial_cash=1_000_000,
        commission_rate=0.0001,  # ETF 低佣金
    )

    # 打印最终结果
    metrics = result.metrics_df
    print("\n回测指标:")
    print(metrics.loc[["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]])
