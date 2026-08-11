"""
第 9 章：基金与资产配置 (Funds & Asset Allocation)（函数式写法）.

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

本示例与 ch09_portfolio.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. on_start 里的状态 (stock_symbol / bond_symbol / target_stock_weight /
   rebalance_interval / days_counter) -> ctx 属性（在 initialize 中赋值）
2. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
3. 自定义方法 Strategy.rebalance -> 模块级 rebalance(ctx) 普通函数（由 on_bar 调用）
4. strategy=RebalanceStrategy -> strategy=on_bar + initialize=initialize

跨 Bar 的状态 (days_counter) 直接挂在 ctx 上读写，作用与类风格的 self 属性一致。
两份示例的回测统计输出应完全一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


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


def initialize(ctx: Any) -> None:
    """
    初始化股债平衡策略状态.

    类风格版把这些状态放在 on_start 里；函数式两处都能写，但推荐 initialize：
    on_start 在快照恢复 (Warm Start) 后会再次触发，跨 Bar 状态写在那里会被重置，
    覆盖掉已从快照恢复的 days_counter（再平衡节奏会从头重数）。
    initialize 只在策略构造时执行一次，不受恢复影响。

    注意：类风格版没有设置 warmup_period，本示例也不设置——
    预热期会改变被跳过的 Bar 数量，凭空加上会让两版输出不再一致。
    """
    ctx.stock_symbol = "510300"
    ctx.bond_symbol = "511010"
    ctx.target_stock_weight = 0.6
    ctx.rebalance_interval = 20  # 每 20 个交易日 (约1个月) 再平衡一次
    ctx.days_counter = 0


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    # 仅在股票的 Bar 触发逻辑 (避免同一天触发两次)
    if bar.symbol != ctx.stock_symbol:
        return

    ctx.days_counter += 1

    # 初始建仓 或 触发再平衡
    if ctx.days_counter == 1 or ctx.days_counter % ctx.rebalance_interval == 0:
        rebalance(ctx)


def rebalance(ctx: Any) -> None:
    """执行再平衡；等价于类风格的自定义方法 Strategy.rebalance."""
    # 获取当前总资产 (现金 + 持仓市值)
    # 注意：这里简化处理，假设当前时刻已获取到所有资产的最新价格
    # 在实盘中可能需要先查询所有持仓市值
    total_value = ctx.equity

    if total_value <= 0:
        return

    ctx.log(f"执行再平衡... 总资产: {total_value:.2f}")

    # 计算目标市值
    target_stock_val = total_value * ctx.target_stock_weight
    target_bond_val = total_value * (1 - ctx.target_stock_weight)

    # 调整仓位
    # order_target_value 会自动计算买卖数量
    ctx.order_target_value(ctx.stock_symbol, target_stock_val)
    ctx.order_target_value(ctx.bond_symbol, target_bond_val)


if __name__ == "__main__":
    df = generate_portfolio_data()

    print("开始运行第 9 章 股债平衡策略示例...")

    result = aq.run_backtest(
        strategy=on_bar,
        initialize=initialize,
        data=df,
        initial_cash=1_000_000,
        commission_rate=0.0001,  # ETF 低佣金
    )

    # 打印最终结果
    metrics = result.metrics_df
    print("\n回测指标:")
    print(metrics.loc[["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]])
