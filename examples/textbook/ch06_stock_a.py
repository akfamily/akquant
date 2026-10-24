"""
第 6 章：A 股交易实战 (T+1 与 涨跌停).

本示例展示了如何处理中国 A 股市场特有的交易规则：
1. **T+1 交易制度**：当天买入的股票，第二个交易日才能卖出。
2. **涨跌停限制**：涨停板无法买入，跌停板无法卖出。
3. **最小交易单位**：买入必须是 100 股的整数倍 (手)。

策略逻辑：
- 每天开盘尝试买入
- 每天收盘尝试卖出
- 观察 T+1 限制如何阻止当日卖出
"""

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar, Strategy


# 模拟数据生成 (包含涨跌停场景)
def generate_mock_data(length: int = 20) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-03", periods=length, freq="B")

    # 构造价格序列
    prices = np.full(length, 100.0)

    # 第 3 天：涨停 (假设涨停价为 110.0)
    prices[2] = 110.0

    # 第 5 天：跌停 (假设跌停价为 90.0)
    prices[4] = 90.0

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices + 2,
            "low": prices - 2,
            "close": prices,
            "volume": 100000,
            "symbol": "600000",
        }
    )

    # 手动设置涨跌停状态 (通过 extra 字段模拟，或者由引擎根据昨收自动判定)
    # 在真实回测中，AKQuant 会根据昨日收盘价自动计算涨跌停
    # 这里我们通过特定的价格行为来触发引擎的涨跌停逻辑
    # 注意：AKQuant 的涨跌停判定依赖于配置的 limit_up_price / limit_down_price
    # 或者通过 use_china_market() 自动启用规则

    return df


class TPlusOneStrategy(Strategy):
    """T+1 策略演示."""

    def on_bar(self, bar: Bar) -> None:
        """收到 Bar 事件的回调."""
        symbol = bar.symbol

        # 获取账户持仓详情
        # position.quantity: 总持仓
        # position.available: 可用持仓 (T+1 解锁后)
        pos = self.get_position(symbol)
        avail = self.get_available_position(symbol)

        self.log(f"当前持仓: 总={pos}, 可用={avail}, 价格={bar.close}")

        # 1. 尝试买入 (T+0)
        if pos == 0:
            self.log("尝试买入 100 股...")
            self.buy(symbol, 100)

        # 2. 尝试卖出 (T+1)
        # 注意：如果当天刚买入，avail 应该为 0，卖单会被拒绝或挂起
        elif pos > 0:
            if avail > 0:
                self.log(f"可用持仓 {avail} > 0，尝试卖出...")
                self.sell(symbol, avail)
            else:
                self.log("可用持仓为 0 (受 T+1 限制)，无法卖出！")


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 6 章示例策略...")

    # 启用 ChinaMarket 模式 (关键！)
    # 这会自动开启 T+1、印花税等规则
    # aq.set_context(market="cn_stock")

    result = aq.run_backtest(
        strategy=TPlusOneStrategy,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,  # 印花税 (仅卖出收取)
        t_plus_one=True,  # 显式开启 T+1
    )
