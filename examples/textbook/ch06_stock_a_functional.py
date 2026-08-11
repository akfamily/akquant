"""
第 6 章：A 股交易实战 (T+1 与 涨跌停)（函数式写法）.

本示例展示了如何处理中国 A 股市场特有的交易规则：
1. **T+1 交易制度**：当天买入的股票，第二个交易日才能卖出。
2. **涨跌停限制**：涨停板无法买入，跌停板无法卖出。
3. **最小交易单位**：买入必须是 100 股的整数倍 (手)。

策略逻辑：
- 每天开盘尝试买入
- 每天收盘尝试卖出
- 观察 T+1 限制如何阻止当日卖出

本示例与 ch06_stock_a.py 的交易逻辑、参数、数据源完全一致，
只把策略入口从类风格换成函数式：

1. Strategy.on_bar 方法 -> 模块级 on_bar(ctx, bar) 函数
2. strategy=TPlusOneStrategy -> strategy=on_bar

注意：类风格的 TPlusOneStrategy 没有 __init__、也没有任何实例状态，
所以函数式版只需要一个 on_bar，不必为了对称而写一个空的 initialize。

T+1、涨跌停、最小交易单位这些规则由引擎按配置强制执行，与策略写成类还是
函数无关：函数式策略同样受完整交易规则约束。两份示例的回测统计输出应完全
一致，这说明函数式与类风格在引擎层等价。
"""

from typing import Any

import akquant as aq
import numpy as np
import pandas as pd
from akquant import Bar


# 模拟数据生成 (包含涨跌停场景)
def generate_mock_data(length: int = 20) -> pd.DataFrame:
    """生成模拟数据."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=length, freq="D")

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


def on_bar(ctx: Any, bar: Bar) -> None:
    """收到 Bar 事件的回调，等价于类风格的 on_bar；self 全部换成 ctx."""
    symbol = bar.symbol

    # 获取账户持仓详情
    # position.quantity: 总持仓
    # position.available: 可用持仓 (T+1 解锁后)
    pos = ctx.get_position(symbol)
    avail = ctx.get_available_position(symbol)

    ctx.log(f"当前持仓: 总={pos}, 可用={avail}, 价格={bar.close}")

    # 1. 尝试买入 (T+0)
    if pos == 0:
        ctx.log("尝试买入 100 股...")
        ctx.buy(symbol, 100)

    # 2. 尝试卖出 (T+1)
    # 注意：如果当天刚买入，avail 应该为 0，卖单会被拒绝或挂起
    elif pos > 0:
        if avail > 0:
            ctx.log(f"可用持仓 {avail} > 0，尝试卖出...")
            ctx.sell(symbol, avail)
        else:
            ctx.log("可用持仓为 0 (受 T+1 限制)，无法卖出！")


if __name__ == "__main__":
    df = generate_mock_data()

    print("开始运行第 6 章示例策略...")

    # 启用 ChinaMarket 模式 (关键！)
    # 这会自动开启 T+1、印花税等规则
    # aq.set_context(market="cn_stock")

    result = aq.run_backtest(
        strategy=on_bar,
        data=df,
        initial_cash=100_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.001,  # 印花税 (仅卖出收取)
        t_plus_one=True,  # 显式开启 T+1
    )
